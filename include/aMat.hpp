/**
 * @file aMat.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Duc Tran     hantran@cs.utah.edu
 * @author Milinda Fernando milinda@cs.utah.edu
 *
 * @brief A sparse matrix class for adaptive finite elements. 
 * 
 * @version 0.1
 * @date 2018-11-07
 * 
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 * 
 */

#include <Eigen/Dense>

#include <mpi.h>
#include <omp.h>

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>

#include <vector>
#include <fstream>
#include <algorithm>

#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include <iostream>

// alternatives for vectorization, alignment = cache line
#ifdef VECTORIZED_AVX512
    #define SIMD_LENGTH (512/(sizeof(DT) * 8)) // length of vector register = 512 bytes
    #define ALIGNMENT 64
#elif VECTORIZED_AVX256
    #define SIMD_LENGTH (256/(sizeof(DT) * 8)) // length of vector register = 256 bytes
    #define ALIGNMENT 64
#elif VECTORIZED_OPENMP
    #define SIMD_LENGTH (512/(sizeof(DT) * 8)) // could be deleted, openMP automatically detect length of vector register
    #define ALIGNMENT 16
#elif VECTORIZED_OPENMP_PADDING
    #define SIMD_LENGTH (512/(sizeof(DT) * 8)) // could be deleted, openMP automatically detect length of vector register
    #define ALIGNMENT 64
#else
    // this is not used since we do not align memory
    #define ALIGNMENT 16
#endif

// number of nonzero terms in the matrix (used in matrix-base and block Jacobi preconditioning)
// e.g. in a structure mesh, eight of 20-node quadratic elements sharing the node
// --> 81 nodes (3 dofs/node) constitue one row of the stiffness matrix
#define NNZ (81*3)

// weight factor for penalty method in applying BC
#define PENALTY_FACTOR 100

namespace par {

    /**@brief method of applying boundary conditions */
    enum class BC_METH { BC_IMATRIX, BC_PENALTY };

    /**@brief types of error used in functions of aMat class */
    enum class Error { SUCCESS,
                       INDEX_OUT_OF_BOUNDS,
                       UNKNOWN_ELEMENT_TYPE,
                       UNKNOWN_ELEMENT_STATUS,
                       NULL_L2G_MAP,
                       GHOST_NODE_NOT_FOUND,
                       UNKNOWN_MAT_TYPE,
                       WRONG_COMMUNICATION,
                       UNKNOWN_DOF_TYPE,
                       UNKNOWN_CONSTRAINT,
                       UNKNOWN_BC_METH,
                       NOT_IMPLEMENTED,
                       GLOBAL_DOF_ID_NOT_FOUND };

    /**@brief  */
    enum class DOF_TYPE { FREE, PRESCRIBED, UNDEFINED };

    /**@brief timers for profiling */
    enum class PROFILER { MATVEC = 0, MATVEC_MUL, MATVEC_ACC,
                        PETSC_ASS, PETSC_MATVEC, PETSC_KfcUc,
                        LAST };



    //==============================================================================================================
    // Class AsyncExchangeCtx is downloaded from Dendro-5.0 with permission from the author (Milinda Fernando)
    // Dendro-5.0 is written by Milinda Fernando and Hari Sundar;
    class AsyncExchangeCtx {
    private:
        /** pointer to the variable which perform the ghost exchange */
        void* m_uiBuffer;

        /** pointer to the send buffer*/
        void* m_uiSendBuf;

        /** pointer to the receive buffer*/
        void* m_uiRecvBuf;

        /** list of request*/
        std::vector<MPI_Request*> m_uiRequests;

    public:
        /**@brief creates an async ghost exchange context*/
        AsyncExchangeCtx( const void* var ) {
            m_uiBuffer  = (void*)var;
            m_uiSendBuf = nullptr;
            m_uiRecvBuf = nullptr;
            m_uiRequests.clear();
        }

        /**@brief allocates send buffer for ghost exchange */
        void allocateSendBuffer(size_t bytes) { m_uiSendBuf = malloc(bytes); }

        /**@brief allocates recv buffer for ghost exchange */
        void allocateRecvBuffer(size_t bytes) { m_uiRecvBuf = malloc(bytes); }

        /**@brief allocates send buffer for ghost exchange */
        void deAllocateSendBuffer() {
            free( m_uiSendBuf );
            m_uiSendBuf = nullptr;
        }

        /**@brief allocates recv buffer for ghost exchange */
        void deAllocateRecvBuffer() {
            free( m_uiRecvBuf );
            m_uiRecvBuf = nullptr;
        }

        /**@brief */
        void* getSendBuffer() { return m_uiSendBuf; }

        /**@brief */
        void* getRecvBuffer() { return m_uiRecvBuf; }

        /**@brief */
        const void* getBuffer() { return m_uiBuffer; }

        /**@brief */
        std::vector<MPI_Request*>& getRequestList() { return m_uiRequests; }

        /**@brief */
        bool operator== (AsyncExchangeCtx other) const { return( m_uiBuffer == other.m_uiBuffer ); }

        ~AsyncExchangeCtx() {
            /*for(unsigned int i=0;i<m_uiRequests.size();i++)
              {
              delete m_uiRequests[i];
              m_uiRequests[i]=nullptr;
              }
              m_uiRequests.clear();*/
        }
    }; // class AsyncExchangeCtx



    //==============================================================================================================
    // Class MatRecord
    //      DT => type of data stored in matrix (eg: double). LI => size of local index.
    template <typename Dt, typename Li>
    class MatRecord {
    private:
        unsigned int m_uiRank;
        Li m_uiRowId;
        Li m_uiColId;
        Dt m_dtVal;
    public:
        MatRecord(){
            m_uiRank = 0;
            m_uiRowId = 0;
            m_uiColId = 0;
            m_dtVal = 0;
        }
        MatRecord(unsigned int rank, Li rowId, Li colId, Dt val){
            m_uiRank = rank;
            m_uiRowId = rowId;
            m_uiColId = colId;
            m_dtVal = val;
        }

        unsigned int getRank() const { return m_uiRank; }
        Li getRowId() const { return m_uiRowId; }
        Li getColId() const { return m_uiColId; }
        Dt getVal()   const { return m_dtVal; }

        void setRank(  unsigned int rank ) { m_uiRank = rank; }
        void setRowId( Li rowId ) { m_uiRowId = rowId; }
        void setColId( Li colId ) { m_uiColId = colId; }
        void setVal(   Dt val ) {   m_dtVal = val; }

        bool operator == (MatRecord const &other) const {
            return ((m_uiRank == other.getRank())&&(m_uiRowId == other.getRowId())&&(m_uiColId == other.getColId()));
        }

        bool operator < (MatRecord const &other) const {
            if (m_uiRank < other.getRank()) return true;
            else if (m_uiRank == other.getRank()) {
                if (m_uiRowId < other.getRowId()) return true;
                else if (m_uiRowId == other.getRowId()) {
                    if (m_uiColId < other.getColId()) return true;
                    else return false;
                }
                else return false;
            }
            else {
                return false;
            }
        }

        bool operator <= (MatRecord const &other) const { return (((*this) < other) || (*this) == other); }

        ~MatRecord() {}
    }; // class MatRecord



    //==============================================================================================================
    // DT => type of data stored in matrix (eg: double). LI => size of local index.
    // class is used to define MPI_Datatype for MatRecord
    template <typename dt, typename li>
    class MPI_datatype_matrecord{
    public:
        static MPI_Datatype value(){
            static bool first = true;
            static MPI_Datatype mpiDatatype;
            if (first){
                first=false;
                MPI_Type_contiguous(sizeof(MatRecord<dt,li>),MPI_BYTE,&mpiDatatype);
                MPI_Type_commit(&mpiDatatype);
            }
            return mpiDatatype;
        }
    }; // class MPI_datatype_matrecord



    //==============================================================================================================
    // Class ConstrainedRecord
    //      DT => type of data stored in matrix (eg: double)
    template <typename DT, typename GI>
    class ConstrainedRecord {
        private:
        GI dofId; // global dof Id that is constrained (i.e. Dirichlet BC)
        DT preVal;// prescribed valua for the constrained dof

        public:
        ConstrainedRecord() {
            dofId = 0;
            preVal = 0.0;
        }

        GI get_dofId() const { return dofId; }
        DT get_preVal() const {return preVal; }

        void set_dofId(GI id) { dofId = id; }
        void set_preVal(DT value) { preVal = value; }

        bool operator == (ConstrainedRecord const &other) const {
            return (dofId == other.get_dofId());
        }
        bool operator < (ConstrainedRecord const &other) const {
            if (dofId < other.get_dofId()) return true;
            else return false;
        }
        bool operator <= (ConstrainedRecord const &other) const {
            return (((*this) < other) || ((*this) == other));
        }

        ~ConstrainedRecord() {}
    };// class ConstrainedDof



    //==============================================================================================================
    // Class profiler_t is downloaded from Dendro-5.0 with permission from the author (Milinda Fernando)
    // Dendro-5.0 is written by Milinda Fernando and Hari Sundar;
    class profiler_t {
        public:
            long double	seconds;  // openmp wall time
            long long p_flpops; // papi floating point operations
            long double snap; // snap shot of the cumilative time.
            long long num_calls; // number of times the timer stop function called.

        protected:
            long double	  _pri_seconds;  // openmp wall time
            long long _pri_p_flpops; // papi floating point operations

        public:
            profiler_t (){
                seconds  = 0.0;   // openmp wall time
                p_flpops =   0;   // papi floating point operations
                snap =0.0;
                num_calls=0;

                _pri_seconds  = 0.0;
                _pri_p_flpops =   0;
            }
            virtual ~profiler_t () {}

            void start(){
                _pri_seconds = omp_get_wtime();
                flops_papi();
            }
            void stop(){
                seconds -= _pri_seconds;
                p_flpops -= _pri_p_flpops;
                snap-=_pri_seconds;

                _pri_seconds = omp_get_wtime();
                flops_papi();

                seconds  += _pri_seconds;
                p_flpops += _pri_p_flpops;
                snap     += _pri_seconds;
                //num_calls++;
            }
            void clear(){
                seconds  = 0.0;
                p_flpops =   0;
                snap=0.0;
                num_calls=0;

                _pri_seconds  = 0.0;
                _pri_p_flpops =   0;
            }
            void snapreset(){
                snap=0.0;
                num_calls=0;
            }

        private:
            void  flops_papi(){
                #ifdef HAVE_PAPI
                    int 		retval;
                    float rtime, ptime, mflops;
                    retval  = PAPI_flops(&rtime, &ptime, &_pri_p_flpops, &mflops);
                    // assert (retval == PAPI_OK);
                #else
                    _pri_p_flpops =   0;
                #endif
            }
    };



    //==============================================================================================================
    // Class aMat
    // DT => type of data stored in matrix (eg: double). GI => size of global index. LI => size of local index
    template <typename DT, typename GI, typename LI>
    class aMat {

        #ifdef AMAT_PROFILER
        public:
        /**@brief list of profilers for timing different tasks */
        std::vector<profiler_t> timing_aMat = std::vector<profiler_t>(static_cast<int>(PROFILER::LAST));

        /**@brief reset variables for timing*/
        void reset_profile_counters(){
            for( unsigned int i = 0; i < timing_aMat.size(); i++){
                //printf("i= %d\n",i);
                timing_aMat[i].clear();
                timing_aMat[i].start();
            }
        }

        /**@brief print out timing */
        void profile_dump(std::ostream& s){

            long double t_rank, t_max;

            // get the time of task
            t_rank = timing_aMat[static_cast<int>(PROFILER::MATVEC)].seconds;

            // get the max time among all ranks
            MPI_Reduce(&t_rank, &t_max, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, m_comm);

            // display the time
            if (m_uiRank == 0){
                s << "time of matvec: = " << t_max << "\n";
            }
        }
        #endif

        public:
        typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> EigenMat;
        typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> EigenVec;

        protected:
        /**@brief communicator used within aMat */
        MPI_Comm m_comm;

        /**@brief my rank */
        unsigned int m_uiRank;

        /**@brief total number of ranks */
        unsigned int m_uiSize;

        /**@brief method of applying Dirichlet BC */
        BC_METH m_BcMeth;

        /**@brief (local) number of DoFs owned by rank */
        LI m_uiNumDofs;

        /**@brief (global) number of DoFs owned by all ranks */
        GI m_ulNumDofsGlobal;

        /**@brief start of global ID of owned dofs */
        GI m_ulGlobalDofStart;

        /**@brief end of global ID of owned dofs (# owned dofs = m_ulGlobalDofEnd - m_ulGlobalDofStart + 1) */
        GI m_ulGlobalDofEnd;

        /**@brief total dofs inclulding ghost */
        LI m_uiNumDofsTotal;

        /**@brief (local) number of elements owned by rank */
        LI m_uiNumElems;

        /**@brief map from element to global DoF: m_ulpMap[eid][local_id]  = global_id */
        GI** m_ulpMap;

        /**@brief number of dofs per element */
        const LI* m_uiDofsPerElem;

        /**@brief map of constrained DOFs: 0 = free dof; 1 = constrained dof */
        unsigned int** m_uipBdrMap;

        /**@brief map of values prescribed on constrained DOFs */
        DT** m_dtPresValMap;

        /**@brief list of constrained DOFs owned by my rank */
        std::vector<GI> ownedConstrainedDofs;

        /**@brief list of values prescribed at constrained DOFs owned by my rank */
        std::vector<DT> ownedPrescribedValues;

        /**@brief list of free DOFs owned by my rank */
        std::vector<GI> ownedFreeDofs;

        /**@brief KfcUc = Kfc * Uc, used to apply bc for rhs */
        Vec KfcUcVec;

        //**@brief penalty number */
        DT m_dtTraceK;

        public:
        /**@brief constructor to initialize variables of aMat */
        aMat( BC_METH bcType = BC_METH::BC_IMATRIX);

        /**@brief destructor of aMat */
        ~aMat();

        /**@brief set communicator for aMat */
        Error set_comm(MPI_Comm comm){
            m_comm = comm;
            MPI_Comm_rank(comm, (int*)&m_uiRank);
            MPI_Comm_size(comm, (int*)&m_uiSize);
            return Error::SUCCESS;
        }

        /**@brief set mapping from element local node to global node */
        virtual Error set_map( const LI           n_elements_on_rank,
                                const LI* const * element_to_rank_map,
                                const LI        * dofs_per_element,
                                const LI          n_all_dofs_on_rank, // Note: includes ghost dofs
                                const GI        * rank_to_global_map,
                                const GI          owned_global_dof_range_begin,
                                const GI          owned_global_dof_range_end,
                                const GI          n_global_dofs ) = 0;

        /**@brief update map when cracks created */
        virtual Error update_map(const LI* new_to_old_rank_map,
                                const LI old_n_all_dofs_on_rank,
                                const GI* old_rank_to_global_map,
                                const LI n_elements_on_rank,
                                const LI* const * element_to_rank_map,
                                const LI* dofs_per_element,
                                const LI n_all_dofs_on_rank,
                                const GI* rank_to_global_map,
                                const GI owned_global_dof_range_begin,
                                const GI owned_global_dof_range_end,
                                const GI n_global_dofs) = 0;

        /**@brief assemble element matrix to global matrix */
        virtual Error set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim ) = 0;

        /**@brief assembly load vector to global vector: same for both matrix-free and matrix-based */
        Error petsc_set_element_vec( Vec vec, LI eid, EigenVec e_vec, LI block_i, InsertMode mode = ADD_VALUES );

        /**@brief set boundary data, numConstraints is the global number of constrains */
        virtual Error set_bdr_map(GI* constrainedDofs, DT* prescribedValues, LI numConstraints) = 0;

        /**@brief apply Dirichlet bc: matrix-free --> apply bc on rhs, matrix-based --> apply bc on rhs and matrix */
        virtual Error apply_bc( Vec rhs ) = 0;

        /**@brief invoke basic PETSc solver, "out" is solution vector */
        virtual Error petsc_solve( const Vec rhs, Vec out ) const = 0;

        /**@brief allocate memory for a PETSc vector "vec", initialized by alpha */
        Error petsc_create_vec( Vec &vec, PetscScalar alpha = 0.0 ) const;

        /**@brief free memory allocated for PETSc vector*/
        Error petsc_destroy_vec( Vec & vec ) const {
            VecDestroy( &vec );
            return Error::SUCCESS;
        }

        /**@brief write PETSc vector "vec" to filename "fvec" */
        Error petsc_dump_vec( Vec vec, const char* filename = nullptr ) const;

        /**@brief begin assembling the matrix  */
        virtual Error petsc_init_mat( MatAssemblyType mode ) const = 0;

        /**@brief complete assembling the matrix  */
        virtual Error petsc_finalize_mat( MatAssemblyType mode ) const = 0;

        /**@brief write global matrix to filename "fvec", matrix-free --> not yet implemented */
        virtual Error petsc_dump_mat( const char* filename = nullptr ) = 0;

    }; // class aMat



    //==============================================================================================================
    // class aMatFree derived from base class aMat
    // DT => type of data stored in matrix (eg: double). GI => size of global index. LI => size of local index
    template <typename DT, typename GI, typename LI>
    class aMatFree : public aMat<DT,GI,LI> {
        public:
        using typename aMat<DT,GI,LI>::EigenMat;
        using typename aMat<DT,GI,LI>::EigenVec;

        using aMat<DT,GI,LI>::m_comm;
        using aMat<DT,GI,LI>::m_uiRank;
        using aMat<DT,GI,LI>::m_uiSize;
        using aMat<DT,GI,LI>::m_BcMeth;

        using aMat<DT,GI,LI>::m_uiNumDofs;
        using aMat<DT,GI,LI>::m_ulNumDofsGlobal;
        using aMat<DT,GI,LI>::m_ulGlobalDofStart;
        using aMat<DT,GI,LI>::m_ulGlobalDofEnd;
        using aMat<DT,GI,LI>::m_uiNumDofsTotal;
        using aMat<DT,GI,LI>::m_uiNumElems;

        using aMat<DT,GI,LI>::m_ulpMap;
        using aMat<DT,GI,LI>::m_uiDofsPerElem;
        using aMat<DT,GI,LI>::m_uipBdrMap;
        using aMat<DT,GI,LI>::m_dtPresValMap;
        using aMat<DT,GI,LI>::ownedConstrainedDofs;
        using aMat<DT,GI,LI>::ownedPrescribedValues;
        using aMat<DT,GI,LI>::ownedFreeDofs;
        using aMat<DT,GI,LI>::KfcUcVec;
        using aMat<DT,GI,LI>::m_dtTraceK;

        #ifdef AMAT_PROFILER
        using aMat<DT,GI,LI>::timing_aMat;
        #endif

        protected:
        /**@brief storage of element matrices */
        std::vector< DT* >* m_epMat;

        /**@brief map from local DoF of element to local DoF: m_uipLocalMap[eid][element_node]  = local dof ID */
        //const LI* const*  m_uipLocalMap;
        LI** m_uipLocalMap;

        /**@brief map from local DoF to global DoF */
        //const GI* m_ulpLocal2Global;
        GI* m_ulpLocal2Global;

        /**@brief number of DoFs owned by each rank, NOT include ghost DoFs */
        std::vector<LI> m_uivLocalDofCounts;

        /**@brief number of elements owned by each rank */
        std::vector<LI> m_uivLocalElementCounts;

        /**@brief exclusive scan of (local) number of DoFs */
        std::vector<GI> m_ulvLocalDofScan;

        /**@brief exclusive scan of (local) number of elements */
        std::vector<GI> m_ulvLocalElementScan;

        /**@brief number of ghost DoFs owned by "pre" processes (whose ranks are smaller than m_uiRank) */
        LI m_uiNumPreGhostDofs;

        /**@brief total number of ghost DoFs owned by "post" processes (whose ranks are larger than m_uiRank) */
        LI m_uiNumPostGhostDofs;

        /**@brief number of DoFs sent to each process (size = m_uiSize) */
        std::vector<LI> m_uivSendDofCounts;

        /**@brief offsets (i.e. exclusive scan) of m_uiSendNodeCounts */
        std::vector<LI> m_uivSendDofOffset;

        /**@brief local DoF IDs to be sent (size = total number of nodes to be sent */
        std::vector<LI> m_uivSendDofIds;

        /**@brief process IDs that I send data to */
        std::vector<unsigned int> m_uivSendRankIds;

        /**@brief number of DoFs to be received from each process (size = m_uiSize) */
        std::vector<LI> m_uivRecvDofCounts;

        /**@brief offsets (i.e. exclusive scan) of m_uiRecvNodeCounts */
        std::vector<LI> m_uivRecvDofOffset;

        /**@brief process IDs that I receive data from */
        std::vector<unsigned int> m_uivRecvRankIds;

        /**@brief local dof-ID starting of pre-ghost nodes, always = 0 */
        LI m_uiDofPreGhostBegin;

        /**@brief local dof-ID ending of pre-ghost nodes */
        LI m_uiDofPreGhostEnd;

        /**@brief local dof-ID starting of nodes owned by me */
        LI m_uiDofLocalBegin;

        /**@brief local dof-ID ending of nodes owned by me */
        LI m_uiDofLocalEnd;

        /**@brief local dof-ID starting of post-ghost nodes */
        LI m_uiDofPostGhostBegin;

        /**@brief local dof-ID ending of post-ghost nodes */
        LI m_uiDofPostGhostEnd;

        /**@brief total number of dofs including ghost dofs and dofs owned by me */
        //LI m_uiNumDofsTotal;

        /**@brief max number of DoFs per element */
        LI m_uiMaxDofsPerBlock;

        /**@brief max number of pads to be added to the end of ve */
        LI m_uiMaxNumPads;

        /**@brief MPI communication tag*/
        int m_iCommTag;

        /**@brief ghost exchange context*/
        std::vector<AsyncExchangeCtx> m_vAsyncCtx;

        /**@brief matrix record for block jacobi matrix*/
        std::vector<MatRecord<DT,LI>> m_vMatRec;

        /**@brief used to save constrained dofs when applying BCs in matvec */
        DT* Uc;

        /**@brief number of owned constraints */
        LI n_owned_constraints;

        #ifdef HYBRID_PARALLEL
            /**@brief max number of omp threads */
            unsigned int m_uiNumThreads;
            /**@brief elemental vectors used in matvec */
            DT** m_veBufs;
            DT** m_ueBufs;
        #else
            /**@brief elemental vectors used in matvec */
            DT* ve;
            DT* ue;
        #endif


        public:
        /**@brief constructor to initialize variables of aMatFree */
        aMatFree(BC_METH bcType = BC_METH::BC_IMATRIX);

        /**@brief destructor of aMatFree */
        ~aMatFree();

        /**@brief overidden version of aMat::set_map */
        Error set_map( const LI       n_elements_on_rank,
                    const LI* const * element_to_rank_map,
                    const LI        * dofs_per_element,
                    const LI          n_all_dofs_on_rank, // Note: includes ghost dofs
                    const GI        * rank_to_global_map,
                    const GI          owned_global_dof_range_begin,
                    const GI          owned_global_dof_range_end,
                    const GI          n_global_dofs );

        /**@brief overidden version of aMat::update_map */
        Error update_map(const LI* new_to_old_rank_map,
                        const LI old_n_all_dofs_on_rank,
                        const GI* old_rank_to_global_map,
                        const LI n_elements_on_rank,
                        const LI* const * element_to_rank_map,
                        const LI* dofs_per_element,
                        const LI n_all_dofs_on_rank,
                        const GI* rank_to_global_map,
                        const GI owned_global_dof_range_begin,
                        const GI owned_global_dof_range_end,
                        const GI n_global_dofs);

        /**@brief overidden version of aMatFree */
        Error set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim );

        /**@brief overidden version of aMat::set_bdr_map */
        Error set_bdr_map(GI* constrainedDofs, DT* prescribedValues, LI numConstraints);

        /**@brief overidden version of aMat::apply_bc */
        Error apply_bc( Vec rhs ){
            apply_bc_rhs(rhs);
            return Error::SUCCESS;
        }

        /**@brief overidden version of aMat::petsc_solve */
        Error petsc_solve( const Vec rhs, Vec out ) const;

        /**@brief */
        Error petsc_init_mat( MatAssemblyType mode ) const {
            printf("petsc_init_mat is not applied for matrix-free\n");
            return Error::NOT_IMPLEMENTED;
        }

        /**@brief */
        Error petsc_finalize_mat( MatAssemblyType mode ) const {
            printf("petsc_finalize_mat is not applied for matrix-free\n");
            return Error::NOT_IMPLEMENTED;
        }

        /**@brief display global matrix to filename using matrix free approach */
        Error petsc_dump_mat( const char* filename );

        /**@brief pointer function points to MatMult_mt */
        std::function<PetscErrorCode(Mat,Vec,Vec)>* get_MatMult_func(){

            std::function<PetscErrorCode(Mat,Vec,Vec)>* f= new std::function<PetscErrorCode(Mat, Vec, Vec)>();

            (*f) = [this](Mat A, Vec u, Vec v){
                this->MatMult_mf(A, u, v);
                return 0;
            };
            return f;
        }

        /**@brief pointer function points to MatGetDiagonal_mf */
        std::function<PetscErrorCode(Mat,Vec)>* get_MatGetDiagonal_func(){

            std::function<PetscErrorCode(Mat,Vec)>* f= new std::function<PetscErrorCode(Mat, Vec)>();

            (*f) = [this](Mat A, Vec d){
                this->MatGetDiagonal_mf(A, d);
                return 0;
            };
            return f;
        }

        /**@brief pointer function points to MatGetDiagonalBlock_mf */
        std::function<PetscErrorCode(Mat, Mat*)>* get_MatGetDiagonalBlock_func(){

            std::function<PetscErrorCode(Mat,Mat*)>* f= new std::function<PetscErrorCode(Mat, Mat*)>();

            (*f) = [this](Mat A, Mat* a){
                this->MatGetDiagonalBlock_mf(A, a);
                return 0;
            };
            return f;
        }

        protected:
        /**@brief apply Dirichlet BCs to the rhs vector */
        Error apply_bc_rhs( Vec rhs );

        /**@brief build scatter-gather map (used for communication) and local-to-local map (used for matvec) */
        Error buildScatterMap();

        /**@brief return true if DoF "enid" of element "eid" is owned by this rank, false otherwise */
        bool is_local_node(LI eid, LI enid) const {
            const LI nid = m_uipLocalMap[eid][enid];
            if( nid >= m_uiDofLocalBegin && nid < m_uiDofLocalEnd ) {
                return true;
            } else {
                return false;
            }
        }

        /**@brief allocate memory for "vec", size includes ghost DoFs if isGhosted=true, initialized by alpha */
        Error create_vec( DT* &vec, bool isGhosted = false, DT alpha = (DT)0.0 ) const;

        /**@brief free memory allocated for vec and set vec to null */
        Error destroy_vec(DT* &vec);

        /**@brief copy local (size = m_uiNumDofs) to corresponding positions of gVec (size = m_uiNumDofsTotal) */
        Error local_to_ghost(DT*  gVec, const DT* local) const;

        /**@brief copy gVec (size = m_uiNumDofsTotal) to local (size = m_uiNumDofs) */
        Error ghost_to_local(DT* local, const DT* gVec) const;

        /**@brief matrix-free version of set_element_matrix: copy element matrix and store in m_pMat */
        Error copy_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim );

        /**@brief get diagonal terms of structure matrix by accumulating diagonal of element matrices */
        Error mat_get_diagonal(DT* diag, bool isGhosted = false);

        /**@brief get diagonal terms with ghosted vector diag */
        Error mat_get_diagonal_ghosted(DT* diag);

        /**@brief return the rank who owns gId */
        unsigned int globalId_2_rank(GI gId) const;

        /**@brief get diagonal block matrix (sparse matrix) */
        Error mat_get_diagonal_block(std::vector<MatRecord<DT,LI>> &diag_blk);

        /**@brief get max number of DoF per block (over all elements)*/
        // this function is not needed since max_dof_per_block is computed in set_map and it is unchanged later on
        //Error get_max_dof_per_block();

        /**@brief allocate memory for ue and ve used for elemental matrix-vector multiplication */
        Error allocate_ue_ve();

        /**@brief begin: owned DoFs send, ghost DoFs receive, called before matvec() */
        Error ghost_receive_begin(DT* vec);

        /**@brief end: ghost DoFs receive, called before matvec() */
        Error ghost_receive_end(DT* vec);

        /**@brief begin: ghost DoFs send, owned DoFs receive and accumulate to current data, called after matvec() */
        Error ghost_send_begin(DT* vec);

        /**@brief end: ghost DoFs send, owned DoFs receive and accumulate to current data, called after matvec() */
        Error ghost_send_end(DT* vec);

        /**@brief v = K * u (K is not assembled, but directly using elemental K_e's).  v (the result) must be allocated by the caller.
         * @param[in] isGhosted = true, if v and u are of size including ghost DoFs
         * @param[in] isGhosted = false, if v and u are of size NOT including ghost DoFs
         * */
        Error matvec(DT* v, const DT* u, bool isGhosted = false);

        /**@brief v = K * u; v and u are of size including ghost DoFs. */
        #ifdef HYBRID_PARALLEL
            Error matvec_ghosted_OMP(DT* v, DT* u);
        #else
            Error matvec_ghosted_noOMP(DT* v, DT* u);
        #endif

        /**@brief matrix-free version of MatMult of PETSc */
        PetscErrorCode MatMult_mf(Mat A, Vec u, Vec v);

        /**@brief matrix-free version of MatGetDiagonal of PETSc */
        PetscErrorCode MatGetDiagonal_mf( Mat A, Vec d );

        /**@brief matrix-free version of MatGetDiagonalBlock of PETSc */
        PetscErrorCode MatGetDiagonalBlock_mf( Mat A, Mat* a );

        /**@brief apply Dirichlet BCs to diagonal vector used for Jacobi preconditioner */
        Error apply_bc_diagonal(Vec rhs);

        /**@brief apply Dirichlet BCs to block diagonal matrix */
        Error apply_bc_blkdiag(Mat* blkdiagMat);

        /**@brief allocate an aligned memory */
        DT* create_aligned_array(unsigned int alignment, unsigned int length);

        /**@brief deallocate an aligned memory */
        void delete_algined_array(DT* array);

        /**@brief compute number of paddings inserted at the end of each column of elemental block matrix */
        /* LI get_column_paddings(size_t alignment, LI column_length){
            if ((column_length % (alignment/sizeof(DT))) != 0){
                return (alignment/sizeof(DT)) - (column_length % (alignment/sizeof(DT)));
            } else {
                return 0;
            }
        } */

    }; //class aMatFree



    //==============================================================================================================
    // class aMatBased derived from base class aMat
    // DT => type of data stored in matrix (eg: double). GI => size of global index. LI => size of local index
    template <typename DT, typename GI, typename LI>
    class aMatBased : public aMat<DT,GI,LI> {
        public:
        using typename aMat<DT,GI,LI>::EigenMat;
        using typename aMat<DT,GI,LI>::EigenVec;

        using aMat<DT,GI,LI>::m_comm;
        using aMat<DT,GI,LI>::m_uiRank;
        using aMat<DT,GI,LI>::m_uiSize;
        using aMat<DT,GI,LI>::m_BcMeth;

        using aMat<DT,GI,LI>::m_uiNumDofs;
        using aMat<DT,GI,LI>::m_ulNumDofsGlobal;
        using aMat<DT,GI,LI>::m_ulGlobalDofStart;
        using aMat<DT,GI,LI>::m_ulGlobalDofEnd;
        using aMat<DT,GI,LI>::m_uiNumDofsTotal;
        using aMat<DT,GI,LI>::m_uiNumElems;

        using aMat<DT,GI,LI>::m_ulpMap;
        using aMat<DT,GI,LI>::m_uiDofsPerElem;
        using aMat<DT,GI,LI>::m_uipBdrMap;
        using aMat<DT,GI,LI>::m_dtPresValMap;
        using aMat<DT,GI,LI>::ownedConstrainedDofs;
        using aMat<DT,GI,LI>::ownedPrescribedValues;
        using aMat<DT,GI,LI>::ownedFreeDofs;
        using aMat<DT,GI,LI>::KfcUcVec;
        using aMat<DT,GI,LI>::m_dtTraceK;

        #ifdef AMAT_PROFILER
        using aMat<DT,GI,LI>::timing_aMat;
        #endif

        protected:
        /**@brief assembled stiffness matrix */
        Mat m_pMat;

        public:
        /**@brief constructor to initialize variables of aMatBased */
        aMatBased( BC_METH bcType = BC_METH::BC_IMATRIX);

        /**@brief destructor of aMatBased */
        ~aMatBased();

        /**@brief overriden version for matrix-based approach */
        Error set_map( const LI       n_elements_on_rank,
                    const LI* const * element_to_rank_map,
                    const LI        * dofs_per_element,
                    const LI          n_all_dofs_on_rank, // Note: includes ghost dofs
                    const GI        * rank_to_global_map,
                    const GI          owned_global_dof_range_begin,
                    const GI          owned_global_dof_range_end,
                    const GI          n_global_dofs );

        /**@brief overridden version for matrix-based approach */
        Error update_map(const LI*        new_to_old_rank_map,
                        const LI          old_n_all_dofs_on_rank,
                        const GI*         old_rank_to_global_map,
                        const LI          n_elements_on_rank,
                        const LI* const * element_to_rank_map,
                        const LI*         dofs_per_element,
                        const LI          n_all_dofs_on_rank,
                        const GI*         rank_to_global_map,
                        const GI          owned_global_dof_range_begin,
                        const GI          owned_global_dof_range_end,
                        const GI          n_global_dofs);

        /**@brief overidden version of aMatFree */
        Error set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim );

        /**@brief overidden version of aMat::set_bdr_map */
        Error set_bdr_map(GI* constrainedDofs, DT* prescribedValues, LI numConstraints);

        /**@brief overidden version of aMat::apply_bc */
        Error apply_bc( Vec rhs ){
            apply_bc_rhs(rhs);
            apply_bc_mat();
            return Error::SUCCESS;
        }

        /**@brief overidden version of aMat::petsc_solve */
        Error petsc_solve( const Vec rhs, Vec out ) const;

        /**@brief overidden version */
        par::Error petsc_init_mat( MatAssemblyType mode ) const {
            MatAssemblyBegin( m_pMat, mode );
            return Error::SUCCESS;
        }

        /**@brief overidden version */
        Error petsc_finalize_mat( MatAssemblyType mode ) const {
            MatAssemblyEnd( m_pMat, mode );
            return Error::SUCCESS;
        }

        /**@brief overidden version */
        // Note: can't be 'const' because may call matvec which may need MPI data to be stored...
        Error petsc_dump_mat( const char* filename = nullptr );

        protected:
        /**@brief matrix-based version of set_element_matrix */
        Error petsc_set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, InsertMode = ADD_VALUES );

        /**@brief apply Dirichlet BCs by modifying the rhs vector, also used for diagonal vector in Jacobi precondition*/
        Error apply_bc_rhs( Vec rhs );

        /**@brief apply Dirichlet BCs by modifying the matrix "m_pMat" */
        Error apply_bc_mat();

    }; // class aMatBased



    //==============================================================================================================
    // context for aMat
    template <typename DT, typename GI, typename LI>
    struct aMatCTX {
        par::aMatFree<DT,GI,LI> * aMatPtr;
    };


    // matrix shell to use aMat::MatMult_mf
    template<typename DT,typename GI, typename LI>
    PetscErrorCode aMat_matvec( Mat A, Vec u, Vec v )
    {
        aMatCTX<DT,GI, LI> * pCtx;
        MatShellGetContext( A, &pCtx );

        par::aMatFree<DT, GI, LI> * pLap = pCtx->aMatPtr;
        std::function<PetscErrorCode( Mat, Vec, Vec )>* f = pLap->get_MatMult_func();
        (*f)( A, u , v );
        delete f;
        return 0;
    }


    // matrix shell to use aMat::MatGetDiagonal_mf
    template<typename DT,typename GI, typename LI>
    PetscErrorCode aMat_matgetdiagonal( Mat A, Vec d )
    {
        aMatCTX<DT,GI,LI> * pCtx;
        MatShellGetContext(A, &pCtx);

        par::aMatFree<DT,GI,LI> * pLap = pCtx->aMatPtr;
        std::function<PetscErrorCode(Mat, Vec)>* f = pLap->get_MatGetDiagonal_func();
        (*f)(A, d);
        delete f;
        return 0;
    }


    // matrix shell to use aMat::MatGetDiagonalBlock_mf
    template<typename DT,typename GI, typename LI>
    PetscErrorCode aMat_matgetdiagonalblock( Mat A, Mat* a )
    {
        aMatCTX<DT,GI,LI> * pCtx;
        MatShellGetContext(A, &pCtx);

        par::aMatFree<DT,GI,LI> * pLap = pCtx->aMatPtr;
        std::function<PetscErrorCode(Mat, Mat*)>* f = pLap->get_MatGetDiagonalBlock_func();
        (*f)(A, a);
        delete f;
        return 0;
    }



    //==============================================================================================================
    // aMat constructor
    template <typename DT,typename GI, typename LI>
    aMat<DT,GI,LI>::aMat( BC_METH bcType ) {
        m_BcMeth           = bcType;        // method of apply bc
        m_uiNumDofs        = 0;             // number of owned dofs
        m_ulNumDofsGlobal  = 0;             // total number of global dofs of all ranks
        m_uiNumElems       = 0;             // number of owned elements
        m_uiNumDofsTotal   = 0;             // total number of owned dofs + ghost dofs
        m_ulGlobalDofStart = 0;
        m_ulGlobalDofEnd   = 0;

        m_ulpMap           = nullptr;       // element-to-global map
        m_uiDofsPerElem    = nullptr;       // number of dofs per element
        m_comm             = MPI_COMM_NULL; // communication of aMat

        m_uipBdrMap = nullptr;
        m_dtPresValMap = nullptr;
    } // aMat::constructor


    template <typename DT,typename GI, typename LI>
    aMat<DT,GI,LI>::~aMat() {
        // free memory allocated for global map
        if (m_ulpMap != nullptr){
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                if (m_ulpMap[eid] != nullptr) {
                    delete [] m_ulpMap[eid];
                }
            }
            delete [] m_ulpMap;
        }

        // free memory allocated for m_uiBdrMap
        if (m_uipBdrMap != nullptr){
            for (LI eid = 0; eid < m_uiNumElems; eid++) {
                if (m_uipBdrMap[eid] != nullptr) delete [] m_uipBdrMap[eid];
            }
            delete [] m_uipBdrMap;
        }

        // free memory allocated for m_dtPresValMap
        if (m_dtPresValMap != nullptr) {
            for (LI eid = 0; eid < m_uiNumElems; eid++) {
                if (m_dtPresValMap[eid] != nullptr) delete [] m_dtPresValMap[eid];
            }
            delete [] m_dtPresValMap;
        }
    } // aMat::destructor


    // 16.Dec.2019: change type of e_vec from DT* to EigenVec to be consistent with element matrix
    // Note: for force vector, there is no block_j (as in stiffness matrix)
    template <typename DT, typename GI, typename LI>
    Error aMat<DT,GI,LI>::petsc_set_element_vec(Vec vec, LI eid, EigenVec e_vec, LI block_i, InsertMode mode){

        LI num_dofs_per_block = e_vec.size();
        assert(e_vec.size() == e_vec.rows()); // since EigenVec is defined as matrix with 1 column

        PetscScalar value;
        PetscInt rowId;

        for (LI r = 0; r < num_dofs_per_block; ++r) {

            // this ONLY WORKS with assumption that all blocks have the same number of dofs (that is true for R-XFEM ?)
            rowId = m_ulpMap[eid][block_i * num_dofs_per_block + r];
            value = e_vec(r);
            VecSetValue(vec, rowId, value, mode);
        }

        return Error::SUCCESS;
    } // aMat::petsc_set_element_vec


    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_create_vec(Vec &vec, PetscScalar alpha) const {
        VecCreate(m_comm, &vec);
        if (m_uiSize>1) {
            VecSetType(vec,VECMPI);
            VecSetSizes(vec, m_uiNumDofs, PETSC_DECIDE);
            VecSet(vec, alpha);
        } else {
            VecSetType(vec,VECSEQ);
            VecSetSizes(vec, m_uiNumDofs, PETSC_DECIDE);
            VecSet(vec, alpha);
        }
        return Error::SUCCESS;
    } // aMat::petsc_create_vec


    template <typename DT, typename GI, typename LI>
    Error aMat<DT,GI,LI>::petsc_dump_vec( Vec vec, const char* filename ) const {
        if( filename == nullptr ) {
            VecView( vec, PETSC_VIEWER_STDOUT_WORLD );
        }
        else {
            PetscViewer viewer;
            // write to ASCII file
            PetscViewerASCIIOpen( m_comm, filename, &viewer );
            // write to file readable by Matlab (filename must be name.m)
            //PetscViewerPushFormat( viewer, PETSC_VIEWER_ASCII_MATLAB ); // comment when what to write in normal ASCII text
            VecView( vec, viewer );
            PetscViewerDestroy( &viewer );
        }
        return Error::SUCCESS;
    } // aMat::petsc_dump_vec



    //==============================================================================================================
    template <typename DT,typename GI, typename LI>
    aMatFree<DT,GI,LI>::aMatFree( BC_METH bcType ) : aMat<DT,GI,LI>(bcType) {
        m_epMat    = nullptr;   // element matrices (Eigen matrix), used in matrix-free
        m_iCommTag = 0;         // tag for sends & receives used in matvec and mat_get_diagonal_block_seq

        m_uipLocalMap = nullptr;

        Uc = nullptr;

        m_uiMaxDofsPerBlock = 0;

        #ifdef HYBRID_PARALLEL
            // (thread local) ve and ue
            m_veBufs = nullptr;
            m_ueBufs = nullptr;
            // max of omp threads
            m_uiNumThreads = omp_get_max_threads();
        #else
            ue = nullptr;
            ve = nullptr;
        #endif
    }// aMatFree::constructor


    template <typename DT,typename GI, typename LI>
    aMatFree<DT,GI,LI>::~aMatFree() {
        // free memory allocated for element-to-local map (allocated in set_map)
        if (m_uipLocalMap != nullptr){
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                if (m_uipLocalMap[eid] != nullptr){
                    delete [] m_uipLocalMap[eid];
                }
            }
            delete [] m_uipLocalMap;
        }

        // free memory allocated for rank-to-global map (allocated in buildScatterMap)
        if (m_ulpLocal2Global != nullptr){
            delete [] m_ulpLocal2Global;
        }

        // free memory allocated for m_epMat storing elemental matrices
        if (m_epMat != nullptr){
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                for (LI bid = 0; bid < m_epMat[eid].size(); bid++){
                    if (m_epMat[eid][bid] != nullptr){
                        // delete the block matrix bid
                        delete_algined_array(m_epMat[eid][bid]);
                    }
                }
                // clear the content of vector of DT* and resize to 0
                m_epMat[eid].clear();
            }
            // delete the array created by new in set_map
            delete [] m_epMat;

            // free memory allocated for Uc
            if (n_owned_constraints > 0) {
                delete [] Uc;
            }
        }

        // free memory allocated for storing elemental vectors ue and ve
        #ifdef HYBRID_PARALLEL
            for (unsigned int tid = 0; tid < m_uiNumThreads; tid++){
                if (m_ueBufs[tid] != nullptr) delete_algined_array(m_ueBufs[tid]);
                if (m_veBufs[tid] != nullptr) delete_algined_array(m_veBufs[tid]);
            }
            if (m_ueBufs != nullptr) {
                free(m_ueBufs);
            }
            if (m_veBufs != nullptr) {
                free(m_veBufs);
            }
        #else
            if (ue != nullptr){
                delete_algined_array(ue);
            }
            if (ve != nullptr) {
                delete_algined_array(ve);
            }
        #endif
    } // aMatFree::destructor


    template <typename DT,typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::set_map( const LI       n_elements_on_rank,
                                    const LI * const * element_to_rank_map,
                                    const LI         * dofs_per_element,
                                    const LI           n_all_dofs_on_rank,
                                    const GI         * rank_to_global_map,
                                    const GI           owned_global_dof_range_begin,
                                    const GI           owned_global_dof_range_end,
                                    const GI           n_global_dofs ){
        // number of owned elements
        m_uiNumElems = n_elements_on_rank;

        // number of owned dofs
        m_uiNumDofs = owned_global_dof_range_end - owned_global_dof_range_begin + 1;

        // number of dofs of ALL ranks, currently this is only used in aMatFree::petsc_dump_mat()
        m_ulNumDofsGlobal = n_global_dofs;

        // these are assertion in buildScatterMap
        m_ulGlobalDofStart = owned_global_dof_range_begin;
        m_ulGlobalDofEnd = owned_global_dof_range_end;
        m_uiNumDofsTotal = n_all_dofs_on_rank;

        // point to provided array giving number of dofs of each element
        m_uiDofsPerElem = dofs_per_element;

        // 2020.05.23 no longer use provided Local2Global map, will build in buildScatterMap
        //m_ulpLocal2Global = rank_to_global_map;

        // create global map based on provided local map and Local2Global
        m_ulpMap = new GI* [m_uiNumElems];
        for( LI eid = 0; eid < m_uiNumElems; eid++ ){
            m_ulpMap[eid] = new GI [m_uiDofsPerElem[eid]];
        }
        for( LI eid = 0; eid < m_uiNumElems; eid++ ){
            for( LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++ ){
                m_ulpMap[eid][nid] = rank_to_global_map[element_to_rank_map[eid][nid]];
            }
        }

        // 05.21.20: create local map that will be built in buildScatterMap
        m_uipLocalMap = new LI* [m_uiNumElems];
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            m_uipLocalMap[eid] = new LI [m_uiDofsPerElem[eid]];
        }

        // clear elemental matrices stored in m_epMat (if it is not empty)
        if( m_epMat != nullptr) {
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                for (LI bid = 0; bid < m_epMat[eid].size(); bid++){
                    if (m_epMat[eid][bid] != nullptr){
                        free(m_epMat[eid][bid]);
                    }
                }
                m_epMat[eid].clear();
            }
            delete [] m_epMat;
            m_epMat = nullptr;
        }

        // allocate m_epMat as an array with size equals number of owned elements
        // we do not know how many blocks and size of blocks for each element at this time
        m_epMat = new std::vector<DT*> [m_uiNumElems];

        // 2020.05.21: no longer use the provided localMap, will build it in buildScatterMap
        //m_uipLocalMap = element_to_rank_map;

        // build scatter map for communication before and after matvec
        // 05.21.20: also build m_uipLocalMap
        buildScatterMap();

        // compute the largest number of dofs per block, to be used for allocation of ue and ve...
        // ASSUME that initially every element has only one block, AND the size of block is unchanged during crack growth
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            if (m_uiMaxDofsPerBlock < m_uiDofsPerElem[eid]) m_uiMaxDofsPerBlock = m_uiDofsPerElem[eid];
        }

        // get number of pads added to ve (where ve = block_matrix * ue)
        #ifdef VECTORIZED_OPENMP_PADDING
            assert((ALIGNMENT % sizeof(DT)) == 0);
            if ((m_uiMaxDofsPerBlock % (ALIGNMENT/sizeof(DT))) != 0){
                m_uiMaxNumPads = (ALIGNMENT/sizeof(DT)) - (m_uiMaxDofsPerBlock % (ALIGNMENT/sizeof(DT)));
            } else {
                m_uiMaxNumPads = 0;
            }
        #endif

        // allocate memory for ue and ve used in elemental matrix-vector multiplication
        allocate_ue_ve();

        return Error::SUCCESS;
    } // aMatFree::set_map


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::update_map(const LI* new_to_old_rank_map,
                                    const LI old_n_all_dofs_on_rank,
                                    const GI* old_rank_to_global_map,
                                    const LI n_elements_on_rank,
                                    const LI* const * element_to_rank_map,
                                    const LI* dofs_per_element,
                                    const LI n_all_dofs_on_rank,
                                    const GI* rank_to_global_map,
                                    const GI owned_global_dof_range_begin,
                                    const GI owned_global_dof_range_end,
                                    const GI n_global_dofs) {

        // Number of owned elements should not be changed (extra dofs are enriched)
        assert(m_uiNumElems == n_elements_on_rank);

        // point to new provided array giving number of dofs of each element
        m_uiDofsPerElem = dofs_per_element;

        // 2020.05.23 no longer use provided Local2Global map, will build in buildScatterMap
        //m_ulpLocal2Global = rank_to_global_map;

        // delete the current global map in order to increase the size of 2nd dimension (i.e. number of dofs per element)
        if (m_ulpMap != nullptr){
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                delete [] m_ulpMap[eid];
            }
        }
        // reallocate according to the new number of dofs per element
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            m_ulpMap[eid] = new GI [m_uiDofsPerElem[eid]];
        }
        // new global map
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            for (LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++){
                m_ulpMap[eid][nid] = rank_to_global_map[element_to_rank_map[eid][nid]];
            }
        }

        // 2020.05.21: delete the current local map in order to increase the size of 2nd dimension (i.e. number of dofs per element)
        if (m_uipLocalMap != nullptr){
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                delete [] m_uipLocalMap[eid];
            }
        }
        // reallocate according to the new number of dofs per element
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            m_uipLocalMap[eid] = new LI [m_uiDofsPerElem[eid]];
        }
        // new local map will be re-build in buildScatterMap which will called below

        // update number of owned dofs
        m_uiNumDofs = owned_global_dof_range_end - owned_global_dof_range_begin + 1;

        // update total dofs of all ranks, currently is only used by aMatFree::petsc_dump_mat()
        m_ulNumDofsGlobal = n_global_dofs;

        /*unsigned long nl = m_uiNumDofs;
        unsigned long ng;
        MPI_Allreduce( &nl, &ng, 1, MPI_LONG, MPI_SUM, m_comm );
        assert( n_global_dofs == ng );*/

        // update variables for assertion in buildScatterMap
        m_ulGlobalDofStart = owned_global_dof_range_begin;
        m_ulGlobalDofEnd = owned_global_dof_range_end;
        m_uiNumDofsTotal = n_all_dofs_on_rank;

        // Note: currently we delete all old element matrices and need to add new element matrices again
        // Todo: only add newly formed blocks?
        if( m_epMat != nullptr) {
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                for (LI bid = 0; bid < m_epMat[eid].size(); bid++){
                    if (m_epMat[eid][bid] != nullptr){
                        free(m_epMat[eid][bid]);
                    }
                }
                m_epMat[eid].clear();
            }
            // we do not delete [] m_epMat because the number of elements do not change when map is updated
        }

        // 2020.05.21: no longer use the provided local Map, will build it in buildScatterMap
        //m_uipLocalMap = element_to_rank_map;

        // build scatter map
        buildScatterMap();

        // compute the largest number of dofs per elements, will be used for allocation ue and ve...
        //get_max_dof_per_elem();

        return Error::SUCCESS;
    } // aMatFree::update_map()


    template <typename DT,typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::buildScatterMap() {
        /* Assumptions: We assume that the global nodes are continuously partitioned across processors. */

        // save the total dofs (owned + ghost) provided in set_map for assertion
        // m_uiNumDofsTotal will be re-computed based on: m_ulpMap, m_uiNumDofs, m_uiNumElems
        const LI m_uiNumDofsTotal_received_in_setmap = m_uiNumDofsTotal;

        // save the global number of dofs (of all ranks) provided in set_map for assertion
        // m_ulNumDofsGlobal will be re-computed based on: m_uiNumDofs
        const GI m_ulNumDofsGlobal_received_in_setmap = m_ulNumDofsGlobal;

        if ( m_ulpMap == nullptr ) { return Error::NULL_L2G_MAP; }

        m_uivLocalDofCounts.clear();
        m_uivLocalElementCounts.clear();
        m_ulvLocalDofScan.clear();
        m_ulvLocalElementScan.clear();

        m_uivLocalDofCounts.resize(m_uiSize);
        m_uivLocalElementCounts.resize(m_uiSize);
        m_ulvLocalDofScan.resize(m_uiSize);
        m_ulvLocalElementScan.resize(m_uiSize);

        // gather local counts
        //MPI_Allgather(&m_uiNumDofs, 1, MPI_INT, &(*(m_uivLocalDofCounts.begin())), 1, MPI_INT, m_comm);
        //MPI_Allgather(&m_uiNumElems, 1, MPI_INT, &(*(m_uivLocalElementCounts.begin())), 1, MPI_INT, m_comm);
        MPI_Allgather(&m_uiNumDofs, 1, MPI_INT, m_uivLocalDofCounts.data(), 1, MPI_INT, m_comm);
        MPI_Allgather(&m_uiNumElems, 1, MPI_INT, m_uivLocalElementCounts.data(), 1, MPI_INT, m_comm);

        // scan local counts to determine owned-range:
        // range of global ID of owned dofs = [m_ulvLocalDofScan[m_uiRank], m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)
        m_ulvLocalDofScan[0] = 0;
        m_ulvLocalElementScan[0] = 0;
        for (unsigned int p = 1; p < m_uiSize; p++) {
            m_ulvLocalDofScan[p] = m_ulvLocalDofScan[p-1] + m_uivLocalDofCounts[p-1];
            m_ulvLocalElementScan[p] = m_ulvLocalElementScan[p-1] + m_uivLocalElementCounts[p-1];
        }

        // global number of dofs of all ranks
        m_ulNumDofsGlobal = m_ulvLocalDofScan[m_uiSize - 1] + m_uivLocalDofCounts[m_uiSize - 1];
        assert( m_ulNumDofsGlobal == m_ulNumDofsGlobal_received_in_setmap );

        // dofs are not owned by me: stored in pre or post lists
        std::vector<GI> preGhostGIds;
        std::vector<GI> postGhostGIds;
        for (LI eid = 0; eid < m_uiNumElems; eid++) {
            for (LI i = 0; i < m_uiDofsPerElem[eid]; i++) {
                // global ID
                const GI global_dof_id = m_ulpMap[eid][i];
                if (global_dof_id < m_ulvLocalDofScan[m_uiRank]) {
                    // dofs with global ID < owned-range --> pre-ghost dofs
                    assert( global_dof_id < m_ulGlobalDofStart ); // m_ulGlobalDofStart was passed in set_map
                    preGhostGIds.push_back( global_dof_id );
                } else if (global_dof_id >= (m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)){
                    // dofs with global ID > owned-range --> post-ghost dofs
                    // note: m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs - 1 = m_ulGlobalDofEnd
                    assert( global_dof_id > m_ulGlobalDofEnd ); // m_ulGlobalDofEnd was passed in set_map
                    postGhostGIds.push_back( global_dof_id );
                } else {
                    assert (( global_dof_id >= m_ulvLocalDofScan[m_uiRank] )
                            && ( global_dof_id < ( m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs )));
                    assert(( global_dof_id >= m_ulGlobalDofStart ) && ( global_dof_id <= m_ulGlobalDofEnd ));
                }
            }
        }

        // sort in ascending order
        std::sort(preGhostGIds.begin(), preGhostGIds.end());
        std::sort(postGhostGIds.begin(), postGhostGIds.end());

        // remove consecutive duplicates and erase all after .end()
        preGhostGIds.erase(std::unique(preGhostGIds.begin(), preGhostGIds.end()), preGhostGIds.end());
        postGhostGIds.erase(std::unique(postGhostGIds.begin(), postGhostGIds.end()), postGhostGIds.end());

        // number of pre and post ghost dofs
        m_uiNumPreGhostDofs = preGhostGIds.size();
        m_uiNumPostGhostDofs = postGhostGIds.size();

        // range of local ID of pre-ghost dofs = [0, m_uiDofPreGhostEnd)
        m_uiDofPreGhostBegin = 0;
        m_uiDofPreGhostEnd = m_uiNumPreGhostDofs;

        // range of local ID of owned dofs = [m_uiDofLocalBegin, m_uiDofLocalEnd)
        m_uiDofLocalBegin = m_uiDofPreGhostEnd;
        m_uiDofLocalEnd = m_uiDofLocalBegin + m_uiNumDofs;

        // range of local ID of post-ghost dofs = [m_uiDofPostGhostBegin, m_uiDofPostGhostEnd)
        m_uiDofPostGhostBegin = m_uiDofLocalEnd;
        m_uiDofPostGhostEnd = m_uiDofPostGhostBegin + m_uiNumPostGhostDofs;

        // total number of dofs including ghost dofs
        m_uiNumDofsTotal = m_uiNumDofs + m_uiNumPreGhostDofs + m_uiNumPostGhostDofs;

        // note: m_uiNumDofsTotal was passed in set_map --> assert if what we received is what we compute here
        assert(m_uiNumDofsTotal == m_uiNumDofsTotal_received_in_setmap);

        // determine owners of pre- and post-ghost dofs
        std::vector<unsigned int> preGhostOwner;
        std::vector<unsigned int> postGhostOwner;
        preGhostOwner.resize(m_uiNumPreGhostDofs);
        postGhostOwner.resize(m_uiNumPostGhostDofs);

        // pre-ghost
        unsigned int pcount = 0; // processor counter, start from 0
        LI gcount = 0; // counter of ghost dof
        while (gcount < m_uiNumPreGhostDofs) {
            // global ID of pre-ghost dof gcount
            GI global_dof_id = preGhostGIds[gcount];
            while ((pcount < m_uiRank) &&
                    (!((global_dof_id >= m_ulvLocalDofScan[pcount])
                    && (global_dof_id < (m_ulvLocalDofScan[pcount] + m_uivLocalDofCounts[pcount]))))) {
                // global_dof_id is not owned by pcount
                pcount++;
            }
            // check if global_dof_id is really in the range of global ID of dofs owned by pcount
            if (!((global_dof_id >= m_ulvLocalDofScan[pcount])
                && (global_dof_id < (m_ulvLocalDofScan[pcount] + m_uivLocalDofCounts[pcount])))) {
                std::cout << "m_uiRank: " << m_uiRank << " pre ghost gid : " << global_dof_id << " was not found in any processor" << std::endl;
                return Error::GHOST_NODE_NOT_FOUND;
            }
            preGhostOwner[gcount] = pcount;
            gcount++;
        }

        // post-ghost
        pcount = m_uiRank; // processor counter, start from my rank
        gcount = 0;
        while (gcount < m_uiNumPostGhostDofs) {
                // global ID of post-ghost dof gcount
                GI global_dof_id = postGhostGIds[gcount];
                while ((pcount < m_uiSize) &&
                        (!((global_dof_id >= m_ulvLocalDofScan[pcount])
                        && (global_dof_id < (m_ulvLocalDofScan[pcount] + m_uivLocalDofCounts[pcount]))))){
                    // global_dof_id is not owned by pcount
                    pcount++;
                }
                // check if global_dof_id is really in the range of global ID of dofs owned by pcount
                if (!((global_dof_id >= m_ulvLocalDofScan[pcount])
                    && (global_dof_id < (m_ulvLocalDofScan[pcount] + m_uivLocalDofCounts[pcount])))) {
                    std::cout << "m_uiRank: " << m_uiRank << " post ghost gid : " << global_dof_id << " was not found in any processor" << std::endl;
                    return Error::GHOST_NODE_NOT_FOUND;
                }
                postGhostOwner[gcount] = pcount;
                gcount++;
            }


        LI * sendCounts = new LI [m_uiSize];
        LI * recvCounts = new LI [m_uiSize];
        LI * sendOffset = new LI [m_uiSize];
        LI * recvOffset = new LI [m_uiSize];

        // Note: the send here is just for use in MPI_Alltoallv, it is NOT the send in communications between processors later
        for (unsigned int i = 0; i < m_uiSize; i++) {
            // many of these will be zero, only non zero for processors that own my ghost nodes
            sendCounts[i] = 0;
        }

        // count number of pre-ghost dofs to corresponding owners
        for (LI i = 0; i < m_uiNumPreGhostDofs; i++) {
            // preGhostOwner[i] = rank who owns the ith pre-ghost dof
            sendCounts[preGhostOwner[i]] += 1;
        }

        // count number of post-ghost dofs to corresponding owners
        for (LI i = 0; i < m_uiNumPostGhostDofs; i++) {
            // postGhostOwner[i] = rank who owns the ith post-ghost dof
            sendCounts[postGhostOwner[i]] += 1;
        }

        // get recvCounts by transposing the matrix of sendCounts
        MPI_Alltoall(sendCounts, 1, MPI_UNSIGNED, recvCounts, 1, MPI_UNSIGNED, m_comm);

        // compute offsets from sends
        sendOffset[0]=0;
        recvOffset[0]=0;
        for(unsigned int i = 1; i < m_uiSize; i++) {
            sendOffset[i] = sendOffset[i-1] + sendCounts[i-1];
            recvOffset[i] = recvOffset[i-1] + recvCounts[i-1];
        }

        // size of sendBuf = # ghost dofs (i.e. # dofs I need but not own)
        // later, this is used as number of dofs that I need to receive from corresponding rank before doing matvec
        std::vector<GI> sendBuf;
        sendBuf.resize(sendOffset[m_uiSize-1] + sendCounts[m_uiSize-1]); //size also = (m_uiNumPreGhostDofs + m_uiNumPostGhostDofs)

        // size of recvBuf = sum of dofs each other rank needs from me
        // later, this is used as the number of dofs that I need to send to ranks (before doing matvec) who need them as ghost dofs
        std::vector<GI> recvBuf;
        recvBuf.resize(recvOffset[m_uiSize-1] + recvCounts[m_uiSize-1]);

        // put global ID of pre- and post-ghost dofs to sendBuf
        for (LI i = 0; i < m_uiNumPreGhostDofs; i++)
            sendBuf[i] = preGhostGIds[i];
        for(LI i = 0; i < m_uiNumPostGhostDofs; i++)
            sendBuf[i + m_uiNumPreGhostDofs] = postGhostGIds[i];

        for(unsigned int i = 0; i < m_uiSize; i++) {
            sendCounts[i] *= sizeof(GI);
            sendOffset[i] *= sizeof(GI);
            recvCounts[i] *= sizeof(GI);
            recvOffset[i] *= sizeof(GI);
        }

        // exchange the global ID of ghost dofs with ranks who own them
        //MPI_Alltoallv(&(*(sendBuf.begin())), (int*)sendCounts, (int*)sendOffset, MPI_BYTE,
        //                &(*(recvBuf.begin())), (int*)recvCounts, (int*)recvOffset, MPI_BYTE, m_comm);
        MPI_Alltoallv(sendBuf.data(), (int*)sendCounts, (int*)sendOffset, MPI_BYTE,
                        recvBuf.data(), (int*)recvCounts, (int*)recvOffset, MPI_BYTE, m_comm);

        for(unsigned int i = 0; i < m_uiSize; i++) {
            sendCounts[i] /= sizeof(GI);
            sendOffset[i] /= sizeof(GI);
            recvCounts[i] /= sizeof(GI);
            recvOffset[i] /= sizeof(GI);
        }


        // convert global Ids in recvBuf (dofs that I need to send to before matvec) to local Ids
        m_uivSendDofIds.resize(recvBuf.size());

        for(LI i = 0; i < recvBuf.size(); i++) {
            // global ID of recvBuf[i]
            const GI global_dof_id = recvBuf[i];
            // check if global_dof_id is really owned by my rank, if not then something went wrong with sendBuf above
            if (global_dof_id < m_ulvLocalDofScan[m_uiRank]  || global_dof_id >=  (m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)) {
                std::cout<<" m_uiRank: "<<m_uiRank<< "scatter map error : "<<__func__<<std::endl;
                Error::GHOST_NODE_NOT_FOUND;
            }
            // also check with data passed in set_map
            assert((global_dof_id >= m_ulGlobalDofStart) && (global_dof_id <= m_ulGlobalDofEnd));
            // convert global id to local id (id local to rank)
            m_uivSendDofIds[i] = m_uiNumPreGhostDofs + (global_dof_id - m_ulvLocalDofScan[m_uiRank]);
        }

        m_uivSendDofCounts.resize(m_uiSize);
        m_uivSendDofOffset.resize(m_uiSize);
        m_uivRecvDofCounts.resize(m_uiSize);
        m_uivRecvDofOffset.resize(m_uiSize);

        for (unsigned int i = 0; i < m_uiSize; i++) {
            m_uivSendDofCounts[i] = recvCounts[i];
            m_uivSendDofOffset[i] = recvOffset[i];
            m_uivRecvDofCounts[i] = sendCounts[i];
            m_uivRecvDofOffset[i] = sendOffset[i];
        }

        // identify ranks that I need to send to and ranks that I will receive from
        for (unsigned int i = 0; i < m_uiSize; i++){
            if (m_uivSendDofCounts[i] > 0){
                m_uivSendRankIds.push_back(i);
            }
            if (m_uivRecvDofCounts[i] > 0){
                m_uivRecvRankIds.push_back(i);
            }
        }

        // 2020.05.21: build rank-to-global map and element-to-rank map
        // local vector = [0, ..., (m_uiNumPreGhostDofs - 1), --> ghost nodes owned by someone before me
        // m_uiNumPreGhostDofs, ..., (m_uiNumPreGhostDofs + m_uiNumDofs - 1), --> nodes owned by me
        // (m_uiNumPreGhostDofs + m_uiNumDofs), ..., (m_uiNumPreGhostDofs + m_uiNumDofs + m_uiNumPostGhostDofs - 1)] --> nodes owned by someone after me
        m_ulpLocal2Global = new GI [m_uiNumDofsTotal];
        LI local_dof_id;
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            for (LI i = 0; i < m_uiDofsPerElem[eid]; i++){
                // global Id of i
                const GI global_dof_id = m_ulpMap[eid][i];
                if (global_dof_id >= m_ulvLocalDofScan[m_uiRank] &&
                    global_dof_id < (m_ulvLocalDofScan[m_uiRank] + m_uivLocalDofCounts[m_uiRank])) {
                    // global_dof_id is owned by me
                    local_dof_id = global_dof_id - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;

                } else if (global_dof_id < m_ulvLocalDofScan[m_uiRank]){
                    // global_dof_id is owned by someone before me
                    const LI lookUp = std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), global_dof_id) - preGhostGIds.begin();
                    local_dof_id = lookUp;

                } else if (global_dof_id >= (m_ulvLocalDofScan[m_uiRank] + m_uivLocalDofCounts[m_uiRank])){
                    // global_dof_id is owned by someone after me
                    const LI lookUp = std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), global_dof_id) - postGhostGIds.begin();
                    local_dof_id = (m_uiNumPreGhostDofs + m_uiNumDofs) + lookUp;
                } else {
                    std::cout << " m_uiRank: " << m_uiRank << "scatter map error : " << __func__ << std::endl;
                    Error::GLOBAL_DOF_ID_NOT_FOUND;
                }
                m_uipLocalMap[eid][i] = local_dof_id;
                m_ulpLocal2Global[local_dof_id] = global_dof_id;
            }
        }

        /* for (LI eid = 0; eid < m_uiNumElems; eid++){
            for (LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++){
                printf("m_uipLocalMap[r%d,e%d,n%d]= %d\n",m_uiRank,eid,nid,m_uipLocalMap[eid][nid]);
            }
        } */

        delete [] sendCounts;
        delete [] recvCounts;
        delete [] sendOffset;
        delete [] recvOffset;

        return Error::SUCCESS;
    } // aMatFree::buildScatterMap()


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::create_vec( DT* &vec, bool isGhosted /* = false */, DT alpha /* = 0.0 */ ) const {
        if (isGhosted){
            vec = new DT[m_uiNumDofsTotal];
        } else {
            vec = new DT[m_uiNumDofs];
        }
        // initialize
        if (isGhosted) {
            for (LI i = 0; i < m_uiNumDofsTotal; i++){
                vec[i] = alpha;
            }
        } else {
            for (LI i = 0; i < m_uiNumDofs; i++){
                vec[i] = alpha;
            }
        }
        return Error::SUCCESS;
    } // aMatFree::create_vec


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::destroy_vec(DT* &vec) {
        if (vec != nullptr) {
            delete[] vec;
            vec = nullptr;
        }
        return Error::SUCCESS;
    } // aMatFree::destroy_vec


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::local_to_ghost( DT*  gVec, const DT* local ) const {
        for (LI i = 0; i < m_uiNumDofsTotal; i++){
            if ((i >= m_uiNumPreGhostDofs) && (i < m_uiNumPreGhostDofs + m_uiNumDofs)) {
                gVec[i] = local[i - m_uiNumPreGhostDofs];
            }
            else {
                gVec[i] = 0.0;
            }
        }
        return Error::SUCCESS;
    } // aMatFree::local_to_ghost


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::ghost_to_local(DT* local, const DT* gVec) const {
        for (LI i = 0; i < m_uiNumDofs; i++) {
            local[i] = gVec[i + m_uiNumPreGhostDofs];
        }
        return Error::SUCCESS;
    } // aMatFree::ghost_to_local


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim ){
        aMatFree<DT,GI,LI>::copy_element_matrix(eid, e_mat, block_i, block_j, blocks_dim);
        return Error::SUCCESS;
    } //aMatFree::set_element_matrix


    template <typename DT,typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::copy_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim ) {

        // resize the vector of blocks for element eid
        m_epMat[eid].resize(blocks_dim * blocks_dim, nullptr);

        // 1D-index of (block_i, block_j)
        LI index = (block_i * blocks_dim) + block_j;

        // allocate memory to store e_mat (e_mat is one of blocks of the elemental matrix of element eid)
        LI num_dofs_per_block = e_mat.rows();
        assert (num_dofs_per_block == e_mat.cols());
        
        // allocate and align memory for elemental matrices
        #if defined(VECTORIZED_AVX512) || defined(VECTORIZED_AVX256) || defined(VECTORIZED_OPENMP)
        m_epMat[eid][index] = create_aligned_array(ALIGNMENT, (num_dofs_per_block * num_dofs_per_block));

        #elif VECTORIZED_OPENMP_PADDING
        // number of paddings inserted to the end of each column of elemental block matrix
        assert((ALIGNMENT % sizeof(DT)) == 0);
        unsigned int nPads = 0;
        //nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
        if ((num_dofs_per_block % (ALIGNMENT/sizeof(DT))) != 0){
            nPads = (ALIGNMENT/sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT/sizeof(DT)));
        }

        // allocate block matrix with added paddings
        m_epMat[eid][index] = create_aligned_array(ALIGNMENT, ((num_dofs_per_block + nPads) * num_dofs_per_block));

        #else
        m_epMat[eid][index] = (DT*)malloc((num_dofs_per_block * num_dofs_per_block) * sizeof(DT));

        #endif
        
        // store block matrix in column-major for simpd, row-major for non-simd
        LI ind = 0;

        #if defined(VECTORIZED_AVX512) || defined(VECTORIZED_AVX256) || defined(VECTORIZED_OPENMP)
        for (LI c = 0; c < num_dofs_per_block; c++){
            for (LI r = 0; r < num_dofs_per_block; r++){
                m_epMat[eid][index][ind] = e_mat(r,c);
                ind++;
                //if (eid == 0) printf("e_mat[%d][%d,%d]= %f\n",eid,r,c,e_mat(r,c));
            }
        }

        #elif VECTORIZED_OPENMP_PADDING
        for (LI c = 0; c < num_dofs_per_block; c++){
            for (LI r = 0; r < num_dofs_per_block; r++){
                m_epMat[eid][index][c * (num_dofs_per_block + nPads) + r] = e_mat(r,c);
            }
        }

        #else
        for (LI r = 0; r < num_dofs_per_block; r++){
            for (LI c = 0; c < num_dofs_per_block; c++){
                m_epMat[eid][index][ind] = e_mat(r,c);
                ind++;
            }
        }
        #endif
        
        // compute the trace of matrix for penalty method
        if (m_BcMeth == BC_METH::BC_PENALTY){
            for (LI r = 0; r < num_dofs_per_block; r++) m_dtTraceK += e_mat(r,r);
        }

        return Error::SUCCESS;
    }// aMatFree::copy_element_matrix


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::mat_get_diagonal(DT* diag, bool isGhosted){
        if (isGhosted) {
            mat_get_diagonal_ghosted(diag);
        } else {
            DT* g_diag;
            create_vec(g_diag, true, 0.0);
            mat_get_diagonal_ghosted(g_diag);
            ghost_to_local(diag, g_diag);
            delete[] g_diag;
        }
        return Error::SUCCESS;
    }// aMatFree::mat_get_diagonal


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::mat_get_diagonal_ghosted(DT* diag){
        LI rowID;

        #ifdef VECTORIZED_OPENMP_PADDING
        unsigned int nPads = 0;
        #endif

        for (LI eid = 0; eid < m_uiNumElems; eid++){
            // number of blocks in each direction (i.e. blocks_dim)
            LI blocks_dim = (LI)sqrt(m_epMat[eid].size());
            
            // number of block must be a square of blocks_dim
            assert((blocks_dim*blocks_dim) == m_epMat[eid].size());
            
            // number of dofs per block, must be the same for all blocks
            const LI num_dofs_per_block = m_uiDofsPerElem[eid]/blocks_dim;

            #ifdef VECTORIZED_OPENMP_PADDING
            //nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
            if ((num_dofs_per_block % (ALIGNMENT/sizeof(DT))) != 0){
                nPads = (ALIGNMENT/sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT/sizeof(DT)));
            }
            #endif

            LI block_row_offset = 0;
            for (LI block_i = 0; block_i < blocks_dim; block_i++) {

                // only get diagonals of diagonal blocks
                LI index = block_i * blocks_dim + block_i;

                // diagonal block must be non-zero
                assert (m_epMat[eid][index] != nullptr);

                for (LI r = 0; r < num_dofs_per_block; r++){
                    // local (rank) row ID
                    rowID = m_uipLocalMap[eid][block_row_offset + r];

                    // get diagonal of elemental block matrix
                    #ifdef VECTORIZED_OPENMP_PADDING
                    diag[rowID] += m_epMat[eid][index][r * (num_dofs_per_block + nPads) + r];
                    #else
                    // diagonals are the same for both simd (column-major) and non-simd (row-major)
                    diag[rowID] += m_epMat[eid][index][r * num_dofs_per_block + r];
                    #endif
                }
                block_row_offset += num_dofs_per_block;
            }
        }

        // communication between ranks
        ghost_send_begin(diag);
        ghost_send_end(diag);

        // 05/01/2020 this is done in apply_bc_diagonal which is not quite efficient because all elements
        // are checked to see if any dof is constrained. Howerver it is not so bad because that function is called only once
        /* LI local_Id;
        for (LI nid = 0; nid < n_owned_constraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;
            if (m_BcMeth == BC_METH::BC_IMATRIX){
                // replace the current diagonal of Kcc block by 1.0
                diag[local_Id] = 1.0;
            } else if (m_BcMeth == BC_METH::BC_PENALTY){
                // add M to the current diagonal of Kcc
                diag[local_Id] = PENALTY_FACTOR * m_dtTraceK;
            }
        } */

        return Error::SUCCESS;
    }// aMatFree::mat_get_diagonal_ghosted


    // return rank that owns global gId
    template <typename DT, typename GI, typename LI>
    unsigned int aMatFree<DT, GI, LI>::globalId_2_rank(GI gId) const {
        unsigned int rank;
        if (gId >= m_ulvLocalDofScan[m_uiSize - 1]){
            rank = m_uiSize - 1;
        } else {
            for (unsigned int i = 0; i < (m_uiSize - 1); i++){
                if (gId >= m_ulvLocalDofScan[i] && gId < m_ulvLocalDofScan[i+1] && (i < (m_uiSize -1))) {
                    rank = i;
                    break;
                }
            }
        }
        return rank;
    } // aMatFree::globalId_2_rank


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT, GI, LI>::mat_get_diagonal_block(std::vector<MatRecord<DT,LI>> &diag_blk){
        LI blocks_dim;
        GI glo_RowId, glo_ColId;
        LI loc_RowId, loc_ColId;
        LI rowID, colID;
        unsigned int rank_r, rank_c;
        DT value;
        LI ind = 0;

        #ifdef VECTORIZED_OPENMP_PADDING
        unsigned int nPads = 0;
        #endif

        std::vector<MatRecord<DT,LI>> matRec1;
        std::vector<MatRecord<DT,LI>> matRec2;

        MatRecord<DT,LI> matr;

        m_vMatRec.clear();
        diag_blk.clear();

        for (LI eid = 0; eid < m_uiNumElems; eid++){

            // number of blocks in row (or column)
            blocks_dim = (LI)sqrt(m_epMat[eid].size());
            assert (blocks_dim * blocks_dim == m_epMat[eid].size());

            LI block_row_offset = 0;
            LI block_col_offset = 0;

            const LI num_dofs_per_block = m_uiDofsPerElem[eid]/blocks_dim;

            #ifdef VECTORIZED_OPENMP_PADDING
            //nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
            if ((num_dofs_per_block % (ALIGNMENT/sizeof(DT))) != 0){
                nPads = (ALIGNMENT/sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT/sizeof(DT)));
            }
            #endif

            for (LI block_i = 0; block_i < blocks_dim; block_i++){

                for (LI block_j = 0; block_j < blocks_dim; block_j++){

                    LI index = block_i * blocks_dim + block_j;

                    if (m_epMat[eid][index] != nullptr){

                        for (LI r = 0; r < num_dofs_per_block; r++){

                            // local row Id (include ghost nodes)
                            rowID = m_uipLocalMap[eid][block_row_offset + r];

                            // global row Id
                            glo_RowId = m_ulpMap[eid][block_row_offset + r];

                            //rank who owns global row Id
                            rank_r = globalId_2_rank(glo_RowId);

                            //local ID in that rank (not include ghost nodes)
                            loc_RowId = (glo_RowId - m_ulvLocalDofScan[rank_r]);

                            for (LI c = 0; c < num_dofs_per_block; c++){
                                // local column Id (include ghost nodes)
                                colID = m_uipLocalMap[eid][block_col_offset + c];

                                // global column Id
                                glo_ColId = m_ulpMap[eid][block_col_offset + c];

                                // rank who owns global column Id
                                rank_c = globalId_2_rank(glo_ColId);

                                // local column Id in that rank (not include ghost nodes)
                                loc_ColId = (glo_ColId - m_ulvLocalDofScan[rank_c]);

                                if( rank_r == rank_c ){
                                    // put all data in a MatRecord object
                                    matr.setRank(rank_r);
                                    matr.setRowId(loc_RowId);
                                    matr.setColId(loc_ColId);
                                    #if defined(VECTORIZED_AVX512) || defined(VECTORIZED_AVX256) || defined(VECTORIZED_OPENMP)
                                        // elemental block matrix stored in column-major
                                        matr.setVal(m_epMat[eid][index][(c * num_dofs_per_block) + r]);
                                    #elif VECTORIZED_OPENMP_PADDING
                                        // paddings are inserted at columns' ends
                                        matr.setVal(m_epMat[eid][index][c * (num_dofs_per_block + nPads) + r]);
                                    #else
                                        // elemental block matrix stored in row-major
                                        matr.setVal(m_epMat[eid][index][(r * num_dofs_per_block) + c]);
                                    #endif

                                    if ((rowID >= m_uiDofLocalBegin) && (rowID < m_uiDofLocalEnd) && (colID >= m_uiDofLocalBegin) && (colID < m_uiDofLocalEnd)){
                                        // add to diagonal block of my rank
                                        assert( rank_r == m_uiRank );
                                        matRec1.push_back(matr);
                                    }
                                    else {
                                        // add to matRec for sending to rank who owns this matrix term
                                        assert( rank_r != m_uiRank);
                                        matRec2.push_back(matr);
                                    }
                                }

                            }
                        }
                    } // if block is not null

                    block_col_offset += num_dofs_per_block;

                } // for block_j

                block_row_offset += num_dofs_per_block;

            } // for block_i

        } // for (eid = 0:m_uiNumElems)

        // sorting matRec2
        std::sort(matRec2.begin(), matRec2.end());

        // accumulate if 2 components of matRec2 are equal, then reduce the size
        ind = 0;
        while (ind < matRec2.size()) {
            matr.setRank(matRec2[ind].getRank());
            matr.setRowId(matRec2[ind].getRowId());
            matr.setColId(matRec2[ind].getColId());

            value = matRec2[ind].getVal();
            // since matRec is sorted, we keep increasing i for all members that are equal
            while (((ind + 1) < matRec2.size()) && (matRec2[ind] == matRec2[ind + 1])) {
                // accumulate value
                value += matRec2[ind + 1].getVal();
                // move i to the next member
                ind++;
            }
            matr.setVal(value);

            // append the matr (with accumulated value) to m_vMatRec
            m_vMatRec.push_back(matr);

            // move i to the next member
            ind++;
        }

        LI* sendCounts = new LI[m_uiSize];
        LI* recvCounts = new LI[m_uiSize];
        LI* sendOffset = new LI[m_uiSize];
        LI* recvOffset = new LI[m_uiSize];

        for (unsigned int i = 0; i < m_uiSize; i++){
            sendCounts[i] = 0;
            recvCounts[i] = 0;
        }

        // number of elements sending to each rank
        for (LI i = 0; i < m_vMatRec.size(); i++){
            sendCounts[m_vMatRec[i].getRank()] ++;
        }

        // number of elements receiving from each rank
        MPI_Alltoall(sendCounts, 1, MPI_UNSIGNED, recvCounts, 1, MPI_UNSIGNED, m_comm);

        sendOffset[0] = 0;
        recvOffset[0] = 0;
        for (unsigned int i = 1; i < m_uiSize; i++){
            sendOffset[i] = sendCounts[i-1] + sendOffset[i-1];
            recvOffset[i] = recvCounts[i-1] + recvOffset[i-1];
        }

        // allocate receive buffer
        std::vector<MatRecord<DT,LI>> recv_buff;
        recv_buff.resize(recvCounts[m_uiSize-1]+recvOffset[m_uiSize-1]);

        // send to all other ranks
        for (unsigned int i = 0; i < m_uiSize; i++){
            if (sendCounts[i] == 0) continue;
            const MPI_Datatype dtype = par::MPI_datatype_matrecord<DT,LI>::value();
            MPI_Send(&m_vMatRec[sendOffset[i]], sendCounts[i], dtype, i, m_iCommTag, m_comm);
        }

        // receive from all other ranks
        for (unsigned int i = 0; i < m_uiSize; i++){
            if (recvCounts[i] == 0) continue;
            const MPI_Datatype dtype = par::MPI_datatype_matrecord<DT,LI>::value();
            MPI_Status status;
            MPI_Recv(&recv_buff[recvOffset[i]], recvCounts[i], dtype, i, m_iCommTag, m_comm, &status);
        }

        m_iCommTag++;

        // add the received data to matRec1
        for (LI i = 0; i < recv_buff.size(); i++){
            if (recv_buff[i].getRank() != m_uiRank) {
                return Error::WRONG_COMMUNICATION;
            } else {
                matr.setRank(recv_buff[i].getRank());
                matr.setRowId(recv_buff[i].getRowId());
                matr.setColId(recv_buff[i].getColId());
                matr.setVal(recv_buff[i].getVal());

                matRec1.push_back(matr);
            }
        }

        // sorting matRec1
        std::sort(matRec1.begin(), matRec1.end());

        // accumulate value if 2 components of matRec1 are equal, then reduce the size
        ind = 0;
        while (ind < matRec1.size()) {
            matr.setRank(matRec1[ind].getRank());
            matr.setRowId(matRec1[ind].getRowId());
            matr.setColId(matRec1[ind].getColId());

            DT val = matRec1[ind].getVal();
            // since matRec1 is sorted, we keep increasing i for all members that are equal
            while (((ind + 1) < matRec1.size()) && (matRec1[ind] == matRec1[ind + 1])) {
                // accumulate value
                val += matRec1[ind + 1].getVal();
                // move i to the next member
                ind++;
            }
            matr.setVal(val);

            // append the matr (with accumulated value) to diag_blk
            diag_blk.push_back(matr);

            // move i to the next member
            ind++;
        }

        delete [] sendCounts;
        delete [] recvCounts;
        delete [] sendOffset;
        delete [] recvOffset;

        return Error::SUCCESS;
    } // aMatFree::mat_get_diagonal_block


    /* template <typename DT, typename  GI, typename LI>
    Error aMatFree<DT,GI,LI>::get_max_dof_per_block(){
        LI num_dofs, blocks_dim;

        for (LI eid = 0; eid < m_uiNumElems; eid++){
            // number of blocks in row (or column)
            blocks_dim = (LI)sqrt(m_epMat[eid].size());
            assert (blocks_dim * blocks_dim == m_epMat[eid].size());

            // number of dofs per block
            assert((m_uiDofsPerElem[eid]%blocks_dim) == 0);
            num_dofs = m_uiDofsPerElem[eid]/blocks_dim;

            if (m_uiMaxDofsPerBlock < num_dofs) m_uiMaxDofsPerBlock = num_dofs;
        }

        // get number of pads added to ve (where ve = block_matrix * ue)
        #ifdef OMP_SIMD_PADDING
            assert((ALIGNMENT % sizeof(DT)) == 0);
            if ((m_uiMaxDofsPerBlock % (ALIGNMENT/sizeof(DT))) != 0){
                m_uiMaxNumPads = (ALIGNMENT/sizeof(DT)) - (m_uiMaxDofsPerBlock % (ALIGNMENT/sizeof(DT)));
            } else {
                m_uiMaxNumPads = 0;
            }
        #endif
        printf("m_uiMaxNumPads= %d\n",m_uiMaxNumPads);

        return Error::SUCCESS;
    } */ // aMatFree::get_max_dof_per_block


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::allocate_ue_ve(){
        #ifdef HYBRID_PARALLEL
            // ue and ve are local to each thread, they will be allocated after #pragma omp parallel
            // at the moment, we only allocate an array (of size = number of threads) of DT*
            m_ueBufs = (DT**)malloc(m_uiNumThreads * sizeof(DT*));
            m_veBufs = (DT**)malloc(m_uiNumThreads * sizeof(DT*));
            for (unsigned int i = 0; i < m_uiNumThreads; i++){
                m_ueBufs[i] = nullptr;
                m_veBufs[i] = nullptr;
            }

        #else
            #if defined(VECTORIZED_AVX512) || defined(VECTORIZED_AVX256) || defined(VECTORIZED_OPENMP)
                // allocate and align ue and ve as normal
                ve = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock);
                ue = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock);
            #elif VECTORIZED_OPENMP_PADDING
                // allocate and align ve = MaxDofsPerBlock + MaxNumPads, ue as normal
                ve = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock + m_uiMaxNumPads);
                ue = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock);
            #else
                // allocate ve and ue without alignment
                ue = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
                ve = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
            #endif
        #endif

        return Error::SUCCESS;
    } // aMatFree::allocate_ue_ve


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::ghost_receive_begin(DT* vec) {
        if (m_uiSize == 1)
            return Error::SUCCESS;

        // exchange context for vec
        AsyncExchangeCtx ctx((const void*)vec);

        // total number of DoFs to be sent
        const LI total_send = m_uivSendDofOffset[m_uiSize-1] + m_uivSendDofCounts[m_uiSize-1];
        assert(total_send == m_uivSendDofIds.size());

        // total number of DoFs to be received
        const LI total_recv = m_uivRecvDofOffset[m_uiSize-1] + m_uivRecvDofCounts[m_uiSize-1];

        // send data of owned DoFs to corresponding ghost DoFs in all other ranks
        if (total_send > 0){
            // allocate memory for sending buffer
            ctx.allocateSendBuffer(sizeof(DT) * total_send);
            // get the address of sending buffer
            DT* send_buf = (DT*)ctx.getSendBuffer();
            // put all sending values to buffer
            for (LI i = 0; i < total_send; i++){
                send_buf[i] = vec[m_uivSendDofIds[i]];
            }
            for (unsigned int r = 0; r < m_uivSendRankIds.size(); r++){
                unsigned int i = m_uivSendRankIds[r]; // rank that I will send to
                // send to rank i
                MPI_Request* req = new MPI_Request();
                MPI_Isend(&send_buf[m_uivSendDofOffset[i]], m_uivSendDofCounts[i] * sizeof(DT), MPI_BYTE, i, m_iCommTag, m_comm, req);
                // put output request req of sending into the Request list of ctx
                ctx.getRequestList().push_back(req);
            }
        }

        // received data for ghost DoFs from all other ranks
        if (total_recv > 0){
            ctx.allocateRecvBuffer(sizeof(DT) * total_recv);
            DT* recv_buf = (DT*) ctx.getRecvBuffer();
            
            for (unsigned int r = 0; r < m_uivRecvRankIds.size(); r++){
                unsigned int i = m_uivRecvRankIds[r]; // rank that I will receive from
                MPI_Request* req = new MPI_Request();
                MPI_Irecv(&recv_buf[m_uivRecvDofOffset[i]], m_uivRecvDofCounts[i] * sizeof(DT), MPI_BYTE, i, m_iCommTag, m_comm, req);
                // put output request req of receiving into Request list of ctx
                ctx.getRequestList().push_back(req);
            }
        }
        // save the ctx of v for later access
        m_vAsyncCtx.push_back(ctx);
        // get a different value of tag if we have another ghost_exchange for a different vec
        m_iCommTag++;

        return Error::SUCCESS;
    } // aMatFree::ghost_receive_begin


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::ghost_receive_end(DT* vec) {
        if (m_uiSize == 1)
            return Error::SUCCESS;

        // get the context associated with vec
        unsigned int ctx_index;
        for (unsigned int i = 0; i < m_vAsyncCtx.size(); i++){
            if (vec == (DT*)m_vAsyncCtx[i].getBuffer()){
                ctx_index = i;
                break;
            }
        }
        AsyncExchangeCtx ctx = m_vAsyncCtx[ctx_index];

        // wait for all sends and receives finish
        MPI_Status sts;
        // total number of sends and receives have issued
        int num_req = ctx.getRequestList().size();
        for (unsigned int i =0; i < num_req; i++) {
            MPI_Wait(ctx.getRequestList()[i], &sts);
        }

        //const unsigned  int total_recv = m_uivRecvDofOffset[m_uiSize-1] + m_uivRecvDofCounts[m_uiSize-1];

        DT* recv_buf = (DT*) ctx.getRecvBuffer();

        // received values are now put at pre-ghost and post-ghost positions of vec
        std::memcpy(vec, recv_buf, m_uiNumPreGhostDofs * sizeof(DT));
        std::memcpy(&vec[m_uiNumPreGhostDofs + m_uiNumDofs], &recv_buf[m_uiNumPreGhostDofs], m_uiNumPostGhostDofs * sizeof(DT));

        // free memory of send and receive buffers of ctx
        ctx.deAllocateRecvBuffer();
        ctx.deAllocateSendBuffer();

        // erase the context associated with ctx in m_vAsyncCtx
        m_vAsyncCtx.erase(m_vAsyncCtx.begin() + ctx_index);

        return Error::SUCCESS;
    } // aMatFree::ghost_receive_end


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::ghost_send_begin(DT* vec) {
        if (m_uiSize == 1)
            return Error::SUCCESS;

        AsyncExchangeCtx ctx((const void*)vec);

        // number of owned dofs to be received from other ranks (i.e. number of dofs that I sent out before doing matvec)
        const LI total_recv = m_uivSendDofOffset[m_uiSize-1] + m_uivSendDofCounts[m_uiSize-1];
        // number of dofs to be sent back to their owners (i.e. number of dofs that I received before doing matvec)
        const LI total_send = m_uivRecvDofOffset[m_uiSize-1] + m_uivRecvDofCounts[m_uiSize-1];

        // receive data
        if (total_recv > 0){
            ctx.allocateRecvBuffer(sizeof(DT) * total_recv);
            DT* recv_buf = (DT*) ctx.getRecvBuffer();
            for (unsigned int r = 0; r < m_uivSendRankIds.size(); r++){
                unsigned int i = m_uivSendRankIds[r];
                //printf("rank %d receives from %d, after matvec\n",m_uiRank,i);
                MPI_Request* req = new MPI_Request();
                MPI_Irecv(&recv_buf[m_uivSendDofOffset[i]], m_uivSendDofCounts[i]*sizeof(DT), MPI_BYTE, i, m_iCommTag, m_comm, req);
                ctx.getRequestList().push_back(req);
            }
        }
        
        // send data of ghost DoFs to ranks owning the DoFs
        if (total_send > 0){
            ctx.allocateSendBuffer(sizeof(DT) * total_send);
            DT* send_buf = (DT*) ctx.getSendBuffer();

            // pre-ghost DoFs
            for (LI i = 0; i < m_uiNumPreGhostDofs; i++){
                send_buf[i] = vec[i];
            }
            // post-ghost DoFs
            for (LI i = m_uiNumPreGhostDofs + m_uiNumDofs; i < m_uiNumPreGhostDofs + m_uiNumDofs + m_uiNumPostGhostDofs; i++){
                send_buf[i - m_uiNumDofs] = vec[i];
            }
            for (unsigned int r = 0; r < m_uivRecvRankIds.size(); r++){
                unsigned int i = m_uivRecvRankIds[r];
                //printf("rank %d sends to %d, after matvec\n",m_uiRank,i);
                MPI_Request* req = new MPI_Request();
                MPI_Isend(&send_buf[m_uivRecvDofOffset[i]], m_uivRecvDofCounts[i] * sizeof(DT), MPI_BYTE, i, m_iCommTag, m_comm, req);
                ctx.getRequestList().push_back(req);
            }
        }
        m_vAsyncCtx.push_back(ctx);
        m_iCommTag++; // get a different value if we have another ghost_exchange for a different vec
        return Error::SUCCESS;
    } // aMatFree::ghost_send_begin


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::ghost_send_end(DT* vec) {
        if (m_uiSize == 1)
            return Error::SUCCESS;

        unsigned int ctx_index;
        for (unsigned i = 0; i < m_vAsyncCtx.size(); i++){
            if (vec == (DT*)m_vAsyncCtx[i].getBuffer()){
                ctx_index = i;
                break;
            }
        }
        AsyncExchangeCtx ctx = m_vAsyncCtx[ctx_index];
        int num_req = ctx.getRequestList().size();

        MPI_Status sts;
        for (unsigned int i = 0; i < num_req; i++){
            MPI_Wait(ctx.getRequestList()[i],&sts);
        }

        //const unsigned  int total_recv = m_uivSendDofOffset[m_uiSize-1] + m_uivSendDofCounts[m_uiSize-1];
        DT* recv_buf = (DT*) ctx.getRecvBuffer();

        // accumulate the received data at the positions that I sent out before doing matvec
        // these positions are indicated in m_uivSendDofIds[]
        // note: size of recv_buf[] = size of m_uivSendDofIds[]
        for (unsigned int i = 0; i < m_uiSize; i++){
            for (LI j = 0; j < m_uivSendDofCounts[i]; j++){
                // bug fixed on 2020.05.23:
                //vec[m_uivSendDofIds[m_uivSendDofOffset[i]] + j] += recv_buf[m_uivSendDofOffset[i] + j];
                vec[ m_uivSendDofIds[m_uivSendDofOffset[i] + j] ] += recv_buf[m_uivSendDofOffset[i] + j];
            }
        }

        // free memory of send and receive buffers of ctx
        ctx.deAllocateRecvBuffer();
        ctx.deAllocateSendBuffer();

        // erase the contex associated wit ctx in m_vAsyncCtx
        m_vAsyncCtx.erase(m_vAsyncCtx.begin() + ctx_index);

        return Error::SUCCESS;
    } // aMatFree::ghost_send_end


    template <typename  DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::matvec(DT* v, const DT* u, bool isGhosted) {
        #ifdef AMAT_PROFILER
            timing_aMat[static_cast<int>(PROFILER::MATVEC)].start();
        #endif
        if ( isGhosted ) {
            // std::cout << "GHOSTED MATVEC" << std::endl;
            #ifdef HYBRID_PARALLEL
                matvec_ghosted_OMP(v, (DT*)u);

            #else
                matvec_ghosted_noOMP(v, (DT*)u);
            #endif
        } else {
            // std::cout << "NON GHOSTED MATVEC" << std::endl;
            DT* gv;
            DT* gu;
            // allocate memory for gv and gu including ghost dof's
            create_vec(gv, true, 0.0);
            create_vec(gu, true, 0.0);
            // copy u to gu
            local_to_ghost(gu, u);

            #ifdef HYBRID_PARALLEL
                matvec_ghosted_OMP(v, (DT*)u);
            #else
                matvec_ghosted_noOMP(v, (DT*)u);
            #endif

            // copy gv to v
            ghost_to_local(v, gv);

            delete[] gv;
            delete[] gu;
        }
        #ifdef AMAT_PROFILER
            timing_aMat[static_cast<int>(PROFILER::MATVEC)].stop();
        #endif
        return Error::SUCCESS;

    } // aMatFree::matvec

    #ifdef HYBRID_PARALLEL

    template <typename  DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::matvec_ghosted_OMP( DT* v, DT* u ) {

        // initialize v (size of v = m_uiNodesPostGhostEnd = m_uiNumDofsTotal)
        for (LI i = 0; i < m_uiDofPostGhostEnd; i++){
            v[i] = 0.0;
        }

        // apply BC (is moved to MatMult_mf, 2020.05.25)
        // this must be done before communication so that ranks that do not own constraint dofs have correct bc
        /* LI local_Id;
        for (LI nid = 0; nid < n_owned_constraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;
            if (m_BcMeth == BC_METH::BC_IMATRIX){
                // save Uc and set u(Uc) = 0
                Uc[nid] = u[local_Id];
                u[local_Id] = 0.0;
            } else if (m_BcMeth == BC_METH::BC_PENALTY){
                // save Uc and multiply with penalty coefficient
                Uc[nid] = u[local_Id] * PENALTY_FACTOR * m_dtTraceK;
            }
        } */
        // end of apply BC

        // send data from owned nodes to ghost nodes (of other processors) to get ready for computing v = Ku
        ghost_receive_begin(u);
        ghost_receive_end(u);

        // multiply [ve] = [ke][ue] for all elements
        #pragma omp parallel
        {
        LI rowID, colID;
        LI blocks_dim, num_dofs_per_block;

        // get thread id
        const unsigned int tId = omp_get_thread_num();

        // number of pads used in padding
        #ifdef VECTORIZED_OPENMP_PADDING
        unsigned int nPads = 0;
        #endif

        // allocate private ve and ue (if not allocated yet)
        if (m_veBufs[tId] == nullptr){
            #if defined(VECTORIZED_AVX512) || defined(VECTORIZED_AVX256) || defined(VECTORIZED_OPENMP)
                // allocate and align ue and ve as normal
                m_veBufs[tId] = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock);
                m_ueBufs[tId] = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock);

            #elif VECTORIZED_OPENMP_PADDING
                // allocate and align ve = MaxDofsPerBlock + MaxNumPads, ue as normal
                m_veBufs[tId] = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock + m_uiMaxNumPads);
                m_ueBufs[tId] = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock);

            #else
                // allocate ve and ue without alignment
                m_veBufs[tId] = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
                m_ueBufs[tId] = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
            #endif

        }
        DT* ueLocal = m_ueBufs[tId];
        DT* veLocal = m_veBufs[tId];

        #pragma omp for
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            // get number of blocks of element eid
            blocks_dim = (LI)sqrt(m_epMat[eid].size());
            LI block_row_offset = 0;
            LI block_col_offset = 0;
            
            // number of dofs per block must be the same for all blocks
            num_dofs_per_block = m_uiDofsPerElem[eid]/blocks_dim;

            // compute needed pads added to the end of each column of block matrix
            #ifdef VECTORIZED_OPENMP_PADDING
            //nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
            if ((num_dofs_per_block % (ALIGNMENT/sizeof(DT))) != 0){
                nPads = (ALIGNMENT/sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT/sizeof(DT)));
            }
            #endif

            for (LI block_i = 0; block_i < blocks_dim; block_i++){

                for (LI block_j = 0; block_j < blocks_dim; block_j++){

                    LI block_ID = block_i * blocks_dim + block_j;
                    
                    if (m_epMat[eid][block_ID] != nullptr){
                        // extract block-element vector ue from structure vector u, and initialize ve
                        for (LI c = 0; c < num_dofs_per_block; c++) {
                            colID = m_uipLocalMap[eid][block_col_offset + c];
                            ueLocal[c] = u[colID];
                            veLocal[c] = 0.0;
                        }

                        // ve = elemental matrix * ue
                        #ifdef VECTORIZED_AVX512
                            for (LI c = 0; c < num_dofs_per_block; c++){
                                unsigned int remain = num_dofs_per_block % SIMD_LENGTH;
                                unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                                DT* x = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                                const DT alpha = ueLocal[c];
                                DT* y = veLocal;
                                // broadcast alpha to form alpha vector
                                __m512d alphaVec = _mm512_set1_pd(alpha);
                                // vector operation y += alpha * x;
                                for (LI j = 0; j < n_intervals; j++){
                                    // load components of x to xVec
                                    __m512d xVec = _mm512_loadu_pd(&x[SIMD_LENGTH * j]);
                                    // load components of y to yVec
                                    __m512d yVec = _mm512_loadu_pd(&y[SIMD_LENGTH * j]);
                                    // vector multiplication tVec = alphaVec * xVec
                                    __m512d tVec = _mm512_mul_pd(xVec, alphaVec);
                                    // accumulate tVec to yVec
                                    yVec = _mm512_add_pd(tVec, yVec);
                                    // store yVec to y
                                    _mm512_storeu_pd(&y[SIMD_LENGTH * j], yVec);
                                }
                                // scalar operation for the remainder
                                if (remain != 0){
                                    double* xVec_remain = new double[SIMD_LENGTH];
                                    double* yVec_remain = new double[SIMD_LENGTH];
                                    for (LI j = 0; j < remain; j++){
                                        xVec_remain[j] = x[j + n_intervals * SIMD_LENGTH];
                                        yVec_remain[j] = y[j + n_intervals * SIMD_LENGTH];
                                    }
                                    for (unsigned int j = remain; j < SIMD_LENGTH; j++){
                                        xVec_remain[j] = 0.0;
                                        yVec_remain[j] = 0.0;
                                    }
                                    __m512d xVec = _mm512_loadu_pd(&xVec_remain[0]);
                                    __m512d yVec = _mm512_loadu_pd(&yVec_remain[0]);
                                    __m512d tVec = _mm512_mul_pd(xVec, alphaVec);
                                    yVec = _mm512_add_pd(tVec, yVec);
                                    _mm512_storeu_pd(&yVec_remain[0], yVec);
                                    for (unsigned int j = 0; j < remain; j++){
                                        y[j + n_intervals * SIMD_LENGTH] = yVec_remain[j];
                                    }
                                    delete [] xVec_remain;
                                    delete [] yVec_remain;
                                }
                            }
                            
                        #elif VECTORIZED_AVX256
                            for (LI c = 0; c < num_dofs_per_block; c++){
                                unsigned int remain = num_dofs_per_block % SIMD_LENGTH;
                                unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                                DT* x = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                                const DT alpha = ueLocal[c];
                                DT* y = veLocal;
                                // broadcast alpha to form alpha vector
                                __m256d alphaVec = _mm256_set1_pd(alpha);
                                // vector operation y += alpha * x;
                                for (LI i = 0; i < n_intervals; i++){
                                    // load components of x to xVec
                                    __m256d xVec = _mm256_load_pd(&x[SIMD_LENGTH * i]);
                                    
                                    // load components of y to yVec
                                    __m256d yVec = _mm256_load_pd(&y[SIMD_LENGTH * i]);
                                    // vector multiplication tVec = alphaVec * xVec
                                    __m256d tVec = _mm256_mul_pd(xVec, alphaVec);
                                    // accumulate tVec to yVec
                                    yVec = _mm256_add_pd(tVec, yVec);
                                    // store yVec to y
                                    _mm256_storeu_pd(&y[SIMD_LENGTH * i], yVec);
                                }
                                
                                // scalar operation for the remainder
                                if (remain != 0){
                                    
                                    double* xVec_remain = new double[SIMD_LENGTH];
                                    double* yVec_remain = new double[SIMD_LENGTH];
                                    for (unsigned int i = 0; i < remain; i++){
                                        xVec_remain[i] = x[i + n_intervals * SIMD_LENGTH];
                                        yVec_remain[i] = y[i + n_intervals * SIMD_LENGTH];
                                    }
                                    for (unsigned int i = remain; i < SIMD_LENGTH; i++){
                                        xVec_remain[i] = 0.0;
                                        yVec_remain[i] = 0.0;
                                    }
                                    __m256d xVec = _mm256_load_pd(&xVec_remain[0]);
                                    __m256d yVec = _mm256_load_pd(&yVec_remain[0]);
                                    __m256d tVec = _mm256_mul_pd(xVec, alphaVec);
                                    yVec = _mm256_add_pd(tVec, yVec);
                                    _mm256_storeu_pd(&yVec_remain[0], yVec);
                                    for (unsigned int i = 0; i < remain; i++){
                                        y[i + n_intervals * SIMD_LENGTH] = yVec_remain[i];
                                    }
                                    delete [] xVec_remain;
                                    delete [] yVec_remain;
                                }
                            }

                        #elif VECTORIZED_OPENMP
                            for (LI c = 0; c < num_dofs_per_block; c++){
                                const DT alpha = ueLocal[c];
                                const DT* x = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                                #pragma omp simd aligned(x, veLocal : ALIGNMENT) safelen(512)
                                for (LI r = 0; r < num_dofs_per_block; r++){
                                    veLocal[r] += alpha * x[r];
                                }
                            }

                        #elif VECTORIZED_OPENMP_PADDING
                            for (LI c = 0; c < num_dofs_per_block; c++){
                                const DT alpha = ueLocal[c];
                                const DT* x = &m_epMat[eid][block_ID][c * (num_dofs_per_block + nPads)];
                                #pragma omp simd aligned(x, veLocal : ALIGNMENT) safelen(512)
                                for (LI r = 0; r < (num_dofs_per_block + nPads); r++){
                                    veLocal[r] += alpha * x[r];
                                }
                            }

                        #else
                            #pragma novector noparallel nounroll
                            for (LI r = 0; r < num_dofs_per_block; r++){
                                #pragma novector noparallel nounroll
                                for (LI c = 0; c < num_dofs_per_block; c++){
                                    veLocal[r] += m_epMat[eid][block_ID][(r * num_dofs_per_block) + c] * ueLocal[c];
                                }
                            }

                        #endif

                        // accumulate element vector ve to structure vector v
                        for (LI r = 0; r < num_dofs_per_block; r++){
                            rowID = m_uipLocalMap[eid][block_row_offset + r];
                            //printf("v[%d]= %f\n",r,v[r]);
                            #pragma omp atomic
                            v[rowID] += veLocal[r];
                        }
                    }
                    // move to next block in j direction (u changes)
                    block_col_offset += num_dofs_per_block;
                }
                // move to next block in i direction (v changes)
                block_row_offset += num_dofs_per_block;
            }
        }
        } //pragma omp parallel

        // send data from ghost nodes back to owned nodes after computing v
        ghost_send_begin(v);
        ghost_send_end(v);

        // apply BC (is moved to MatMult_mf, 2020.05.25)
        // this must be done after communication to finalize value of constrained dofs owned by me
        /* for (LI nid = 0; nid < n_owned_constraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;
            if (m_BcMeth == BC_METH::BC_IMATRIX){
                //set v(Uc) = Uc which is saved before doing matvec
                v[local_Id] = Uc[nid];
            } else if (m_BcMeth == BC_METH::BC_PENALTY){
                // accumulate v(Uc) = v(Uc) + Uc[nid] where Uc[nid] = u[local_Id] * PENALTY_FACTOR * m_dtTraceK;
                v[local_Id] += Uc[nid];
            }
        } */
        // end of apply BC

        return Error::SUCCESS;
    } // aMatFree::matvec_ghosted_OMP


    #else


    template <typename  DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::matvec_ghosted_noOMP( DT* v, DT* u ) {

        LI blocks_dim, num_dofs_per_block;

        // initialize v (size of v = m_uiNodesPostGhostEnd = m_uiNumDofsTotal)
        for (LI i = 0; i < m_uiDofPostGhostEnd; i++){
            v[i] = 0.0;
        }

        LI rowID, colID;

        // apply BC: save Uc and set u(Uc) = 0 (is moved to MatMult_mf, 2020.05.25)
        // this must be done before communication so that ranks that do not own constraint dofs have correct bc
        /* LI local_Id;
        for (LI nid = 0; nid < n_owned_constraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;
            if (m_BcMeth == BC_METH::BC_IMATRIX){
                // save Uc and set u(Uc) = 0
                Uc[nid] = u[local_Id];
                u[local_Id] = 0.0;
            } else if (m_BcMeth == BC_METH::BC_PENALTY){
                // save Uc and multiply with penalty coefficient
                Uc[nid] = u[local_Id] * PENALTY_FACTOR * m_dtTraceK;
            }
        } */
        // end of apply BC

        // send data from owned nodes to ghost nodes (of other processors) to get ready for computing v = Ku
        ghost_receive_begin(u);
        ghost_receive_end(u);

        // number of pads used in padding
        #ifdef VECTORIZED_OPENMP_PADDING
        unsigned int nPads = 0;
        #endif

        // multiply [ve] = [ke][ue] for all elements
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            blocks_dim = (LI)sqrt(m_epMat[eid].size());
            LI block_row_offset = 0;
            LI block_col_offset = 0;

            // number of dofs per block must be the same for all blocks
            num_dofs_per_block = m_uiDofsPerElem[eid]/blocks_dim;

            // compute number of paddings inserted to the end of each column of block matrix
            #ifdef VECTORIZED_OPENMP_PADDING
            //nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
            if ((num_dofs_per_block % (ALIGNMENT/sizeof(DT))) != 0){
                nPads = (ALIGNMENT/sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT/sizeof(DT)));
            }
            #endif

            for (LI block_i = 0; block_i < blocks_dim; block_i++){

                for (LI block_j = 0; block_j < blocks_dim; block_j++){

                    LI block_ID = block_i * blocks_dim + block_j;

                    if (m_epMat[eid][block_ID] != nullptr){
                        #ifdef AMAT_PROFILER
                            timing_aMat[static_cast<int>(PROFILER::MATVEC_MUL)].start();
                        #endif
                        // extract block-element vector ue from structure vector u, and initialize ve
                        for (LI c = 0; c < num_dofs_per_block; c++) {
                            colID = m_uipLocalMap[eid][block_col_offset + c];
                            ue[c] = u[colID];
                            ve[c] = 0.0;
                        }

                        // ve = elemental matrix * ue
                        #ifdef VECTORIZED_AVX512
                            for (LI c = 0; c < num_dofs_per_block; c++){
                                unsigned int remain = num_dofs_per_block % SIMD_LENGTH;
                                unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                                DT* x = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                                const DT alpha = ue[c];
                                DT* y = ve;
                                // broadcast alpha to form alpha vector
                                __m512d alphaVec = _mm512_set1_pd(alpha);
                                // vector operation y += alpha * x;
                                for (LI j = 0; j < n_intervals; j++){
                                    // load components of x to xVec
                                    __m512d xVec = _mm512_loadu_pd(&x[SIMD_LENGTH * j]);
                                    // load components of y to yVec
                                    __m512d yVec = _mm512_loadu_pd(&y[SIMD_LENGTH * j]);
                                    // vector multiplication tVec = alphaVec * xVec
                                    __m512d tVec = _mm512_mul_pd(xVec, alphaVec);
                                    // accumulate tVec to yVec
                                    yVec = _mm512_add_pd(tVec, yVec);
                                    // store yVec to y
                                    _mm512_storeu_pd(&y[SIMD_LENGTH * j], yVec);
                                }
                                // scalar operation for the remainder
                                if (remain != 0){
                                    double* xVec_remain = new double[SIMD_LENGTH];
                                    double* yVec_remain = new double[SIMD_LENGTH];
                                    for (LI j = 0; j < remain; j++){
                                        xVec_remain[j] = x[j + n_intervals * SIMD_LENGTH];
                                        yVec_remain[j] = y[j + n_intervals * SIMD_LENGTH];
                                    }
                                    for (unsigned int j = remain; j < SIMD_LENGTH; j++){
                                        xVec_remain[j] = 0.0;
                                        yVec_remain[j] = 0.0;
                                    }
                                    __m512d xVec = _mm512_loadu_pd(&xVec_remain[0]);
                                    __m512d yVec = _mm512_loadu_pd(&yVec_remain[0]);
                                    __m512d tVec = _mm512_mul_pd(xVec, alphaVec);
                                    yVec = _mm512_add_pd(tVec, yVec);
                                    _mm512_storeu_pd(&yVec_remain[0], yVec);
                                    for (unsigned int j = 0; j < remain; j++){
                                        y[j + n_intervals * SIMD_LENGTH] = yVec_remain[j];
                                    }
                                    delete [] xVec_remain;
                                    delete [] yVec_remain;
                                }
                            }

                        #elif VECTORIZED_AVX256

                            for (LI c = 0; c < num_dofs_per_block; c++){

                                unsigned int remain = num_dofs_per_block % SIMD_LENGTH;
                                unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                                DT* x = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                                const DT alpha = ue[c];
                                DT* y = ve;
                                // broadcast alpha to form alpha vector
                                __m256d alphaVec = _mm256_set1_pd(alpha);
                                // vector operation y += alpha * x;
                                for (LI i = 0; i < n_intervals; i++){
                                    // load components of x to xVec
                                    __m256d xVec = _mm256_load_pd(&x[SIMD_LENGTH * i]);

                                    // load components of y to yVec
                                    __m256d yVec = _mm256_load_pd(&y[SIMD_LENGTH * i]);
                                    // vector multiplication tVec = alphaVec * xVec
                                    __m256d tVec = _mm256_mul_pd(xVec, alphaVec);
                                    // accumulate tVec to yVec
                                    yVec = _mm256_add_pd(tVec, yVec);
                                    // store yVec to y
                                    _mm256_storeu_pd(&y[SIMD_LENGTH * i], yVec);
                                }

                                // scalar operation for the remainder
                                if (remain != 0){

                                    double* xVec_remain = new double[SIMD_LENGTH];
                                    double* yVec_remain = new double[SIMD_LENGTH];
                                    for (unsigned int i = 0; i < remain; i++){
                                        xVec_remain[i] = x[i + n_intervals * SIMD_LENGTH];
                                        yVec_remain[i] = y[i + n_intervals * SIMD_LENGTH];
                                    }
                                    for (unsigned int i = remain; i < SIMD_LENGTH; i++){
                                        xVec_remain[i] = 0.0;
                                        yVec_remain[i] = 0.0;
                                    }
                                    __m256d xVec = _mm256_load_pd(&xVec_remain[0]);
                                    __m256d yVec = _mm256_load_pd(&yVec_remain[0]);
                                    __m256d tVec = _mm256_mul_pd(xVec, alphaVec);
                                    yVec = _mm256_add_pd(tVec, yVec);
                                    _mm256_storeu_pd(&yVec_remain[0], yVec);
                                    for (unsigned int i = 0; i < remain; i++){
                                        y[i + n_intervals * SIMD_LENGTH] = yVec_remain[i];
                                    }
                                    delete [] xVec_remain;
                                    delete [] yVec_remain;
                                }
                            }
                        #elif VECTORIZED_OPENMP
                            for (LI c = 0; c < num_dofs_per_block; c++){
                                const DT alpha = ue[c];
                                const DT* x = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                                DT* y = ve;
                                #pragma omp simd aligned(x, y : ALIGNMENT) safelen(512)
                                for (LI r = 0; r < num_dofs_per_block; r++){
                                    y[r] += alpha * x[r];
                                }
                            }
                        #elif VECTORIZED_OPENMP_PADDING
                            for (LI c = 0; c < num_dofs_per_block; c++){
                                const DT alpha = ue[c];
                                const DT* x = &m_epMat[eid][block_ID][c * (num_dofs_per_block + nPads)];
                                DT* y = ve;
                                #pragma omp simd aligned(x, y : ALIGNMENT) safelen(512)
                                for (LI r = 0; r < (num_dofs_per_block + nPads); r++){
                                    y[r] += alpha * x[r];
                                }
                            }
                        #else
                            #pragma novector noparallel nounroll
                            for (LI r = 0; r < num_dofs_per_block; r++){
                                #pragma novector noparallel nounroll
                                for (LI c = 0; c < num_dofs_per_block; c++){
                                    ve[r] += m_epMat[eid][block_ID][(r * num_dofs_per_block) + c] * ue[c];
                                }
                            }
                        #endif
                        #ifdef AMAT_PROFILER
                            timing_aMat[static_cast<int>(PROFILER::MATVEC_MUL)].stop();
                        #endif

                        #ifdef AMAT_PROFILER
                            timing_aMat[static_cast<int>(PROFILER::MATVEC_ACC)].start();
                        #endif
                        // accumulate element vector ve to structure vector v
                        for (LI r = 0; r < num_dofs_per_block; r++){
                            rowID = m_uipLocalMap[eid][block_row_offset + r];
                            v[rowID] += ve[r];
                        }
                        #ifdef AMAT_PROFILER
                            timing_aMat[static_cast<int>(PROFILER::MATVEC_ACC)].stop();
                        #endif
                    }
                    // move to next block in j direction (u changes)
                    block_col_offset += num_dofs_per_block;
                }
                // move to next block in i direction (v changes)
                block_row_offset += num_dofs_per_block;
            }
        }

        // send data from ghost nodes back to owned nodes after computing v
        ghost_send_begin(v);
        ghost_send_end(v);

        // apply BC (is moved to MatMult_mf, 2020.05.25)
        // this must be done after communication to finalize value of constrained dofs owned by me
        /* for (LI nid = 0; nid < n_owned_constraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;
            if (m_BcMeth == BC_METH::BC_IMATRIX){
                //set v(Uc) = Uc which is saved before doing matvec
                v[local_Id] = Uc[nid];
            } else if (m_BcMeth == BC_METH::BC_PENALTY){
                // accumulate v(Uc) = v(Uc) + Uc[nid] where Uc[nid] = u[local_Id] * PENALTY_FACTOR * m_dtTraceK;
                v[local_Id] += Uc[nid];
            }
        } */
        // end of apply BC

        return Error::SUCCESS;
    } // aMatFree::matvec_ghosted_noOMP

    #endif


    template <typename  DT, typename GI, typename LI>
    PetscErrorCode aMatFree<DT,GI,LI>::MatMult_mf( Mat A, Vec u, Vec v ) {

        PetscScalar * vv; // this allows vv to be considered as regular vector
        PetscScalar * uu;

        LI local_Id;
        // VecZeroEntries(v);

        VecGetArray(v, &vv);
        VecGetArrayRead(u,(const PetscScalar**)&uu);

        DT* vvg;
        DT* uug;

        // allocate memory for vvg and uug including ghost nodes
        create_vec(vvg, true, 0);
        create_vec(uug, true, 0);

        // copy data of uu (not-ghosted) to uug
        local_to_ghost(uug, uu);

        // apply BC: save value of U_c, then make U_c = 0
        const LI numConstraints = ownedConstrainedDofs.size();
        DT* Uc = new DT [numConstraints];
        for (LI nid = 0; nid < numConstraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;
            Uc[nid] = uug[local_Id];
            uug[local_Id] = 0.0;
        }
        // end of apply BC

        // vvg = K * uug
        matvec(vvg, uug, true); // this gives V_f = (K_ff * U_f) + (K_fc * 0) = K_ff * U_f

        // apply BC: now set V_c = U_c which was saved in U'_c
        for (LI nid = 0; nid < numConstraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;
            vvg[local_Id] = Uc[nid];
        }
        delete [] Uc;
        // end of apply BC

        ghost_to_local(vv,vvg);

        delete [] vvg;
        delete [] uug;

        VecRestoreArray(v,&vv);

        return 0;
    }// aMatFree::MatMult_mf


    template<typename DT, typename GI, typename LI>
    PetscErrorCode aMatFree<DT,GI,LI>::MatGetDiagonal_mf(Mat A, Vec d){
        // point to data of PETSc vector d
        PetscScalar* dd;
        VecGetArray(d, &dd);
        //PetscInt N;
        //VecGetSize(d,&N);
        //std::cout<<" N: "<<N<<std::endl;

        // allocate regular vector used for get_diagonal() in aMatFree
        double* ddg;
        create_vec(ddg, true, 0);

        // get diagonal of matrix and put into ddg
        mat_get_diagonal(ddg, true);

        // copy ddg (ghosted) into (non-ghosted) dd
        ghost_to_local(dd, ddg);
        /*for (LI i = 0; i < m_uiNumDofs; i++){
            printf("[%d,%d,%f]\n",m_uiRank,i,dd[i]);
        }*/

        // deallocate ddg
        destroy_vec(ddg);

        // update data of PETSc vector d
        VecRestoreArray(d, &dd);

        // apply Dirichlet boundary condition
        apply_bc_diagonal( d );

        VecAssemblyBegin( d );
        VecAssemblyEnd( d );

        return 0;
    }// aMatFree::MatGetDiagonal_mf


    template<typename DT, typename GI, typename LI>
    PetscErrorCode aMatFree<DT,GI,LI>::MatGetDiagonalBlock_mf(Mat A, Mat* a){
        LI local_size = m_uiNumDofs;

        //LI local_rowID, local_colID;
        //DT value;
        //PetscScalar* aa;

        // sparse block diagonal matrix
        std::vector<MatRecord<DT,LI>> ddg;
        mat_get_diagonal_block(ddg);

        /*std::ofstream myfile;
        myfile.open("blk_diag.dat");
        for (LI r = 0; r < local_size; r++){
            for (LI c = 0; c < local_size; c++){
                //printf("[r%d][%d,%d]= %f; ",m_uiRank,r,c,aag[r][c]);
                myfile << "[r" << m_uiRank << "][" << r << "," << c << "]= " << aag[r][c] << " , ";
            }
            //printf("\n");
            myfile << std::endl;
        }*/

        // create Petsc sequential matrix
        //Mat B;

        //printf("local_size = %d, size of ddg = %d\n", local_size, ddg.size());

        MatCreateSeqAIJ(PETSC_COMM_SELF, local_size, local_size, NNZ, PETSC_NULL, a); // todo: 27 only good for dofs_per_node = 1
        //MatSetOption(*(a), MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        //MatSetUp(*(a));
        //MatZeroEntries(B);

        // set values ...
        std::vector<PetscScalar> values;
        std::vector<PetscInt> colIndices;
        PetscInt rowID;

        std::sort(ddg.begin(),ddg.end());

        LI ind = 0;
        while (ind < ddg.size()) {
            assert(ddg[ind].getRank() == m_uiRank);
            rowID = ddg[ind].getRowId();

            // clear data used for previous rowID
            colIndices.clear();
            values.clear();

            // push the first value
            colIndices.push_back(ddg[ind].getColId());
            values.push_back(ddg[ind].getVal());

            // push other values having same rowID
            while (((ind+1) < ddg.size()) && (ddg[ind].getRowId() == ddg[ind+1].getRowId())){
                colIndices.push_back(ddg[ind+1].getColId());
                values.push_back(ddg[ind+1].getVal());
                ind++;
            }

            // set values for rowID
            //MatSetValuesLocal(B, 1, &rowID, colIndices.size(), colIndices.data(), values.data(), INSERT_VALUES);
            MatSetValues((*a), 1, &rowID, colIndices.size(), colIndices.data(), values.data(), INSERT_VALUES);

            // move to next rowID
            ind++;
        }

        /*for (LI i = 0; i < ddg.size(); i++){
            assert(ddg[i].getRank() == m_uiRank);
            local_rowID = ddg[i].getRowId();
            local_colID = ddg[i].getColId();
            value = ddg[i].getVal();
            //MatSetValue(B, local_rowID, local_colID, value, INSERT_VALUES);
            MatSetValuesLocal(B, 1, &local_rowID, 1, &local_colID, &value, INSERT_VALUES);
        }*/

        // apply boundary condition for block diagonal matrix
        apply_bc_blkdiag(a);

        MatAssemblyBegin((*a), MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd((*a), MAT_FINAL_ASSEMBLY);

        /* PetscViewer viewer;
        const char filename [256] = "blkmatrix.dat";
        PetscViewerASCIIOpen(m_comm, filename, &viewer);
        MatView((*a),viewer);
        PetscViewerDestroy(&viewer); */

        /*char filename[1024];
        sprintf(filename, "Bmatrix_%d.dat", m_uiRank);
        PetscViewer viewer;
        PetscViewerASCIIOpen( PETSC_COMM_SELF, filename, &viewer );
        MatView((*a), viewer);
        PetscViewerDestroy(&viewer);*/

        //MatDuplicate(B, MAT_COPY_VALUES, a);
        //a = &B;

        //MatDestroy(&B);

        // dense block diagonal matrix
        /*Mat B;
        MatCreateSeqDense(PETSC_COMM_SELF,local_size,local_size,nullptr,&B);
        MatDenseGetArray(B, &aa);
        double** aag;
        create_mat(aag, false, 0.0); //allocate, not include ghost nodes
        mat_get_diagonal_block_dense(aag);
        for (LI i = 0; i < local_size; i++){
            for (LI j = 0; j < local_size; j++){
                aa[(i*local_size) + j] = aag[i][j];
            }
        }
        destroy_mat(aag,local_size);

        // update data for B
        MatDenseRestoreArray(B, &aa);

        // copy B to a
        MatDuplicate(B, MAT_COPY_VALUES, a);

        MatDestroy(&B);*/

        return 0;
    } // aMatFree::MatGetDiagonalBlock_mf


    template <typename DT,typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::set_bdr_map(GI* constrainedDofs, DT* prescribedValues, LI numConstraints){

        // extract constrained dofs owned by me
        LI local_Id;
        GI global_Id;

        for (LI i = 0; i < numConstraints; i++){
            global_Id = constrainedDofs[i];
            if ((global_Id >= m_ulvLocalDofScan[m_uiRank]) && (global_Id < m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)) {
                ownedConstrainedDofs.push_back(global_Id);
                ownedPrescribedValues.push_back(prescribedValues[i]);
            }
        }

        // construct elemental map of boundary condition
        m_uipBdrMap = new unsigned int* [m_uiNumElems];
        m_dtPresValMap = new DT* [m_uiNumElems];
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            m_uipBdrMap[eid] = new unsigned int[m_uiDofsPerElem[eid]];
            m_dtPresValMap[eid] = new DT [m_uiDofsPerElem[eid]];
        }

        for (LI eid = 0; eid < m_uiNumElems; eid++){
            for (LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++){
                global_Id = m_ulpMap[eid][nid];
                LI index;
                for (index = 0; index < numConstraints; index++) {
                    if (global_Id == constrainedDofs[index]) {
                        m_uipBdrMap[eid][nid] = 1;
                        m_dtPresValMap[eid][nid] = prescribedValues[index];
                        break;
                    }
                }
                if (index == numConstraints){
                    m_uipBdrMap[eid][nid] = 0;
                    m_dtPresValMap[eid][nid] = -1E16; // for testing
                    if ((global_Id >= m_ulvLocalDofScan[m_uiRank]) && (global_Id < m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)) {
                        ownedFreeDofs.push_back(global_Id);
                    }
                }
            }
        }

        //if (ownedFreeDofs_unsorted.size() > 0){
        if (ownedFreeDofs.size() > 0){
            std::sort(ownedFreeDofs.begin(), ownedFreeDofs.end());
            ownedFreeDofs.erase(std::unique(ownedFreeDofs.begin(),ownedFreeDofs.end()), ownedFreeDofs.end());
        }

        if (m_BcMeth == BC_METH::BC_IMATRIX){
            // allocate KfcUc with size = m_uiNumDofs, this will be subtracted from rhs to apply bc
            this->petsc_create_vec( KfcUcVec );

        } else if (m_BcMeth == BC_METH::BC_PENALTY){
            // initialize the trace of matrix, used to form the (big) penalty number
            m_dtTraceK = 0.0;

        } else {
            return Error::UNKNOWN_BC_METH;
        }

        // allocate memory for Uc used in matvec_ghosted when applying BC
        n_owned_constraints = ownedConstrainedDofs.size();
        if (n_owned_constraints > 0){
            Uc = new DT [n_owned_constraints];
        }

        return Error::SUCCESS;
    }// aMatFree::set_bdr_map


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::apply_bc_diagonal(Vec diag) {
        PetscInt rowId;

        for (LI eid = 0; eid < m_uiNumElems; eid++){
            for (LI r = 0; r < m_uiDofsPerElem[eid]; r++){
                if (m_uipBdrMap[eid][r] == 1){
                    // global row ID
                    rowId = m_ulpMap[eid][r];
                    // 05/01/2020: add the case of penalty method for apply bc
                    if (m_BcMeth == BC_METH::BC_IMATRIX){
                        VecSetValue(diag, rowId, 1.0, INSERT_VALUES);
                    } else if (m_BcMeth == BC_METH::BC_PENALTY){
                        VecSetValue(diag, rowId, (PENALTY_FACTOR * m_dtTraceK), INSERT_VALUES);
                    }
                }
            }
        }
        return Error::SUCCESS;
    } // aMatFree::apply_bc_diagonal


    // apply Dirichlet BC to block diagonal matrix
    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::apply_bc_blkdiag(Mat* blkdiagMat) {
        LI num_dofs_per_elem;
        PetscInt loc_rowId, loc_colId;
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            // total number of dofs per element eid
            num_dofs_per_elem = m_uiDofsPerElem[eid];

            // loop on all dofs of element
            for (LI r = 0; r < num_dofs_per_elem; r++) {
                loc_rowId = m_uipLocalMap[eid][r];
                // 05.21.20: bug loc_rowId <= m_uiDofLocalEnd is fixed
                if ((loc_rowId >= m_uiDofLocalBegin) && (loc_rowId < m_uiDofLocalEnd)) {
                    if (m_uipBdrMap[eid][r] == 1) {
                        for (LI c = 0; c < num_dofs_per_elem; c++) {
                            loc_colId = m_uipLocalMap[eid][c];
                            // 05.21.20: bug loc_rowId <= m_uiDofLocalEnd is fixed
                            if ((loc_colId >= m_uiDofLocalBegin) && (loc_colId < m_uiDofLocalEnd)) {
                                if (loc_rowId == loc_colId) {
                                    // 05/01/2020: add the case of penalty method for apply bc
                                    if (m_BcMeth == BC_METH::BC_IMATRIX){
                                        MatSetValue(*blkdiagMat, (loc_rowId - m_uiNumPreGhostDofs),
                                                (loc_colId - m_uiNumPreGhostDofs), 1.0, INSERT_VALUES);
                                    } else if (m_BcMeth == BC_METH::BC_PENALTY){
                                        MatSetValue(*blkdiagMat, (loc_rowId - m_uiNumPreGhostDofs),
                                                (loc_colId - m_uiNumPreGhostDofs), (PENALTY_FACTOR * m_dtTraceK), INSERT_VALUES);
                                    }

                                } else {
                                    // 05/01/2020: only for identity-matrix method, not for penalty method
                                    if (m_BcMeth == BC_METH::BC_IMATRIX){
                                        MatSetValue(*blkdiagMat, (loc_rowId - m_uiNumPreGhostDofs),
                                                (loc_colId - m_uiNumPreGhostDofs), 0.0, INSERT_VALUES);
                                    }
                                }
                            }
                        }
                    } else {
                        for (LI c = 0; c < num_dofs_per_elem; c++) {
                            loc_colId = m_uipLocalMap[eid][c];
                            // 05.21.20: bug loc_rowId <= m_uiDofLocalEnd is fixed
                            if ((loc_colId >= m_uiDofLocalBegin) && (loc_colId < m_uiDofLocalEnd)) {
                                if (m_uipBdrMap[eid][c] == 1) {
                                    // 05/01/2020: only for identity-matrix method, not for penalty method
                                    if (m_BcMeth == BC_METH::BC_IMATRIX){
                                        MatSetValue(*blkdiagMat, (loc_rowId - m_uiNumPreGhostDofs),
                                                (loc_colId - m_uiNumPreGhostDofs), 0.0, INSERT_VALUES);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return Error::SUCCESS;
    } // aMatFree::apply_bc_blkdiag


    // rhs[i] = Uc_i if i is on boundary of Dirichlet condition, Uc_i is the prescribed value on boundary
    // rhs[i] = rhs[i] - sum_{j=1}^{nc}{K_ij * Uc_j} if i is a free dof
    //          where nc is the total number of boundary dofs and K_ij is stiffness matrix
    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::apply_bc_rhs(Vec rhs){

        // set rows associated with constrained dofs to be equal to Uc
        PetscInt global_Id;
        PetscScalar value, value1, value2;

        // compute KfcUc
        if (m_BcMeth == BC_METH::BC_IMATRIX){
            LI block_dims;
            LI num_dofs_per_block;
            LI block_index;
            std::vector<PetscScalar> KfcUc_elem;
            std::vector<PetscInt> row_Indices_KfcUc_elem;
            PetscInt rowId;
            PetscScalar temp;
            bool bdrFlag, rowFlag;

            #ifdef VECTORIZED_OPENMP_PADDING
            unsigned int nPads = 0;
            #endif

            for (LI eid = 0; eid < m_uiNumElems; eid++){
                block_dims = (LI)sqrt(m_epMat[eid].size());
                assert((block_dims*block_dims) == m_epMat[eid].size());
                LI block_row_offset = 0;
                LI block_col_offset = 0;
                num_dofs_per_block = m_uiDofsPerElem[eid]/block_dims;

                #ifdef VECTORIZED_OPENMP_PADDING
                //nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
                if ((num_dofs_per_block % (ALIGNMENT/sizeof(DT))) != 0){
                    nPads = (ALIGNMENT/sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT/sizeof(DT)));
                }
                #endif

                // clear the vectors storing values of KfcUc for element eid
                KfcUc_elem.clear();
                row_Indices_KfcUc_elem.clear();

                for (LI block_i = 0; block_i < block_dims; block_i++){
                    for (LI block_j = 0; block_j < block_dims; block_j++){
                        block_index = block_i * block_dims + block_j;
                        // continue if block_index is not nullptr
                        if (m_epMat[eid][block_index] != nullptr){
                            for (LI r = 0; r < num_dofs_per_block; r++){
                                rowFlag = false;
                                // continue if row is associated with a free dof
                                if (m_uipBdrMap[eid][block_i * num_dofs_per_block + r] == 0){
                                    rowId = m_ulpMap[eid][block_i * num_dofs_per_block + r];
                                    temp = 0;
                                    // loop over columns of the element matrix (block)
                                    for (LI c = 0; c < num_dofs_per_block; c++){
                                        // continue if column is associated with a constrained dof
                                        if (m_uipBdrMap[eid][block_j * num_dofs_per_block + c] == 1){
                                            // accumulate Kfc[r,c]*Uc[c]
                                            #if defined(VECTORIZED_AVX512) || defined(VECTORIZED_AVX256) || defined(VECTORIZED_OPENMP)
                                                // block m_epMat[eid][block_index] is stored in column-major
                                                temp += m_epMat[eid][block_index][(c*num_dofs_per_block) + r] *
                                                        m_dtPresValMap[eid][block_j * num_dofs_per_block + c];
                                            #elif VECTORIZED_OPENMP_PADDING
                                                temp += m_epMat[eid][block_index][c*(num_dofs_per_block + nPads) + r] *
                                                        m_dtPresValMap[eid][block_j * num_dofs_per_block + c];
                                            #else
                                                // block m_epMat[eid][block_index] is stored in row-major
                                                temp += m_epMat[eid][block_index][(r*num_dofs_per_block) + c] *
                                                        m_dtPresValMap[eid][block_j * num_dofs_per_block + c];
                                            #endif
                                            rowFlag = true; // this rowId has constrained column dof
                                            bdrFlag = true; // this element matrix has KfcUc
                                        }
                                    }
                                    if (rowFlag){
                                        row_Indices_KfcUc_elem.push_back(rowId);
                                        KfcUc_elem.push_back(-1.0*temp);
                                    }
                                }
                            }
                            if (bdrFlag){
                                VecSetValues(KfcUcVec, row_Indices_KfcUc_elem.size(), row_Indices_KfcUc_elem.data(), KfcUc_elem.data(), ADD_VALUES);
                            }
                        } // m_epMat[eid][index] != nullptr
                    } // for block_j
                } // for block_i
            } // for eid
        } // if (m_BcMeth == BC_METH::BC_IMATRIX)

        // modify Fc
        for (LI nid = 0; nid < ownedConstrainedDofs.size(); nid++){
            global_Id = ownedConstrainedDofs[nid];
            if (m_BcMeth == BC_METH::BC_IMATRIX){
                value = ownedPrescribedValues[nid];
                VecSetValue(rhs, global_Id, value, INSERT_VALUES);
            } else if (m_BcMeth == BC_METH::BC_PENALTY){
                value = PENALTY_FACTOR * m_dtTraceK * ownedPrescribedValues[nid];
                VecSetValue(rhs, global_Id, value, INSERT_VALUES);
            }

        }

        // modify Ff for the case of BC_IMATRIX
        if (m_BcMeth == BC_METH::BC_IMATRIX){
            // need to finalize vector KfcUcVec before extracting its value
            VecAssemblyBegin( KfcUcVec );
            VecAssemblyEnd( KfcUcVec );

            for (LI nid = 0; nid < ownedFreeDofs.size(); nid++){
                global_Id = ownedFreeDofs[nid];
                VecGetValues(KfcUcVec, 1, &global_Id, &value1);
                VecGetValues(rhs, 1, &global_Id, &value2);
                value = value1 + value2;
                VecSetValue(rhs, global_Id, value, INSERT_VALUES);
            }
            VecDestroy(&KfcUcVec);
        }

        //petsc_destroy_vec(KfcUcVec);

        return Error::SUCCESS;
    } // aMatFree::apply_bc_rhs


    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::petsc_solve( const Vec rhs, Vec out ) const {
        // PETSc shell matrix
        Mat pMatFree;
        // get context to aMatFree
        aMatCTX<DT,GI,LI> ctx;
        // point back to aMatFree
        ctx.aMatPtr =  (aMatFree<DT,GI,LI>*)this;

        // create matrix shell
        MatCreateShell( m_comm, m_uiNumDofs, m_uiNumDofs, PETSC_DETERMINE, PETSC_DETERMINE, &ctx, &pMatFree );

        // set operation for matrix-vector multiplication using aMatFree::MatMult_mf
        MatShellSetOperation( pMatFree, MATOP_MULT, (void(*)(void))aMat_matvec<DT,GI,LI> );

        // set operation for geting matrix diagonal using aMatFree::MatGetDiagonal_mf
        MatShellSetOperation( pMatFree, MATOP_GET_DIAGONAL, (void(*)(void))aMat_matgetdiagonal<DT,GI,LI> );

        // set operation for geting block matrix diagonal using aMatFree::MatGetDiagonalBlock_mf
        MatShellSetOperation( pMatFree, MATOP_GET_DIAGONAL_BLOCK, (void(*)(void))aMat_matgetdiagonalblock<DT,GI,LI> );

        // abstract Krylov object, linear solver context
        KSP ksp;
        // abstract preconditioner object, pre conditioner context
        PC  pc;

        // default KSP context
        KSPCreate( m_comm, &ksp );

        // set the matrix associated the linear system
        KSPSetOperators( ksp, pMatFree, pMatFree );

        // set default solver (e.g. KSPCG, KSPFGMRES, ...)
        // could be overwritten at runtime using -ksp_type <type>
        KSPSetType(ksp, KSPCG);
        KSPSetFromOptions(ksp);

        KSPSetTolerances(ksp,1E-12,1E-12,PETSC_DEFAULT, 20000);

        // set default preconditioner (e.g. PCJACOBI, PCBJACOBI, ...)
        // could be overwritten at runtime using -pc_type <type>
        KSPGetPC( ksp, &pc );
        PCSetType( pc, PCJACOBI );
        PCSetFromOptions( pc );

        // solve the system
        KSPSolve( ksp, rhs, out );

        // clean up
        KSPDestroy( &ksp );

        return Error::SUCCESS;
    } // aMatFree::petsc_solve


    /**@brief: allocate an aligned memory */
    template <typename DT, typename GI, typename LI>
    DT* aMatFree<DT,GI,LI>::create_aligned_array(unsigned int alignment, unsigned int length){

        DT* array;

        #ifdef USE_WINDOWS
            array = (DT*)_aligned_malloc(length * sizeof(DT), alignment);
        #else
            int err;
            err = posix_memalign((void**)&array, alignment, length * sizeof(DT));
            if (err){
                return nullptr;
            }
            // supported (and recommended) by Intel compiler:
            //array = (DT*)_mm_malloc(length * sizeof(DT), alignment);
        #endif

        return array;
    } // aMatFree::create_aligned_array


    /**@brief: deallocate an aligned memory */
    template <typename DT, typename GI, typename LI>
    inline void aMatFree<DT,GI,LI>::delete_algined_array(DT* array){
        #ifdef USE_WINDOWS
            _aligned_free(array);
        #else
            free(array);
        #endif
    } // aMatFree::delete_aligned_array

    template <typename DT, typename GI, typename LI>
    Error aMatFree<DT,GI,LI>::petsc_dump_mat( const char* filename ){
        // this prints out the global matrix by using matvec(u,v) in oder to test the implementation of matvec(u,v)
        // algorithm:
        // u0 = [1 0 0 ... 0] --> matvec(v0,u0) will give v0 is the first column of the global matrix
        // u1 = [0 1 0 ... 0] --> matvec(v1,u1) will give v1 is the second column of the global matrix
        // u2 = [0 0 1 ... 0] --> matvec(v2,u2) will give v2 is the third column of the global matrix
        // ... up to the last column of the global matrix

        // create matrix computed by matrix-free
        Mat m_pMatFree;
        MatCreate(m_comm, &m_pMatFree);
        MatSetSizes(m_pMatFree, m_uiNumDofs, m_uiNumDofs, PETSC_DECIDE, PETSC_DECIDE);
        if(m_uiSize > 1) {
            MatSetType(m_pMatFree, MATMPIAIJ);
            MatMPIAIJSetPreallocation( m_pMatFree, NNZ, PETSC_NULL, NNZ, PETSC_NULL );
        } else {
            MatSetType(m_pMatFree, MATSEQAIJ);
            MatSeqAIJSetPreallocation(m_pMatFree, NNZ, PETSC_NULL);
        }
        MatSetOption(m_pMatFree, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

        // create vectors ui and vi preparing for multiple matrix-vector multiplications
        DT* ui;
        DT* vi;
        ui = new DT [m_uiNumDofsTotal];
        vi = new DT [m_uiNumDofsTotal];
        LI localId;
        PetscScalar value;
        PetscInt rowId, colId;

        // loop over all dofs of the global vector
        for (GI globalId = 0; globalId < m_ulNumDofsGlobal; globalId++){
            // initialize input ui and output vi
            for (LI i = 0; i < m_uiNumDofsTotal; i++){
                ui[i] = 0.0;
                vi[i] = 0.0;
            }

            // check if globalId is owned by me --> set input ui = [0...1...0] (if not ui = [0...0])
            if ((globalId >= m_ulvLocalDofScan[m_uiRank]) && (globalId < (m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs))){
                localId = m_uiNumPreGhostDofs + (globalId - m_ulvLocalDofScan[m_uiRank]);
                ui[localId] = 1.0;
            }

            // doing vi = matrix * ui
            matvec(vi, ui, true);

            // set vi to (globalId)th column of the matrix
            colId = globalId;
            for (LI r = 0; r < m_uiNumDofs; r++){
                rowId = m_ulpLocal2Global[r + m_uiNumPreGhostDofs];
                value = vi[r + m_uiNumPreGhostDofs];
                if (fabs(value) > 1e-16){
                    MatSetValue(m_pMatFree, rowId, colId, value, ADD_VALUES);
                }
            }
        }

        MatAssemblyBegin(m_pMatFree, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(m_pMatFree, MAT_FINAL_ASSEMBLY);

        // display matrix
        if (filename == nullptr){
            MatView(m_pMatFree, PETSC_VIEWER_STDOUT_WORLD);
        } else {
            PetscViewer viewer;
            PetscViewerASCIIOpen( m_comm, filename, &viewer );
            // write to file readable by Matlab (filename must be filename.m in order to execute in Matlab)
            //PetscViewerPushFormat( viewer, PETSC_VIEWER_ASCII_MATLAB );
            MatView( m_pMatFree, viewer );
            PetscViewerDestroy( &viewer );
        }

        delete [] ui;
        delete [] vi;
        MatDestroy(&m_pMatFree);

        return Error::SUCCESS;
    }//aMatFree::petsc_dump_mat

    //==============================================================================================================
    template <typename DT,typename GI, typename LI>
    aMatBased<DT,GI,LI>::aMatBased( BC_METH bcType ) : aMat<DT,GI,LI>(bcType) {
        m_pMat = nullptr;       // "global" matrix
    } // aMatBased::constructor

    template <typename DT,typename GI, typename LI>
    aMatBased<DT,GI,LI>::~aMatBased() {
        // MatDestroy(&m_pMat);
    } // aMatBased::destructor

    template <typename DT,typename GI, typename LI>
    Error aMatBased<DT,GI,LI>::set_map( const LI      n_elements_on_rank,
                                        const LI * const * element_to_rank_map,
                                        const LI         * dofs_per_element,
                                        const LI           n_all_dofs_on_rank,
                                        const GI         * rank_to_global_map,
                                        const GI           owned_global_dof_range_begin,
                                        const GI           owned_global_dof_range_end,
                                        const GI           n_global_dofs ){

        // number of owned elements
        m_uiNumElems = n_elements_on_rank;

        // number  of owned dofs
        m_uiNumDofs = owned_global_dof_range_end - owned_global_dof_range_begin + 1;

        // number of dofs of ALL ranks, currently this is not used in aMatBased
        m_ulNumDofsGlobal = n_global_dofs;

        // these will be used in set_bdr_map to pick the constraints owned by rank
        m_ulGlobalDofStart = owned_global_dof_range_begin;
        m_ulGlobalDofEnd = owned_global_dof_range_end;
        m_uiNumDofsTotal = n_all_dofs_on_rank;

        // point to provided array giving number of dofs of each element
        m_uiDofsPerElem = dofs_per_element;

        // create global map based on provided local map and Local2Global
        m_ulpMap = new GI* [m_uiNumElems];
        for( LI eid = 0; eid < m_uiNumElems; eid++ ){
            m_ulpMap[eid] = new GI [m_uiDofsPerElem[eid]];
        }
        for( LI eid = 0; eid < m_uiNumElems; eid++ ){
            for( LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++ ){
                m_ulpMap[eid][nid] = rank_to_global_map[element_to_rank_map[eid][nid]];
            }
        }

        MatCreate(m_comm, &m_pMat);
        MatSetSizes(m_pMat, m_uiNumDofs, m_uiNumDofs, PETSC_DECIDE, PETSC_DECIDE);
        if(m_uiSize > 1) {
            MatSetType(m_pMat, MATMPIAIJ);
            MatMPIAIJSetPreallocation( m_pMat, NNZ, PETSC_NULL, NNZ, PETSC_NULL );
        }
        else {
            MatSetType(m_pMat, MATSEQAIJ);
            MatSeqAIJSetPreallocation(m_pMat, NNZ, PETSC_NULL);
        }
        // this will disable on preallocation errors (but not good for performance)
        MatSetOption(m_pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

        return Error::SUCCESS;
    } // aMatBased::set_map


    template <typename DT, typename GI, typename LI>
    Error aMatBased<DT,GI,LI>::update_map(const LI* new_to_old_rank_map,
                                        const LI old_n_all_dofs_on_rank,
                                        const GI* old_rank_to_global_map,
                                        const LI n_elements_on_rank,
                                        const LI* const * element_to_rank_map,
                                        const LI* dofs_per_element,
                                        const LI n_all_dofs_on_rank,
                                        const GI* rank_to_global_map,
                                        const GI owned_global_dof_range_begin,
                                        const GI owned_global_dof_range_end,
                                        const GI n_global_dofs) {

        // It is assumed that total number of owned elements is unchanged
        assert(m_uiNumElems == n_elements_on_rank);

        // point to new provided array giving number of dofs of each element
        m_uiDofsPerElem = dofs_per_element;

        // update new global map (number dofs per element is changed, but number of owned elements is intact)
        if (m_ulpMap != nullptr){
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                delete[] m_ulpMap[eid];
            }
        }
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            m_ulpMap[eid] = new GI[m_uiDofsPerElem[eid]];
        }
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            for (LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++){
                m_ulpMap[eid][nid] = rank_to_global_map[element_to_rank_map[eid][nid]];
            }
        }

        // update number of owned dofs
        m_uiNumDofs = owned_global_dof_range_end - owned_global_dof_range_begin + 1;

        // update total dofs of all ranks, currently not use by aMatBased
        m_ulNumDofsGlobal = n_global_dofs;

        /*unsigned long nl = m_uiNumDofs;
        unsigned long ng;
        MPI_Allreduce( &nl, &ng, 1, MPI_LONG, MPI_SUM, m_comm );
        assert( n_global_dofs == ng );*/

        // allocate new (larger) matrix of size m_uiNumDofs
        if( m_pMat != nullptr ){
            MatDestroy( &m_pMat );
            m_pMat = nullptr;
        }
        MatCreate( m_comm, &m_pMat );
        MatSetSizes( m_pMat, m_uiNumDofs, m_uiNumDofs, PETSC_DECIDE, PETSC_DECIDE );
        if( m_uiSize > 1 ) {
            // initialize matrix
            MatSetType(m_pMat, MATMPIAIJ);
            MatMPIAIJSetPreallocation(m_pMat, 30, PETSC_NULL, 30, PETSC_NULL);
        }
        else {
            MatSetType(m_pMat, MATSEQAIJ);
            MatSeqAIJSetPreallocation(m_pMat, 30, PETSC_NULL);
        }

        return Error::SUCCESS;
    } // aMatBased::update_map()


    template <typename DT, typename GI, typename LI>
    Error aMatBased<DT,GI,LI>::set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim ){
        aMatBased<DT,GI,LI>::petsc_set_element_matrix(eid, e_mat, block_i, block_j, ADD_VALUES);
        return Error::SUCCESS;
    } // aMatBased::set_element_matrix


    // use with Eigen, matrix-based, set every row of the matrix (faster than set every term of the matrix)
    template <typename DT,typename GI, typename LI>
    Error aMatBased<DT,GI,LI>::petsc_set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, InsertMode mode ) {

        #ifdef AMAT_PROFILER
            timing_aMat[static_cast<int>(PROFILER::PETSC_ASS)].start();
        #endif

        // this is number of dofs per block:
        LI num_dofs_per_block = e_mat.rows();
        assert(num_dofs_per_block == e_mat.cols());

        // assemble global matrix (petsc matrix)
        // now set values ...
        std::vector<PetscScalar> values(num_dofs_per_block);
        std::vector<PetscInt> colIndices(num_dofs_per_block);
        PetscInt rowId;
        for (LI r = 0; r < num_dofs_per_block; ++r) {
            // this ONLY WORKS with assumption that all blocks have the same number of dofs (that is true for RXFEM ?)
            rowId = m_ulpMap[eid][block_i * num_dofs_per_block + r];
            for (LI c = 0; c < num_dofs_per_block; ++c) {
                colIndices[c] = m_ulpMap[eid][block_j * num_dofs_per_block + c];
                values[c] = e_mat(r,c);
            } // c
            //MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), colIndices.data(), values.data(), mode);
        } // r

        // compute the trace of matrix for penalty method
        if (m_BcMeth == BC_METH::BC_PENALTY){
            for (LI r = 0; r < num_dofs_per_block; r++) m_dtTraceK += e_mat(r,r);
        }

        #ifdef AMAT_PROFILER
            timing_aMat[static_cast<int>(PROFILER::PETSC_ASS)].stop();
        #endif

        return Error::SUCCESS;
    } // aMatBased::petsc_set_element_matrix


    template <typename DT, typename GI, typename LI>
    Error aMatBased<DT,GI,LI>::petsc_dump_mat( const char* filename /* = nullptr */ ) {

        if( m_pMat == nullptr ) {
            std::cout << "Matrix has not yet been allocated, can't display...\n";
            return Error::SUCCESS;
        }

        PetscBool assembled = PETSC_FALSE;
        MatAssembled( m_pMat, &assembled );
        if( !assembled ) {
            std::cout << "Matrix has not yet been assembled, can't display...\n";
            return Error::SUCCESS;
        }

        if( filename == nullptr ) {
            MatView( m_pMat, PETSC_VIEWER_STDOUT_WORLD );
        }
        else {
            PetscViewer viewer;
            PetscViewerASCIIOpen( m_comm, filename, &viewer );
            // write to file readable by Matlab (filename must be filename.m in order to execute in Matlab)
            //PetscViewerPushFormat( viewer, PETSC_VIEWER_ASCII_MATLAB );
            MatView( m_pMat, viewer );
            PetscViewerDestroy( &viewer );
        }
        return Error::SUCCESS;
    } // aMatBased::petsc_dump_mat


    template <typename DT,typename GI, typename LI>
    Error aMatBased<DT,GI,LI>::set_bdr_map(GI* constrainedDofs, DT* prescribedValues, LI numConstraints){

        // extract constrained dofs owned by me
        LI local_Id;
        GI global_Id;

        for (LI i = 0; i < numConstraints; i++){
            global_Id = constrainedDofs[i];
            if ((global_Id >= m_ulGlobalDofStart) && (global_Id <= m_ulGlobalDofEnd)) {
                ownedConstrainedDofs.push_back(global_Id);
                ownedPrescribedValues.push_back(prescribedValues[i]);
            }
        }

        // construct elemental map of boundary condition
        m_uipBdrMap = new unsigned int* [m_uiNumElems];
        m_dtPresValMap = new DT* [m_uiNumElems];
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            m_uipBdrMap[eid] = new unsigned int[m_uiDofsPerElem[eid]];
            m_dtPresValMap[eid] = new DT [m_uiDofsPerElem[eid]];
        }

        for (LI eid = 0; eid < m_uiNumElems; eid++){
            for (LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++){
                global_Id = m_ulpMap[eid][nid];
                LI index;
                for (index = 0; index < numConstraints; index++) {
                    if (global_Id == constrainedDofs[index]) {
                        m_uipBdrMap[eid][nid] = 1;
                        m_dtPresValMap[eid][nid] = prescribedValues[index];
                        break;
                    }
                }
                if (index == numConstraints){
                    m_uipBdrMap[eid][nid] = 0;
                    m_dtPresValMap[eid][nid] = -1E16; // for testing
                    if ((global_Id >= m_ulGlobalDofStart) && (global_Id <= m_ulGlobalDofEnd)) {
                        ownedFreeDofs.push_back(global_Id);
                    }
                }
            }
        }

        //if (ownedFreeDofs_unsorted.size() > 0){
        if (ownedFreeDofs.size() > 0){
            std::sort(ownedFreeDofs.begin(), ownedFreeDofs.end());
            ownedFreeDofs.erase(std::unique(ownedFreeDofs.begin(),ownedFreeDofs.end()), ownedFreeDofs.end());
        }

        if (m_uiRank == 0){
            if (m_BcMeth == BC_METH::BC_IMATRIX){
                printf("Apply BC: use identity-matrix method\n");
            } else if (m_BcMeth == BC_METH::BC_PENALTY){
                printf("Apply BC: use penaly method\n");
            } else {
                return Error::UNKNOWN_BC_METH;
            }
        }

        if (m_BcMeth == BC_METH::BC_IMATRIX){
            // allocate KfcUc with size = m_uiNumDofs, this will be subtracted from rhs to apply bc
            this->petsc_create_vec( KfcUcVec );

        } else if (m_BcMeth == BC_METH::BC_PENALTY){
            // initialize the trace of matrix, used to form the (big) penalty number
            m_dtTraceK = 0.0;
        }

        return Error::SUCCESS;
    }// aMatBased::set_bdr_map


    // apply Dirichlet bc by modifying matrix, only used in matrix-based approach
    template <typename DT, typename GI, typename LI>
    Error aMatBased<DT,GI,LI>::apply_bc_mat() {
        LI num_dofs_per_elem;
        PetscInt rowId, colId;

        for (LI eid = 0; eid < m_uiNumElems; eid ++) {
            num_dofs_per_elem = m_uiDofsPerElem[eid];
            if (m_BcMeth == BC_METH::BC_IMATRIX){
                for (LI r = 0; r < num_dofs_per_elem; r++) {
                    rowId = m_ulpMap[eid][r];
                    if (m_uipBdrMap[eid][r] == 1) {
                        for (LI c = 0; c < num_dofs_per_elem; c++) {
                            colId = m_ulpMap[eid][c];
                            if (colId == rowId) {
                                MatSetValue(m_pMat, rowId, colId, 1.0, INSERT_VALUES);
                            } else {
                                MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                            }
                        }
                    } else {
                        for (LI c = 0; c < num_dofs_per_elem; c++) {
                            colId = m_ulpMap[eid][c];
                            if (m_uipBdrMap[eid][c] == 1) {
                                MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                            }
                        }
                    }
                }
            } else if (m_BcMeth == BC_METH::BC_PENALTY){
                for (LI r = 0; r < num_dofs_per_elem; r++){
                    rowId = m_ulpMap[eid][r];
                    if (m_uipBdrMap[eid][r] == 1) {
                        for (LI c = 0; c < num_dofs_per_elem; c++) {
                            colId = m_ulpMap[eid][c];
                            if (colId == rowId) {
                                MatSetValue(m_pMat, rowId, colId, PENALTY_FACTOR * m_dtTraceK, INSERT_VALUES);
                            }
                        }
                    }
                }
            }
        }
        return Error::SUCCESS;
    } // aMatBased::apply_bc_mat


    template <typename DT, typename GI, typename LI>
    Error aMatBased<DT,GI,LI>::apply_bc_rhs(Vec rhs){

        // set rows associated with constrained dofs to be equal to Uc
        PetscInt global_Id;
        PetscScalar value, value1, value2;

        // compute KfcUc
        if (m_BcMeth == BC_METH::BC_IMATRIX){
            #ifdef AMAT_PROFILER
                timing_aMat[static_cast<int>(PROFILER::PETSC_KfcUc)].start();
            #endif
            Vec uVec, vVec;
            this->petsc_create_vec( uVec );
            this->petsc_create_vec( vVec );

            // [uVec] contains the prescribed values at location of constrained dofs
            for (LI r = 0; r < ownedConstrainedDofs.size(); r++){
                value = ownedPrescribedValues[r];
                global_Id = ownedConstrainedDofs[r];
                VecSetValue(uVec, global_Id, value, INSERT_VALUES);
            }

            // multiply [K][uVec] = [vVec] where locations of free dofs equal to [Kfc][Uc]
            MatMult(m_pMat, uVec, vVec);
            //dump_mat();
            VecAssemblyBegin( vVec );
            VecAssemblyEnd( vVec );

            // extract the values of [Kfc][Uc] from vVec and set to KfcUcVec
            std::vector<PetscScalar> KfcUc_values(ownedFreeDofs.size());
            std::vector<PetscInt> KfcUc_indices(ownedFreeDofs.size());
            for (LI r = 0; r < ownedFreeDofs.size(); r++){
                KfcUc_indices[r] = ownedFreeDofs[r];
            }
            VecGetValues(vVec, KfcUc_indices.size(), KfcUc_indices.data(), KfcUc_values.data());
            for (LI r = 0; r < ownedFreeDofs.size(); r++){
                KfcUc_values[r] = -1.0*KfcUc_values[r];
            }

            VecSetValues(KfcUcVec, KfcUc_indices.size(), KfcUc_indices.data(), KfcUc_values.data(), ADD_VALUES);

            //petsc_destroy_vec(uVec);
            //petsc_destroy_vec(vVec);
            VecDestroy(&uVec);
            VecDestroy(&vVec);

            #ifdef AMAT_PROFILER
                timing_aMat[static_cast<int>(PROFILER::PETSC_KfcUc)].stop();
            #endif
        } // if (m_BcMeth == BC_METH::BC_IMATRIX)

        // modify Fc
        for (LI nid = 0; nid < ownedConstrainedDofs.size(); nid++){
            global_Id = ownedConstrainedDofs[nid];
            if (m_BcMeth == BC_METH::BC_IMATRIX){
                value = ownedPrescribedValues[nid];
                VecSetValue(rhs, global_Id, value, INSERT_VALUES);
            } else if (m_BcMeth == BC_METH::BC_PENALTY){
                value = PENALTY_FACTOR * m_dtTraceK * ownedPrescribedValues[nid];
                VecSetValue(rhs, global_Id, value, INSERT_VALUES);
            }

        }

        // modify Ff for the case of BC_IMATRIX
        if (m_BcMeth == BC_METH::BC_IMATRIX){
            // need to finalize vector KfcUcVec before extracting its value
            VecAssemblyBegin( KfcUcVec );
            VecAssemblyEnd( KfcUcVec );

            for (LI nid = 0; nid < ownedFreeDofs.size(); nid++){
                global_Id = ownedFreeDofs[nid];
                VecGetValues(KfcUcVec, 1, &global_Id, &value1);
                VecGetValues(rhs, 1, &global_Id, &value2);
                value = value1 + value2;
                VecSetValue(rhs, global_Id, value, INSERT_VALUES);
            }
            VecDestroy(&KfcUcVec);
        }

        //petsc_destroy_vec(KfcUcVec);

        return Error::SUCCESS;
    } // aMatBased::apply_bc_rhs


    template <typename DT, typename GI, typename LI>
    Error aMatBased<DT,GI,LI>::petsc_solve( const Vec rhs, Vec out ) const {
        // abstract Krylov object, linear solver context
        KSP ksp;
        // abstract preconditioner object, pre conditioner context
        PC  pc;
        // default KSP context
        KSPCreate( m_comm, &ksp );

        // set default solver (e.g. KSPCG, KSPFGMRES, ...)
        // could be overwritten at runtime using -ksp_type <type>
        KSPSetType(ksp, KSPCG);
        KSPSetFromOptions(ksp);

        KSPSetTolerances(ksp,1E-12,1E-12,PETSC_DEFAULT, 20000);

        // set the matrix associated the linear system
        KSPSetOperators(ksp, m_pMat, m_pMat);

        // set default preconditioner (e.g. PCJACOBI, PCBJACOBI, ...)
        // could be overwritten at runtime using -pc_type <type>
        KSPGetPC(ksp,&pc);
        PCSetType(pc, PCJACOBI);
        PCSetFromOptions(pc);

        // solve the system
        KSPSolve(ksp, rhs, out); // solve the linear system

        // clean up
        KSPDestroy( &ksp );

        return Error::SUCCESS;
    } // aMatBased::petsc_solve

} // end of namespace par