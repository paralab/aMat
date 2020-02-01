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

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>

#include <vector>
#include <fstream>
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////////////////////

#define AMAT_MAX_CRACK_LEVEL      0                          // number of cracks allowed in 1 element
#define AMAT_MAX_EMAT_PER_ELEMENT (1u<<AMAT_MAX_CRACK_LEVEL) // max number of cracked elements on each element

//////////////////////////////////////////////////////////////////////////////////////////////

namespace par {

    enum class AMAT_TYPE { PETSC_SPARSE, MAT_FREE };

    enum class Error { SUCCESS,
                       INDEX_OUT_OF_BOUNDS,
                       UNKNOWN_ELEMENT_TYPE,
                       UNKNOWN_ELEMENT_STATUS,
                       NULL_L2G_MAP,
                       GHOST_NODE_NOT_FOUND,
                       UNKNOWN_MAT_TYPE,
                       WRONG_COMMUNICATION,
                       UNKNOWN_DOF_TYPE,
                       UNKNOWN_CONSTRAINT };

    enum class DOF_TYPE { FREE, PRESCRIBED, UNDEFINED };

    //////////////////////////////////////////////////////////////////////////////////////////////
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
        inline void allocateSendBuffer(size_t bytes) { m_uiSendBuf = malloc(bytes); }

        /**@brief allocates recv buffer for ghost exchange */
        inline void allocateRecvBuffer(size_t bytes) { m_uiRecvBuf = malloc(bytes); }

        /**@brief allocates send buffer for ghost exchange */
        inline void deAllocateSendBuffer() {
            free( m_uiSendBuf );
            m_uiSendBuf = nullptr;
        }

        /**@brief allocates recv buffer for ghost exchange */
        inline void deAllocateRecvBuffer() {
            free( m_uiRecvBuf );
            m_uiRecvBuf = nullptr;
        }

        /**@brief */
        inline void* getSendBuffer() { return m_uiSendBuf; }

        /**@brief */
        inline void* getRecvBuffer() { return m_uiRecvBuf; }

        /**@brief */
        inline const void* getBuffer() { return m_uiBuffer; }

        /**@brief */
        inline std::vector<MPI_Request*>& getRequestList() { return m_uiRequests; }

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

    //////////////////////////////////////////////////////////////////////////////////////////////
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

        inline unsigned int getRank() const { return m_uiRank; }
        inline Li getRowId() const { return m_uiRowId; }
        inline Li getColId() const { return m_uiColId; }
        inline Dt getVal()   const { return m_dtVal; }

        inline void setRank(  unsigned int rank ) { m_uiRank = rank; }
        inline void setRowId( Li rowId ) { m_uiRowId = rowId; }
        inline void setColId( Li colId ) { m_uiColId = colId; }
        inline void setVal(   Dt val ) {   m_dtVal = val; }

        bool operator == (MatRecord const &other) const {
            return ((m_uiRank == other.getRank())&&(m_uiRowId == other.getRowId())&&(m_uiColId == other.getColId()));
        }

        bool operator < (MatRecord const &other) const {
            if (m_uiRank < other.getRank()) {
                return true;
            }
            else if (m_uiRank == other.getRank()) {
                if (m_uiRowId < other.getRowId()) {
                    return true;
                }
                else if (m_uiRowId == other.getRowId()) {
                    if (m_uiColId < other.getColId()) {
                        return true;
                    }
                    else {
                        return false;
                    }
                }
                else {
                    return false;
                }
            }
            else {
                return false;
            }
        }

        bool operator <= (MatRecord const &other) const { return (((*this) < other) || (*this) == other); }

        ~MatRecord() {}

    }; // class MatRecord

    // class BoundRecord, designed to be used for get_boundary_dofs(), but no longer in use
    template <typename Dt, typename Gi, typename Li>
    class BoundRecord {
    private:
        // rank who owns boundary dof
        unsigned int m_uiRank;
        // Dof type
        //par::DOF_TYPE m_DofType;
        // global Id of boundary dof
        Gi m_uiDofId;
        // value prescribed at boundary dof
        Dt m_dtPresVal;
    public:
        BoundRecord() {
            m_uiRank = 0;
            m_uiDofId = 0;
            m_dtPresVal = 0;
        }
        BoundRecord(unsigned int rank, Li dofId, Dt presVal){
            m_uiRank = rank;
            m_uiDofId = dofId;
            m_dtPresVal = presVal;
        }
        ~BoundRecord(){}

        inline unsigned int getRank() const {return m_uiRank;}
        //inline par::DOF_TYPE getDofType() const {return m_DofType;}
        inline Li getDofId() const {return m_uiDofId;}
        inline Dt getPresVal() const {return m_dtPresVal;}

        inline void setRank(unsigned int rank) {m_uiRank = rank;}
        //inlinde void setDofType(par::DOF_TYPE dofType) {m_DofType = dofType;}
        inline void setDofId(Li dofId) {m_uiDofId = dofId;}
        inline void setPresVal(Dt presVal) {m_dtPresVal = presVal;}

        bool operator == (BoundRecord const &other) const {
            return ((m_uiRank == other.getRank())&&(m_uiDofId == other.getDofId()));
        }
        bool operator < (BoundRecord const &other) const {
            if (m_uiRank < other.getRank()) {
                return true;
            }
            else if (m_uiRank == other.getRank()) {
                if (m_uiDofId < other.getDofId()) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                return false;
            }
        }
        bool operator <= (BoundRecord const &other) const { return (((*this) < other) || (*this) == other); }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Class aMat
    //
    // DT => type of data stored in matrix (eg: double). GI => size of global index. LI => size of local index

    template <typename DT, typename GI, typename LI>
    class aMat {

        typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> EigenMat;
        typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> EigenVec;

    protected:
        /**@brief Flag to use matrix-free or matrix-based method*/
        AMAT_TYPE m_MatType;

        /**@brief communicator used within aMat */
        MPI_Comm m_comm;
        /**@brief my rank */
        unsigned int m_uiRank;
        /**@brief total number of ranks */
        unsigned int m_uiSize;

        /**@brief (local) number of DoFs owned by rank */
        LI m_uiNumNodes;
        /**@brief (global) number of DoFs owned by all ranks */
        GI m_ulNumNodesGlobal;
        /**@brief start of global ID of owned dofs, just for assertion */
        GI m_ulGlobalNodeStart_assert;
        /**@brief end of global ID of owned dofs, just for assertion */
        GI m_ulGlobalNodeEnd_assert;
        /**@brief total dofs inclulding ghost, just for assertion */
        LI m_uiTotalNodes_assert;
        /**@brief (local) number of elements owned by rank */
        LI m_uiNumElems;
        /**@brief max number of DoFs per element*/
        LI m_uiMaxNodesPerElem;


        /**@brief assembled stiffness matrix */
        Mat m_pMat;
        /**@brief storage of element matrices */
        std::vector<EigenMat>* m_epMat;


        /**@brief map from local DoF of element to global DoF: m_ulpMap[eid][local_id]  = global_id */
        GI** m_ulpMap;

        /**@brief number of dofs per element */
        const LI* m_uiNodesPerElem;

        /**@brief map from local DoF of element to local DoF: m_uiMap[eid][element_node]  = local node-ID */
        const LI* const*  m_uipLocalMap;

        /**@brief map from local DoF to global DoF */
        const GI* m_ulpLocal2Global;

        /**@brief number of DoFs owned by each rank, NOT include ghost DoFs */
        std::vector<LI> m_uivLocalNodeCounts;

        /**@brief number of elements owned by each rank */
        std::vector<LI> m_uivLocalElementCounts;

        /**@brief exclusive scan of (local) number of DoFs */
        std::vector<GI> m_uivLocalNodeScan;

        /**@brief exclusive scan of (local) number of elements */
        std::vector<GI> m_uivLocalElementScan;

        /**@brief number of ghost DoFs owned by "pre" processes (whose ranks are smaller than m_uiRank) */
        LI m_uiNumPreGhostNodes;

        /**@brief total number of ghost DoFs owned by "post" processes (whose ranks are larger than m_uiRank) */
        LI m_uiNumPostGhostNodes;

        /**@brief number of DoFs sent to each process (size = m_uiSize) */
        std::vector<LI> m_uivSendNodeCounts;

        /**@brief offsets (i.e. exclusive scan) of m_uiSendNodeCounts */
        std::vector<LI> m_uivSendNodeOffset;

        /**@brief local DoF IDs to be sent (size = total number of nodes to be sent */
        std::vector<LI> m_uivSendNodeIds;

        /**@brief number of DoFs to be received from each process (size = m_uiSize) */
        std::vector<LI> m_uivRecvNodeCounts;

        /**@brief offsets (i.e. exclusive scan) of m_uiRecvNodeCounts */
        std::vector<LI> m_uivRecvNodeOffset;

        /**@brief local node-ID starting of pre-ghost nodes, always = 0 */
        LI m_uiNodePreGhostBegin;

        /**@brief local node-ID ending of pre-ghost nodes */
        LI m_uiNodePreGhostEnd;

        /**@brief local node-ID starting of nodes owned by me */
        LI m_uiNodeLocalBegin;

        /**@brief local node-ID ending of nodes owned by me */
        LI m_uiNodeLocalEnd;

        /**@brief local node-ID starting of post-ghost nodes */
        LI m_uiNodePostGhostBegin;

        /**@brief local node-ID ending of post-ghost nodes */
        LI m_uiNodePostGhostEnd;

        /**@brief total number of nodes including ghost nodes and nodes owned by me */
        LI m_uiNumNodesTotal;

        /**@brief MPI communication tag*/
        int m_iCommTag;

        /**@brief ghost exchange context*/
        std::vector<AsyncExchangeCtx> m_vAsyncCtx;

        /**@brief matrix record for block jacobi matrix*/
        std::vector<MatRecord<DT,LI>> m_vMatRec;

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

        /**@KfcUc (= Kfc * Uc) used to apply bc for rhs */
        Vec KfcUcVec;


        /**@brief TEMPORARY VARIABLES FOR DEBUGGING */
        Mat m_pMat_matvec; // matrix created by matvec() to compare with m_pMat

        /**@brief VARIABLES ARE NO LONGER USED */
        std::vector<BoundRecord<DT,GI,LI>> m_vBdrRec; // boundary dof Ids and prescribed values for Dirichlet bc
        std::vector<GI> m_vFreeNodes; // free dof Ids
        std::vector< par::DOF_TYPE > m_DofType;
        std::vector< DT > m_dPreDofValue;
        std::vector< GI > m_GlobalDofId;

    public:

        /**@brief constructor to initialize variables of aMat */
        aMat( AMAT_TYPE matType );

        /**@brief destructor of aMat */
        ~aMat();

        inline par::Error set_comm(MPI_Comm comm){
            m_comm = comm;
            MPI_Comm_rank(comm, (int*)&m_uiRank);
            MPI_Comm_size(comm, (int*)&m_uiSize);
            return Error::SUCCESS;
        }

        /**@brief set mapping from element local node to global node */
        par::Error set_map( const LI          n_elements_on_rank,
                            const LI* const * element_to_rank_map,
                            const LI        * dofs_per_element,
                            const LI          n_all_dofs_on_rank, // Note: includes ghost dofs
                            const GI        * rank_to_global_map,
                            const GI          owned_global_dof_range_begin,
                            const GI          owned_global_dof_range_end,
                            const GI          n_global_dofs );

        /**@brief update map when cracks created */
        par::Error update_map(const LI* new_to_old_rank_map, const LI old_n_all_dofs_on_rank, const LI* old_rank_to_global_map,
                              const LI n_elements_on_rank, const GI* const * element_to_rank_map, const LI* dofs_per_element,
                              const LI n_all_dofs_on_rank, const LI* rank_to_global_map, const GI owned_global_dof_range_begin,
                              const GI owned_global_dof_range_end, const GI n_global_dofs);

        /**@brief build scatter-gather map (used for communication) and local-to-local map (used for matvec) */
        par::Error buildScatterMap();

        /**@brief return number of DoFs owned by this rank*/
        inline unsigned int get_local_num_nodes() const {
            return m_uiNumNodes;
        }
        /**@brief return total number of DoFs (owned DoFs + ghost DoFs)*/
        inline unsigned int get_total_local_num_nodes() const {
            return m_uiNumNodesTotal;
        }
        /**@brief return number of elements owned by this rank*/
        inline unsigned int get_local_num_elements() const {
            return m_uiNumElems;
        }
        /**@brief return number of nodes of element eid */
        inline unsigned int get_nodes_per_element (unsigned int eid) const {
            return m_uiNodesPerElem[eid];
        }
        /**@brief return the map from DoF of element to local ID of vector (included ghost DoFs) */
        inline const LI * const * get_e2local_map() const {
            return m_uipLocalMap;
        }
        /**@brief return the map from DoF of element to global ID */
        inline GI** get_e2global_map() const {
            return m_ulpMap;
        }
        /**@brief return the ID of first pre-ghost DoF */
        inline unsigned int get_pre_ghost_begin() const {
            return m_uiNodePreGhostBegin;
        }
        /**@brief return the ID that is 1 bigger than last pre-ghost DoF */
        inline unsigned int get_pre_ghost_end() const {
            return m_uiNodePreGhostEnd;
        }
        /**@brief return the ID of first post-ghost DoF */
        inline unsigned int get_post_ghost_begin() const {
            return m_uiNodePostGhostBegin;
        }
        /**@brief return the ID that is 1 bigger than last post-ghost DoF */
        inline unsigned int get_post_ghost_end() const {
            return m_uiNodePostGhostEnd;
        }
        /**@brief return the ID of first DoF owned by this rank*/
        inline unsigned int get_local_begin() const {
            return m_uiNodeLocalBegin;
        }
        /**@brief return the Id that is 1 bigger than last DoF owned by this rank*/
        inline unsigned int get_local_end() const {
            return m_uiNodeLocalEnd;
        }
        /**@brief return true if DoF "enid" of element "eid" is owned by this rank, false otherwise */
        inline bool is_local_node(unsigned int eid, unsigned int enid) const {
            const unsigned int nid = (const unsigned int)m_uipLocalMap[eid][enid];
            if( nid >= m_uiNodeLocalBegin && nid < m_uiNodeLocalEnd ) {
                return true;
            }
            else {
                return false;
            }
        }

        /**@brief begin assembling the matrix "m_pMat", called after MatSetValues */
        inline par::Error petsc_init_mat( MatAssemblyType mode ) const {
            MatAssemblyBegin( m_pMat, mode );
            return Error::SUCCESS;
        }
        /**@brief complete assembling the matrix "m_pMat", called before using the matrix */
        inline par::Error petsc_finalize_mat( MatAssemblyType mode ) const {
            MatAssemblyEnd( m_pMat, mode );
            return Error::SUCCESS;
        }
        /**@brief begin assembling the petsc vec (defined outside aMat) */
        inline par::Error petsc_init_vec( Vec vec ) const {
            VecAssemblyBegin( vec );
            return Error::SUCCESS;
        }
        /**@brief end assembling the petsc vec (defined outside aMat) */
        inline par::Error petsc_finalize_vec( Vec vec ) const {
            VecAssemblyEnd( vec );
            return Error::SUCCESS;
        }

        /**@brief allocate memory for a PETSc vector "vec", initialized by alpha */
        par::Error petsc_create_vec( Vec &vec, PetscScalar alpha = 0.0 ) const;

        /**@brief assembly global load vector */
        //par::Error petsc_set_element_vec( Vec vec, LI eid, DT* e_vec, LI block_i, LI block_j, InsertMode mode = ADD_VALUES );
        par::Error petsc_set_element_vec( Vec vec, LI eid, EigenVec e_vec, LI block_i, InsertMode mode = ADD_VALUES );

        /**@brief assembly element matrix to structural matrix (for matrix-based method) */
        par::Error petsc_set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, InsertMode = ADD_VALUES );

        /**@brief: write PETSc matrix "m_pMat" to "filename"
         * @param[in] filename: name of file to write matrix to.  If nullptr, then write to stdout.
         */
        par::Error dump_mat( const char* filename = nullptr ); // Note: can't be 'const' because may call matvec which may need MPI data to be stored...

        /**@brief: write PETSc vector "vec" to filename "fvec"
         * @param[in] vec      : petsc vector to write to file
         * @param[in] filename : name of file to write vector to.  If nullptr, then dump to std out.
         */
        par::Error dump_vec( Vec vec, const char* filename = nullptr ) const;

        /**@brief get diagonal of m_pMat and put to vec */
        par::Error petsc_get_diagonal( const Vec & vec ) const;

        /**@brief free memory allocated for PETSc vector*/
        par::Error petsc_destroy_vec( Vec & vec ) const;

        /**@brief allocate memory for "vec", size includes ghost DoFs if isGhosted=true, initialized by alpha */
        par::Error create_vec( DT* &vec, bool isGhosted = false, DT alpha = (DT)0.0 ) const;

        /**@brief allocate memory for "mat", size includes ghost DoFs if isGhosted=true, initialized by alpha */
        par::Error create_mat( DT** &mat, bool isGhosted = false, DT alpha = (DT)0.0 );

        /**@brief copy local to corresponding positions of gVec (size including ghost DoFs) */
        par::Error local_to_ghost(DT*  gVec, const DT* local) const;

        /**@brief copy gVec (size including ghost DoFs) to local (size of local DoFs) */
        par::Error ghost_to_local(DT* local, const DT* gVec) const;

        /**@brief copy element matrix and store in m_mats, used for matrix-free method */
        par::Error copy_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim );

        /**@brief get diagonal terms of structure matrix by accumulating diagonal of element matrices */
        par::Error mat_get_diagonal(DT* diag, bool isGhosted = false);

        /**@brief get diagonal terms with ghosted vector diag */
        par::Error mat_get_diagonal_ghosted(DT* diag);

        /**@brief compute the rank who owns gId */
        unsigned int globalId_2_rank(GI gId) const;

        /**@brief get diagonal block matrix (sparse matrix) */
        par::Error mat_get_diagonal_block(std::vector<MatRecord<DT,LI>> &diag_blk);

        /**@brief get max number of DoF per element*/
        par::Error get_max_dof_per_elem();

        /**@brief free memory allocated for vec and set vec to null */
        par::Error destroy_vec(DT* &vec);

        /**@brief free memory allocated for matrix mat and set mat to null */
        par::Error destroy_mat(DT** &mat, const unsigned int nrow);

        /**@brief begin: owned DoFs send, ghost DoFs receive, called before matvec() */
        par::Error ghost_receive_begin(DT* vec);

        /**@brief end: ghost DoFs receive, called before matvec() */
        par::Error ghost_receive_end(DT* vec);

        /**@brief begin: ghost DoFs send, owned DoFs receive and accumulate to current data, called after matvec() */
        par::Error ghost_send_begin(DT* vec);

        /**@brief end: ghost DoFs send, owned DoFs receive and accumulate to current data, called after matvec() */
        par::Error ghost_send_end(DT* vec);

        /**@brief v = K * u (K is not assembled, but directly using elemental K_e's).  v (the result) must be allocated by the caller.
         * @param[in] isGhosted = true, if v and u are of size including ghost DoFs
         * @param[in] isGhosted = false, if v and u are of size NOT including ghost DoFs
         * */
        par::Error matvec(DT* v, const DT* u, bool isGhosted = false);

        // FIXME: internal only call (move to private)?  Use matvec instead?
        /**@brief v = K * u; v and u are of size including ghost DoFs. */
        par::Error matvec_ghosted(DT* v, DT* u);

        /**@brief matrix-free version of MatMult of PETSc */
        PetscErrorCode MatMult_mf(Mat A, Vec u, Vec v);

        /**@brief matrix-free version of MatGetDiagonal of PETSc */
        PetscErrorCode MatGetDiagonal_mf( Mat A, Vec d );

        /**@brief matrix-free version of MatGetDiagonalBlock of PETSc */
        PetscErrorCode MatGetDiagonalBlock_mf( Mat A, Mat* a );

        /**@brief pointer function points to MatMult_mt */
        inline std::function<PetscErrorCode(Mat,Vec,Vec)>* get_MatMult_func(){

            std::function<PetscErrorCode(Mat,Vec,Vec)>* f= new std::function<PetscErrorCode(Mat, Vec, Vec)>();

            (*f) = [this](Mat A, Vec u, Vec v){
                this->MatMult_mf(A, u, v);
                return 0;
            };
            return f;
        }

        /**@brief pointer function points to MatGetDiagonal_mf */
        inline std::function<PetscErrorCode(Mat,Vec)>* get_MatGetDiagonal_func(){

            std::function<PetscErrorCode(Mat,Vec)>* f= new std::function<PetscErrorCode(Mat, Vec)>();

            (*f) = [this](Mat A, Vec d){
                this->MatGetDiagonal_mf(A, d);
                return 0;
            };
            return f;
        }

        /**@brief pointer function points to MatGetDiagonalBlock_mf */
        inline std::function<PetscErrorCode(Mat, Mat*)>* get_MatGetDiagonalBlock_func(){

            std::function<PetscErrorCode(Mat,Mat*)>* f= new std::function<PetscErrorCode(Mat, Mat*)>();

            (*f) = [this](Mat A, Mat* a){
                this->MatGetDiagonalBlock_mf(A, a);
                return 0;
            };
            return f;
        }

        /**@brief set boundary data */
        par::Error set_bdr_map(GI* constrainedDofs, DT* prescribedValues, LI numConstraints);

        /**@brief apply Dirichlet BCs by modifying the matrix "m_pMat" */
        par::Error apply_bc_mat();

        /**@brief apply Dirichlet BCs to diagonal vector used for Jacobi preconditioner */
        par::Error apply_bc_diagonal(Vec rhs);

        /**@brief apply Dirichlet BCs to block diagonal matrix */
        par::Error apply_bc_blkdiag(Mat* blkdiagMat);

        /**@brief apply Dirichlet BCs by modifying the rhs vector, also used for diagonal vector in Jacobi precondition*/
        par::Error apply_bc_rhs( Vec rhs );

        /**@brief: invoke basic PETSc solver, "out" is solution vector */
        par::Error petsc_solve( const Vec rhs, Vec out ) const;


        /**@brief ********* FUNCTIONS FOR DEBUGGING **************************************************/
        inline void echo_rank() const {
            printf("echo from rank= %d\n", m_uiRank);
        }

        inline par::Error petsc_init_mat_matvec( MatAssemblyType mode ) const {
            MatAssemblyBegin( m_pMat_matvec, mode );
            return Error::SUCCESS;
        }

        inline par::Error petsc_finalize_mat_matvec( MatAssemblyType mode ) const{
            MatAssemblyEnd( m_pMat_matvec, mode );
            return Error::SUCCESS;
        }

        inline par::Error set_Local2Global( GI* local_to_global ){
            m_ulpLocal2Global = local_to_global;
            return Error::SUCCESS;
        }

        /**@brief create pestc matrix with size of m_uniNumNodes^2, used in testing matvec() */
        par::Error petsc_create_matrix_matvec();

        /**@brief assemble matrix term by term so that we can control not to assemble "almost zero" terms*/
        par::Error set_element_matrix_term_by_term( LI eid, EigenMat e_mat, InsertMode mode = ADD_VALUES );

        /**@brief compare 2 matrices */
        par::Error petsc_compare_matrix();

        /**@brief compute the norm of the diffference of 2 matrices*/
        par::Error petsc_norm_matrix_difference();

        /**@brief print out to file matrix "m_pMat_matvec" (matrix created by using matvec() to multiply
         * m_pMat with series of vectors [1 0 0...]
         * @param[in] filename : name of file to write vector to.  If nullptr, then dump to std out.
         */
        par::Error dump_mat_matvec( const char* filename = nullptr ) const;

        /**@brief y = m_pMat * x */
        par::Error petsc_matmult( Vec x, Vec result );

        /**@brief set entire vector "vec" to the column "nonzero_row" of matrix m_pMat_matvec, to compare with m_pMat*/
        par::Error petsc_set_matrix_matvec( DT* vec, unsigned int nonzero_row, InsertMode mode = ADD_VALUES );

        /**@brief: test only: display all components of vector on screen */
        par::Error print_vector( const DT* vec, bool ghosted = false );

        /**@brief: test only: display all element matrices (for purpose of debugging) */
        par::Error print_matrix();

        /**@brief: transform vec to pestc vector (for comparison between matrix-free and matrix-based)*/
        par::Error transform_to_petsc_vector(const DT* vec, Vec petsc_vec, bool ghosted = false);

        /**@brief: apply zero Dirichlet boundary condition on nodes dictated by dirichletBMap */
        par::Error set_vector_bc(DT* vec, unsigned int eid, const GI **dirichletBMap);

        /**@brief get diagonal block matrix for sequential code (no communication among ranks) */
        par::Error mat_get_diagonal_block_seq(DT **diag_blk);

        /**@brief copy gMat (size including ghost DoFs) to lMat (size of local DoFs) */
        par::Error ghost_to_local_mat(DT**  lMat, DT** gMat) const;

        /**@brief assemble element matrix to global matrix for matrix-based, not using Eigen */
        par::Error petsc_set_element_matrix( LI eid, DT *e_mat, InsertMode mode = ADD_VALUES );

        par::Error print_mepMat();

        /**@brief get diagonal block matrix (dense matrix) */
        par::Error mat_get_diagonal_block_dense(DT **diag_blk);

        /**@brief ********** FUNCTIONS ARE NO LONGER IN USE, JUST FOR REFERENCE *********************/
        /**@brief assembly element matrix to structure matrix, multiple levels of twining
         * @param[in] e_mat : element stiffness matrices (pointer)
         * @param[in] twin_level: level of twinning (0 no crack, 1 one crack, 2 two cracks, 3 three cracks)
         * */
        par::Error set_element_matrices( LI eid, const EigenMat* e_mat, unsigned int twin_level, InsertMode mode = ADD_VALUES);
        par::Error petsc_set_element_matrix( LI eid, const EigenMat e_mat, LI e_mat_id, InsertMode mode = ADD_VALUES );

    }; // end class aMat


    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // context for aMat

    template <typename DT, typename GI, typename LI>
    struct aMatCTX {
        par::aMat<DT,GI,LI> * aMatPtr;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    // matrix shell to use aMat::MatMult_mf
    template<typename DT,typename GI, typename LI>
    PetscErrorCode aMat_matvec( Mat A, Vec u, Vec v )
    {
        aMatCTX<DT,GI, LI> * pCtx;
        MatShellGetContext( A, &pCtx );

        par::aMat<DT, GI, LI> * pLap = pCtx->aMatPtr;
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

        par::aMat<DT,GI,LI> * pLap = pCtx->aMatPtr;
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

        par::aMat<DT,GI,LI> * pLap = pCtx->aMatPtr;
        std::function<PetscErrorCode(Mat, Mat*)>* f = pLap->get_MatGetDiagonalBlock_func();
        (*f)(A, a);
        delete f;
        return 0;
    }

    // aMat constructor
    template <typename DT,typename GI, typename LI>
    aMat<DT,GI,LI>::aMat( AMAT_TYPE matType ){
        m_MatType          = matType;       // set type of matrix (matrix-based or matrix-free)
        m_uiNumNodes       = 0;             // umber of local dofs
        m_ulNumNodesGlobal = 0;             // number of global dofs
        m_uiNumElems       = 0;             // number of local elements
        m_ulpMap           = nullptr;       // local-to-global map
        m_uiNodesPerElem   = nullptr;       // number of dofs per element
        m_epMat            = nullptr;       // element matrices (Eigen matrix), used in matrix-free
        m_pMat             = nullptr;       // structure matrix, used in matrix-based
        m_comm             = MPI_COMM_NULL; // communication of aMat
        if( matType == AMAT_TYPE::MAT_FREE ){
            m_iCommTag = 0;         // tag for sends & receives used in matvec and mat_get_diagonal_block_seq
        }
        //m_uipBdrMap = nullptr;
        //m_dtPresValMap = nullptr;
    }// constructor

    template <typename DT,typename GI, typename LI>
    aMat<DT,GI,LI>::~aMat()
    {
        delete [] m_epMat;
        if( m_MatType == AMAT_TYPE::MAT_FREE ){
            for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
                delete [] m_ulpMap[eid]; //delete array
            }
            delete [] m_ulpMap;
        }
        else if( m_MatType == AMAT_TYPE::PETSC_SPARSE ) {
            //MatDestroy(&m_pMat);
        }

        /*for (LI eid = 0; eid < m_uiNumElems; eid++){
            if (m_uipBdrMap[eid] != nullptr) delete [] m_uipBdrMap[eid];
            if (m_dtPresValMap[eid] != nullptr) delete [] m_dtPresValMap[eid];
        }
        delete [] m_uipBdrMap;
        delete [] m_dtPresValMap;*/
    } // ~aMat

    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::set_map( const LI           n_elements_on_rank,
                                        const LI * const * element_to_rank_map,
                                        const LI         * dofs_per_element,
                                        const LI           n_all_dofs_on_rank,
                                        const GI         * rank_to_global_map,
                                        const GI           owned_global_dof_range_begin,
                                        const GI           owned_global_dof_range_end,
                                        const GI           n_global_dofs ){

        m_uiNumElems = n_elements_on_rank; // This is number of owned element
        m_uiNumNodes = owned_global_dof_range_end - owned_global_dof_range_begin + 1;

        m_ulNumNodesGlobal = n_global_dofs; // currently this is not used
        m_ulGlobalNodeStart_assert = owned_global_dof_range_begin; // this will be used for assertion in buildScatterMap
        m_ulGlobalNodeEnd_assert = owned_global_dof_range_end; // this will be used for assertion in buildScatterMap
        m_uiTotalNodes_assert = n_all_dofs_on_rank; // this will be used for assertion in buildScatterMap

        // point to provided local map
        m_uiNodesPerElem = dofs_per_element;

        // point to rank_to_global map
        m_ulpLocal2Global = rank_to_global_map;

        m_ulpMap = new GI* [m_uiNumElems];
        for( LI eid = 0; eid < m_uiNumElems; eid++ ){
            m_ulpMap[eid] = new GI [m_uiNodesPerElem[eid]];
        }
        for( LI eid = 0; eid < m_uiNumElems; eid++ ){
            for( unsigned int nid = 0; nid < m_uiNodesPerElem[eid]; nid++ ){
                m_ulpMap[eid][nid] = rank_to_global_map[element_to_rank_map[eid][nid]];
            }
        }

        if( m_MatType == AMAT_TYPE::PETSC_SPARSE ){
            MatCreate(m_comm, &m_pMat);
            MatSetSizes(m_pMat, m_uiNumNodes, m_uiNumNodes, PETSC_DECIDE, PETSC_DECIDE);
            if(m_uiSize > 1) {
                MatSetType(m_pMat, MATMPIAIJ);
                MatMPIAIJSetPreallocation( m_pMat, 30, PETSC_NULL, 30, PETSC_NULL );
            }
            else {
                MatSetType(m_pMat, MATSEQAIJ);
                MatSeqAIJSetPreallocation(m_pMat, 30, PETSC_NULL);
            }
            // this will disable on preallocation errors (but not good for performance)
            MatSetOption(m_pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

            //LocaMap is required by buildScatterMap which is used for apply boundary condition
            m_uipLocalMap = element_to_rank_map;
            buildScatterMap();
        }
        else if( m_MatType == AMAT_TYPE::MAT_FREE ){
            if( m_epMat != nullptr) {
                delete [] m_epMat;
                m_epMat = nullptr;
            }
            m_epMat = new std::vector<EigenMat>[m_uiNumElems];

            // point to provided local Map
            m_uipLocalMap = element_to_rank_map;

            buildScatterMap();

            get_max_dof_per_elem();

        }
        else {
            std::cout << "ERROR: mat type is unknown: " << (int)m_MatType << "\n";
            return Error::UNKNOWN_MAT_TYPE;
        }
        return Error::SUCCESS;

    } //set_map


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::update_map(const LI* new_to_old_rank_map, const LI old_n_all_dofs_on_rank,
                          const LI* old_rank_to_global_map, const LI n_elements_on_rank,
                          const GI* const * element_to_rank_map, const LI* dofs_per_element,
                          const LI n_all_dofs_on_rank, const LI* rank_to_global_map,
                          const GI owned_global_dof_range_begin, const GI owned_global_dof_range_end, const GI n_global_dofs) {

        // It is assumed that total number of owned elements is unchanged
        assert(m_uiNumElems == n_elements_on_rank);

        // delete old global map m_ulpMap[eid][:], keep [eid] since number of total elements does not change
        if (m_ulpMap != nullptr){
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                delete[] m_ulpMap[eid];
            }
        }

        // point to new dofs_per_element map
        m_uiNodesPerElem = dofs_per_element;

        // update new rank_to_global map
        m_ulpLocal2Global = rank_to_global_map;

        // allocate with new dofs per element, and update new global map
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            m_ulpMap[eid] = new GI[m_uiNodesPerElem[eid]];
        }
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            for (LI nid = 0; nid < m_uiNodesPerElem[eid]; nid++){
                m_ulpMap[eid][nid] = rank_to_global_map[element_to_rank_map[eid][nid]];
            }
        }

        // update total number of owned dofs
        m_uiNumNodes = owned_global_dof_range_end - owned_global_dof_range_begin;

        // update total number of dofs on all ranks, currently not in use
        m_ulNumNodesGlobal = n_global_dofs;

        /*unsigned long nl = m_uiNumNodes;
        unsigned long ng;
        MPI_Allreduce( &nl, &ng, 1, MPI_LONG, MPI_SUM, m_comm );
        assert( n_global_dofs == ng );*/

        // update variables for assertion in buildScatterMap
        m_ulGlobalNodeStart_assert = owned_global_dof_range_begin;
        m_ulGlobalNodeEnd_assert = owned_global_dof_range_end;
        m_uiTotalNodes_assert = n_all_dofs_on_rank;

        // allocate new (larger) matrix of size m_uiNumNodes
        if( m_MatType == AMAT_TYPE::PETSC_SPARSE ) {
            if( m_pMat != nullptr ){
                MatDestroy( &m_pMat );
                m_pMat = nullptr;
            }
            MatCreate( m_comm, &m_pMat );
            MatSetSizes( m_pMat, m_uiNumNodes, m_uiNumNodes, PETSC_DECIDE, PETSC_DECIDE );
            if( m_uiSize > 1 ) {
                // initialize matrix
                MatSetType(m_pMat, MATMPIAIJ);
                MatMPIAIJSetPreallocation(m_pMat, 30, PETSC_NULL, 30, PETSC_NULL);
            }
            else {
                MatSetType(m_pMat, MATSEQAIJ);
                MatSeqAIJSetPreallocation(m_pMat, 30, PETSC_NULL);
            }

            //LocaMap is required by buildScatterMap which is used for apply boundary condition
            m_uipLocalMap = element_to_rank_map;
            buildScatterMap();
        }
        else if (m_MatType == AMAT_TYPE::MAT_FREE){
            // Note: currently we delete all old element matrices and need to add new element matrices again
            // Todo: only add newly formed blocks?
            for (LI eid = 0; eid < m_uiNumElems; eid++){
                m_epMat[eid].clear();
            }

            // point to new local map
            m_uipLocalMap = element_to_rank_map;

            // build scatter map
            buildScatterMap();
        }

        return Error::SUCCESS;

    } // update_map()


    // build scatter map
    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::buildScatterMap() {
        /* Assumptions: We assume that the global nodes are continuously partitioned across processors.
           Currently we do not account for twin elements
           "node" is actually "dof" because the map is in terms of dofs */

        if( m_ulpMap == nullptr ) { return Error::NULL_L2G_MAP; }

        m_uivLocalNodeCounts.clear();
        m_uivLocalElementCounts.clear();
        m_uivLocalNodeScan.clear();
        m_uivLocalElementScan.clear();

        m_uivLocalNodeCounts.resize(m_uiSize);
        m_uivLocalElementCounts.resize(m_uiSize);
        m_uivLocalNodeScan.resize(m_uiSize);
        m_uivLocalElementScan.resize(m_uiSize);

        // gather local counts
        MPI_Allgather(&m_uiNumNodes, 1, MPI_INT, &(*(m_uivLocalNodeCounts.begin())), 1, MPI_INT, m_comm);
        MPI_Allgather(&m_uiNumElems, 1, MPI_INT, &(*(m_uivLocalElementCounts.begin())), 1, MPI_INT, m_comm);

        // scan local counts to determine owned-range:
        // range of global ID of owned dofs = [m_uivLocalNodeScan[m_uiRank], m_uivLocalNodeScan[m_uiRank] + m_uiNumNodes)
        m_uivLocalNodeScan[0] = 0;
        m_uivLocalElementScan[0] = 0;
        for (unsigned int p = 1; p < m_uiSize; p++) {
            m_uivLocalNodeScan[p] = m_uivLocalNodeScan[p-1] + m_uivLocalNodeCounts[p-1];
            m_uivLocalElementScan[p] = m_uivLocalElementScan[p-1] + m_uivLocalElementCounts[p-1];
        }

        // dofs are not owned by me: stored in pre or post lists
        std::vector<GI> preGhostGIds;
        std::vector<GI> postGhostGIds;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++) {
            unsigned int num_nodes = m_uiNodesPerElem[eid];
            for (unsigned int i = 0; i < num_nodes; i++) {
                // global ID
                const unsigned int nid = m_ulpMap[eid][i];
                if (nid < m_uivLocalNodeScan[m_uiRank]) {
                    // dofs with global ID < owned-range --> pre-ghost dofs
                    assert( nid < m_ulGlobalNodeStart_assert);
                    preGhostGIds.push_back(nid);
                } else if (nid >= (m_uivLocalNodeScan[m_uiRank] + m_uiNumNodes)){
                    // dofs with global ID > owned-range --> post-ghost dofs
                    assert( nid > m_ulGlobalNodeEnd_assert);
                    postGhostGIds.push_back(nid);
                } else {
                    assert ((nid >= m_uivLocalNodeScan[m_uiRank])  && (nid< (m_uivLocalNodeScan[m_uiRank] + m_uiNumNodes)));
                    assert((nid >= m_ulGlobalNodeStart_assert) && (nid <= m_ulGlobalNodeEnd_assert));
                }
            }
        }

        // sort in ascending order
        std::sort(preGhostGIds.begin(), preGhostGIds.end());
        std::sort(postGhostGIds.begin(), postGhostGIds.end());

        // remove consecutive duplicates and erase all after .end()
        preGhostGIds.erase(std::unique(preGhostGIds.begin(), preGhostGIds.end()), preGhostGIds.end());
        postGhostGIds.erase(std::unique(postGhostGIds.begin(), postGhostGIds.end()), postGhostGIds.end());

        // number of ghost dofs
        m_uiNumPreGhostNodes = preGhostGIds.size();
        m_uiNumPostGhostNodes = postGhostGIds.size();

        // range of local ID of pre-ghost dofs = [0, m_uiNodePreGhostEnd)
        m_uiNodePreGhostBegin = 0;
        m_uiNodePreGhostEnd = m_uiNumPreGhostNodes;

        // range of local ID of owned dofs = [m_uiNodeLocalBegin, m_uiNodeLocalEnd)
        m_uiNodeLocalBegin = m_uiNodePreGhostEnd;
        m_uiNodeLocalEnd = m_uiNodeLocalBegin + m_uiNumNodes;

        // range of local ID of post-ghost dofs = [m_uiNodePostGhostBegin, m_uiNodePostGhostEnd)
        m_uiNodePostGhostBegin = m_uiNodeLocalEnd;
        m_uiNodePostGhostEnd = m_uiNodePostGhostBegin + m_uiNumPostGhostNodes;

        // total number of dofs including ghost dofs
        m_uiNumNodesTotal = m_uiNumNodes + m_uiNumPreGhostNodes + m_uiNumPostGhostNodes;
        assert(m_uiNumNodesTotal == m_uiTotalNodes_assert);

        // determine owners of pre- and post-ghost dofs
        std::vector<unsigned int> preGhostOwner;
        std::vector<unsigned int> postGhostOwner;
        preGhostOwner.resize(m_uiNumPreGhostNodes);
        postGhostOwner.resize(m_uiNumPostGhostNodes);

        // pre-ghost
        unsigned int pcount = 0; // processor count, start from 0
        unsigned int gcount = 0; // global ID count
        while (gcount < m_uiNumPreGhostNodes) {
            // global ID of pre-ghost dof gcount
            unsigned int nid = preGhostGIds[gcount];
            while ((pcount < m_uiRank) &&
                   (!((nid >= m_uivLocalNodeScan[pcount]) && (nid < (m_uivLocalNodeScan[pcount] + m_uivLocalNodeCounts[pcount]))))) {
                // nid is not in the range of global ID of dofs owned by pcount
                pcount++;
            }
            // check if nid is really in the range of global ID of dofs owned by pcount
            if (!((nid >= m_uivLocalNodeScan[pcount]) && (nid < (m_uivLocalNodeScan[pcount] + m_uivLocalNodeCounts[pcount])))) {
                std::cout << "m_uiRank: " << m_uiRank << " pre ghost gid : " << nid << " was not found in any processor" << std::endl;
                return Error::GHOST_NODE_NOT_FOUND;
            }
            preGhostOwner[gcount] = pcount;
            gcount++;
        }

        // post-ghost
        pcount = m_uiRank; // start from my rank
        gcount = 0;
        while(gcount < m_uiNumPostGhostNodes)
            {
                // global ID of post-ghost dof gcount
                unsigned int nid = postGhostGIds[gcount];
                while ((pcount < m_uiSize) &&
                       (!((nid >= m_uivLocalNodeScan[pcount]) && (nid < (m_uivLocalNodeScan[pcount] + m_uivLocalNodeCounts[pcount]))))){
                    // nid is not the range of global ID of dofs owned by pcount
                    pcount++;
                }
                // check if nid is really in the range of global ID of dofs owned by pcount
                if (!((nid >= m_uivLocalNodeScan[pcount]) && (nid < (m_uivLocalNodeScan[pcount] + m_uivLocalNodeCounts[pcount])))) {
                    std::cout << "m_uiRank: " << m_uiRank << " post ghost gid : " << nid << " was not found in any processor" << std::endl;
                    return Error::GHOST_NODE_NOT_FOUND;
                }
                postGhostOwner[gcount] = pcount;
                gcount++;
            }

        unsigned int * sendCounts = new unsigned int[m_uiSize];
        unsigned int * recvCounts = new unsigned int[m_uiSize];
        unsigned int * sendOffset = new unsigned int[m_uiSize];
        unsigned int * recvOffset = new unsigned int[m_uiSize];

        // Note: the send here is just for use in MPI_Alltoallv, it is NOT the send in communications between processors later
        for (unsigned int i = 0; i < m_uiSize; i++) {
            // many of these will be zero, only non zero for processors that own my ghost nodes
            sendCounts[i] = 0;
        }

        // count number of pre-ghost dofs to corresponding owners
        for (unsigned int i = 0; i < m_uiNumPreGhostNodes; i++) {
            sendCounts[preGhostOwner[i]] += 1;
        }

        // count number of post-ghost dofs to corresponding owners
        for (unsigned int i = 0; i < m_uiNumPostGhostNodes; i++) {
            sendCounts[postGhostOwner[i]] += 1;
        }

        // get recvCounts by transposing the matrix of sendCounts
        MPI_Alltoall(sendCounts, 1, MPI_UNSIGNED, recvCounts, 1, MPI_UNSIGNED, m_comm);

        // compute offsets from sends
        sendOffset[0]=0;
        recvOffset[0]=0;
        for(unsigned int i=1; i<m_uiSize; i++) {
            sendOffset[i] = sendOffset[i-1] + sendCounts[i-1];
            recvOffset[i] = recvOffset[i-1] + recvCounts[i-1];
        }

        std::vector<GI> sendBuf;
        std::vector<GI> recvBuf;

        // total elements to be sent => global ID of ghost dofs that are owned by destination rank
        sendBuf.resize(sendOffset[m_uiSize-1] + sendCounts[m_uiSize-1]);
        // total elements to be received => global ID of owned dofs that are ghost in source rank
        recvBuf.resize(recvOffset[m_uiSize-1] + recvCounts[m_uiSize-1]);

        // put global ID of pre- and post-ghost dofs to sendBuf
        for(unsigned int i = 0; i < m_uiNumPreGhostNodes; i++)
            sendBuf[i] = preGhostGIds[i];
        for(unsigned int i = 0; i < m_uiNumPostGhostNodes; i++)
            sendBuf[i + m_uiNumPreGhostNodes] = postGhostGIds[i];

        for(unsigned int i = 0; i < m_uiSize; i++) {
            sendCounts[i] *= sizeof(GI);
            sendOffset[i] *= sizeof(GI);
            recvCounts[i] *= sizeof(GI);
            recvOffset[i] *= sizeof(GI);
        }

        // exchange the global ID of ghost dofs with ranks who own them
        MPI_Alltoallv(&(*(sendBuf.begin())), (int*)sendCounts, (int*)sendOffset, MPI_BYTE,
                      &(*(recvBuf.begin())), (int*)recvCounts, (int*)recvOffset, MPI_BYTE, m_comm);

        for(unsigned int i = 0; i < m_uiSize; i++) {
            sendCounts[i] /= sizeof(GI);
            sendOffset[i] /= sizeof(GI);
            recvCounts[i] /= sizeof(GI);
            recvOffset[i] /= sizeof(GI);
        }

        // compute local ID of owned dofs i (i = 0..recBuf.size()) that need to be send data to where i is a ghost dof
        m_uivSendNodeIds.resize(recvBuf.size());

        for(unsigned int i = 0; i < recvBuf.size(); i++) {
            // global ID of recvBuf[i]
            const unsigned int gid = recvBuf[i];
            // check if gid is really owned by my rank (if not then something goes wrong with sendBuf above
            if (gid < m_uivLocalNodeScan[m_uiRank]  || gid >=  (m_uivLocalNodeScan[m_uiRank] + m_uiNumNodes)) {
                std::cout<<" m_uiRank: "<<m_uiRank<< "scatter map error : "<<__func__<<std::endl;
                par::Error::GHOST_NODE_NOT_FOUND;
            }
            assert((gid >= m_ulGlobalNodeStart_assert) && (gid <= m_ulGlobalNodeEnd_assert));
            // local ID
            m_uivSendNodeIds[i] = m_uiNumPreGhostNodes + (gid - m_uivLocalNodeScan[m_uiRank]);
        }

        m_uivSendNodeCounts.resize(m_uiSize);
        m_uivSendNodeOffset.resize(m_uiSize);
        m_uivRecvNodeCounts.resize(m_uiSize);
        m_uivRecvNodeOffset.resize(m_uiSize);

        for (unsigned int i = 0; i < m_uiSize; i++) {
            m_uivSendNodeCounts[i] = recvCounts[i];
            m_uivSendNodeOffset[i] = recvOffset[i];
            m_uivRecvNodeCounts[i] = sendCounts[i];
            m_uivRecvNodeOffset[i] = sendOffset[i];
        }

        // assert local map m_uipLocalMap[eid][nid]
        // structure displ vector = [0, ..., (m_uiNumPreGhostNodes - 1), --> ghost nodes owned by someone before me
        //    m_uiNumPreGhostNodes, ..., (m_uiNumPreGhostNodes + m_uiNumNodes - 1), --> nodes owned by me
        //    (m_uiNumPreGhostNodes + m_uiNumNodes), ..., (m_uiNumPreGhostNodes + m_uiNumNodes + m_uiNumPostGhostNodes - 1)] --> nodes owned by someone after me
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            unsigned int num_nodes = m_uiNodesPerElem[eid];

            for (unsigned int i = 0; i < num_nodes; i++){
                const unsigned int nid = m_ulpMap[eid][i];
                if (nid >= m_uivLocalNodeScan[m_uiRank] &&
                    nid < (m_uivLocalNodeScan[m_uiRank] + m_uivLocalNodeCounts[m_uiRank])) {
                    // nid is owned by me
                    assert(m_uipLocalMap[eid][i] == nid - m_uivLocalNodeScan[m_uiRank] + m_uiNumPreGhostNodes);
                } else if (nid < m_uivLocalNodeScan[m_uiRank]){
                    // nid is owned by someone before me
                    const unsigned int lookUp = std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), nid) - preGhostGIds.begin();
                    assert(m_uipLocalMap[eid][i] == lookUp);
                } else if (nid >= (m_uivLocalNodeScan[m_uiRank] + m_uivLocalNodeCounts[m_uiRank])){
                    // nid is owned by someone after me
                    const unsigned int lookUp = std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), nid) - postGhostGIds.begin();
                    assert(m_uipLocalMap[eid][i] ==  (m_uiNumPreGhostNodes + m_uiNumNodes) + lookUp);
                }
            }
        }
        delete [] sendCounts;
        delete [] recvCounts;
        delete [] sendOffset;
        delete [] recvOffset;
        return Error::SUCCESS;
    } // buildScatterMap()


    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::set_bdr_map(GI* constrainedDofs, DT* prescribedValues, LI numConstraints){

        // extract constrained dofs owned by me
        LI local_Id;
        GI global_Id;

        for (LI i = 0; i < numConstraints; i++){
            global_Id = constrainedDofs[i];
            if ((global_Id >= m_uivLocalNodeScan[m_uiRank]) && (global_Id < m_uivLocalNodeScan[m_uiRank] + m_uiNumNodes)) {
                ownedConstrainedDofs.push_back(global_Id);
                ownedPrescribedValues.push_back(prescribedValues[i]);
            }
        }

        // construct elemental map of boundary condition
        m_uipBdrMap = new unsigned int* [m_uiNumElems];
        m_dtPresValMap = new DT* [m_uiNumElems];
        for (LI eid = 0; eid < m_uiNumElems; eid++){
            m_uipBdrMap[eid] = new unsigned int[m_uiNodesPerElem[eid]];
            m_dtPresValMap[eid] = new DT [m_uiNodesPerElem[eid]];
        }

        for (LI eid = 0; eid < m_uiNumElems; eid++){
            for (LI nid = 0; nid < m_uiNodesPerElem[eid]; nid++){
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
                    if ((global_Id >= m_uivLocalNodeScan[m_uiRank]) && (global_Id < m_uivLocalNodeScan[m_uiRank] + m_uiNumNodes)) {
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

        // allocate KfcUc, this will be subtracted from rhs to apply bc
        petsc_create_vec(KfcUcVec);

        return Error::SUCCESS;
    }//set_bdr_map


    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_create_vec(Vec &vec, PetscScalar alpha) const {
        VecCreate(m_comm, &vec);
        if (m_uiSize>1) {
            VecSetType(vec,VECMPI);
            VecSetSizes(vec, m_uiNumNodes, PETSC_DECIDE);
            VecSet(vec, alpha);
        } else {
            VecSetType(vec,VECSEQ);
            VecSetSizes(vec, m_uiNumNodes, PETSC_DECIDE);
            VecSet(vec, alpha);
        }
        return Error::SUCCESS;
    } // petsc_create_vec


    // 16.Dec.2019: change type of e_vec from DT* to EigenVec to be consistent with element matrix
    // Note: for force vector, there is no block_j (as in stiffness matrix)
    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_set_element_vec(Vec vec, LI eid, EigenVec e_vec, LI block_i, InsertMode mode){

        //unsigned int num_nodes = m_uiNodesPerElem[eid];
        unsigned int num_rows = e_vec.size();
        assert(e_vec.size() == e_vec.rows()); // since EigenVec is defined as matrix with 1 column

        PetscScalar value;
        PetscInt rowId;

        for (unsigned int r = 0; r < num_rows; ++r) {

            // this ONLY WORKS with assumption that all blocks have the same number of dofs (that is true for R-XFEM ?)
            rowId = m_ulpMap[eid][block_i * num_rows + r];
            value = e_vec(r);
            VecSetValue(vec, rowId, value, mode);
        }

        return Error::SUCCESS;
    } // petsc_set_element_vec


    // use with Eigen, matrix-based, set every row of the matrix (faster than set every term of the matrix)
    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, InsertMode mode ) {

        // this is number of dofs per block:
        unsigned int num_rows = e_mat.rows();
        assert(num_rows == e_mat.cols());

        // assemble global matrix (petsc matrix)
        // now set values ...
        std::vector<PetscScalar> values(num_rows);
        std::vector<PetscInt> colIndices(num_rows);
        PetscInt rowId;
        for (unsigned int r = 0; r < num_rows; ++r) {
            // this ONLY WORKS with assumption that all blocks have the same number of dofs (that is true for RXFEM ?)
            rowId = m_ulpMap[eid][block_i * num_rows + r];
            for (unsigned int c = 0; c < num_rows; ++c) {
                colIndices[c] = m_ulpMap[eid][block_j * num_rows + c];
                values[c] = e_mat(r,c);
            } // c
            //MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), colIndices.data(), values.data(), mode);
        } // r

        // prepare for bc of rhs
        std::vector<PetscScalar> KfcUcValues;
        std::vector<PetscInt> rowIndices;
        PetscScalar temp;
        bool bdrFlag, rowFlag;
        // loop over rows of the element matrix (block)
        for (LI r = 0; r < num_rows; r++){
            rowFlag = false;
            // continue if row is associated with a free dof
            if (m_uipBdrMap[eid][block_i * num_rows + r] == 0){
                rowId = m_ulpMap[eid][block_i * num_rows + r];
                temp = 0;
                // loop over columns of the element matrix (block)
                for (LI c = 0; c < num_rows; c++){
                    // continue if column is associated with a constrained dof
                    if (m_uipBdrMap[eid][block_j * num_rows + c] == 1){
                        // accumulate Kfc[r,c]*Uc[c]
                        temp += e_mat(r,c) * m_dtPresValMap[eid][block_j * num_rows + c];
                        rowFlag = true; // this rowId has constrained column dof
                        bdrFlag = true; // this element matrix has KfcUc
                    }
                }
                if (rowFlag){
                    rowIndices.push_back(rowId);
                    KfcUcValues.push_back(-1.0*temp);
                }
            }
        }
        if (bdrFlag){
            VecSetValues(KfcUcVec, rowIndices.size(), rowIndices.data(), KfcUcValues.data(), ADD_VALUES);
        }

        return Error::SUCCESS;
    } // petsc_set_element_matrix


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::dump_mat( const char* filename /* = nullptr */ ) {

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
    } // dump_mat


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::dump_vec( Vec vec, const char* filename ) const {
        if( filename == nullptr ) {
            VecView( vec, PETSC_VIEWER_STDOUT_WORLD );
        }
        else {
            PetscViewer viewer;
            // write to ASCII file
            PetscViewerASCIIOpen( m_comm, filename, &viewer );
            VecView( vec, viewer );
            PetscViewerDestroy( &viewer );
        }
        return Error::SUCCESS;
    } // dump_vec


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_get_diagonal( const Vec & vec ) const {
        MatGetDiagonal( m_pMat, vec );
        return Error::SUCCESS;
    } //petsc_get_diagonal


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_destroy_vec( Vec & vec ) const {
        VecDestroy( &vec );
        return Error::SUCCESS;
    }


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::create_vec( DT* &vec, bool isGhosted /* = false */, DT alpha /* = 0.0 */ ) const {
        if (isGhosted){
            vec = new DT[m_uiNumNodesTotal];
        } else {
            vec = new DT[m_uiNumNodes];
        }
        // initialize
        if (isGhosted) {
            for (unsigned int i = 0; i < m_uiNumNodesTotal; i++){
                vec[i] = alpha;
            }
        } else {
            for (unsigned int i = 0; i < m_uiNumNodes; i++){
                vec[i] = alpha;
            }
        }
        return Error::SUCCESS;
    } // create_vec

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::create_mat( DT** &mat, bool isGhosted /* = false */, DT alpha /* = 0.0 */ ){
        if( isGhosted ){
            mat = new DT*[m_uiNumNodesTotal];
            for (unsigned int i = 0; i < m_uiNumNodesTotal; i++){
                mat[i] = new DT[m_uiNumNodesTotal];
            }
            for (unsigned int i = 0; i < m_uiNumNodesTotal; i++){
                for (unsigned int j = 0; j < m_uiNumNodesTotal; j++){
                    mat[i][j] = alpha;
                }
            }
        }
        else {
            mat = new DT *[m_uiNumNodes];
            for (unsigned int i = 0; i < m_uiNumNodes; i++) {
                mat[i] = new DT[m_uiNumNodes];
            }
            for (unsigned int i = 0; i < m_uiNumNodes; i++){
                for (unsigned int j = 0; j < m_uiNumNodes; j++){
                    mat[i][j] = alpha;
                }
            }
        }
        return Error::SUCCESS;
    }

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::local_to_ghost( DT*  gVec, const DT* local ) const {
        for (unsigned int i = 0; i < m_uiNumNodesTotal; i++){
            if ((i >= m_uiNumPreGhostNodes) && (i < m_uiNumPreGhostNodes + m_uiNumNodes)) {
                gVec[i] = local[i - m_uiNumPreGhostNodes];
            }
            else {
                gVec[i] = 0.0;
            }
        }
        return Error::SUCCESS;
    } // local_to_ghost

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::ghost_to_local(DT* local, const DT* gVec) const {
        for (unsigned int i = 0; i < m_uiNumNodes; i++) {
            local[i] = gVec[i + m_uiNumPreGhostNodes];
        }
        return Error::SUCCESS;
    } // ghost_to_local


    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::copy_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim ) {
        // resize the vector of blocks for element eid
        m_epMat[eid].resize(blocks_dim * blocks_dim);

        // 1D-index of (block_i, block_j)
        LI index = (block_i * blocks_dim) + block_j;

        // copy e_mat to the location of index
        m_epMat[eid][index] = e_mat;

        // prepare for bc of rhs
        LI num_rows = e_mat.rows();
        assert(num_rows == e_mat.cols());
        std::vector<PetscScalar> KfcUcValues;
        std::vector<PetscInt> rowIndices;
        PetscInt rowId;
        PetscScalar temp;
        bool bdrFlag, rowFlag;

        // loop over rows of the element matrix (block)
        for (LI r = 0; r < num_rows; r++){
            rowFlag = false;
            // continue if row is associated with a free dof
            if (m_uipBdrMap[eid][block_i * num_rows + r] == 0){
                rowId = m_ulpMap[eid][block_i * num_rows + r];
                temp = 0;
                // loop over columns of the element matrix (block)
                for (LI c = 0; c < num_rows; c++){
                    // continue if column is associated with a constrained dof
                    if (m_uipBdrMap[eid][block_j * num_rows + c] == 1){
                        // accumulate Kfc[r,c]*Uc[c]
                        temp += e_mat(r,c) * m_dtPresValMap[eid][block_j * num_rows + c];
                        rowFlag = true; // this rowId has constrained column dof
                        bdrFlag = true; // this element matrix has KfcUc
                    }
                }
                if (rowFlag){
                    rowIndices.push_back(rowId);
                    KfcUcValues.push_back(-1.0*temp);
                }
            }
        }

        if (bdrFlag){
            VecSetValues(KfcUcVec, rowIndices.size(), rowIndices.data(), KfcUcValues.data(), ADD_VALUES);
        }

        return Error::SUCCESS;
    }// copy_element_matrix

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::mat_get_diagonal(DT* diag, bool isGhosted){
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
    }// mat_get_diagonal

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::mat_get_diagonal_ghosted(DT* diag){

        EigenMat e_mat;
        LI rowID;

        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){

            // number of blocks in each direction
            LI blocks_dim = (LI)sqrt(m_epMat[eid].size());
            assert((blocks_dim*blocks_dim) == m_epMat[eid].size());

            LI block_row_offset = 0;

            for (LI block_i = 0; block_i < blocks_dim; block_i++) {

                // only get diagonals of diagonal blocks
                LI index = block_i * blocks_dim + block_i;

                // diagonal block must not be null
                assert (m_epMat[eid][index].size() != 0);

                e_mat = m_epMat[eid][index];
                assert(e_mat.rows() == e_mat.cols());

                // number of dofs per block
                const LI num_dofs_per_block = e_mat.rows();
                assert(num_dofs_per_block == m_uiNodesPerElem[eid]/blocks_dim);

                for (LI r = 0; r < num_dofs_per_block; r++){

                    // local (rank) row ID
                    rowID = m_uipLocalMap[eid][block_row_offset + r];

                    // accumulate the diagonal term
                    diag[rowID] += e_mat(r,r);
                }
                block_row_offset += num_dofs_per_block;
            }
        }

        // communication between ranks
        ghost_send_begin(diag);
        ghost_send_end(diag);

        return Error::SUCCESS;
    }// mat_get_diagonal_ghosted

    // return rank that owns global gId
    template <typename DT, typename GI, typename LI>
    unsigned int aMat<DT, GI, LI>::globalId_2_rank(GI gId) const {
        unsigned int rank;
        if (gId >= m_uivLocalNodeScan[m_uiSize - 1]){
            rank = m_uiSize - 1;
        } else {
            for (unsigned int i = 0; i < (m_uiSize - 1); i++){
                if (gId >= m_uivLocalNodeScan[i] && gId < m_uivLocalNodeScan[i+1] && (i < (m_uiSize -1))) {
                    rank = i;
                    break;
                }
            }
        }
        return rank;
    }


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT, GI, LI>::mat_get_diagonal_block(std::vector<MatRecord<DT,LI>> &diag_blk){
        LI blocks_dim;
        EigenMat e_mat;
        GI glo_RowId, glo_ColId;
        LI loc_RowId, loc_ColId;
        LI rowID, colID;
        unsigned int rank_r, rank_c;
        DT value;
        LI ind = 0;

        std::vector<MatRecord<DT,LI>> matRec1;
        std::vector<MatRecord<DT,LI>> matRec2;

        MatRecord<DT,LI> matr;

        m_vMatRec.clear();
        diag_blk.clear();

        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){

            // number of blocks in row (or column)
            blocks_dim = (LI)sqrt(m_epMat[eid].size());
            assert (blocks_dim * blocks_dim == m_epMat[eid].size());

            LI block_row_offset = 0;
            LI block_col_offset = 0;
            LI num_dofs_per_block;

            for (LI block_i = 0; block_i < blocks_dim; block_i++){

                for (LI block_j = 0; block_j < blocks_dim; block_j++){

                    LI index = block_i * blocks_dim + block_j;

                    if (m_epMat[eid][index].size() != 0){

                        e_mat = m_epMat[eid][index];
                        assert(e_mat.rows() == e_mat.cols());

                        // number of dofs per block
                        num_dofs_per_block = e_mat.rows();
                        assert(num_dofs_per_block == m_uiNodesPerElem[eid]/blocks_dim);

                        for (LI r = 0; r < num_dofs_per_block; r++){

                            // local row Id (include ghost nodes)
                            rowID = m_uipLocalMap[eid][block_row_offset + r];

                            // global row Id
                            glo_RowId = m_ulpMap[eid][block_row_offset + r];

                            //rank who owns global row Id
                            rank_r = globalId_2_rank(glo_RowId);

                            //local ID in that rank (not include ghost nodes)
                            loc_RowId = (glo_RowId - m_uivLocalNodeScan[rank_r]);

                            for (LI c = 0; c < num_dofs_per_block; c++){
                                // local column Id (include ghost nodes)
                                colID = m_uipLocalMap[eid][block_col_offset + c];

                                // global column Id
                                glo_ColId = m_ulpMap[eid][block_col_offset + c];

                                // rank who owns global column Id
                                rank_c = globalId_2_rank(glo_ColId);

                                // local column Id in that rank (not include ghost nodes)
                                loc_ColId = (glo_ColId - m_uivLocalNodeScan[rank_c]);

                                if( rank_r == rank_c ){
                                    // put all data in a MatRecord object
                                    matr.setRank(rank_r);
                                    matr.setRowId(loc_RowId);
                                    matr.setColId(loc_ColId);
                                    matr.setVal(e_mat(r,c));

                                    if ((rowID >= m_uiNodeLocalBegin) && (rowID < m_uiNodeLocalEnd) && (colID >= m_uiNodeLocalBegin) && (colID < m_uiNodeLocalEnd)){
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


        unsigned int* sendCounts = new unsigned int[m_uiSize];
        unsigned int* recvCounts = new unsigned int[m_uiSize];
        unsigned int* sendOffset = new unsigned int[m_uiSize];
        unsigned int* recvOffset = new unsigned int[m_uiSize];

        for (unsigned int i = 0; i < m_uiSize; i++){
            sendCounts[i] = 0;
            recvCounts[i] = 0;
        }

        // number of elements sending to each rank
        for (unsigned int i = 0; i < m_vMatRec.size(); i++){
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
        for (unsigned int i = 0; i < recv_buff.size(); i++){
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
    } // mat_get_diagonal_block

    template <typename DT, typename  GI, typename LI>
    par::Error aMat<DT,GI,LI>::get_max_dof_per_elem(){
        unsigned int num_nodes;
        unsigned int max_dpe = 0;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            num_nodes = m_uiNodesPerElem[eid];
            if (max_dpe < num_nodes) max_dpe = num_nodes;
        }
        m_uiMaxNodesPerElem = max_dpe;
        return Error::SUCCESS;
    }// get_max_dof_per_elem

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::destroy_vec(DT* &vec) {
        if (vec != nullptr) {
            delete[] vec;
            vec = nullptr;
        }
        return Error::SUCCESS;
    }

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::destroy_mat(DT** &mat, const unsigned int nrow){
        if (mat != nullptr){
            for (unsigned int i = 0; i < nrow; i++){
                if (mat[i] != nullptr){
                    delete[](mat[i]);
                    mat[i] = nullptr;
                }
            }
            delete[](mat);
            mat = nullptr;
        }
        return Error::SUCCESS;
    }


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::ghost_receive_begin(DT* vec) {

        if(m_uiSize==1)
            return par::Error::SUCCESS;

        // exchange context for vec
        AsyncExchangeCtx ctx((const void*)vec);

        // total number of DoFs to be sent
        const unsigned int total_send = m_uivSendNodeOffset[m_uiSize-1] + m_uivSendNodeCounts[m_uiSize-1];

        // total number of DoFs to be received
        const unsigned  int total_recv = m_uivRecvNodeOffset[m_uiSize-1] + m_uivRecvNodeCounts[m_uiSize-1];

        // send data of owned DoFs to corresponding ghost DoFs in all other ranks
        if (total_send > 0){
            // allocate memory for sending buffer
            ctx.allocateSendBuffer(sizeof(DT) * total_send);
            // get the address of sending buffer
            DT* send_buf = (DT*)ctx.getSendBuffer();
            // put all sending values to buffer
            for (unsigned int i = 0; i < total_send; i++){
                send_buf[i] = vec[m_uivSendNodeIds[i]];
            }
            for (unsigned int i = 0; i < m_uiSize; i++){
                // if nothing to send to rank i then skip
                if (m_uivSendNodeCounts[i] == 0) continue;
                // send to rank i
                MPI_Request* req = new MPI_Request();
                MPI_Isend(&send_buf[m_uivSendNodeOffset[i]], m_uivSendNodeCounts[i] * sizeof(DT), MPI_BYTE, i, m_iCommTag, m_comm, req);
                // put output request req of sending into the Request list of ctx
                ctx.getRequestList().push_back(req);
            }
        }

        // received data for ghost DoFs from all other ranks
        if (total_recv > 0){
            ctx.allocateRecvBuffer(sizeof(DT) * total_recv);
            DT* recv_buf = (DT*) ctx.getRecvBuffer();
            for (unsigned int i = 0; i < m_uiSize; i++){
                if (m_uivRecvNodeCounts[i] == 0) continue;
                MPI_Request* req = new MPI_Request();
                MPI_Irecv(&recv_buf[m_uivRecvNodeOffset[i]], m_uivRecvNodeCounts[i] * sizeof(DT), MPI_BYTE, i, m_iCommTag, m_comm, req);
                // pout output request req of receiving into Request list of ctx
                ctx.getRequestList().push_back(req);
            }
        }
        // save the ctx of v for later access
        m_vAsyncCtx.push_back(ctx);
        // get a different value of tag if we have another ghost_exchange for a different vec
        m_iCommTag++;
        return Error::SUCCESS;
    } //ghost_receive_begin

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::ghost_receive_end(DT* vec) {

        if(m_uiSize==1)
            return par::Error::SUCCESS;

        // get the context associated with vec
        unsigned int ctx_index;
        for (unsigned i = 0; i < m_vAsyncCtx.size(); i++){
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

        //const unsigned  int total_recv = m_uivRecvNodeOffset[m_uiSize-1] + m_uivRecvNodeCounts[m_uiSize-1];

        DT* recv_buf = (DT*) ctx.getRecvBuffer();
        // copy values of pre-ghost nodes from recv_buf to vec
        std::memcpy(vec, recv_buf, m_uiNumPreGhostNodes*sizeof(DT));
        // copy values of post-ghost nodes from recv_buf to vec
        std::memcpy(&vec[m_uiNumPreGhostNodes + m_uiNumNodes], &recv_buf[m_uiNumPreGhostNodes], m_uiNumPostGhostNodes*sizeof(DT));

        // free memory of send and receive buffers
        ctx.deAllocateRecvBuffer();
        ctx.deAllocateSendBuffer();

        m_vAsyncCtx.erase(m_vAsyncCtx.begin() + ctx_index);
        return Error::SUCCESS;
    } // ghost_receive_end


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::ghost_send_begin(DT* vec) {

        if(m_uiSize==1)
            return par::Error::SUCCESS;

        AsyncExchangeCtx ctx((const void*)vec);

        // number of DoFs to be received (= number of DoFs that are sent before calling matvec)
        const unsigned  int total_recv = m_uivSendNodeOffset[m_uiSize-1] + m_uivSendNodeCounts[m_uiSize-1];
        // number of DoFs to be sent (= number of DoFs that are received before calling matvec)
        const unsigned  int total_send = m_uivRecvNodeOffset[m_uiSize-1] + m_uivRecvNodeCounts[m_uiSize-1];

        // receive data for owned DoFs that are sent back to my rank from other ranks (after matvec is done)
        if (total_recv > 0){
            ctx.allocateRecvBuffer(sizeof(DT) * total_recv);
            DT* recv_buf = (DT*) ctx.getRecvBuffer();
            for (unsigned int i = 0; i < m_uiSize; i++){
                if (m_uivSendNodeCounts[i] == 0) continue;
                MPI_Request* req = new MPI_Request();
                MPI_Irecv(&recv_buf[m_uivSendNodeOffset[i]], m_uivSendNodeCounts[i]*sizeof(DT), MPI_BYTE, i, m_iCommTag, m_comm, req);
                ctx.getRequestList().push_back(req);
            }
        }

        // send data of ghost DoFs to ranks owning the DoFs
        if (total_send > 0){
            ctx.allocateSendBuffer(sizeof(DT) * total_send);
            DT* send_buf = (DT*) ctx.getSendBuffer();

            // pre-ghost DoFs
            for (unsigned int i = 0; i < m_uiNumPreGhostNodes; i++){
                send_buf[i] = vec[i];
            }
            // post-ghost DoFs
            for (unsigned int i = m_uiNumPreGhostNodes + m_uiNumNodes; i < m_uiNumPreGhostNodes + m_uiNumNodes + m_uiNumPostGhostNodes; i++){
                send_buf[i - m_uiNumNodes] = vec[i];
            }
            for (unsigned int i = 0; i < m_uiSize; i++){
                if (m_uivRecvNodeCounts[i] == 0) continue;
                MPI_Request* req = new MPI_Request();
                MPI_Isend(&send_buf[m_uivRecvNodeOffset[i]], m_uivRecvNodeCounts[i] * sizeof(DT), MPI_BYTE, i, m_iCommTag, m_comm, req);
                ctx.getRequestList().push_back(req);
            }
        }
        m_vAsyncCtx.push_back(ctx);
        m_iCommTag++; // get a different value if we have another ghost_exchange for a different vec
        return Error::SUCCESS;
    } // ghost_send_begin

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::ghost_send_end(DT* vec) {

        if(m_uiSize==1)
            return par::Error::SUCCESS;

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
        for(unsigned int i=0;i<num_req;i++)
            {
                MPI_Wait(ctx.getRequestList()[i],&sts);
            }

        //const unsigned  int total_recv = m_uivSendNodeOffset[m_uiSize-1] + m_uivSendNodeCounts[m_uiSize-1];
        DT* recv_buf = (DT*) ctx.getRecvBuffer();

        for (unsigned int i = 0; i < m_uiSize; i++){
            for (unsigned int j = 0; j < m_uivSendNodeCounts[i]; j++){
                vec[m_uivSendNodeIds[m_uivSendNodeOffset[i]] + j] += recv_buf[m_uivSendNodeOffset[i] + j];
            }
        }
        ctx.deAllocateRecvBuffer();
        ctx.deAllocateSendBuffer();
        m_vAsyncCtx.erase(m_vAsyncCtx.begin() + ctx_index);
        return Error::SUCCESS;
    } // ghost_send_end

    template <typename  DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::matvec( DT* v, const DT* u, bool isGhosted /* = false */ ) {
        if( isGhosted ) {
            // std::cout << "GHOSTED MATVEC" << std::endl;
            matvec_ghosted(v, (DT*)u);
        }
        else {
            // std::cout << "NON GHOSTED MATVEC" << std::endl;
            DT* gv;
            DT* gu;
            // allocate memory for gv and gu including ghost dof's
            create_vec(gv, true, 0.0);
            create_vec(gu, true, 0.0);
            // copy u to gu
            local_to_ghost(gu, u);

            matvec_ghosted(gv, gu);
            // copy gv to v
            ghost_to_local(v, gv);

            delete[] gv;
            delete[] gu;
        }
        return Error::SUCCESS;
    } // matvec


    // matvec (v = K * u) embeded applying bc by modifying matrix
    template <typename  DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::matvec_ghosted( DT* v, DT* u ) {

        LI blocks_dim, num_dofs_per_bolck;
        DT* ue;
        DT* ve;
        EigenMat emat;

        // initialize v (size of v = m_uiNodesPostGhostEnd = m_uiNumNodesTotal)
        for (unsigned int i = 0; i < m_uiNodePostGhostEnd; i++){
            v[i] = 0.0;
        }

        /*printf("rank %d, before matvec u = \n",m_uiRank);
        for (unsigned int i = 0; i < m_uiNumNodesTotal; i++){
            printf("u[%d]= %f\n",i,u[i]);
        }*/

        // allocate memory for element vectors
        ue = new DT[m_uiMaxNodesPerElem];
        ve = new DT[m_uiMaxNodesPerElem];

        LI rowID, colID;

        // apply BC: save Uc and set u(Uc) = 0 (could be moved to MatMult_mf)
        const LI numConstraints = ownedConstrainedDofs.size();
        LI local_Id;
        DT* Uc = new DT [numConstraints];
        for (LI nid = 0; nid < numConstraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_uivLocalNodeScan[m_uiRank] + m_uiNumPreGhostNodes;
            Uc[nid] = u[local_Id];
            u[local_Id] = 0.0;
        }
        // end of apply BC

        // send data from owned nodes to ghost nodes (of other processors) to get ready for computing v = Ku
        ghost_receive_begin(u);
        ghost_receive_end(u);

        // multiply [ve] = [ke][ue] for all elements
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            blocks_dim = (LI)sqrt(m_epMat[eid].size());
            LI block_row_offset = 0;
            LI block_col_offset = 0;
            LI num_dofs_per_block;

            for (LI block_i = 0; block_i < blocks_dim; block_i++){

                for (LI block_j = 0; block_j < blocks_dim; block_j++){

                    LI index = block_i * blocks_dim + block_j;

                    if (m_epMat[eid][index].size() != 0){

                        emat = m_epMat[eid][index];
                        assert(emat.rows() == emat.cols());

                        num_dofs_per_block = emat.rows();
                        assert(num_dofs_per_block == m_uiNodesPerElem[eid]/blocks_dim);

                        // extract block-element vector ue from structure vector u
                        for (LI c = 0; c < num_dofs_per_block; c++) {
                            colID = m_uipLocalMap[eid][block_col_offset + c];
                            ue[c] = u[colID];
                        }

                        // apply bc by modifying matrix according to {Kfc = 0; Kcf = 0; Kcc = I}
                        // BUG here: Kcc = 1 on diagonals is accumulated, thus at the end it is NOT 1
                        /*for (LI r = 0; r < num_dofs_per_block; r++){
                            if (m_uipBdrMap[eid][block_row_offset + r] == 1){
                                for (LI c = 0; c < num_dofs_per_block; c++){
                                    if ((c == r) && (block_i == block_j)){
                                        // set [Kcc] = [I]
                                        emat(r,c) = 1.0;
                                    } else {
                                        // set [Kcf] = [0]
                                        emat(r,c) = 0.0;
                                    }
                                }
                            } else {
                                for (LI c = 0; c < num_dofs_per_block; c++){
                                    if (m_uipBdrMap[eid][block_col_offset + c] == 1){
                                        // set [Kfc] = [0]
                                        emat(r,c) = 0.0;
                                    }
                                }
                            }
                        }*/

                        // multiply block-element matrix with block-element vector
                        for (LI i = 0; i < num_dofs_per_block; i++){
                            ve[i] = 0.0;
                            for (unsigned int j = 0; j < num_dofs_per_block; j++){
                                ve[i] += emat(i,j) * ue[j];
                            }
                        }
                        // accumulate element vector ve to structure vector v
                        for (unsigned int r = 0; r < num_dofs_per_block; r++){
                            rowID = m_uipLocalMap[eid][block_row_offset + r];
                            v[rowID] += ve[r];
                        }
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

        // apply BC: set v(Uc) = Uc which is saved before doing matvec (could be moved to MatMult_mf)
        for (LI nid = 0; nid < numConstraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_uivLocalNodeScan[m_uiRank] + m_uiNumPreGhostNodes;
            v[local_Id] = Uc[nid];
        }
        delete [] Uc;
        // end of apply BC

        delete [] ue;
        delete [] ve;

        /*printf("rank %d, before matvec v = \n",m_uiRank);
        for (unsigned int i = 0; i < m_uiNumNodesTotal; i++){
            printf("v[%d]= %f\n",i,v[i]);
        }*/

        return Error::SUCCESS;
    } // matvec_ghosted

    template <typename  DT, typename GI, typename LI>
    PetscErrorCode aMat<DT,GI,LI>::MatMult_mf( Mat A, Vec u, Vec v ) {

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
        /* const LI numConstraints = ownedConstrainedDofs.size();
        DT* Uc = new DT [numConstraints];
        for (LI nid = 0; nid < numConstraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_uivLocalNodeScan[m_uiRank] + m_uiNumPreGhostNodes;
            Uc[nid] = uug[local_Id];
            uug[local_Id] = 0.0;
        } */
        // end of apply BC

        // vvg = K * uug
        matvec(vvg, uug, true); // this gives V_f = (K_ff * U_f) + (K_fc * 0) = K_ff * U_f

        // apply BC: now set V_c = U_c which was saved in U'_c
        /* for (LI nid = 0; nid < numConstraints; nid++){
            local_Id = ownedConstrainedDofs[nid] - m_uivLocalNodeScan[m_uiRank] + m_uiNumPreGhostNodes;
            vvg[local_Id] = Uc[nid];
        }
        delete [] Uc; */
        // end of apply BC

        ghost_to_local(vv,vvg);

        delete [] vvg;
        delete [] uug;

        VecRestoreArray(v,&vv);

        return 0;
    }// MatMult_mf


    template<typename DT, typename GI, typename LI>
    PetscErrorCode aMat<DT,GI,LI>::MatGetDiagonal_mf(Mat A, Vec d){

        // point to data of PETSc vector d
        PetscScalar* dd;
        VecGetArray(d, &dd);
        //PetscInt N;
        //VecGetSize(d,&N);
        //std::cout<<" N: "<<N<<std::endl;

        // allocate regular vector used for get_diagonal() in aMat
        double* ddg;
        create_vec(ddg, true, 0);

        // get diagonal of matrix and put into ddg
        mat_get_diagonal(ddg, true);

        // copy ddg (ghosted) into (non-ghosted) dd
        ghost_to_local(dd, ddg);
        /*for (unsigned int i = 0; i < m_uiNumNodes; i++){
            printf("[%d,%d,%f]\n",m_uiRank,i,dd[i]);
        }*/

        // deallocate ddg
        destroy_vec(ddg);

        // update data of PETSc vector d
        VecRestoreArray(d, &dd);

        // apply Dirichlet boundary condition
        apply_bc_diagonal( d );
        petsc_init_vec( d );
        petsc_finalize_vec( d );

        return 0;

    }// MatGetDiagonal_mf


    template<typename DT, typename GI, typename LI>
    PetscErrorCode aMat<DT,GI,LI>::MatGetDiagonalBlock_mf(Mat A, Mat* a){

        unsigned int local_size = get_local_num_nodes();

        //LI local_rowID, local_colID;
        //DT value;
        //PetscScalar* aa;

        // sparse block diagonal matrix
        std::vector<MatRecord<DT,LI>> ddg;
        mat_get_diagonal_block(ddg);

        /*std::ofstream myfile;
        myfile.open("blk_diag.dat");
        for (unsigned int r = 0; r < local_size; r++){
            for (unsigned int c = 0; c < local_size; c++){
                //printf("[r%d][%d,%d]= %f; ",m_uiRank,r,c,aag[r][c]);
                myfile << "[r" << m_uiRank << "][" << r << "," << c << "]= " << aag[r][c] << " , ";
            }
            //printf("\n");
            myfile << std::endl;
        }*/

        // create Petsc sequential matrix
        //Mat B;

        MatCreateSeqAIJ(PETSC_COMM_SELF, local_size, local_size, 27, PETSC_NULL, a); // todo: 27 only good for dofs_per_node = 1
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
        for (unsigned int i = 0; i < local_size; i++){
            for (unsigned int j = 0; j < local_size; j++){
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
    }


    // apply Dirichlet bc by modifying matrix, only used in matrix-based approach
    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::apply_bc_mat() {
        unsigned int num_nodes;

        PetscInt rowId, colId;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid ++) {
            num_nodes = m_uiNodesPerElem[eid];
            for (unsigned int r = 0; r < num_nodes; r++) {
                rowId = m_ulpMap[eid][r];
                if (m_uipBdrMap[eid][r] == 1) {
                    for (unsigned int c = 0; c < num_nodes; c++) {
                        colId = m_ulpMap[eid][c];
                        if (colId == rowId) {
                            MatSetValue(m_pMat, rowId, colId, 1.0, INSERT_VALUES);
                        } else {
                            MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                        }
                    }
                } else {
                    for (unsigned int c = 0; c < num_nodes; c++) {
                        colId = m_ulpMap[eid][c];
                        if (m_uipBdrMap[eid][c] == 1) {
                            MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                        }
                    }
                }
            }
        }
        return Error::SUCCESS;
    } // apply_bc_mat

    // apply Dirichlet bc to diagonal vector used in Jacobi preconditioning
    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::apply_bc_diagonal(Vec diag) {
        unsigned int num_nodes;
        PetscInt rowId;

        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            num_nodes = m_uiNodesPerElem[eid];
            for (unsigned int r = 0; r < num_nodes; r++){
                if (m_uipBdrMap[eid][r] == 1){
                    // global row ID
                    rowId = m_ulpMap[eid][r];
                    VecSetValue(diag, rowId, 1.0, INSERT_VALUES);
                }
            }
        }
        return Error::SUCCESS;
    }


    // apply Dirichlet BC to block diagonal matrix
    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::apply_bc_blkdiag(Mat* blkdiagMat) {
        unsigned int num_nodes;
        PetscInt loc_rowId, loc_colId;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            // total number of dofs per element eid
            num_nodes = m_uiNodesPerElem[eid];
            // loop on all dofs of element
            for (unsigned int r = 0; r < num_nodes; r++) {
                loc_rowId = m_uipLocalMap[eid][r];
                if ((loc_rowId >= m_uiNodeLocalBegin) && (loc_rowId <= m_uiNodeLocalEnd)) {
                    if (m_uipBdrMap[eid][r] == 1) {
                        for (unsigned int c = 0; c < num_nodes; c++) {
                            loc_colId = m_uipLocalMap[eid][c];
                            if ((loc_colId >= m_uiNodeLocalBegin) && (loc_colId <= m_uiNodeLocalEnd)) {
                                if (loc_rowId == loc_colId) {
                                    MatSetValue(*blkdiagMat, (loc_rowId - m_uiNumPreGhostNodes),
                                                (loc_colId - m_uiNumPreGhostNodes), 1.0, INSERT_VALUES);
                                } else {
                                    MatSetValue(*blkdiagMat, (loc_rowId - m_uiNumPreGhostNodes),
                                                (loc_colId - m_uiNumPreGhostNodes), 0.0, INSERT_VALUES);
                                }
                            }
                        }
                    } else {
                        for (unsigned int c = 0; c < num_nodes; c++) {
                            loc_colId = m_uipLocalMap[eid][c];
                            if ((loc_colId >= m_uiNodeLocalBegin) && (loc_colId <= m_uiNodeLocalEnd)) {
                                if (m_uipBdrMap[eid][c] == 1) {
                                    MatSetValue(*blkdiagMat, (loc_rowId - m_uiNumPreGhostNodes),
                                                (loc_colId - m_uiNumPreGhostNodes), 0.0, INSERT_VALUES);
                                }
                            }
                        }
                    }
                }
            }
        }
        return Error::SUCCESS;
    }

    // apply Dirichlet bc by modifying rhs:
    // rhs[i] = Uc_i if i is on boundary of Dirichlet condition, Uc_i is the prescribed value on boundary
    // rhs[i] = rhs[i] - sum_{j=1}^{nc}{K_ij * Uc_j} if i is a free dof
    //          where nc is the total number of boundary dofs and K_ij is stiffness matrix
    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::apply_bc_rhs(Vec rhs){

        // FIXME: Do we need to handle apply_bc_rhs() differently depending on matrix type?
        //
        // if (m_MatType == AMAT_TYPE::MAT_FREE){
        // }
        // else if (m_MatType == AMAT_TYPE::PETSC_SPARSE){
        //     // todo apply bc for rhs in matrix-based method
        // }
        // else {
        //     return Error::UNKNOWN_MAT_TYPE;
        // }

        // set rows associated with constrained dofs to be equal to Uc
        LI num_nodes;
        PetscInt global_Id;
        PetscScalar value, value1, value2;

        for (LI nid = 0; nid < ownedConstrainedDofs.size(); nid++){
            global_Id = ownedConstrainedDofs[nid];
            value = ownedPrescribedValues[nid];
            VecSetValue(rhs, global_Id, value, INSERT_VALUES);
        }

        // need to finalize vector KfcUcVec before extracting its value
        petsc_init_vec(KfcUcVec);
        petsc_finalize_vec(KfcUcVec);

        for (LI nid = 0; nid < ownedFreeDofs.size(); nid++){
            global_Id = ownedFreeDofs[nid];
            VecGetValues(KfcUcVec, 1, &global_Id, &value1);
            VecGetValues(rhs, 1, &global_Id, &value2);
            value = value1 + value2;
            VecSetValue(rhs, global_Id, value, INSERT_VALUES);
        }

        VecDestroy(&KfcUcVec);

        return Error::SUCCESS;
    } // apply_bc_rhs

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_solve( const Vec rhs, Vec out ) const {

        if( m_MatType == AMAT_TYPE::MAT_FREE ) {

            // PETSc shell matrix
            Mat pMatFree;
            // get context to aMat
            aMatCTX<DT,GI,LI> ctx;
            // point back to aMat
            ctx.aMatPtr =  (aMat<DT,GI,LI>*)this; // FIXME: casting away "const", better way to do this?

            // create matrix shell
            MatCreateShell( m_comm, m_uiNumNodes, m_uiNumNodes, PETSC_DETERMINE, PETSC_DETERMINE, &ctx, &pMatFree );

            // set operation for matrix-vector multiplication using aMat::MatMult_mf
            MatShellSetOperation( pMatFree, MATOP_MULT, (void(*)(void))aMat_matvec<DT,GI,LI> );

            // set operation for geting matrix diagonal using aMat::MatGetDiagonal_mf
            MatShellSetOperation( pMatFree, MATOP_GET_DIAGONAL, (void(*)(void))aMat_matgetdiagonal<DT,GI,LI> );

            // set operation for geting block matrix diagonal using aMat::MatGetDiagonalBlock_mf
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

            // set default preconditioner (e.g. PCJACOBI, PCBJACOBI, ...)
            // could be overwritten at runtime using -pc_type <type>
            KSPGetPC( ksp, &pc );
            PCSetType( pc, PCJACOBI );
            PCSetFromOptions( pc );

            // solve the system
            KSPSolve( ksp, rhs, out );

            // clean up
            KSPDestroy( &ksp );

        }
        else { // Normal PETSc solve (ie, not matrix free)
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
        }

        return Error::SUCCESS;

    } // petsc_solve


    /**@brief ********* FUNCTIONS FOR DEBUGGING **************************************************/

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_create_matrix_matvec(){
        MatCreate(m_comm, &m_pMat_matvec);
        MatSetSizes(m_pMat_matvec, m_uiNumNodes, m_uiNumNodes, PETSC_DECIDE, PETSC_DECIDE);
        if (m_uiSize > 1) {
            MatSetType(m_pMat_matvec, MATMPIAIJ);
            MatMPIAIJSetPreallocation(m_pMat_matvec, 30, PETSC_NULL, 30, PETSC_NULL);
        } else {
            MatSetType(m_pMat_matvec, MATSEQAIJ);
            MatSeqAIJSetPreallocation(m_pMat_matvec, 30, PETSC_NULL);
        }
        MatZeroEntries(m_pMat_matvec);
        return Error :: SUCCESS;
    } // petsc_create_matrix_matvec

    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::set_element_matrix_term_by_term( LI eid, EigenMat e_mat, InsertMode mode /* = ADD_VALUES*/ ) {

        assert( e_mat.rows()== e_mat.cols() );
        unsigned int num_rows = e_mat.rows();

        // assemble global matrix (petsc matrix)
        // now set values ...
        PetscScalar value;
        PetscInt rowId, colId;

        for (unsigned int r = 0; r < num_rows; ++r) {
            rowId = m_ulpMap[eid][r];
            for (unsigned int c = 0; c < num_rows; ++c) {
                colId = m_ulpMap[eid][c];
                value = e_mat(r,c);
                if (fabs(value) > 1e-16)
                    MatSetValue(m_pMat, rowId, colId, value, mode);
            }
        }
        return Error::SUCCESS;
    } // set_element_matrix_term_by_term

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>:: petsc_compare_matrix(){
        PetscBool flg;
        MatEqual(m_pMat, m_pMat_matvec, &flg);
        if (flg == PETSC_TRUE) {
            printf("Matrices are equal\n");
        } else {
            printf("Matrices are NOT equal\n");
        }
        return Error::SUCCESS;
    } // petsc_compare_matrix

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>:: petsc_norm_matrix_difference(){
        PetscScalar a = -1.0;
        PetscReal nrm;

        // subtraction: m_pMat = m_pMat + a * m_pMat_matvec
        MatAXPY(m_pMat, a, m_pMat_matvec, SAME_NONZERO_PATTERN);

        // compute norm of the matrix difference
        //MatNorm(m_pMat, NORM_FROBENIUS, &nrm);
        MatNorm(m_pMat, NORM_INFINITY, &nrm);

        printf("Norm of matrix difference = %10.6f\n", nrm);
        return Error::SUCCESS;
    } // petsc_norm_matrix_difference


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::dump_mat_matvec( const char* filename /* = nullptr */ ) const {

        if( filename == nullptr ) {
            MatView( m_pMat_matvec, PETSC_VIEWER_STDOUT_WORLD );
        }
        else {

            // write matrix m_pMat_matvec to file
            PetscViewer viewer;
            PetscViewerASCIIOpen( m_comm, filename, &viewer );

            // write to file readable by Matlab
            PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);

            MatView( m_pMat_matvec, viewer );
            PetscViewerDestroy( &viewer );
        }

        return Error::SUCCESS;
    } // dump_mat_matvec

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_matmult( Vec x, Vec result ){
        MatMult( m_pMat, x, result );
        return Error::SUCCESS;
    } // petsc_matmult

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>:: petsc_set_matrix_matvec( DT* vec, unsigned int global_column, InsertMode mode /* = ADD_VALUES */ ) {

        PetscScalar value;
        PetscInt rowId;
        PetscInt colId;

        // set elements of vector to the corresponding column of matrix
        colId = global_column;
        for (unsigned int i = 0; i < m_uiNumNodes; i++){
            value = vec[i];
            //rowId = local_to_global[i];
            rowId = m_ulpLocal2Global[i];
            // std::cout << "setting: " << rowId << "," << colId << std::endl;
            if (fabs(value) > 1e-16) {
                MatSetValue(m_pMat_matvec, rowId, colId, value, mode);
            }
        }

        return Error::SUCCESS;
    } // petsc_set_matrix_matvec

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::print_vector( const DT* vec, bool ghosted /* = false */ ){
        if( ghosted ){
            // size of vec includes ghost DoFs, print local DoFs only
            for (unsigned int i = m_uiNodeLocalBegin; i < m_uiNodeLocalEnd; i++){
                printf("rank %d, v[%d] = %10.5f \n", m_uiRank, i - m_uiNumPreGhostNodes, vec[i]);
            }
        }
        else {
            // vec is only for local DoFs
            printf("here, rank %d, m_uiNumNodes = %d\n", m_uiRank, m_uiNumNodes);
            for (unsigned int i = m_uiNodeLocalBegin; i < m_uiNodeLocalEnd; i++){
                printf("rank %d, v[%d] = %10.5f \n", m_uiRank, i, vec[i-m_uiNumPreGhostNodes]);
            }
        }
        return Error::SUCCESS;
    } // print_vector

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::print_matrix(){
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            unsigned int row = m_epMat[eid].rows();
            unsigned int col = m_epMat[eid].cols();
            printf("rank= %d, eid= %d, row= %d, col= %d, m_epMat= \n", m_uiRank, eid, row, col);
            for (unsigned int r = 0; r < row; r++){
                for (unsigned int c = 0; c < col; c++){
                    printf("%10.3f\n",m_epMat[eid](r,c));
                }
            }
        }
        return Error::SUCCESS;
    } // print_matrix

    // transform vec to petsc vector for comparison between matrix-free and matrix-based methods
    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::transform_to_petsc_vector(const DT* vec, Vec petsc_vec, bool ghosted) {
        PetscScalar value;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            unsigned int num_nodes = m_uiNodesPerElem[eid];

            for (unsigned int i = 0; i < num_nodes; i++){
                const unsigned int nidG = m_ulpMap[eid][i]; // global node
                unsigned int nidL = m_uipLocalMap[eid][i];  // local node

                if (nidL >= m_uiNumPreGhostNodes && nidL < m_uiNumPreGhostNodes + m_uiNumNodes) {
                    // nidL is owned by me
                    if (ghosted){
                        value = vec[nidL];
                    } else {
                        value = vec[nidL - m_uiNumPreGhostNodes];
                    }
                    VecSetValue(petsc_vec, nidG, value, INSERT_VALUES);
                }
            }
        }
        return Error::SUCCESS;
    } // transform_to_petsc_vector

    // explicitly apply Dirichlet boundary conditions on structure vector
    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::set_vector_bc(DT* vec, unsigned int eid, const GI **dirichletBMap){
        unsigned int num_nodes = m_uiNodesPerElem[eid];
        unsigned int rowId, boundrow;
        //for (unsigned int r = 0; r < num_nodes * dof; r++){
        for (unsigned int r = 0; r < num_nodes; r++){
            rowId = m_uipLocalMap[eid][r];
            boundrow = dirichletBMap[eid][r];
            if (boundrow == 1) {
                // boundary node
                vec[rowId] = 0.0;
            }
        }
        return Error::SUCCESS;
    }// set_vector_bc


    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::mat_get_diagonal_block_seq(DT **diag_blk){
        //Note: diag_blk is already allocated with size of [m_uiNumNodes][m_uiNumNodes]
        unsigned int num_nodes;
        EigenMat e_mat;
        LI rowID, colID;

        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            num_nodes = m_uiNodesPerElem[eid];
            // get element matrix
            e_mat = m_epMat[eid];
            for (unsigned int r = 0; r < num_nodes; r++){
                rowID = m_uipLocalMap[eid][r];
                if ((rowID >= m_uiNodeLocalBegin) && (rowID < m_uiNodeLocalEnd)){
                    // only assembling if rowID is owned by my rank
                    for (unsigned int c = 0; c < num_nodes; c++) {
                        colID = m_uipLocalMap[eid][c];
                        if ((colID >= m_uiNodeLocalBegin) && (colID < m_uiNodeLocalEnd)) {
                            // only assembling if colID is owned by my rank
                            diag_blk[rowID - m_uiNumPreGhostNodes][colID - m_uiNumPreGhostNodes] += e_mat(r, c);
                        }
                    }
                }
            }
        }
        return Error::SUCCESS;
    }// mat_get_diagonal_block_seq

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::ghost_to_local_mat(DT** lMat, DT** gMat) const {
        for (unsigned int r = 0; r < m_uiNumNodes; r++){
            for (unsigned int c = 0; c < m_uiNumNodes; c++){
                lMat[r][c] = gMat[r + m_uiNumPreGhostNodes][c + m_uiNumPreGhostNodes];
            }
        }
        return Error::SUCCESS;
    }// ghost_to_local_mat



    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_set_element_matrix( LI eid, DT* e_mat, InsertMode mode /* = ADD_VALUES */ ) {
        unsigned int num_nodes = m_uiNodesPerElem[eid];

        // now set values ...
        //std::vector<PetscScalar> values(num_nodes * dof);
        std::vector<PetscScalar> values(num_nodes);
        //std::vector<PetscInt> colIndices(num_nodes * dof);
        std::vector<PetscInt> colIndices(num_nodes);

        PetscInt rowId;

        unsigned int index = 0;
        for (unsigned int r = 0; r < num_nodes; ++r) {
            rowId = m_ulpMap[eid][r];
            for (unsigned int c = 0; c < num_nodes; ++c) {
                colIndices[c] = m_ulpMap[eid][c];
                values[c] = e_mat[index];
                index++;
            } // c
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
            // values.clear();
            // colIndices.clear();
        } // r

        return Error::SUCCESS;
    } // petsc_set_element_matrix


    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::print_mepMat() {
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            for (unsigned int block = 0; block < m_epMat[eid].size(); block++){
                printf("rank %d, eid %d, block %d\n",m_uiRank,eid,block);
                for (unsigned int r = 0; r < m_epMat[eid][block].rows(); r++){
                    for (unsigned int c = 0; c < m_epMat[eid][block].rows(); c++){
                        printf("[%d,%d]= %f; ",r,c,m_epMat[eid][block](r,c));
                    }
                    printf("\n");
                }
            }
        }
        return Error::SUCCESS;
    }

    template <typename DT, typename GI, typename LI>
    par::Error aMat<DT, GI, LI>::mat_get_diagonal_block_dense(DT **diag_blk){
        LI blocks_dim;
        EigenMat e_mat;
        GI glo_RowId, glo_ColId;
        LI loc_RowId, loc_ColId;
        LI rowID, colID;
        unsigned int rank_r, rank_c;

        std::vector<MatRecord<DT,LI>> matRec;
        MatRecord<DT,LI> matr;

        m_vMatRec.clear();

        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){

            // number of blocks in row (or column)
            blocks_dim = (LI)sqrt(m_epMat[eid].size());
            assert (blocks_dim * blocks_dim == m_epMat[eid].size());

            LI block_row_offset = 0;
            LI block_col_offset = 0;
            LI num_dofs_per_block;

            for (LI block_i = 0; block_i < blocks_dim; block_i++){

                for (LI block_j = 0; block_j < blocks_dim; block_j++){

                    LI index = block_i * blocks_dim + block_j;

                    if (m_epMat[eid][index].size() != 0){

                        e_mat = m_epMat[eid][index];
                        assert(e_mat.rows() == e_mat.cols());

                        // number of dofs per block
                        num_dofs_per_block = e_mat.rows();
                        assert(num_dofs_per_block == m_uiNodesPerElem[eid]/blocks_dim);

                        for (LI r = 0; r < num_dofs_per_block; r++){

                            // local row Id (include ghost nodes)
                            rowID = m_uipLocalMap[eid][block_row_offset + r];

                            // global row Id
                            glo_RowId = m_ulpMap[eid][block_row_offset + r];

                            //rank who owns global row Id
                            rank_r = globalId_2_rank(glo_RowId);

                            //local ID in that rank (not include ghost nodes)
                            loc_RowId = (glo_RowId - m_uivLocalNodeScan[rank_r]);

                            for (LI c = 0; c < num_dofs_per_block; c++){
                                // local column Id (include ghost nodes)
                                colID = m_uipLocalMap[eid][block_col_offset + c];

                                // global column Id
                                glo_ColId = m_ulpMap[eid][block_col_offset + c];

                                // rank who owns global column Id
                                rank_c = globalId_2_rank(glo_ColId);

                                // local column Id in that rank (not include ghost nodes)
                                loc_ColId = (glo_ColId - m_uivLocalNodeScan[rank_c]);

                                // assemble to block diagonal matrix if both i and j belong to my rank
                                if ((rowID >= m_uiNodeLocalBegin) && (rowID < m_uiNodeLocalEnd) && (colID >= m_uiNodeLocalBegin) && (colID < m_uiNodeLocalEnd)){
                                    diag_blk[rowID - m_uiNumPreGhostNodes][colID - m_uiNumPreGhostNodes] += e_mat(r,c);
                                }

                                // set to matrix record when both i and j belong to the same rank
                                if ((rank_r == rank_c)&&(rank_r != m_uiRank)){
                                    matr.setRank(rank_r);
                                    matr.setRowId(loc_RowId);
                                    matr.setColId(loc_ColId);
                                    matr.setVal(e_mat(r,c));
                                    matRec.push_back(matr);
                                }
                            }
                        }
                    } // if block is not null

                    block_col_offset += num_dofs_per_block;

                } // for block_j

                block_row_offset += num_dofs_per_block;

            } // for block_i

        } // for (eid = 0:m_uiNumElems)

        std::sort(matRec.begin(), matRec.end());

        // accumulate value if 2 components of matRec are equal, then reduce the size of matRec
        unsigned int i = 0;
        while (i < matRec.size()) {
            matr.setRank(matRec[i].getRank());
            matr.setRowId(matRec[i].getRowId());
            matr.setColId(matRec[i].getColId());

            DT val = matRec[i].getVal();
            // since matRec is sorted, we keep increasing i for all members that are equal
            while (((i + 1) < matRec.size()) && (matRec[i] == matRec[i + 1])) {
                // accumulate value
                val += matRec[i + 1].getVal();
                // move i to the next member
                i++;
            }
            matr.setVal(val);

            // append the matr (with accumulated value) to m_vMatRec
            m_vMatRec.push_back(matr);

            // move i to the next member
            i++;
        }


        unsigned int* sendCounts = new unsigned int[m_uiSize];
        unsigned int* recvCounts = new unsigned int[m_uiSize];
        unsigned int* sendOffset = new unsigned int[m_uiSize];
        unsigned int* recvOffset = new unsigned int[m_uiSize];

        for (unsigned int i = 0; i < m_uiSize; i++){
            sendCounts[i] = 0;
            recvCounts[i] = 0;
        }

        // number of elements sending to each rank
        for (unsigned int i = 0; i < m_vMatRec.size(); i++){
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

        // accumulated to block diagonal matrix
        for (unsigned int i = 0; i < recv_buff.size(); i++){
            if (recv_buff[i].getRank() != m_uiRank) {
                return Error::WRONG_COMMUNICATION;
            } else {
                loc_RowId = recv_buff[i].getRowId();
                loc_ColId = recv_buff[i].getColId();
                diag_blk[loc_RowId][loc_ColId] += recv_buff[i].getVal();
            }
        }

        delete [] sendCounts;
        delete [] recvCounts;
        delete [] sendOffset;
        delete [] recvOffset;

        return Error::SUCCESS;
    } //mat_get_diagonal_block_dense

    /**@brief ********** FUNCTIONS ARE NO LONGER IN USE, JUST FOR REFERENCE *********************/
    // e_mat is an array of EigenMat with the size dictated by twin_level (e.g. twin_level = 1, then size of e_mat is 2)
    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::set_element_matrices( LI eid, const EigenMat* e_mat, unsigned int twin_level, InsertMode mode /* = ADD_VALUES */ ) {

        unsigned int num_nodes = m_uiNodesPerElem[eid];

        // number of twinning matrices (e.g. twin_level = 2 then numEMat = 4)
        unsigned int numEMat = (1u<<twin_level);

        // since e_mat is dynamic, then first casting it to void* so that we can move for each element of e_mat
        const void* eMat= (const void*)e_mat;
        for (unsigned int i=0; i<numEMat; i++) {
            size_t bytes=0;
            if (num_nodes==4){
                 bytes = sizeof(Eigen::Matrix<DT,4,4>);
                 petsc_set_element_matrix(eid,(*(Eigen::Matrix<DT,4,4>*)eMat), i, mode);

            } else if(num_nodes==8) {
                bytes = sizeof(Eigen::Matrix<DT,8,8>);
                petsc_set_element_matrix(eid,(*(Eigen::Matrix<DT,8,8>*)eMat), i, mode);

            }else {
                return Error::UNKNOWN_ELEMENT_TYPE;
            }

            // move to next block (each block has size of bytes)
            eMat= (char*)eMat + bytes;
        }
        return Error::SUCCESS;
    } // set_element_matrices

    // used with set_element_matrices for the case of one eid but multiple matrices
    template <typename DT,typename GI, typename LI>
    par::Error aMat<DT,GI,LI>::petsc_set_element_matrix(LI eid, const EigenMat e_mat, LI e_mat_id, InsertMode mode) {

        unsigned int num_nodes = m_uiNodesPerElem[eid];

        assert(e_mat.rows()==e_mat.cols());
        unsigned int num_rows = e_mat.rows(); // num_rows = num_nodes * dof

        // copy element matrix
        m_epMat[eid * AMAT_MAX_EMAT_PER_ELEMENT + e_mat_id ] = e_mat;

        // now set values ...
        std::vector<PetscScalar> values(num_rows);
        std::vector<PetscInt> colIndices(num_rows);
        PetscInt rowId;

        //unsigned int index = 0;
        for (unsigned int r = 0; r < num_rows; ++r) {
            //rowId = m_ulpMap[eid][r]; // map in terms of dof
            //rowId = dof*m_ulpMap[eid][r/dof] + r%dof; // map in terms of nodes, 1 matrix per eid (old version)
            //rowId = dof*m_ulpMap[eid][e_mat_id * num_nodes + r/dof] + r%dof; // map in terms of nodes, multiple matrices per eid
            rowId = m_ulpMap[eid][e_mat_id * num_nodes + r]; // map in terms of nodes, multiple matrices per eid
            for (unsigned int c = 0; c < num_rows; ++c) {
                //colIndices[c] = m_ulpMap[eid][c];
                //colIndices[c] = dof*m_ulpMap[eid][c/dof] + c%dof;
                //colIndices[c] = dof*m_ulpMap[eid][e_mat_id * num_nodes + c/dof] + c%dof;
                colIndices[c] = m_ulpMap[eid][e_mat_id * num_nodes + c];
                values[c] = e_mat(r,c);
            } // c
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
        } // r

        return Error::SUCCESS;
    } // petsc_set_element_matrix

} // end namespace par
