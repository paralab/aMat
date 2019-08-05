/**
 * @file aMat.hpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
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
#include <vector>
#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include <mpi.h>
#include <petscksp.h>
#include "Dense"
PETSC_EXTERN PetscErrorCode PCCreate_Jacobi(PC);

#define AMAT_MAX_CRACK_LEVEL 0 // number of cracks allowed in 1 element
#define AMAT_MAX_EMAT_PER_ELEMENT (1u<<AMAT_MAX_CRACK_LEVEL) // max number of cracked elements on each element

namespace par {
    enum class Error {SUCCESS,
                      INDEX_OUT_OF_BOUNDS,
                      UNKNOWN_ELEMENT_TYPE,
                      UNKNOWN_ELEMENT_STATUS,
                      NULL_L2G_MAP,
                      GHOST_NODE_NOT_FOUND
    };
    enum class ElementType {TET, HEX};

    // AsyncExchangeCtx is downloaded from Dendro-5.0 written by Milinda Fernando and Hari Sundar;
    // Using of AsyncExchange was permitted by the author Milinda Fernando
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
            AsyncExchangeCtx(const void* var) {
                m_uiBuffer = (void*)var;
                m_uiSendBuf = NULL;
                m_uiRecvBuf = NULL;
                m_uiRequests.clear();
            }
            /**@brief allocates send buffer for ghost exchange */
            inline void allocateSendBuffer(size_t bytes) {
                m_uiSendBuf = malloc(bytes);
            }
            /**@brief allocates recv buffer for ghost exchange */
            inline void allocateRecvBuffer(size_t bytes) {
                m_uiRecvBuf = malloc(bytes);
            }
            /**@brief allocates send buffer for ghost exchange */
            inline void deAllocateSendBuffer() {
                free(m_uiSendBuf);
                m_uiSendBuf = NULL;
            }
            /**@brief allocates recv buffer for ghost exchange */
            inline void deAllocateRecvBuffer() {
                free(m_uiRecvBuf);
                m_uiRecvBuf = NULL;
            }
            /**@brief */
            inline void* getSendBuffer() {
                return m_uiSendBuf;
            }
            /**@brief */
            inline void* getRecvBuffer() {
                return m_uiRecvBuf;
            }
            /**@brief */
            inline const void* getBuffer() {
                return m_uiBuffer;
            }
            /**@brief */
            inline std::vector<MPI_Request*>& getRequestList(){
                return m_uiRequests;
            }
            /**@brief */
            bool operator== (AsyncExchangeCtx other) const{
                return( m_uiBuffer == other.m_uiBuffer );
            }

            ~AsyncExchangeCtx() {
                /*for(unsigned int i=0;i<m_uiRequests.size();i++)
                 {
                     delete m_uiRequests[i];
                     m_uiRequests[i]=NULL;
                 }
                 m_uiRequests.clear();*/
            }
        };


    template <typename T,typename I>
    class aMat {
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> EigenMat;

    protected:
        /**@brief communicator used within aMat */
        MPI_Comm       m_comm;

        /**@brief my rank */
        unsigned int   m_uiRank;

        /**@brief total number of ranks */
        unsigned int   m_uiSize;

        /**@brief (local) number of DoFs owned by rank */
        unsigned int   m_uiNumNodes;

        /**@brief (global) number of DoFs owned by all ranks */
        unsigned long  m_uiNumNodesGlobal;

        /**@brief (local) number of elements owned by rank */
        unsigned int   m_uiNumElems;

        /**@brief max number of DoFs per element*/
        unsigned int   m_uiMaxNodesPerElem;

        /**@brief assembled stiffness matrix */
        Mat            m_pMat;

        /**@brief storage of element matrices */
        EigenMat*      m_mats;

        /**@brief map from local DoF of element to global DoF: m_ulpMap[eid][local_id]  = global_id */
        I**            m_ulpMap;

        /**@brief map from local DoF of element to local DoF: m_uiMap[eid][element_node]  = local node-ID */
        unsigned int** m_uiMap;

        /**@brief number of DoFs owned by each rank, NOT include ghost DoFs */
        std::vector<unsigned int> m_uiLocalNodeCounts;

        /**@brief number of elements owned by each rank */
        std::vector<unsigned int> m_uiLocalElementCounts;

        /**@brief exclusive scan of (local) number of DoFs */
        std::vector<unsigned int> m_uiLocalNodeScan; // todo use I

        /**@brief exclusive scan of (local) number of elements */
        std::vector<unsigned int> m_uiLocalElementScan;

        /**@brief number of ghost DoFs owned by "pre" processes (whose ranks are smaller than m_uiRank) */
        unsigned int              m_uiNumPreGhostNodes;

        /**@brief total number of ghost DoFs owned by "post" processes (whose ranks are larger than m_uiRank) */
        unsigned int              m_uiNumPostGhostNodes;

        /**@brief number of DoFs sent to each process (size = m_uiSize) */
        std::vector<unsigned int> m_uiSendNodeCounts;

        /**@brief offsets (i.e. exclusive scan) of m_uiSendNodeCounts */
        std::vector<unsigned int> m_uiSendNodeOffset;

        /**@brief local DoF IDs to be sent (size = total number of nodes to be sent */
        std::vector<unsigned int> m_uiSendNodeIds;

        /**@brief number of DoFs to be received from each process (size = m_uiSize) */
        std::vector<unsigned int> m_uiRecvNodeCounts;

        /**@brief offsets (i.e. exclusive scan) of m_uiRecvNodeCounts */
        std::vector<unsigned int> m_uiRecvNodeOffset;

        /**@brief local node-ID starting of pre-ghost nodes, always = 0 */
        unsigned int              m_uiNodePreGhostBegin;

        /**@brief local node-ID ending of pre-ghost nodes */
        unsigned int              m_uiNodePreGhostEnd;

        /**@brief local node-ID starting of nodes owned by me */
        unsigned int              m_uiNodeLocalBegin;

        /**@brief local node-ID ending of nodes owned by me */
        unsigned int              m_uiNodeLocalEnd;

        /**@brief local node-ID starting of post-ghost nodes */
        unsigned int              m_uiNodePostGhostBegin;

        /**@brief local node-ID ending of post-ghost nodes */
        unsigned int              m_uiNodePostGhostEnd;

        /**@brief total number of nodes including ghost nodes and nodes owned by me */
        unsigned int              m_uiNumNodesTotal;

        /**@brief MPI communication tag*/
        int                       m_uiCommTag;

        /**@brief ghost exchange context*/
        std::vector<AsyncExchangeCtx> m_uiAsyncCtx;

        /**@brief VARIABLES USED ONLY IN AMAT, NOT IN DISTMAT */
        par::ElementType*         m_pEtypes; // type of element list

        /**@brief TEMPORARY VARIABLES FOR DEBUGGING */
        Mat                       m_pMat_matvec; // matrix created by matvec() to compare with m_pMat
        I*                        m_uiLocal2Global; // map from local dof to global dof, temporarily used for testing matvec()

    public:

        /**@brief constructor to initialize variables of aMat */
        aMat(unsigned int nelem, par::ElementType* etype, unsigned int n_local, MPI_Comm comm);
        /**@brief destructor of aMat */
        ~aMat();


        /**@brief set mapping from element local node to global node */
        inline par::Error set_map(I** map){
            m_ulpMap = map;
            return Error::SUCCESS;
        }
        /**@brief build scatter-gather map (used for communication) and local-to-local map (used for matvec) */
        par::Error buildScatterMap();// todo buildScatterMap should be in protected


        /**@brief return number of DoFs owned by this rank*/
        inline unsigned int get_local_num_nodes() const {
            return m_uiNumNodes;
        }
        /**@brief return number of elements owned by this rank*/
        inline unsigned int get_local_num_elements() const {
            return m_uiNumElems;
        }
        /**@brief return the map from DoF of element to local ID of vector (included ghost DoFs) */
        inline const unsigned int ** get_e2local_map() const {
            return (const unsigned int **)m_uiMap;
        }
        /**@brief return the map from DoF of element to global ID */
        inline const I ** get_e2global_map() const {
            return (const I **)m_ulpMap;
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
            const unsigned int nid = m_uiMap[eid][enid];
            if (nid >= m_uiNodeLocalBegin && nid < m_uiNodeLocalEnd)
                return true;
            else
                return false;
        }


        /**@brief begin assembling the matrix "m_pMat", called after MatSetValues */
        inline par::Error petsc_init_mat(MatAssemblyType mode) const {
            MatAssemblyBegin(m_pMat, mode);
            return Error::SUCCESS;
        }
        /**@brief complete assembling the matrix "m_pMat", called before using the matrix */
        inline par::Error petsc_finalize_mat(MatAssemblyType mode) const {
            MatAssemblyEnd(m_pMat,mode);
            return Error::SUCCESS;
        }
        /**@brief begin assembling the petsc vec (defined outside aMat) */
        inline par::Error petsc_init_vec(Vec vec) const {
            VecAssemblyBegin(vec);
            return Error::SUCCESS;
        }
        /**@brief end assembling the petsc vec (defined outside aMat) */
        inline par::Error petsc_finalize_vec(Vec vec) const {
            VecAssemblyEnd(vec);
            return Error::SUCCESS;
        }
        /**@brief allocate memory for a PETSc vector "vec", initialized by alpha */
        par::Error petsc_create_vec(Vec &vec, PetscScalar alpha = 0.0) const;
        /**@brief assembly global load vector */
        par::Error petsc_set_element_vec(Vec vec, unsigned int eid, T* e_vec, InsertMode mode = ADD_VALUES);
        /**@brief assembly element matrix to structural matrix (for matrix-based method) */
        par::Error petsc_set_element_matrix(unsigned int eid, T *e_mat, InsertMode mode = ADD_VALUES);
        par::Error petsc_set_element_matrix(unsigned int eid, EigenMat e_mat, InsertMode = ADD_VALUES);
        /**@brief: write PETSc matrix "m_pMat" to filename "fmat" */
        par::Error dump_mat(const char* fmat) const;
        /**@brief: write PETSc vector "vec" to filename "fvec" */
        par::Error dump_vec(const char* fvec, Vec vec) const;
        /**@brief get diagonal of m_pMat and put to vec */
        par::Error petsc_get_diagonal(Vec vec) const;
        /**@brief free memory allocated for PETSc vector*/
        par::Error petsc_destroy_vec(Vec &vec) const;

        /**@brief allocate memory for "vec", size includes ghost DoFs if isGhosted=tru, initialized by alpha */
        par::Error create_vec(T* &vec, bool isGhosted = false, T alpha = (T)0);
        /**@brief copy local to corresponding positions of gVec (size including ghost DoFs) */
        par::Error local_to_ghost(T*  gVec, const T* local);
        /**@brief copy gVec (size including ghost DoFs) to local (size of local DoFs) */
        par::Error ghost_to_local( T*  local, const T* gVec);
        /**@brief copy element matrix and store in m_mats, used for matrix-free method */
        par::Error copy_element_matrix(unsigned int eid, EigenMat e_mat);
        /**@brief get diagonal terms of structure matrix by accumulating diagonal of element matrices */
        par::Error get_diagonal(T* diag, bool isGhosted = false);
        /**@brief get diagonal terms with ghosted vector diag */
        par::Error get_diagonal_ghosted(T* diag);
        /**@brief get max number of DoF per element*/
        par::Error get_max_dof_per_elem();
        /**@brief free memory allocated for vec and set vec to null */
        par::Error destroy_vec(T* &vec);

        /**@brief begin: owned DoFs send, ghost DoFs receive, called before matvec() */
        par::Error ghost_receive_begin(T* vec);
        /**@brief end: ghost DoFs receive, called before matvec() */
        par::Error ghost_receive_end(T* vec);
        /**@brief begin: ghost DoFs send, owned DoFs receive and accumulate to current data, called after matvec() */
        par::Error ghost_send_begin(T* vec);
        /**@brief end: ghost DoFs send, owned DoFs receive and accumulate to current data, called after matvec() */
        par::Error ghost_send_end(T* vec);

        /**@brief v = K * u (K is not assembled, but directly using elemental K_e's)
         * @param[in] isGhosted = true, if v and u are of size including ghost DoFs
         * @param[in] isGhosted = false, if v and u are of size NOT including ghost DoFs
         * */
        par::Error matvec(T* v, const T* u, bool isGhosted = false);
        /**@brief v = K * u; v and u are of size including ghost DoFs*/
        par::Error matvec_ghosted(T* v, T* u);




        /**@brief apply Dirichlet boundary conditions by modifying the matrix "m_pMat" and RHS vector
         * @param[in] dirichletBMap[eid][num_nodes]: indicator of boundary node (1) or interior node (0)
         * */
        par::Error apply_dirichlet(Vec rhs, unsigned int eid, const I** dirichletBMap);
        /**@brief: invoke basic PETSc solver, "out" is solution vector */
        par::Error petsc_solve(const Vec rhs,Vec out) const;


        /**@brief ********* FUNCTIONS FOR DEBUGGING **************************************************/
        inline void echo_rank() const {
            printf("echo from rank= %d\n", m_uiRank);
        }

        inline par::Error petsc_init_mat_matvec(MatAssemblyType mode) const {
            MatAssemblyBegin(m_pMat_matvec, mode);
            return Error::SUCCESS;
        }

        inline par::Error petsc_finalize_mat_matvec(MatAssemblyType mode) const{
            MatAssemblyEnd(m_pMat_matvec,mode);
            return Error::SUCCESS;
        }

        inline par::Error set_Local2Global(I* local_to_global){
            m_uiLocal2Global = local_to_global;
            return Error::SUCCESS;
        }

        /**@brief create pestc matrix with size of m_uniNumNodes^2, used in testing matvec() */
        par::Error petsc_create_matrix_matvec();

        /**@brief assemble matrix term by term so that we can control not to assemble "almost zero" terms*/
        par::Error set_element_matrix_term_by_term(unsigned int eid, EigenMat e_mat, InsertMode mode = ADD_VALUES);

        /**@brief compare 2 matrices */
        par::Error petsc_compare_matrix();

        /**@brief compute the norm of the diffference of 2 matrices*/
        par::Error petsc_norm_matrix_difference();

        /**@brief print out to file matrix "m_pMat_matvec" (matrix created by using matvec() to multiply
         * m_pMat with series of vectors [1 0 0...]
         */
        par::Error dump_mat_matvec(const char* fmat) const;

        /**@brief y = m_pMat * x */
        par::Error petsc_matmult(Vec x, Vec y);

        /**@brief set entire vector "vec" to the column "nonzero_row" of matrix m_pMat_matvec, to compare with m_pMat*/
        par::Error petsc_set_matrix_matvec(T* vec, unsigned int nonzero_row, InsertMode mode = ADD_VALUES);

        /**@brief: test only: display all components of vector on screen */
        par::Error print_vector(const T* vec, bool ghosted = false);

        /**@brief: test only: display all element matrices (for purpose of debugging) */
        par::Error print_matrix();

        /**@brief: transform vec to pestc vector (for comparison between matrix-free and matrix-based)*/
        par::Error transform_to_petsc_vector(const T* vec, Vec petsc_vec, bool ghosted = false);

        /**@brief: apply zero Dirichlet boundary condition on nodes dictated by dirichletBMap */
        par::Error set_vector_bc(T* vec, unsigned int eid, const I **dirichletBMap);




        /**@brief ********** FUNCTIONS USED ONLY IN AMAT, NOT IN DISTMAT ****************************/
        /** @brief number of nodes per element */
        inline static unsigned int nodes_per_element(par::ElementType etype) {
            switch (etype) {
                case par::ElementType::TET:
                    return 4;
                case par::ElementType::HEX:
                    return 8;
                default:
                    return (unsigned int)Error::UNKNOWN_ELEMENT_TYPE;
            }
        }



        /**@brief ********** FUNCTIONS ARE NO LONGER IN USE, JUST FOR REFERENCE *********************/
        inline static unsigned int dofs_per_element(par::ElementType etype, unsigned int estatus) {
            switch (etype) {
                case par::ElementType::TET: {
                    if (estatus == 0) {
                        return 4*3;
                    } else if (estatus == 1) {
                        return 8*3;
                    } else if (estatus == 2) {
                        return 16*3;
                    } else {
                        return (unsigned int)Error::UNKNOWN_ELEMENT_STATUS;
                    }
                }
                case par::ElementType::HEX: {
                    if (estatus == 0) {
                        return 8*3;
                    } else if (estatus == 1) {
                        return 16*3;
                    } else if (estatus == 2) {
                        return 32*3;
                    } else {
                        return (unsigned int)Error::UNKNOWN_ELEMENT_STATUS;
                    }
                }
                default:
                    return (unsigned int)Error::UNKNOWN_ELEMENT_TYPE;
            }
        }

        inline unsigned int get_nodes_per_element (unsigned int eid) const {
            return nodes_per_element(m_pEtypes[eid]);
        }

        /**@brief assembly element matrix to structure matrix, multiple levels of twining
         * @param[in] e_mat : element stiffness matrices (pointer)
         * @param[in] twin_level: level of twinning (0 no crack, 1 one crack, 2 two cracks, 3 three cracks)
         * */
        par::Error set_element_matrices(unsigned int eid, EigenMat* e_mat, unsigned int twin_level, InsertMode mode=ADD_VALUES);

        par::Error petsc_set_element_matrix(unsigned int eid, EigenMat e_mat, unsigned int e_mat_id, InsertMode mode=ADD_VALUES);

    }; // end of class aMat

    /**@brief ********************************************************************************************************/

    // constructor
    template <typename T,typename I>
    aMat<T,I>::aMat(unsigned int nelem, par::ElementType* etype, unsigned int n_local, MPI_Comm comm) {

        m_comm = comm;

        MPI_Comm_rank(comm, (int*)&m_uiRank);
        MPI_Comm_size(comm, (int*)&m_uiSize);

        m_uiNumNodes = n_local;

        unsigned long nl = m_uiNumNodes;

        MPI_Allreduce(&nl, &m_uiNumNodesGlobal, 1, MPI_LONG, MPI_SUM, m_comm);

        MatCreate(m_comm, &m_pMat);
        //MatSetSizes(m_pMat, m_uiNumNodes*m_uiNumDOFperNode, m_uiNumNodes*m_uiNumDOFperNode, PETSC_DECIDE, PETSC_DECIDE);
        MatSetSizes(m_pMat, m_uiNumNodes, m_uiNumNodes, PETSC_DECIDE, PETSC_DECIDE);

        if(m_uiSize > 1) {
            // initialize matrix
            MatSetType(m_pMat, MATMPIAIJ);
            //MatMPIAIJSetPreallocation(m_pMat, 30*m_uiNumDOFperNode , PETSC_NULL, 30*m_uiNumDOFperNode , PETSC_NULL);
            MatMPIAIJSetPreallocation(m_pMat, 30 , PETSC_NULL, 30 , PETSC_NULL);

        } else {
            MatSetType(m_pMat, MATSEQAIJ);
            //MatSeqAIJSetPreallocation(m_pMat, 30*m_uiNumDOFperNode, PETSC_NULL);
            MatSeqAIJSetPreallocation(m_pMat, 30, PETSC_NULL);
        }
        // this will disable on preallocation errors. (but not good for performance)
        MatSetOption(m_pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

        // number of elements per rank
        m_uiNumElems = nelem;

        // etype[eid] = type of element eid
        m_pEtypes = etype;

        // storage of element matrices, each element is allocated totally AMAT_MAX_EMAT_PER_ELEMENT of EigenMat
        //m_mats = new EigenMat[m_uiNumElems * AMAT_MAX_EMAT_PER_ELEMENT]; // could be use if they decide to have multiple matrices for cracked elements
        m_mats = new EigenMat[m_uiNumElems];

        m_ulpMap = NULL;

        m_uiMap = new unsigned int*[m_uiNumElems];
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            m_uiMap[eid] = new unsigned int[nodes_per_element(m_pEtypes[eid])];
        }

    } // constructor

    template <typename T,typename I>
    aMat<T,I>::~aMat()
    {
        delete [] m_mats;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            delete [] m_uiMap[eid]; //delete array
        }
        delete [] m_uiMap;
        //MatDestroy(&m_pMat);
    } // ~aMat



    // build scatter map
    template <typename T,typename I>
    par::Error aMat<T,I>::buildScatterMap() {
        /* Assumptions: We assume that the global nodes are continuously partitioned across processors.
           Currently we do not account for twin elements
           "node" is actually "dof" because the map is in terms of dofs */

        if (m_ulpMap == NULL) return Error::NULL_L2G_MAP;

        m_uiLocalNodeCounts.clear();
        m_uiLocalElementCounts.clear();
        m_uiLocalNodeScan.clear();
        m_uiLocalElementScan.clear();

        m_uiLocalNodeCounts.resize(m_uiSize);
        m_uiLocalElementCounts.resize(m_uiSize);
        m_uiLocalNodeScan.resize(m_uiSize);
        m_uiLocalElementScan.resize(m_uiSize);

        // gather local counts
        MPI_Allgather(&m_uiNumNodes, 1, MPI_INT, &(*(m_uiLocalNodeCounts.begin())), 1, MPI_INT, m_comm);
        MPI_Allgather(&m_uiNumElems, 1, MPI_INT, &(*(m_uiLocalElementCounts.begin())), 1, MPI_INT, m_comm);

        // scan local counts
        m_uiLocalNodeScan[0] = 0;
        m_uiLocalElementScan[0] = 0;
        for (unsigned int p = 1; p < m_uiSize; p++) {
            m_uiLocalNodeScan[p] = m_uiLocalNodeScan[p-1] + m_uiLocalNodeCounts[p-1];
            m_uiLocalElementScan[p] = m_uiLocalElementScan[p-1] + m_uiLocalElementCounts[p-1];
        }

        // nodes are not owned by me: stored in pre or post lists
        std::vector<I> preGhostGIds;
        std::vector<I> postGhostGIds;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++) {
            par::ElementType e_type = m_pEtypes[eid];
            unsigned int num_nodes = aMat::nodes_per_element(e_type);

            for (unsigned int i = 0; i < num_nodes; i++) {
                const unsigned  int nid = m_ulpMap[eid][i];
                if (nid < m_uiLocalNodeScan[m_uiRank]) {
                    // pre-ghost global ID nodes
                    preGhostGIds.push_back(nid);
                } else if (nid >= (m_uiLocalNodeScan[m_uiRank] + m_uiNumNodes)){
                    // post-ghost global ID nodes
                    postGhostGIds.push_back(nid);
                } else {
                    assert ((nid >= m_uiLocalNodeScan[m_uiRank])  && (nid< (m_uiLocalNodeScan[m_uiRank] + m_uiNumNodes)));
                }
            }
        }

        // sort in ascending order
        std::sort(preGhostGIds.begin(), preGhostGIds.end());
        std::sort(postGhostGIds.begin(), postGhostGIds.end());

        // remove consecutive duplicates and erase all after .end()
        preGhostGIds.erase(std::unique(preGhostGIds.begin(), preGhostGIds.end()), preGhostGIds.end());
        postGhostGIds.erase(std::unique(postGhostGIds.begin(), postGhostGIds.end()), postGhostGIds.end());

        // number of ghost nodes
        m_uiNumPreGhostNodes = preGhostGIds.size();
        m_uiNumPostGhostNodes = postGhostGIds.size();

        // index of pre-ghost nodes
        m_uiNodePreGhostBegin = 0;
        m_uiNodePreGhostEnd = m_uiNumPreGhostNodes;

        // index of owned nodes
        m_uiNodeLocalBegin = m_uiNodePreGhostEnd;
        m_uiNodeLocalEnd = m_uiNodeLocalBegin + m_uiNumNodes;

        // index of post-ghost nodes
        m_uiNodePostGhostBegin = m_uiNodeLocalEnd;
        m_uiNodePostGhostEnd = m_uiNodePostGhostBegin + m_uiNumPostGhostNodes;

        // total number of nodes including ghost nodes
        m_uiNumNodesTotal = m_uiNumNodes + m_uiNumPreGhostNodes + m_uiNumPostGhostNodes;

        // owners of pre and post ghost nodes
        std::vector<unsigned int> preGhostOwner;
        std::vector<unsigned int> postGhostOwner;
        preGhostOwner.resize(m_uiNumPreGhostNodes);
        postGhostOwner.resize(m_uiNumPostGhostNodes);

        // pre-ghost
        unsigned int pcount = 0; // processor count, start from 0
        unsigned int gcount = 0; // node id count
        while (gcount < m_uiNumPreGhostNodes) {
            unsigned int nid = preGhostGIds[gcount];
            while ((pcount < m_uiRank) &&
                   (!((nid >= m_uiLocalNodeScan[pcount]) && (nid < (m_uiLocalNodeScan[pcount] + m_uiLocalNodeCounts[pcount]))))) {
                // nid NOT owned by pcount
                pcount++;
            }
            if (!((nid >= m_uiLocalNodeScan[pcount]) && (nid < (m_uiLocalNodeScan[pcount] + m_uiLocalNodeCounts[pcount])))) {
                std::cout << "m_uiRank: " << m_uiRank << " pre ghost gid : " << nid << " was not found in any processor" << std::endl;
                return Error::GHOST_NODE_NOT_FOUND;
            }
            while ((gcount < m_uiNumPreGhostNodes) &&
                   (((nid >= m_uiLocalNodeScan[pcount]) && (nid < (m_uiLocalNodeScan[pcount] + m_uiLocalNodeCounts[pcount]))))) {
                // nid owned by pcount
                preGhostOwner[gcount] = pcount;
                gcount++;
            }
        }

        // post-ghost
        pcount = m_uiRank; // start from my rank
        gcount = 0;
        while(gcount < m_uiNumPostGhostNodes)
        {
            unsigned int nid = postGhostGIds[gcount];
            while ((pcount < m_uiSize) &&
                   (!((nid >= m_uiLocalNodeScan[pcount]) && (nid < (m_uiLocalNodeScan[pcount] + m_uiLocalNodeCounts[pcount]))))){
                // nid is NOT owned by pcount
                pcount++;
            }
            if (!((nid >= m_uiLocalNodeScan[pcount]) && (nid < (m_uiLocalNodeScan[pcount] + m_uiLocalNodeCounts[pcount])))) {
                std::cout << "m_uiRank: " << m_uiRank << " post ghost gid : " << nid << " was not found in any processor" << std::endl;
                return Error::GHOST_NODE_NOT_FOUND;
            }
            while ((gcount < m_uiNumPostGhostNodes) &&
                   (((nid >= m_uiLocalNodeScan[pcount]) && (nid < (m_uiLocalNodeScan[pcount] + m_uiLocalNodeCounts[pcount]))))){
                // nid is owned by pcount
                postGhostOwner[gcount] = pcount;
                gcount++;
            }
        }

        /*for (unsigned i = 0; i < m_uiNumPreGhostNodes; i++) {
            printf("rank %d, preGhostNode %d, id = %d, owner = %d\n", m_uiRank, i, preGhostGIds[i], preGhostOwner[i]);
        }*/

        unsigned int * sendCounts = new unsigned int[m_uiSize];
        unsigned int * recvCounts = new unsigned int[m_uiSize];
        unsigned int * sendOffset = new unsigned int[m_uiSize];
        unsigned int * recvOffset = new unsigned int[m_uiSize];

        // Note: the send here is just for use in MPI_Alltoallv, it is NOT the sends in communications between processors later
        for (unsigned int i = 0; i < m_uiSize; i++) {
            // many of these will be zero, only none zero for processors that own my ghost nodes
            sendCounts[i] = 0;
        }
        for (unsigned int i = 0; i < m_uiNumPreGhostNodes; i++) {
            // accumulate number of nodes that owned by preGhostOwner[i]
            sendCounts[preGhostOwner[i]] += 1;
        }
        for (unsigned int i = 0; i < m_uiNumPostGhostNodes; i++) {
            // accumulate number of nodes that owned by postGhostOwner[i]
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

        std::vector<I> sendBuf;
        std::vector<I> recvBuf;

        // count of total send
        sendBuf.resize(sendOffset[m_uiSize-1] + sendCounts[m_uiSize-1]);
        // count of total recv
        recvBuf.resize(recvOffset[m_uiSize-1] + recvCounts[m_uiSize-1]);

        // put all sending data to sendBuf
        for(unsigned int i = 0; i < m_uiNumPreGhostNodes; i++)
            sendBuf[i] = preGhostGIds[i];
        for(unsigned int i = 0; i < m_uiNumPostGhostNodes; i++)
            sendBuf[i + m_uiNumPreGhostNodes] = postGhostGIds[i];

        for(unsigned int i = 0; i < m_uiSize; i++) {
            sendCounts[i] *= sizeof(I);
            sendOffset[i] *= sizeof(I);
            recvCounts[i] *= sizeof(I);
            recvOffset[i] *= sizeof(I);
        }

        MPI_Alltoallv(&(*(sendBuf.begin())), (int*)sendCounts, (int*)sendOffset, MPI_BYTE,
                      &(*(recvBuf.begin())), (int*)recvCounts, (int*)recvOffset, MPI_BYTE, m_comm);

        for(unsigned int i = 0; i < m_uiSize; i++)
        {
            sendCounts[i] /= sizeof(I);
            sendOffset[i] /= sizeof(I);
            recvCounts[i] /= sizeof(I);
            recvOffset[i] /= sizeof(I);
        }

        // build the scatter map.
        m_uiSendNodeIds.resize(recvBuf.size());

        // local node IDs that need to be sent out
        for(unsigned int i = 0; i < recvBuf.size(); i++) {
            const unsigned int gid = recvBuf[i];
            if (gid < m_uiLocalNodeScan[m_uiRank]  || gid >=  (m_uiLocalNodeScan[m_uiRank] + m_uiNumNodes)) {
                std::cout<<" m_uiRank: "<<m_uiRank<< "scatter map error : "<<__func__<<std::endl;
                par::Error::GHOST_NODE_NOT_FOUND;
            }
            m_uiSendNodeIds[i] = m_uiNumPreGhostNodes + (gid - m_uiLocalNodeScan[m_uiRank]);
        }

        m_uiSendNodeCounts.resize(m_uiSize);
        m_uiSendNodeOffset.resize(m_uiSize);
        m_uiRecvNodeCounts.resize(m_uiSize);
        m_uiRecvNodeOffset.resize(m_uiSize);

        for (unsigned int i = 0; i < m_uiSize; i++) {
            m_uiSendNodeCounts[i] = recvCounts[i];
            m_uiSendNodeOffset[i] = recvOffset[i];
            m_uiRecvNodeCounts[i] = sendCounts[i];
            m_uiRecvNodeOffset[i] = sendOffset[i];
        }

        // build local map m_uiMap[eid][nid]
        // structure displ vector = [0, ..., (m_uiNumPreGhostNodes - 1), --> ghost nodes owned by someone before me
        //    m_uiNumPreGhostNodes, ..., (m_uiNumPreGhostNodes + m_uiNumNodes - 1), --> nodes owned by me
        //    (m_uiNumPreGhostNodes + m_uiNumNodes), ..., (m_uiNumPreGhostNodes + m_uiNumNodes + m_uiNumPostGhostNodes - 1)] --> nodes owned by someone after me
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            par::ElementType e_type = m_pEtypes[eid];
            unsigned int num_nodes = aMat::nodes_per_element(e_type);
            for (unsigned int i = 0; i < num_nodes; i++){
                const unsigned int nid = m_ulpMap[eid][i];
                if (nid >= m_uiLocalNodeScan[m_uiRank] &&
                    nid < (m_uiLocalNodeScan[m_uiRank] + m_uiLocalNodeCounts[m_uiRank])) {
                    // nid is owned by me
                    m_uiMap[eid][i] = nid - m_uiLocalNodeScan[m_uiRank] + m_uiNumPreGhostNodes;
                } else if (nid < m_uiLocalNodeScan[m_uiRank]){
                    // nid is owned by someone before me
                    const unsigned int lookUp = std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), nid) - preGhostGIds.begin();
                    m_uiMap[eid][i] = lookUp;
                } else if (nid >= (m_uiLocalNodeScan[m_uiRank] + m_uiLocalNodeCounts[m_uiRank])){
                    // nid is owned by someone after me
                    const unsigned int lookUp = std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), nid) - postGhostGIds.begin();
                    m_uiMap[eid][i] =  (m_uiNumPreGhostNodes + m_uiNumNodes) + lookUp;
                }
            }
        }
        delete [] sendCounts;
        delete [] recvCounts;
        delete [] sendOffset;
        delete [] recvOffset;
        return Error::SUCCESS;
    } // buildScatterMap




    template <typename T,typename I>
    par::Error aMat<T,I>::petsc_create_vec(Vec &vec, PetscScalar alpha) const {
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

    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_set_element_vec(Vec vec, unsigned int eid, T *e_vec, InsertMode mode){

        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        //unsigned int dof = m_uiNumDOFperNode;

        PetscScalar value;
        PetscInt rowId;
        //unsigned int index = 0;

        //for (unsigned int r = 0; r < num_nodes*dof; ++r) {
        for (unsigned int r = 0; r < num_nodes; ++r) {
            //rowId = dof * m_ulpMap[eid][r/dof] + r % dof;
            rowId = m_ulpMap[eid][r];
            //value = e_vec[index];
            //index++;
            value = e_vec[r];
            VecSetValue(vec, rowId, value, mode);
        }
        return Error::SUCCESS;
    } // petsc_set_element_vec

    template <typename T,typename I>
    par::Error aMat<T,I>::petsc_set_element_matrix(unsigned int eid, T *e_mat, InsertMode mode){

        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        //unsigned int dof = m_uiNumDOFperNode;

        // now set values ...
        //std::vector<PetscScalar> values(num_nodes * dof);
        std::vector<PetscScalar> values(num_nodes);
        //std::vector<PetscInt> colIndices(num_nodes * dof);
        std::vector<PetscInt> colIndices(num_nodes);
        PetscInt rowId;

        unsigned int index = 0;
        //for (unsigned int r = 0; r < num_nodes * dof; ++r) {
        for (unsigned int r = 0; r < num_nodes; ++r) {
            //rowId = dof * m_ulpMap[eid][r/dof] + r % dof;
            rowId = m_ulpMap[eid][r];
            //for (unsigned int c = 0; c < num_nodes * dof; ++c) {
            for (unsigned int c = 0; c < num_nodes; ++c) {
                //colIndices[c] = dof * m_ulpMap[eid][c/dof] + c % dof;
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


    // use with Eigen, matrix-based, set every row of the matrix (faster than set every term of the matrix)
    template <typename T,typename I>
    par::Error aMat<T,I>::petsc_set_element_matrix(unsigned int eid, EigenMat e_mat, InsertMode mode) {

        par::ElementType e_type = m_pEtypes[eid];
        //unsigned int num_nodes = aMat::nodes_per_element(e_type);

        assert(e_mat.rows()==e_mat.cols());
        unsigned int num_rows = e_mat.rows();

        // assemble global matrix (petsc matrix)
        // now set values ...
        std::vector<PetscScalar> values(num_rows);
        std::vector<PetscInt> colIndices(num_rows);
        PetscInt rowId;
        for (unsigned int r = 0; r < num_rows; ++r) {
            rowId = m_ulpMap[eid][r];
            for (unsigned int c = 0; c < num_rows; ++c) {
                colIndices[c] = m_ulpMap[eid][c];
                values[c] = e_mat(r,c);
            } // c
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
        } // r

        return Error::SUCCESS;
    } // petsc_set_element_matrix


    template <typename T, typename I>
    par::Error aMat<T,I>::dump_mat(const char* fmat) const {
        PetscViewer viewer;
        PetscViewerASCIIOpen(m_comm, fmat, &viewer);
        // write to file readable by Matlab (filename must be filename.m in order to execute in Matlab)
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
        MatView(m_pMat,viewer);
        PetscViewerDestroy(&viewer);
        return Error::SUCCESS;
    } // dump_mat


    template <typename T, typename I>
    par::Error aMat<T,I>::dump_vec(const char* fvec, Vec vec) const {
        PetscViewer viewer;
        // write to ASCII file
        PetscViewerASCIIOpen(m_comm, fvec, &viewer);
        VecView(vec,viewer);
        PetscViewerDestroy(&viewer);
        return Error::SUCCESS;
    } // dump_vec


    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_get_diagonal(Vec vec) const {
        MatGetDiagonal(m_pMat, vec);
        return Error::SUCCESS;
    } //petsc_get_diagonal


    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_destroy_vec(Vec &vec) const {
        VecDestroy(&vec);
        return Error::SUCCESS;
    }


    template <typename T, typename I>
    par::Error aMat<T,I>::create_vec(T* &vec, bool isGhosted, T alpha){
        if (isGhosted){
            //vec = new T[m_uiNumNodesTotal * m_uiNumDOFperNode];
            vec = new T[m_uiNumNodesTotal];
        } else {
            //vec = new T[m_uiNumNodes * m_uiNumDOFperNode];
            vec = new T[m_uiNumNodes];
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

    template <typename T, typename I>
    par::Error aMat<T,I>::local_to_ghost(T*  gVec, const T* local){
        for (unsigned int i = 0; i < m_uiNumNodesTotal; i++){
            if ((i >= m_uiNumPreGhostNodes) && (i < m_uiNumPreGhostNodes + m_uiNumNodes)) {
                gVec[i] = local[i - m_uiNumPreGhostNodes];
            } else {
                gVec[i] = 0.0;
            }
        }
        return Error::SUCCESS;
    } // local_to_ghost

    template <typename T, typename I>
    par::Error aMat<T,I>::ghost_to_local(T* local, const T* gVec) {
        for (unsigned int i = 0; i < m_uiNumNodes; i++) {
            local[i] = gVec[i + m_uiNumPreGhostNodes];
        }
        return Error::SUCCESS;
    } // ghost_to_local

    template <typename T,typename I>
    par::Error aMat<T,I>::copy_element_matrix(unsigned int eid, EigenMat e_mat) {
        // store element matrix, will be used for matrix free
        m_mats[eid] = e_mat;
        return Error::SUCCESS;
    }// copy_element_matrix

    template <typename T, typename I>
    par::Error aMat<T,I>::get_diagonal(T* diag, bool isGhosted){
        if (isGhosted) {
            get_diagonal_ghosted(diag);
        } else {
            T* g_diag;
            create_vec(g_diag, true, 0.0);
            get_diagonal_ghosted(g_diag);
            ghost_to_local(diag, g_diag);
            delete[] g_diag;
        }
        return Error::SUCCESS;
    }// get_diagonal

    template <typename T, typename I>
    par::Error aMat<T,I>::get_diagonal_ghosted(T* diag){
        par::ElementType e_type;
        unsigned int num_nodes;
        EigenMat e_mat;
        I rowID;

        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            e_type = m_pEtypes[eid];
            num_nodes = aMat::nodes_per_element(e_type);
            e_mat = m_mats[eid];
            assert(e_mat.rows() == e_mat.cols());
            assert(e_mat.rows() == num_nodes);
            for (unsigned int r = 0; r < num_nodes; r++){
                rowID = m_uiMap[eid][r];
                diag[rowID] += e_mat(r,r);
            }
        }
        ghost_send_begin(diag);
        ghost_send_end(diag);

        return Error::SUCCESS;
    }// get_diagonal_ghosted

    template <typename T, typename  I>
    par::Error aMat<T,I>::get_max_dof_per_elem(){
        par::ElementType e_type;
        unsigned int num_nodes;
        unsigned int max_dpe = 0;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            e_type = m_pEtypes[eid];
            num_nodes = aMat::nodes_per_element(e_type);
            if (max_dpe < num_nodes) max_dpe = num_nodes;
        }
        m_uiMaxNodesPerElem = max_dpe;
        return Error::SUCCESS;
    }// get_max_dof_per_elem

    template <typename T, typename I>
    par::Error aMat<T,I>::destroy_vec(T* &vec) {
        if (vec != NULL) {
            delete[] vec;
            vec = NULL;
        }
        return Error::SUCCESS;
    }


    template <typename T, typename I>
    par::Error aMat<T,I>::ghost_receive_begin(T* vec) {

        // exchange context for vec
        AsyncExchangeCtx ctx((const void*)vec);

        // total number of DoFs to be sent
        const unsigned int total_send = m_uiSendNodeOffset[m_uiSize-1] + m_uiSendNodeCounts[m_uiSize-1];
        // total number of DoFs to be received
        const unsigned  int total_recv = m_uiRecvNodeOffset[m_uiSize-1] + m_uiRecvNodeCounts[m_uiSize-1];

        // send data of owned DoFs to corresponding ghost DoFs in other ranks
        if (total_send > 0){
            // allocate memory for sending buffer
            ctx.allocateSendBuffer(sizeof(T) * total_send);
            // get the address of sending buffer
            T* send_buf = (T*)ctx.getSendBuffer();
            // put all sending values to buffer
            for (unsigned int i = 0; i < total_send; i++){
                send_buf[i] = vec[m_uiSendNodeIds[i]];
            }
            for (unsigned int i = 0; i < m_uiSize; i++){
                // if nothing to send to rank i then skip
                if (m_uiSendNodeCounts[i] == 0) continue;
                // send to rank i
                MPI_Request* req = new MPI_Request();
                MPI_Isend(&send_buf[m_uiSendNodeOffset[i]], m_uiSendNodeCounts[i] * sizeof(T), MPI_BYTE, i, m_uiCommTag, m_comm, req);
                ctx.getRequestList().push_back(req);
            }
        }

        // received data for ghost DoFs
        if (total_recv > 0){
            ctx.allocateRecvBuffer(sizeof(T) * total_recv);
            T* recv_buf = (T*) ctx.getRecvBuffer();
            for (unsigned int i = 0; i < m_uiSize; i++){
                if (m_uiRecvNodeCounts[i] == 0) continue;
                MPI_Request* req = new MPI_Request();
                MPI_Irecv(&recv_buf[m_uiRecvNodeOffset[i]], m_uiRecvNodeCounts[i] * sizeof(T), MPI_BYTE, i, m_uiCommTag, m_comm, req);
                ctx.getRequestList().push_back(req);
            }
        }
        m_uiAsyncCtx.push_back(ctx);
        m_uiCommTag++; // get a different value if we have another ghost_exchange for a different vec
        return Error::SUCCESS;
    } //ghost_receive_begin

    template <typename T, typename I>
    par::Error aMat<T,I>::ghost_receive_end(T* vec) {

        unsigned int ctx_index;
        for (unsigned i = 0; i < m_uiAsyncCtx.size(); i++){
            if (vec == (T*)m_uiAsyncCtx[i].getBuffer()){
                ctx_index = i;
                break;
            }
        }
        AsyncExchangeCtx ctx = m_uiAsyncCtx[ctx_index];
        int num_req = ctx.getRequestList().size();

        MPI_Status sts;
        for (unsigned int i =0; i < num_req; i++) {
            MPI_Wait(ctx.getRequestList()[i], &sts);
        }

        const unsigned  int total_recv = m_uiRecvNodeOffset[m_uiSize-1] + m_uiRecvNodeCounts[m_uiSize-1];
        T* recv_buf = (T*) ctx.getRecvBuffer();
        std::memcpy(vec, recv_buf, m_uiNumPreGhostNodes*sizeof(T));
        std::memcpy(&vec[m_uiNumPreGhostNodes + m_uiNumNodes], &recv_buf[m_uiNumPreGhostNodes], m_uiNumPostGhostNodes*sizeof(T));

        ctx.deAllocateRecvBuffer();
        ctx.deAllocateSendBuffer();

        m_uiAsyncCtx.erase(m_uiAsyncCtx.begin() + ctx_index);
        return Error::SUCCESS;
    } // ghost_receive_end


    template <typename T, typename I>
    par::Error aMat<T,I>::ghost_send_begin(T* vec) {

        AsyncExchangeCtx ctx((const void*)vec);

        const unsigned  int total_recv = m_uiSendNodeOffset[m_uiSize-1] + m_uiSendNodeCounts[m_uiSize-1];
        const unsigned  int total_send = m_uiRecvNodeOffset[m_uiSize-1] + m_uiRecvNodeCounts[m_uiSize-1];

        if (total_recv > 0){
            ctx.allocateRecvBuffer(sizeof(T) * total_recv);
            T* recv_buf = (T*) ctx.getRecvBuffer();
            for (unsigned int i = 0; i < m_uiSize; i++){
                if (m_uiSendNodeCounts[i] == 0) continue;
                MPI_Request* req = new MPI_Request();
                MPI_Irecv(&recv_buf[m_uiSendNodeOffset[i]], m_uiSendNodeCounts[i]*sizeof(T), MPI_BYTE, i, m_uiCommTag, m_comm, req);
                ctx.getRequestList().push_back(req);
            }
        }

        if (total_send > 0){
            ctx.allocateSendBuffer(sizeof(T) * total_send);
            T* send_buf = (T*) ctx.getSendBuffer();

            for (unsigned int i = 0; i < m_uiNumPreGhostNodes; i++){
                send_buf[i] = vec[i];
            }
            for (unsigned int i = m_uiNumPreGhostNodes + m_uiNumNodes; i < m_uiNumPreGhostNodes + m_uiNumNodes + m_uiNumPostGhostNodes; i++){
                send_buf[i - m_uiNumNodes] = vec[i];
            }
            for (unsigned int i = 0; i < m_uiSize; i++){
                if (m_uiRecvNodeCounts[i] == 0) continue;
                MPI_Request* req = new MPI_Request();
                MPI_Isend(&send_buf[m_uiRecvNodeOffset[i]], m_uiRecvNodeCounts[i] * sizeof(T), MPI_BYTE, i, m_uiCommTag, m_comm, req);
                ctx.getRequestList().push_back(req);
            }
        }
        m_uiAsyncCtx.push_back(ctx);
        m_uiCommTag++; // get a different value if we have another ghost_exchange for a different vec
        return Error::SUCCESS;
    } // ghost_send_begin

    template <typename T, typename I>
    par::Error aMat<T,I>::ghost_send_end(T* vec) {

        unsigned int ctx_index;
        for (unsigned i = 0; i < m_uiAsyncCtx.size(); i++){
            if (vec == (T*)m_uiAsyncCtx[i].getBuffer()){
                ctx_index = i;
                break;
            }
        }
        AsyncExchangeCtx ctx = m_uiAsyncCtx[ctx_index];
        int num_req = ctx.getRequestList().size();

        MPI_Status sts;
        for(unsigned int i=0;i<num_req;i++)
        {
            MPI_Wait(ctx.getRequestList()[i],&sts);
        }

        const unsigned  int total_recv = m_uiSendNodeOffset[m_uiSize-1] + m_uiSendNodeCounts[m_uiSize-1];
        T* recv_buf = (T*) ctx.getRecvBuffer();

        for (unsigned int i = 0; i < m_uiSize; i++){
            for (unsigned int j = 0; j < m_uiSendNodeCounts[i]; j++){
                vec[m_uiSendNodeIds[m_uiSendNodeOffset[i]] + j] += recv_buf[m_uiSendNodeOffset[i] + j];
            }
        }
        ctx.deAllocateRecvBuffer();
        ctx.deAllocateSendBuffer();
        m_uiAsyncCtx.erase(m_uiAsyncCtx.begin() + ctx_index);
        return Error::SUCCESS;
    } // ghost_send_end

    template <typename  T, typename I>
    par::Error aMat<T,I>::matvec(T* v, const T* u, bool isGhosted) {
        if (isGhosted) {
            // std::cout << "GHOSTED MATVEC" << std::endl;
            matvec_ghosted(v, (T*)u);
        } else {
            // std::cout << "NON GHOSTED MATVEC" << std::endl;
            T* gv;
            T* gu;
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


    template <typename  T, typename I>
    par::Error aMat<T,I>::matvec_ghosted(T* v, T* u){

        par::ElementType e_type;
        unsigned int num_nodes;
        T* ue;
        T* ve;
        EigenMat emat;

        // initialize v (size of v = m_uiNodesPostGhostEnd = m_uiNumNodesTotal)
        for (unsigned int i = 0; i < m_uiNodePostGhostEnd; i++){
            v[i] = 0.0;
        }

        // allocate memory for element vectors
        ue = new T[m_uiMaxNodesPerElem];
        ve = new T[m_uiMaxNodesPerElem];

        I rowID;

        // send data from owned nodes to ghost nodes (of other processors) to get ready for computing v = Ku
        ghost_receive_begin(u);
        ghost_receive_end(u);

        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            e_type = m_pEtypes[eid];
            num_nodes = aMat::nodes_per_element(e_type); // reminder: "node" is dof

            // extract element vector ue from structure vector u
            //for (unsigned int r = 0; r < num_nodes * dof; ++r) {
            for (unsigned int r = 0; r < num_nodes; ++r) {
                //rowID = dof * m_uiMap[eid][r/dof] + r % dof;
                rowID = m_uiMap[eid][r];
                ue[r] = u[rowID];
            }

            // compute ve = ke * ue
            //elem_matvec(ve, (const T*) ue, eid); // this is old version

            // get element matrix from storage
            emat = m_mats[eid];
            assert(emat.rows() == emat.cols());
            assert(emat.rows() == num_nodes);
            for (unsigned int i = 0; i < emat.rows(); i++){
                ve[i] = 0.0;
                for (unsigned int j = 0; j < emat.cols(); j ++){
                    ve[i] += emat(i,j) * ue[j];
                }
            }

            // accumulate element vector ve to structure vector v
            //for (unsigned int r = 0; r < num_nodes * dof; r++){
            for (unsigned int r = 0; r < num_nodes; r++){
                //rowID = dof * m_uiMap[eid][r/dof] + r % dof;
                rowID = m_uiMap[eid][r];
                v[rowID] += ve[r];
            }
        }

        // send data from ghost nodes back to owned nodes after computing v
        ghost_send_begin(v);
        ghost_send_end(v);

        delete [] ue;
        delete [] ve;
        return Error::SUCCESS;
    } // matvec_ghosted


    // apply essential bc by modifying matrix and rhs vector (petsc vector)
    template <typename T, typename I>
    par::Error aMat<T,I>::apply_dirichlet(Vec rhs, unsigned int eid, const I** dirichletBMap) {
        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        //unsigned int dof = m_uiNumDOFperNode;

        PetscInt rowId, colId, boundrow, boundcol;

        //for (unsigned int r = 0; r < num_nodes*dof; r++) {
        for (unsigned int r = 0; r < num_nodes; r++) {
            //rowId = dof * m_ulpMap[eid][r/dof] + r % dof;
            rowId = m_ulpMap[eid][r];
            //boundrow = dirichletBMap[eid][r/dof]; // if a node is boundary, all dof's of this node are set to 0
            boundrow = dirichletBMap[eid][r];

            if (boundrow == 1) {
                VecSetValue(rhs, rowId, 0.0, INSERT_VALUES);
                //for (unsigned int c = 0; c < num_nodes*dof; c++) {
                for (unsigned int c = 0; c < num_nodes; c++) {
                    //colId = dof * m_ulpMap[eid][c/dof] + c % dof; //fixme: check for cases of dof > 1
                    colId = m_ulpMap[eid][c];
                    if (colId == rowId) {
                        MatSetValue(m_pMat, rowId, colId, 1.0, INSERT_VALUES);
                    } else {
                        MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                    }
                }
            } else {
                //for (unsigned int c = 0; c < num_nodes*dof; c++) {
                for (unsigned int c = 0; c < num_nodes; c++) {
                    //colId = dof * m_ulpMap[eid][c/dof] + c % dof;
                    colId = m_ulpMap[eid][c];
                    //boundcol =  dirichletBMap[eid][c/dof];
                    boundcol =  dirichletBMap[eid][c];
                    if (boundcol == 1) {
                        MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                    }
                }
            }
        }
        return Error::SUCCESS;
    } // apply_dirichlet

    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_solve(const Vec rhs, Vec out) const {

        KSP ksp;     // abstract Krylov object, linear solver context
        PC  pc;      // abstract preconditioner object, pre conditioner context

        KSPCreate(m_comm, &ksp); // create the default KSP context

        // specify solver
        //KSPSetType(ksp,KSPFGMRES); // set particular solver
        KSPSetType(ksp, KSPCG);

        //PCCreate(m_comm,&pc);
        KSPSetOperators(ksp, m_pMat, m_pMat); // set the matrix associated the linear system

        // specify preconditioner
        //KSPGetPC(ksp,&pc);
        //PCSetType(pc, PCJACOBI);

        // these 2 lines are not necessary, just need the above to set preconditioner
        //PCSetFromOptions(pc);
        //KSPSetPC(ksp,pc);

        KSPSetFromOptions(ksp); // set KSP options from the option database
        KSPSolve(ksp, rhs, out); // solve the linear system

        return Error::SUCCESS;
    } // petsc_solve



    /**@brief ********* FUNCTIONS FOR DEBUGGING **************************************************/

    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_create_matrix_matvec(){
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

    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrix_term_by_term(unsigned int eid, EigenMat e_mat, InsertMode mode) {

        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);

        assert(e_mat.rows()== e_mat.cols());
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

    template <typename T, typename I>
    par::Error aMat<T,I>:: petsc_compare_matrix(){
        PetscBool flg;
        MatEqual(m_pMat, m_pMat_matvec, &flg);
        if (flg == PETSC_TRUE) {
            printf("Matrices are equal\n");
        } else {
            printf("Matrices are NOT equal\n");
        }
        return Error::SUCCESS;
    } // petsc_compare_matrix

    template <typename T, typename I>
    par::Error aMat<T,I>:: petsc_norm_matrix_difference(){
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


    template <typename T, typename I>
    par::Error aMat<T,I>::dump_mat_matvec(const char* fmat) const {

        // write matrix m_pMat_matvec to file
        PetscViewer viewer;
        PetscViewerASCIIOpen(m_comm, fmat, &viewer);

        // write to file readable by Matlab
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);

        MatView(m_pMat_matvec,viewer);
        PetscViewerDestroy(&viewer);

        return Error::SUCCESS; // fixme
    } // dump_mat_matvec

    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_matmult(Vec x, Vec y){
        MatMult(m_pMat, x, y);
        return Error::SUCCESS;
    } // petsc_matmult

    template <typename T, typename I>
    par::Error aMat<T,I>:: petsc_set_matrix_matvec(T* vec, unsigned int global_column, InsertMode mode) {

        PetscScalar value;
        PetscInt rowId;
        PetscInt colId;

        // set elements of vector to the corresponding column of matrix
        colId = global_column;
        for (unsigned int i = 0; i < m_uiNumNodes; i++){
            value = vec[i];
            //rowId = local_to_global[i];
            rowId = m_uiLocal2Global[i];
            // std::cout << "setting: " << rowId << "," << colId << std::endl;
            if (fabs(value) > 1e-16)
                MatSetValue(m_pMat_matvec, rowId, colId, value, mode);
        }

        return Error::SUCCESS;
    } // petsc_set_matrix_matvec

    template <typename T, typename I>
    par::Error aMat<T,I>::print_vector(const T* vec, bool ghosted){
        if (ghosted){
            // size of vec includes ghost DoFs, print local DoFs only
            for (unsigned int i = m_uiNodeLocalBegin; i < m_uiNodeLocalEnd; i++){
                printf("rank %d, v[%d] = %10.5f \n", m_uiRank, i - m_uiNumPreGhostNodes, vec[i]);
            }
        } else {
            // vec is only for local DoFs
            printf("here, rank %d, m_uiNumNodes = %d\n", m_uiRank, m_uiNumNodes);
            for (unsigned int i = m_uiNodeLocalBegin; i < m_uiNodeLocalEnd; i++){
                printf("rank %d, v[%d] = %10.5f \n", m_uiRank, i, vec[i-m_uiNumPreGhostNodes]);
            }
        }
        return Error::SUCCESS;
    } // print_vector

    template <typename T, typename I>
    par::Error aMat<T,I>::print_matrix(){
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            unsigned int row = m_mats[eid].rows();
            unsigned int col = m_mats[eid].cols();
            printf("rank= %d, eid= %d, row= %d, col= %d, m_mats= \n", m_uiRank, eid, row, col);
            for (unsigned int r = 0; r < row; r++){
                for (unsigned int c = 0; c < col; c++){
                    printf("%10.3f\n",m_mats[eid](r,c));
                }
            }
        }
        return Error::SUCCESS;
    } // print_matrix

    // transform vec to petsc vector for comparison between matrix-free and matrix-based methods
    template <typename T, typename I>
    par::Error aMat<T,I>::transform_to_petsc_vector(const T* vec, Vec petsc_vec, bool ghosted) {
        PetscScalar value;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            par::ElementType e_type = m_pEtypes[eid];
            unsigned int num_nodes = aMat::nodes_per_element(e_type);
            //unsigned int dof = m_uiNumDOFperNode;

            //for (unsigned int i = 0; i < num_nodes * dof; i++){
            for (unsigned int i = 0; i < num_nodes; i++){
                const unsigned int nidG = m_ulpMap[eid][i]; // global node
                unsigned int nidL = m_uiMap[eid][i];  // local node

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
    template <typename T, typename I>
    par::Error aMat<T,I>::set_vector_bc(T* vec, unsigned int eid, const I **dirichletBMap){
        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        //unsigned int dof = m_uiNumDOFperNode;
        unsigned int rowId, boundrow;
        //for (unsigned int r = 0; r < num_nodes * dof; r++){
        for (unsigned int r = 0; r < num_nodes; r++){
            //rowId = dof * m_uiMap[eid][r/dof] + r % dof;
            rowId = m_uiMap[eid][r];
            //boundrow = dirichletBMap[eid][r/dof];
            boundrow = dirichletBMap[eid][r];
            if (boundrow == 1) {
                // boundary node
                vec[rowId] = 0.0;
            }
        }
        return Error::SUCCESS;
    }// set_vector_bc


    /**@brief ********** FUNCTIONS ARE NO LONGER IN USE, JUST FOR REFERENCE *********************/
    // e_mat is an array of EigenMat with the size dictated by twin_level (e.g. twin_level = 1, then size of e_mat is 2)
    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrices(unsigned int eid, EigenMat* e_mat, unsigned int twin_level, InsertMode mode) {
        par::ElementType e_type = m_pEtypes[eid];

        // number of twinning matrices (e.g. twin_level = 2 then numEMat = 4)
        unsigned int numEMat = (1u<<twin_level);

        // since e_mat is dynamic, then first casting it to void* so that we can move for each element of e_mat
        void* eMat= (void*)e_mat;
        for (unsigned int i=0; i<numEMat; i++) {
            size_t bytes=0;
            if (e_type==ElementType::TET){
                 bytes = sizeof(Eigen::Matrix<T,4,4>);
                 petsc_set_element_matrix(eid,(*(Eigen::Matrix<T,4,4>*)eMat), i, mode);

            } else if(e_type==ElementType::HEX) {
                bytes = sizeof(Eigen::Matrix<T,8,8>);
                petsc_set_element_matrix(eid,(*(Eigen::Matrix<T,8,8>*)eMat), i, mode);

            }else {
                return Error::UNKNOWN_ELEMENT_TYPE;
            }

            // move to next block (each block has size of bytes)
            eMat= (char*)eMat + bytes;
        }
        return Error::SUCCESS;
    } // set_element_matrices

    // used with set_element_matrices for the case of one eid but multiple matrices
    template <typename T,typename I>
    par::Error aMat<T,I>::petsc_set_element_matrix(unsigned int eid, EigenMat e_mat, unsigned int e_mat_id, InsertMode mode) {

        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        //unsigned int dof = m_uiNumDOFperNode;

        assert(e_mat.rows()==e_mat.cols());
        unsigned int num_rows = e_mat.rows(); // num_rows = num_nodes * dof

        // copy element matrix
        m_mats[eid * AMAT_MAX_EMAT_PER_ELEMENT + e_mat_id ] = e_mat;

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

        return Error::SUCCESS; // fixme
    } // petsc_set_element_matrix

}; // end of namespace par