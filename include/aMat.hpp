/**
 * @file aMat.hpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
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
#define AMAT_MAX_CRACK_LEVEL 1 // number of cracks allowed in 1 element
#define AMAT_MAX_EMAT_PER_ELEMENT (1u<<AMAT_MAX_CRACK_LEVEL)
#define AMAT_G2L_OFFSET 3

namespace par {
    enum class Error {SUCCESS,
                      INDEX_OUT_OF_BOUNDS,
                      UNKNOWN_ELEMENT_TYPE,
                      UNKNOWN_ELEMENT_STATUS,
                      NULL_L2G_MAP,
                      GHOST_NODE_NOT_FOUND
    };
    enum class ElementType {TET, HEX};

    class AsyncExchangeCtx {
        private :
            /** pointer to the variable which perform the ghost exchange */
            void* m_uiBuffer;

            /** pointer to the send buffer*/
            void* m_uiSendBuf;

            /** pointer to the receive buffer*/
            void* m_uiRecvBuf;

            /** list of request*/
            std::vector<MPI_Request*>m_uiRequests;
        public:
            /**@brief creates an async ghost exchange context*/
            AsyncExchangeCtx(const void* var)
            {
                m_uiBuffer=(void*)var;
                m_uiSendBuf=NULL;
                m_uiRecvBuf=NULL;
                m_uiRequests.clear();
            }

            /**@brief allocates send buffer for ghost exchange*/
            inline void allocateSendBuffer(size_t bytes)
            {
                m_uiSendBuf = malloc(bytes);
            }

            /**@brief allocates recv buffer for ghost exchange*/
            inline void allocateRecvBuffer(size_t bytes)
            {
                m_uiRecvBuf=malloc(bytes);
            }

            /**@brief allocates send buffer for ghost exchange*/
            inline void deAllocateSendBuffer()
            {
                free(m_uiSendBuf);
                m_uiSendBuf=NULL;
            }

            /**@brief allocates recv buffer for ghost exchange*/
            inline void deAllocateRecvBuffer()
            {
                free(m_uiRecvBuf);
                m_uiRecvBuf=NULL;
            }

            inline void* getSendBuffer() { return m_uiSendBuf;}
            inline void* getRecvBuffer() { return m_uiRecvBuf;}

            inline const void* getBuffer() {return m_uiBuffer;}

            inline std::vector<MPI_Request*>& getRequestList(){ return m_uiRequests;}

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
        /**@brief number of nodes owned by m_uiRank */
        unsigned int      m_uiNumNodes;

        /**@brief number of DOFs per node */
        unsigned int      m_uiNumDOFperNode;

        /**@brief number of globlal nodes */
        unsigned long     m_uiNumNodesGlobal;

        /**@brief communicator */
        MPI_Comm          m_comm;

        /**@brief my rank */
        unsigned int      m_uiRank;

        /**@brief total number of processes */
        unsigned int      m_uiSize;

        /**@brief PETSC global stiffness matrix */
        Mat                 m_pMat;

        /**@brief local-to-global map: m_ulpMap[eid][local_id]  = global_id */
        I**               m_ulpMap;

        /**@brief type of element list */
        par::ElementType*   m_pEtypes;

        /**@brief number of elements belong to m_uiRank */
        unsigned int      m_uiNumElems;

        /**@brief storage of element matrices */
        EigenMat* m_mats;

        /**@brief number of nodes owned by each rank */
        std::vector<unsigned int> m_uiLocalNodeCounts;

        /**@brief number of elements owned by each rank */
        std::vector<unsigned int> m_uiLocalElementCounts;

        // todo use I
        /**@brief exclusive scan of m_uiNumNodes */
        std::vector<unsigned int> m_uiLocalNodeScan;

        /**@brief exclusive scan of m_uiNumElems */
        std::vector<unsigned int> m_uiLocalElementScan;

        /**@brief number of ghost nodes owned by "pre" processes (whose ranks are smaller than m_uiRank) */
        unsigned int m_uiNumPreGhostNodes;

        /**@brief total number of ghost nodes owned by "post" processes (whose ranks are larger than m_uiRank) */
        unsigned int m_uiNumPostGhostNodes;

        /**@brief local node IDs to be sent (size = total number of nodes to be sent */
        std::vector<unsigned int> m_uiSendNodeIds;

        /**@brief number of nodes sent to each process (size = m_uiSize) */
        std::vector<unsigned int> m_uiSendNodeCounts;

        /**@brief offets (i.e. exclusive scan) of m_uiSendNodeCounts */
        std::vector<unsigned int> m_uiSendNodeOffset;

        /**@brief number of nodes to be received from each process (size = m_uiSize) */
        std::vector<unsigned int> m_uiRecvNodeCounts;

        /**@brief offsets (i.e. exclusive scan) of m_uiRecvNodeCounts */
        std::vector<unsigned int> m_uiRecvNodeOffset;

        /**@brief ghost exchange context*/
        std::vector<AsyncExchangeCtx> m_uiAsyncCtx;

        /**@brief MPI communication tag*/
        int m_uiCommTag;

        /**@brief local node-ID starting of pre-ghost nodes, always = 0 */
        unsigned int m_uiNodePreGhostBegin;

        /**@brief local node-ID ending of pre-ghost nodes */
        unsigned int m_uiNodePreGhostEnd;

        /**@brief local node-ID starting of nodes owned by me */
        unsigned int m_uiNodeLocalBegin;

        /**@brief local node-ID ending of nodes owned by me */
        unsigned int m_uiNodeLocalEnd;

        /**@brief local node-ID starting of post-ghost nodes */
        unsigned int m_uiNodePostGhostBegin;

        /**@brief local node-ID ending of post-ghost nodes */
        unsigned int m_uiNodePostGhostEnd;

        /**@brief total number of nodes including ghost nodes and nodes owned by me */
        unsigned int m_uiNumNodesTotal;

        /**@brief local map m_uiMap[eid][element_node]  = local node-ID */
        unsigned int** m_uiMap;


    protected:

        par::Error buildG2LMap();

    public:

        // todo buildScatterMap should be in protected
        /**
            * @brief: build scatter/gather map (used for communication) and local-to-local map (used for matvec)
            * */
        par::Error buildScatterMap();

        /**
            * @brief: number of nodes per element
            * @param[in] element type
            * @param[out] number of nodes
            * */
        static unsigned int nodes_per_element(par::ElementType etype) {
            switch (etype) {
                case par::ElementType::TET:
                    return 4;
                case par::ElementType::HEX:
                    return 8;
                default:
                    return (unsigned int)Error::UNKNOWN_ELEMENT_TYPE;
            }
        }

        /**
         * @brief
         * for now assume all element types have 3 dofs/node, could be changed depending on type
         * for now max of 2 cracks is allowed
         * */
        static unsigned int dofs_per_element(par::ElementType etype, unsigned int estatus) {
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

        /**
         * @brief: initialize variables of aMat
         * @param[in] n_local : number of local nodes
         * @param[in] dof : degree of freedoms per node
         * @param[in] comm: MPI communicator
         * */
        aMat(unsigned int nelem,par::ElementType* etype, unsigned int n_local, unsigned int dof, MPI_Comm comm);

        /**@brief de-constructor for aMat*/
        ~aMat();

        /**
         * @brief creates a petsc vector and initialize with a single scalar value
         * @param
         * */
        par::Error petsc_create_vec(Vec &vec, PetscScalar alpha = 0.0) const;

        /**
         * @brief: begin assembling the matrix, called after MatSetValues
         * @param[in] mode: petsc matrix assembly type
         */
        inline par::Error petsc_init_mat(MatAssemblyType mode) const {
            MatAssemblyBegin(m_pMat, mode);
            return Error::SUCCESS; //fixme
        }

        /**
         * @brief: begin assembling the petsc vec,
         * @param[in] vec: pestc vector
         */
        inline par::Error petsc_init_vec(Vec vec) const {
            VecAssemblyBegin(vec);
            return Error::SUCCESS; //fixme
        }

        /**
         * @brief: complete assembling the matrix, called before using the matrix
         * @param[in] mode: petsc matrix assembly type
         */
        par::Error petsc_finalize_mat(MatAssemblyType mode) const{
            MatAssemblyEnd(m_pMat,mode);
            return Error::SUCCESS; // fixme
        }

        /**
         * @brief: end assembling the petsc vec,
         * @param[in] vec: pestc vector
         * */
        par::Error petsc_finalize_vec(Vec vec) const{
            VecAssemblyEnd(vec);
            return Error::SUCCESS; // fixme
        }

        /**
         * @brief: create vector,
         * @param[in,out] vec: vector
         * @param[in] isGhosted: if true then size of vector includes ghost nodes
         * @param[in] alpha: initialied value for all members of vec
         * */
        par::Error create_vec(T* & vec, bool isGhosted = false, T alpha = (T)0);

        /**
         * @brief: create vector including ghost nodes,
         * @param[in] local: vector not including ghost nodes
         * @param[out] gVec: vector including ghost nodes
         * */
        par::Error local_to_ghost( T* & gVec, const T* local);

        /**
         * @brief: create vector not including ghost nodes,
         * @param[in] gVec: vector including ghost nodes
         * @param[out] local: vector not including ghost nodes
         * */
        par::Error ghost_to_local( T* & local, const T* gVec);

        /**
         * @brief: ghost nodes receive data from other processes, start
         * @param[in,out] vec: vector
         * */
        par::Error ghost_receive_begin(T* vec);

        /**
         * @brief: ghost nodes send data to other processes, start
         * @param[in,out] vec: vector
         * */
        par::Error ghost_send_begin(T* vec);

        /**
         * @brief: ghost nodes receive values from other processes, end
         * @param[in,out] vec: vector
         * */
        par::Error ghost_receive_end(T* vec);

        /**
         * @brief: ghost nodes send data to other processes, end
         * @param[in,out] vec: vector
         * */
        par::Error ghost_send_end(T* vec);

        /**
         * @brief: initial interface, twin is indicator whether the element is cracked
         * */
        par::Error set_element_matrix(unsigned int eid, T* e_mat, InsertMode mode=ADD_VALUES);

        /**
         * @brief: assembly global stiffness matrix
         * @param[in] eid : element ID
         * @param[in] e_mat : element stiffness matrix
         * @param[in] mode = ADD_VALUES : add to existing values of the matrix
         * */
        par::Error set_element_matrix(unsigned int eid, EigenMat e_mat, unsigned int e_mat_id, InsertMode mode=ADD_VALUES);

        /**
         * @brief: assembly global stiffness matrix crack elements with multiple levels
         * @param[in] eid : element ID
         * @param[in] e_mat : element stiffness matrices (pointer)
         * @param[in] twin_level: level of twinning (0 no crack, 1 one crack, 2 two cracks, 3 three cracks)
         * @param[in] mode = ADD_VALUES : add to existing values of the matrix
         * */
        par::Error set_element_matrices(unsigned int eid, EigenMat* e_mat, unsigned int twin_level, InsertMode mode=ADD_VALUES);

        /**
         * @brief: matrix free matvec
         * @param[in] u: global vector
         * @param[in] elem_matvec(T* ve, const T* ue, unsigned int eid): function to compute ve = Ke*ue for element eid
         * @param[out] v: global vector v = Ku
         * */
        par::Error matvec(T* v, T* u, void (*elem_matvec)(T *, const T *, unsigned int));

        /**
        * @brief: set mapping from element local node to global node
        * @param[in] map[eid][local_node_ID]
        * */
        par::Error set_map(I** map){
            m_ulpMap = map;
            return Error::SUCCESS; // fixme to have a specific error type for other cases
        }

        /**
         * @brief: assembly global load vector
         * @param[in/out] vec: petsc vector to assemble into
         * @param[in] eid : element ID
         * @param[in] e_mat : element load vector
         * @param[in] mode = ADD_VALUES : add to existing values of the matrix
         * */
        par::Error petsc_set_element_vector(Vec vec, unsigned int eid, T *e_vec, InsertMode mode = ADD_VALUES);

        /**
         * @brief: write pestsc matrix to ASCII file
         * @param[in/out] fmat: filename to write matrix to
         */
        par::Error dump_mat(const char* fmat) const;

        /**
         * @brief: write petsc vector to ASCII file
         * @param[in/out] fvec: filename to write vector to
         * @param[in] vec : petsc vector to write to file
         * */
        par::Error dump_vec(const char* fvec,Vec vec) const;

        /**
         * @brief apply Dirichlet boundary conditions to the matrix.
         * @param[in/out] rhs: the RHS vector (load vector)
         * @param[in] eid: element number
         * @param[in] dirichletBMap[eid][num_nodes]: indicator of boundary node (1) or interior node (0)
         * */
        par::Error apply_dirichlet(Vec rhs,unsigned int eid,const I** dirichletBMap);

        /**
         * @brief: invoke basic petsc solver
         * @param[in] rhs: petsc RHS vector
         * @param[out] out: petsc solution vector
         * */
        par::Error petsc_solve(const Vec rhs,Vec out) const;

        /**
         * @brief: set Pestc vector
         * @param[in/out] exact_sol: petsc exact solution vector
         * @param[in] eid: element id
         * @param[in] e_sol: elemet exact solution
         * @param[in] mode: petsc insert or add modes
         * */
        par::Error petsc_set_vector(Vec exact_sol, unsigned int eid, T *e_sol, InsertMode mode = INSERT_VALUES) const;

        /**
         * @brief: pestc matrix-vector product y = Ax, where A is petsc global stiffness matrix m_pMat
         */
        par::Error petsc_matmult(Vec x, Vec y);

        /**
         * @brief: display all components of vector on screen (for purpose of debugging)
         */
        par::Error print_vector(const T* vec);

        /**
         * @brief: apply zero Dirichlet boundary condition on nodes dictated by dirichletBMap
         */
        par::Error set_vector_bc(T *vec, unsigned int eid, const I **dirichletBMap);


        /**
         * @brief: transform vec to pestc vector (for purpose of debugging)
         */
        par::Error transform_to_petsc_vector(const T* vec, Vec petsc_vec);

    }; // end of class aMat

    //******************************************************************************************************************

    // constructor
    template <typename T,typename I>
    aMat<T,I>::aMat(unsigned int nelem,par::ElementType* etype, unsigned int n_local, unsigned int dof, MPI_Comm comm) {

        m_comm = comm;

        MPI_Comm_rank(comm, (int*)&m_uiRank);
        MPI_Comm_size(comm, (int*)&m_uiSize);

        m_uiNumNodes = n_local;
        m_uiNumDOFperNode = dof;

        unsigned long nl = m_uiNumNodes;

        MPI_Allreduce(&nl, &m_uiNumNodesGlobal, 1, MPI_LONG, MPI_SUM, m_comm);

        MatCreate(m_comm, &m_pMat);
        MatSetSizes(m_pMat, m_uiNumNodes*m_uiNumDOFperNode, m_uiNumNodes*m_uiNumDOFperNode, PETSC_DECIDE, PETSC_DECIDE);

        if(m_uiSize > 1) {
            // initialize matrix
            MatSetType(m_pMat, MATMPIAIJ);
            MatMPIAIJSetPreallocation(m_pMat, 30*m_uiNumDOFperNode , PETSC_NULL, 30*m_uiNumDOFperNode , PETSC_NULL);

        }else {
            MatSetType(m_pMat, MATSEQAIJ);
            MatSeqAIJSetPreallocation(m_pMat, 30*m_uiNumDOFperNode, PETSC_NULL);


        }
        // this will disable on preallocation errors. (but not good for performance)
        //MatSetOption(m_pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

        // number of elements per rank
        m_uiNumElems = nelem;

        // etype[eid] = type of element eid
        m_pEtypes = etype;

        // copy of element matrices, each element is allocated totally AMAT_MAX_EMAT_PER_ELEMENT of EigenMat
        m_mats = new EigenMat[m_uiNumElems * AMAT_MAX_EMAT_PER_ELEMENT];

        m_ulpMap = NULL;

        m_uiMap = new unsigned int*[m_uiNumElems];
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            m_uiMap[eid] = new unsigned int[nodes_per_element(m_pEtypes[eid])];
        }

    } // constructor



    // destructor
    template <typename T,typename I>
    aMat<T,I>::~aMat()
    {
        delete [] m_mats;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            delete [] m_uiMap[eid]; //delete array
        }
        delete [] m_uiMap;
    } // ~aMat



    // build scatter map
    template <typename T,typename I>
    par::Error aMat<T,I>::buildScatterMap() {
        /*
        * Assumptions
        * 1). We assume that the global nodes are continuously partitioned across processors.
        * currently we do not account for twin elements
        * */

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
        //std::cout<<"local nodes: "<<m_uiNumNodes<<" m_uiSize: "<<m_uiSize<<std::endl;
        MPI_Allgather(&m_uiNumNodes, 1, MPI_INT, &(*(m_uiLocalNodeCounts.begin())), 1, MPI_INT, m_comm);
        MPI_Allgather(&m_uiNumElems, 1, MPI_INT, &(*(m_uiLocalElementCounts.begin())), 1, MPI_INT, m_comm);

        // scan local counts
        m_uiLocalNodeScan[0] = 0;
        m_uiLocalElementScan[0] = 0;

        for (unsigned int p = 1; p < m_uiSize; p++) {
            m_uiLocalNodeScan[p] = m_uiLocalNodeScan[p-1] + m_uiLocalNodeCounts[p-1];
            m_uiLocalElementScan[p] = m_uiLocalElementScan[p-1] + m_uiLocalElementCounts[p-1];
        }

        std::vector<I> preGhostGIds;
        std::vector<I> postGhostGIds;

        // nodes are not owned by me: stored in pre or post lists
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

        /*for (unsigned int i = 0; i < preGhostGIds.size(); i++){
            printf("after erase, rank= %d preGhostGIds= %d\n", m_uiRank, preGhostGIds[i]);
        }
        for (unsigned int i = 0; i < postGhostGIds.size(); i++){
            printf("after erase, rank= %d postGhostGIds= %d\n", m_uiRank, postGhostGIds[i]);
        }*/

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
        for(unsigned int i=1; i<m_uiSize; i++)
        {
            sendOffset[i] = sendOffset[i-1] + sendCounts[i-1];
            recvOffset[i] = recvOffset[i-1] + recvCounts[i-1];
        }

        /*for (unsigned i = 0; i < m_uiSize; i++) {
            printf("rank %d, sendCounts[%d] = %d\n", m_uiRank, i, sendCounts[i]);
        }
        for (unsigned i = 0; i < m_uiSize; i++) {
            printf("rank %d, sendOffset[%d] = %d\n", m_uiRank, i, sendOffset[i]);
        }
        for (unsigned i = 0; i < m_uiSize; i++) {
            printf("rank %d, recvCounts[%d] = %d\n", m_uiRank, i, recvCounts[i]);
        }
        for (unsigned i = 0; i < m_uiSize; i++) {
            printf("rank %d, recvOffset[%d] = %d\n", m_uiRank, i, recvOffset[i]);
        }*/

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

        /*for (unsigned i = 0; i < sendBuf.size(); i++) {
            printf("rank %d, sendBuf[%d] = %d\n", m_uiRank, i, sendBuf[i]);
        }*/

        for(unsigned int i = 0; i < m_uiSize; i++) {
            sendCounts[i] *= sizeof(I);
            sendOffset[i] *= sizeof(I);
            recvCounts[i] *= sizeof(I);
            recvOffset[i] *= sizeof(I);
        }

        MPI_Alltoallv(&(*(sendBuf.begin())), (int*)sendCounts, (int*)sendOffset, MPI_BYTE,
                      &(*(recvBuf.begin())), (int*)recvCounts, (int*)recvOffset, MPI_BYTE, m_comm);

        /*for (unsigned i = 0; i < recvBuf.size(); i++) {
            printf("rank %d, recvBuf[%d] = %d\n", m_uiRank, i, recvBuf[i]);
        }*/

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

        for(unsigned int i = 0; i < m_uiSize; i++) {
            m_uiSendNodeCounts[i] = recvCounts[i];
            m_uiSendNodeOffset[i] = recvOffset[i];
            m_uiRecvNodeCounts[i] = sendCounts[i];
            m_uiRecvNodeOffset[i] = sendOffset[i];
        }

        /*for (unsigned i = 0; i < m_uiSize; i++){
            printf("rank %d, m_uiLocalNodeCounts[%d] = %d, m_uiLocalNodeScan[%d] = %d\n",m_uiRank,i,m_uiLocalNodeCounts[i],i,m_uiLocalNodeScan[i]);
        }
        for (unsigned i = 0; i < m_uiSize; i++){
            printf("rank %d, m_uiLocalElementCounts[%d] = %d, m_uiLocalElementScan[%d] = %d\n",m_uiRank,i,m_uiLocalElementCounts[i],i,m_uiLocalElementScan[i]);
        }*/

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
        /*for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            par::ElementType e_type = m_pEtypes[eid];
            unsigned int num_nodes = aMat::nodes_per_element(e_type);
            for (unsigned int i = 0; i < num_nodes; i++){
                printf("rank= %d, element= %d, m_ulpMap[%d]= %d, m_uiMap[%d]= %d \n", m_uiRank, eid,i, m_ulpMap[eid][i], i, m_uiMap[eid][i]);
            }
        }*/

        /*for(unsigned int i = 0; i < m_uiSize; i++)
        {
            for(unsigned int node=m_uiSendNodeOffset[i];node< (m_uiSendNodeOffset[i]+ m_uiSendNodeCounts[i]);node++)
                printf("m_uiRank: %d  needs to send  local id : %d global id : %d  to proc: %d \n",m_uiRank,m_uiSendNodeIds[node],recvBuf[node],i);
        }*/

        /*for (unsigned i = 0; i < m_uiSize; i++){
            printf("rank %d, m_uiSendNodeCounts[%d] = %d, m_uiSendNodeOffset[%d] = %d\n",m_uiRank,i,m_uiSendNodeCounts[i],i,m_uiSendNodeOffset[i]);
        }
        for (unsigned i = 0; i < m_uiSize; i++){
            printf("rank %d, m_uiRecvNodeCounts[%d] = %d, m_uiRecvNodeOffset[%d] = %d\n", m_uiRank,i,m_uiRecvNodeCounts[i],i,m_uiRecvNodeOffset[i]);
        }
        for (unsigned i = 0; i < m_uiSendNodeIds.size(); i++){
            printf("rank %d, m_uiSendNodeIds[%d] = %d\n",m_uiRank,i,m_uiSendNodeIds[i]);
        }
        printf("rank %d, m_uiNumPreGhostNodes = %d, m_uniPostGhostNodes = %d\n",m_uiRank,m_uiNumPreGhostNodes,m_uiNumPostGhostNodes);*/

        delete [] sendCounts;
        delete [] recvCounts;
        delete [] sendOffset;
        delete [] recvOffset;
        return Error::SUCCESS;
    } // buildScatterMap



    // create pestsc vector
    template <typename T,typename I>
    par::Error aMat<T,I>::petsc_create_vec(Vec &vec, PetscScalar alpha) const
    // the & here because we will allocate memory for vec, and Vec is a pointer by Petsc,
    // this means that we want to modify the address (not the value) of vec
    {
        // initialize rhs vector
        VecCreate(m_comm, &vec);
        if(m_uiSize>1)
        {
            VecSetType(vec,VECMPI);
            VecSetSizes(vec, m_uiNumNodes * m_uiNumDOFperNode, PETSC_DECIDE);
            VecSet(vec, alpha);
        }else {
            VecSetType(vec,VECSEQ);
            VecSetSizes(vec, m_uiNumNodes * m_uiNumDOFperNode, PETSC_DECIDE);
            VecSet(vec, alpha);
        }
        return Error::SUCCESS; // fixme
    } // petsc_create_vec



    // allocate memory for structure vector, if isGhosted then include space for ghost nodes
    template <typename T, typename I>
    par::Error aMat<T,I>::create_vec(T* &vec, bool isGhosted, T alpha){
        if (isGhosted){
            vec = new T[m_uiNumNodesTotal * m_uiNumDOFperNode];
        } else {
            vec = new T[m_uiNumNodes * m_uiNumDOFperNode];
        }
        // initialize
        if (isGhosted) {
            for (unsigned int i = 0; i < m_uiNumNodesTotal * m_uiNumDOFperNode; i++){
                vec[i] = alpha;
            }
        } else {
            for (unsigned int i = 0; i < m_uiNumNodes * m_uiNumDOFperNode; i++){
                vec[i] = alpha;
            }
        }
        return Error::SUCCESS;
    } // create_vec



    // transform structure vector to include ghost nodes: NOT count for DOF yet
    template <typename T, typename I>
    par::Error aMat<T,I>::local_to_ghost( T* & gVec, const T* local){
            gVec = new T[m_uiNumNodesTotal];
        for (unsigned int i = 0; i < m_uiNumNodesTotal; i++){
            if ((i >= m_uiNumPreGhostNodes) && (i < m_uiNumPreGhostNodes + m_uiNumNodes)) {
                gVec[i] = local[i - m_uiNumPreGhostNodes];
            } else {
                gVec[i] = 0.0;
            }
        }
        return Error::SUCCESS;
    } // local_to_ghost



    // transform structure vector to include only local nodes: NOT count for DOF yet
    template <typename T, typename I>
    par::Error aMat<T,I>::ghost_to_local(T* & local, const T *gVec) {
        local = new T[m_uiNumNodes];
        for (unsigned int i = 0; i < m_uiNumNodes; i++){
            local[i] = gVec[i + m_uiNumPreGhostNodes];
        }
        return Error::SUCCESS;
    } // ghost_to_local



    // send and receive before matvec: ranks who own nodes send data to ranks who have ghost nodes
    template <typename T, typename I>
    par::Error aMat<T,I>::ghost_receive_begin(T* vec) {
        AsyncExchangeCtx ctx((const void*)vec);

        // total number of nodes to be sent (not dof yet)
        const unsigned  int total_send = m_uiSendNodeOffset[m_uiSize-1] + m_uiSendNodeCounts[m_uiSize-1];

        // total number of nodes to be received (not dof yet)
        const unsigned  int total_recv = m_uiRecvNodeOffset[m_uiSize-1] + m_uiRecvNodeCounts[m_uiSize-1];

        if (total_send > 0){
            ctx.allocateSendBuffer(sizeof(T) * total_send);
            T* send_buf = (T*) ctx.getSendBuffer();
            for (unsigned int i = 0; i < total_send; i++){
                send_buf[i] = vec[m_uiSendNodeIds[i]];
            }
            for (unsigned int i = 0; i < m_uiSize; i++){
                if (m_uiSendNodeCounts[i] == 0) continue;
                MPI_Request* req = new MPI_Request();
                MPI_Isend(&send_buf[m_uiSendNodeOffset[i]], m_uiSendNodeCounts[i] * sizeof(T), MPI_BYTE, i, m_uiCommTag, m_comm, req);
                ctx.getRequestList().push_back(req);
            }
        }
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



    // send and receive after matvec: ranks who have ghost nodes send back data to ranks who own nodes
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
        for(unsigned int i=0;i<num_req;i++)
        {
            MPI_Wait(ctx.getRequestList()[i],&sts);
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



    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrix(unsigned int eid, T* e_mat, InsertMode mode){

        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        unsigned int dof = m_uiNumDOFperNode;

        // now set values ...
        std::vector<PetscScalar> values(num_nodes*dof);
        std::vector<PetscInt> colIndices(num_nodes*dof);
        PetscInt rowId;

        unsigned int index = 0;
        for (unsigned int r = 0; r < num_nodes*dof; ++r) {
            rowId = dof*m_ulpMap[eid][r/dof] + r%dof;
            for (unsigned int c = 0; c < num_nodes*dof; ++c) {
                colIndices[c] = dof*m_ulpMap[eid][c/dof] + c%dof;
                values[c] = e_mat[index];
                index++;
                //std::cout<<" col: "<<colIndices[c]<<std::endl;
            } // c
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
            // values.clear();
            // colIndices.clear();
        } // r

        return Error::SUCCESS; // fixme
    } // set_element_matrix



    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrix(unsigned int eid, EigenMat e_mat, unsigned int e_mat_id, InsertMode mode) {

        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        unsigned int dof = m_uiNumDOFperNode;

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
            rowId = dof*m_ulpMap[eid][e_mat_id * num_nodes + r/dof] + r%dof; // map in terms of nodes, multiple matrices per eid
            for (unsigned int c = 0; c < num_rows; ++c) {
                //colIndices[c] = m_ulpMap[eid][c];
                //colIndices[c] = dof*m_ulpMap[eid][c/dof] + c%dof;
                colIndices[c] = dof*m_ulpMap[eid][e_mat_id * num_nodes + c/dof] + c%dof;
                values[c] = e_mat(r,c);
            } // c
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
        } // r

        return Error::SUCCESS; // fixme
    } // set_element_matrix



    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrices(unsigned int eid, EigenMat* e_mat, unsigned int twin_level, InsertMode mode) {

        par::ElementType e_type = m_pEtypes[eid];

        // number of twinning matrices (e.g. twin_level = 2 then numEMat = 4)
        unsigned int numEMat=(1u<<twin_level);

        // since e_mat is dynamic, then first casting it to void* so that we can move for each element of e_mat
        void* eMat= (void*)e_mat;
        for(unsigned int i=0; i<numEMat; i++)
        {
            size_t bytes=0;
            if(e_type==ElementType::TET)
            {
                 bytes = sizeof(Eigen::Matrix<T,4,4>);
                 set_element_matrix(eid,(*(Eigen::Matrix<T,4,4>*)eMat), i, mode);

            }else if(e_type==ElementType::HEX)
            {
                bytes = sizeof(Eigen::Matrix<T,8,8>);
                set_element_matrix(eid,(*(Eigen::Matrix<T,8,8>*)eMat), i, mode);

            }else {
                return Error::UNKNOWN_ELEMENT_TYPE;
            }

            // move to next block (each block has size of bytes)
            eMat= (char*)eMat + bytes;
        }
        return Error::SUCCESS; // fixme
    } // set_element_matrices



    // compute v = Ku
    // elem_matvec(T* ve, T* ue, unsigned int eid) is to compute element ve = ke * ue for element eid
    template <typename  T, typename I>
    par::Error aMat<T,I>::matvec(T* v, T* u, void (*elem_matvec)(T *, const T *, unsigned int)){

        par::ElementType e_type;
        unsigned int num_nodes;
        T* ue;
        T* ve;
        unsigned int dof = m_uiNumDOFperNode;

        // initialize v (size of v = m_uiNodesPostGhostEnd = m_uiNumNodesTotal)
        for (unsigned int i = 0; i < m_uiNodePostGhostEnd; i++){
            v[i] = 0.0;
        }

        // find max of number of nodes per element
        unsigned int max_dpe = 0;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++) {
            e_type = m_pEtypes[eid];
            num_nodes = aMat::nodes_per_element(e_type);
            if (max_dpe < num_nodes) max_dpe = num_nodes;
        }
        max_dpe = max_dpe * dof;

        ue = new T[max_dpe];
        ve = new T[max_dpe];

        I rowID;

        // send data from owned nodes to ghost nodes (of other processors) to get ready for computing v = Ku
        ghost_receive_begin(u);
        ghost_receive_end(u);

        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            e_type = m_pEtypes[eid];
            num_nodes = aMat::nodes_per_element(e_type);

            // extract element vector ue from structure vector u
            for (unsigned int r = 0; r < num_nodes * dof; ++r) {
                rowID = dof * m_uiMap[eid][r/dof] + r % dof;
                ue[r] = u[rowID];
            }

            // compute ve = ke * ue
            elem_matvec(ve, (const T*) ue, eid);

            // accumulate element vector ve to structure vector v
            for (unsigned int r = 0; r < num_nodes * dof; r++){
                rowID = dof * m_uiMap[eid][r/dof] + r % dof;
                v[rowID] += ve[r];
            }
        }

        // send data from ghost nodes back to owned nodes after computing v
        ghost_send_begin(v);
        ghost_send_end(v);

        delete [] ue;
        delete [] ve;
        return Error::SUCCESS; // fixme
    } // matvec


    // print out vec whose size including ghost nodes
    template <typename T, typename I>
    par::Error aMat<T,I>::print_vector(const T* vec){
        for (unsigned int i = 0; i < m_uiNumNodesTotal; i++){
            if (i >= m_uiNumPreGhostNodes && i < m_uiNumPreGhostNodes + m_uiNumNodes) {
                printf("rank= %d, v[%d]= %10.3f \n", m_uiRank, i-m_uiNumPreGhostNodes, vec[i]);
            }
        }
        return Error::SUCCESS;
    } // print_vector


    // transform vec to petsc vector for comparison between matrix-free and matrix-based methods
    template <typename T, typename I>
    par::Error aMat<T,I>::transform_to_petsc_vector(const T* vec, Vec petsc_vec) {
        PetscScalar value;
        for (unsigned int eid = 0; eid < m_uiNumElems; eid++){
            par::ElementType e_type = m_pEtypes[eid];
            unsigned int num_nodes = aMat::nodes_per_element(e_type);
            unsigned int dof = m_uiNumDOFperNode;

            for (unsigned int i = 0; i < num_nodes * dof; i++){
                const unsigned int nidG = m_ulpMap[eid][i/dof]; // global node
                unsigned int nidL = m_uiMap[eid][i];  // local node
                if (nidL >= m_uiNumPreGhostNodes && nidL < m_uiNumPreGhostNodes + m_uiNumNodes) {
                    // nidL is owned by me
                    value = vec[nidL];
                    //printf("rank= %d, eid= %d, i= %d, nidG= %d, nidL= %d\n",m_uiRank, eid, i, nidG, nidL);
                    VecSetValue(petsc_vec, nidG, value, INSERT_VALUES);
                }
            }
        }
        return Error::SUCCESS;
    } // transform_to_petsc_vector


    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_set_element_vector(Vec vec, unsigned int eid, T *e_vec, InsertMode mode){

        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        unsigned int dof = m_uiNumDOFperNode;

        PetscScalar value;
        PetscInt rowId;

        unsigned int index = 0;
        for (unsigned int r = 0; r < num_nodes*dof; ++r) {
            rowId = dof * m_ulpMap[eid][r/dof] + r % dof;
            value = e_vec[index];
            index++;
            VecSetValue(vec, rowId, value, mode);
        }

        return Error::SUCCESS; // fixme
    } // petsc_set_element_vector



    template <typename T, typename I>
    par::Error aMat<T,I>::apply_dirichlet(Vec rhs, unsigned int eid, const I** dirichletBMap) {
        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        unsigned int dof = m_uiNumDOFperNode;

        PetscInt rowId, colId, boundrow, boundcol;

        for (unsigned int r = 0; r < num_nodes*dof; r++) {
            rowId = dof * m_ulpMap[eid][r/dof] + r % dof;
            boundrow = dirichletBMap[eid][r/dof]; // if a node is boundary, all dof's of this node are set to 0
            if (boundrow == 1) {
                VecSetValue(rhs, rowId, 0.0, INSERT_VALUES);
                for (unsigned int c = 0; c < num_nodes*dof; c++) {
                    colId = dof * m_ulpMap[eid][c/dof] + c % dof; //fixme: check for cases of dof > 1
                    if (colId == rowId) {
                        MatSetValue(m_pMat, rowId, colId, 1.0, INSERT_VALUES);
                    } else {
                        MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                    }
                }
            } else {
                for (unsigned int c = 0; c < num_nodes*dof; c++) {
                    colId = dof * m_ulpMap[eid][c/dof] + c % dof;
                    boundcol =  dirichletBMap[eid][c/dof];
                    if (boundcol == 1) {
                        MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                    }
                }
            }
        }
        return Error::SUCCESS;
    } // apply_dirichlet



    template <typename T, typename I>
    par::Error aMat<T,I>::dump_mat(const char* fmat) const {

        // write matrix to file
        PetscViewer viewer;
        PetscViewerASCIIOpen(m_comm, fmat, &viewer);
        MatView(m_pMat,viewer);
        PetscViewerDestroy(&viewer);

        return Error::SUCCESS; // fixme
    } // dump_mat



    template <typename T, typename I>
    par::Error aMat<T,I>::dump_vec(const char* fvec, Vec vec) const
    {
        PetscViewer viewer;
        PetscViewerASCIIOpen(m_comm, fvec, &viewer);
        VecView(vec,viewer);
        PetscViewerDestroy(&viewer);

        return Error::SUCCESS; // fixme
    } // dump_vec



    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_solve(const Vec rhs, Vec out) const {

        KSP ksp;     /* linear solver context */
        PC  pc;      /* pre conditioner context */

        KSPCreate(m_comm,&ksp);
        //PCCreate(m_comm,&pc);
        KSPSetOperators(ksp, m_pMat, m_pMat);
        //PCSetFromOptions(pc);
        //KSPSetPC(ksp,pc);
        KSPSetFromOptions(ksp);
        KSPSolve(ksp,rhs,out);

        return Error::SUCCESS; // fixme
    } // petsc_solve



    template <typename T, typename I>
    par::Error aMat<T,I>:: petsc_set_vector(Vec exact_sol, unsigned int eid, T *e_sol, InsertMode mode) const{
        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        unsigned int dof = m_uiNumDOFperNode;

        PetscScalar value;
        PetscInt rowId;

        for (unsigned int r = 0; r < num_nodes*dof; ++r) {
            rowId = dof * m_ulpMap[eid][r/dof] + r % dof;
            value = e_sol[r];
            VecSetValue(exact_sol, rowId, value, mode);
        }

        return Error::SUCCESS; // fixme
    } // petsc_set_vector



    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_matmult(Vec x, Vec y){
        MatMult(m_pMat, x, y);
        return Error::SUCCESS;
    } // petsc_matmult



    // apply Dirichlet boundary conditions
    template <typename T, typename I>
    par::Error aMat<T,I>::set_vector_bc(T *vec, unsigned int eid, const I **dirichletBMap){
        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        unsigned int dof = m_uiNumDOFperNode;
        unsigned int rowId, boundrow;
        for (unsigned int r = 0; r < num_nodes * dof; r++){
            rowId = dof * m_uiMap[eid][r/dof] + r % dof;
            boundrow = dirichletBMap[eid][r/dof];
            if (boundrow == 1) {
                // boundary node
                vec[rowId] = 0.0;
            }
        }
        return Error::SUCCESS;
    }
}; // end of namespace par
