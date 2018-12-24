/**
 * @file aMat.h
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

namespace par {

    enum class Error {SUCCESS, INDEX_OUT_OF_BOUNDS, UNKNOWN_ELEMENT_TYPE};
    enum class ElementType {TET, HEX, TET_TWIN, HEX_TWIN};

    template <typename T,typename I>
    class aMat {

        // what information do we need from the mesh ...
        // all we need is,
        // (eid, el_local_node_id) -> global_node_id
        // we need
        //
        // number of elements x {
        //                        element_type => tet, hex, tet_twin, hex_twin
        //                        [local_node_id] -> global_node_id
        //                      }
        //
        //  eg: Initial mesh with a single element
        //           elist_main = [e_10, type=quad]
        //           elist_sec  = []
        //           twin_element(10);
        //           elist_main = [e_10, type=quad]   -> v1
        //           elist_sec  = [e_10, type=quad]   -> v2
        //           merge_elements()
        //           elist_main = [e_10, type=quad_twin]
        //           elist_sec = []

    protected:
        /**@brief number of local nodes */
        unsigned int      m_uiNumNodes;
        /**@brief number of DOFs per node */
        unsigned int      m_uiNumDOFperNode;
        /**@brief number of globlal nodes */
        unsigned long     m_uiNumNodesGlobal;

        /**@brief */
        MPI_Comm          m_comm;
        /**@brief */
        unsigned int      m_uiRank;
        /**@brief */
        unsigned int      m_uiSize;

        /**@brief PETSC matrix */
        Mat                 m_pMat;
        /**@brief map[e][local_node]  = global_node */
        I**               m_ulpMap;
        par::ElementType*   m_pEtypes;

    public:
        /**
         * @brief:
         * @param[in] n_local:
         * @param[in] dof: degrees of freedoms
         * @param[in] comm: MPI communicator
         * */
        aMat(const unsigned int n_local, const unsigned int dof, MPI_Comm comm);

        /**
         * @brief:
         * @param[in]
         * */
        par::Error set_element_matrix(unsigned int eid, T* e_mat, bool twin, InsertMode mode=ADD_VALUES);

        /**
         * @brief: changing size of matrix
         * @param[in]
         * */
        par::Error twin_element(unsigned int e);

        /**
        * @brief: set block matrix
        * @param[in]
        * */
        par::Error set_block(I index, T* block_matrix, unsigned int sz, unsigned int dof);

        /**
        * @brief: merge twined elements to single
        * @param[in]
        * */
        par::Error merge_elements();

        /**
         * @brief:
         * @param[in]
         * */
        par::Error matvec(std::vector<T>& v_primary, std::vector<T>& v_twin, std::vector<T>& w_primary, std::vector<T>& w_twin);

        static unsigned int nodes_per_element(par::ElementType etype) {
            switch (etype) {
                case par::ElementType::TET: return 4; break;
                case par::ElementType::HEX: return 8; break;
                case par::ElementType::TET_TWIN: return 4; break;
                case par::ElementType::HEX_TWIN: return 8; break;
                default:
                    return (unsigned int)Error::UNKNOWN_ELEMENT_TYPE;
            }
            return (unsigned int)Error::UNKNOWN_ELEMENT_TYPE;
        }

        void set_map(I** map){
            m_ulpMap = map;

        }

    }; // end of class aMat


    template <typename T,typename I>
    aMat<T,I>::aMat(const unsigned int n_local,const unsigned int dof, MPI_Comm comm) {

        PetscBool isAij=PETSC_FALSE, isAijSeq=PETSC_FALSE, isAijPrl=PETSC_TRUE, isSuperLU, isSuperLU_Dist;
        isSuperLU = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU,&isSuperLU);
        isSuperLU_Dist = PETSC_FALSE; // PetscStrcmp(mtype,MATSUPERLU_DIST,&isSuperLU_Dist);

        m_comm = comm;


        MPI_Comm_rank(comm, (int*)&m_uiRank);
        MPI_Comm_size(comm, (int*)&m_uiSize);

        m_uiNumNodes = n_local;
        m_uiNumDOFperNode = dof;

        unsigned long nl = m_uiNumNodes;

        MPI_Allreduce(&nl, &m_uiNumNodesGlobal, 1, MPI_LONG, MPI_SUM, m_comm);

        // initialize matrix
        MatCreate(m_comm, &m_pMat);
        MatSetSizes(m_pMat, m_uiNumNodes*m_uiNumDOFperNode, m_uiNumNodes*m_uiNumDOFperNode, PETSC_DECIDE, PETSC_DECIDE);
        MatSetType(m_pMat, MATMPIAIJ);

        if(m_uiSize > 1) {
            MatMPIAIJSetPreallocation(m_pMat, 26*m_uiNumDOFperNode , PETSC_NULL, 26*m_uiNumDOFperNode , PETSC_NULL);
        }else {
            MatSeqAIJSetPreallocation(m_pMat, 26*m_uiNumDOFperNode , PETSC_NULL);
        }
    }


    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrix(unsigned int eid, T* e_mat, bool twin, InsertMode mode) {

        par::ElementType e_type = m_pEtypes[eid];

        unsigned int num_nodes = aMat::nodes_per_element(e_type);

        unsigned int dof = m_uiNumDOFperNode;

        // now set values ...
        std::vector<PetscScalar> values(num_nodes*dof);
        std::vector<PetscInt> colIndices(num_nodes*dof);
        PetscInt rowId;

        unsigned int index = 0;
        for (unsigned int r = 0; r < num_nodes*dof; ++r) {
            rowId = m_ulpMap[eid][r/dof];
            for (unsigned int c = 0; c < num_nodes*dof; ++c) {
                colIndices[c] = m_ulpMap[eid][c/dof];
                values[c] = e_mat[index];
                index++;

            } // c
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
            // values.clear();
            // colIndices.clear();
        } // r
        return Error::SUCCESS;
    }

}; // end of namespace par
