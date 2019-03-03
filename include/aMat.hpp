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





namespace par {

    enum class Error {SUCCESS, INDEX_OUT_OF_BOUNDS, UNKNOWN_ELEMENT_TYPE, UNKNOWN_ELEMENT_STATUS};
    enum class ElementType {TET, HEX, TET_TWIN, HEX_TWIN};

    template <typename T,typename I>
    class aMat {

    typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> EigenMat;

    protected:
        /**@brief number of local nodes */
        unsigned int      m_uiNumNodes;

        /**@brief number of DOFs per node */
        unsigned int      m_uiNumDOFperNode;

        /**@brief number of globlal nodes */
        unsigned long     m_uiNumNodesGlobal;

        /**@brief communicator */
        MPI_Comm          m_comm;

        /**@brief process ID*/
        unsigned int      m_uiRank;

        /**@brief number of processes */
        unsigned int      m_uiSize;

        /**@brief PETSC matrix */
        Mat                 m_pMat;

        /**@brief map[e][local_node]  = global_node */
        I**               m_ulpMap;

        /**@brief type of element list */
        par::ElementType*   m_pEtypes;

        /**@brief number of elements */
        unsigned int      m_uiNumElem;

        /**@brief storage of element matrices */
        EigenMat* m_mats;

        /**@brief storage of element status */
        unsigned int* m_status;


    public:

        /**
            * @brief: number of nodes per element
            * @param[in] element type
            * @param[out] number of nodes
            * */
        static unsigned int nodes_per_element(par::ElementType etype) {
            switch (etype) {
                case par::ElementType::TET: return 4; break;
                case par::ElementType::HEX: return 8; break;
                case par::ElementType::TET_TWIN: return 4; break;
                case par::ElementType::HEX_TWIN: return 8; break;
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
                    break;
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
                    break;
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
        aMat(const unsigned int n_local, const unsigned int dof, MPI_Comm comm);


        /**
         * @brief creates a petsc vector
         * @param
         * */
        par::Error create_vec(Vec& vec);

        /**
         * @brief: begin assembling the matrix, called after MatSetValues
         * @param[in] mode: petsc matrix assembly type
         */
        inline par::Error petsc_init_mat(MatAssemblyType mode){
            MatAssemblyBegin(m_pMat,mode);
            return Error::SUCCESS; //fixme
        }

        /**
         * @brief: begin assembling the petsc vec,
         * @param[in] vec: pestc vector
         */
        inline par::Error petsc_init_vec(Vec vec){
            VecAssemblyBegin(vec);
            return Error::SUCCESS; //fixme
        }


        /**
         * @brief: complete assembling the matrix, called before using the matrix
         * @param[in] mode: petsc matrix assembly type
         */
        par::Error petsc_finalize_mat(MatAssemblyType mode){
            MatAssemblyEnd(m_pMat,mode);
            return Error::SUCCESS; // fixme
        }
        /**
         * @brief: end assembling the petsc vec,
         * @param[in] vec: pestc vector
         * */
        par::Error petsc_finalize_vec(Vec vec){
            VecAssemblyEnd(vec);
            return Error::SUCCESS; // fixme
        }


        /**
         * @brief: intial interface, twin is indicator whether the element is cracked
         * */
        par::Error set_element_matrix(unsigned int eid, T* e_mat, bool twin, InsertMode mode=ADD_VALUES);

        /**
         * @brief: assembly global stiffness matrix
         * @param[in] element_status : 0 no crack, 1 cracked at level 1, 2 cracked at level 2, 3 cracked at level 3
         * */
        par::Error set_element_matrix(unsigned int eid, EigenMat* e_mat, unsigned int element_status, InsertMode mode=ADD_VALUES);


        /**
         * @brief: assembly global stiffness matrix
         * @param[in] eid : element ID
         * @param[in] e_mat : element stiffness matrix
         * @param[in] mode = ADD_VALUES : add to existing values of the matrix
         * */
        par::Error set_element_matrix(unsigned int eid, EigenMat* e_mat, InsertMode mode=ADD_VALUES);


        /**
         * @brief: assembly global stiffness matrix crack elements with multiple levels
         * @param[in] eid : element ID
         * @param[in] e_mat : element stiffness matrices (pointer)
         * @param[in] twin_level: level of twinning (0 no crack, 1 one crack, 2 two cracks, 3 three cracks)
         * @param[in] mode = ADD_VALUES : add to existing values of the matrix
         * */
        par::Error set_element_matrices(unsigned int eid, EigenMat* e_mat, unsigned int twin_level, InsertMode mode=ADD_VALUES);


        /**
         * @brief: matrix vector multiplication
         * @param[in] u : nodal displacements
         * @param[out] v : [v] = [K][u]
         * */
        par::Error matvec(std::vector<T>& v_primary, std::vector<T>& v_twin, std::vector<T>& w_primary, std::vector<T>& w_twin);


        /**
        * @brief: set mapping from element local node to global node
        * @param[in] map[eid][local_node_ID]
        * */
        par::Error set_map(I** map){
            m_ulpMap = map;

            return Error::SUCCESS; // fixme to have a specific error type for other cases
        }


        /**
        * @brief: set element types
        * @param[in] nelem: number of total local elements
        * @param[in] etype[eid]
        * */
        par::Error set_elem_types(unsigned int nelem, par::ElementType* etype){
            m_uiNumElem = nelem;
            m_pEtypes = etype;

            return Error::SUCCESS; // fixme to have a specific error type for other cases
        }


        /**
         * @brief: assembly global load vector
         * @param[in/out] vec: petsc vector to assemble into
         * @param[in] eid : element ID
         * @param[in] e_mat : element load vector
         * @param[in] twin : if element is twinned or not
         * @param[in] mode = ADD_VALUES : add to existing values of the matrix
         * */
        par::Error set_element_vector(Vec vec,unsigned int eid, T* e_vec, bool twin, InsertMode mode=ADD_VALUES);

        /**
         * @brief: write pestsc matrix to ASCII file
         * @param[in/out] fmat: filename to write matrix to
         */
        par::Error dump_mat(const char* fmat);

        /**
         * @brief: write petsc vector to ASCII file
         * @param[in/out] fvec: filename to write vector to
         * @param[in] vec : petsc vector to write to file
         * */
        par::Error dump_vec(const char* fvec,Vec vec);

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
        par::Error petsc_solve(const Vec rhs,Vec out);

        /**
         * @brief: invoke basic petsc solver
         * @param[in/out] exact_sol: petsc exact solution vector
         * @param[in] eid: element id
         * @param[in] e_sol: elemet exact solution
         * @param[in] mode: petsc insert or add modes
         * */
        par::Error exact_sol(Vec exact_sol, unsigned int eid, T* e_sol, InsertMode mode=INSERT_VALUES);


    }; // end of class aMat



    //******************************************************************************************************************
    template <typename T,typename I>
    aMat<T,I>::aMat(const unsigned int n_local,const unsigned int dof, MPI_Comm comm) {

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
    }

    template <typename T,typename I>
    par::Error aMat<T,I>::create_vec(Vec& vec)
    {
        // initialize rhs vector
        VecCreate(m_comm, &vec);

        if(m_uiSize>1)
        {
            VecSetType(vec,VECMPI);
            VecSetSizes(vec, m_uiNumNodes*m_uiNumDOFperNode, PETSC_DECIDE);

        }else {
            VecSetType(vec,VECSEQ);
            VecSetSizes(vec, m_uiNumNodes*m_uiNumDOFperNode, PETSC_DECIDE);
        }

        return Error::SUCCESS; // fixme to have a specific error type for other cases

    }


    //******************************************************************************************************************
    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrix(unsigned int eid, T* e_mat, bool twin, InsertMode mode){
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
            //std::cout<<"row: "<<rowId<<std::endl;
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

        return Error::SUCCESS; // fixme to have a specific error type for other cases
    }


    //******************************************************************************************************************
    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrix(unsigned int eid, EigenMat* e_mat, unsigned int element_status, InsertMode mode) {

        par::ElementType e_type = m_pEtypes[eid];
        assert(e_mat->rows()==e_mat->cols());
        unsigned int num_dofs = aMat::dofs_per_element(e_type, element_status);

        // copy element matrix
        m_mats[eid] = *e_mat;

        // copy element status
        m_status[eid] = element_status;

        // now set values ...
        std::vector<PetscScalar> values;
        std::vector<PetscInt> colIndices;

        colIndices.resize(num_dofs);
        values.resize(num_dofs);

        PetscInt rowId;
        //unsigned int index = 0;
        for (unsigned int r = 0; r < num_dofs; ++r) {
            rowId = m_ulpMap[eid][r];
            for (unsigned int c = 0; c < num_dofs; ++c) {
                colIndices[c] = m_ulpMap[eid][c];
                values[c] = m_mats[eid](r,c);
            } // c
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
        } // r

        return Error::SUCCESS; // fixme to have a specific error type for other cases
    }


    //******************************************************************************************************************
    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrix(unsigned int eid, EigenMat* e_mat,InsertMode mode) {

        par::ElementType e_type = m_pEtypes[eid];

        assert(e_mat->rows()==e_mat->cols());
        unsigned int num_dofs = e_mat->rows();

        // copy element matrix
        m_mats[eid] = *e_mat;

        // now set values ...
        std::vector<PetscScalar> values(num_dofs);
        std::vector<PetscInt> colIndices(num_dofs);
        PetscInt rowId;

        //unsigned int index = 0;
        for (unsigned int r = 0; r < num_dofs; ++r) {
            rowId = m_ulpMap[eid][r];
            for (unsigned int c = 0; c < num_dofs; ++c) {
                colIndices[c] = m_ulpMap[eid][c];
                values[c] = m_mats[eid](r,c);
            } // c
            MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
        } // r

        return Error::SUCCESS; // fixme to have a specific error type for other cases
    }


    //******************************************************************************************************************
    template <typename T,typename I>
    par::Error aMat<T,I>::set_element_matrices(unsigned int eid, EigenMat* e_mat, unsigned int twin_level, InsertMode mode) {

        // 1u left shift "twin_level" bits.
        unsigned int numEMat=(1u<<twin_level);
        for(unsigned int i=0;i<numEMat;i++)
            set_element_matrix(eid, e_mat[i], mode);

        return Error::SUCCESS; // fixme to have a specific error type for other cases
    }


    //******************************************************************************************************************
    template <typename T, typename I>
    par::Error aMat<T,I>::matvec(std::vector<T>& v_primary, std::vector<T>& v_twin, std::vector<T>& w_primary, std::vector<T>& w_twin){

        Vec v, w;
        PetscScalar value, *array;
        PetscInt i, nlocal, istart, iend;
        PetscViewer viewer;

        // number of (local) dof's
        nlocal = m_uiNumNodes*m_uiNumDOFperNode;

        // initialize vectors
        VecCreate(m_comm, &v);
        VecSetSizes(v, nlocal, PETSC_DECIDE);
        VecSetFromOptions(v);
        VecCreate(m_comm, &w);
        VecSetSizes(w, nlocal, PETSC_DECIDE);
        VecSetFromOptions(w);

        // set values of input vector v_primary for v
        VecGetArray(v, &array);
        for (i = 0; i < nlocal; i++){
            array[i] = v_primary[i];
        }
        //PetscViewerASCIIOpen(m_comm, "v.dat", &viewer);
        //VecView(v,viewer);

        // matrix vector multiplication
        MatMult(m_pMat, v, w);

        //PetscViewerASCIIOpen(m_comm, "w.dat", &viewer);
        //VecView(w,viewer);

        // set values of w for output vector w_primary
        VecGetOwnershipRange(w, &istart, &iend);

        unsigned int index = 0;
        for (i = istart; i <iend; i++){
            VecGetValues(w,1,&i,&value);
            w_primary[index] = value;
            index++;
        }

        PetscViewerDestroy(&viewer);
        VecDestroy(&v);
        VecDestroy(&w);
        return Error::SUCCESS;
    }


    //******************************************************************************************************************
    template <typename T, typename I>
    par::Error aMat<T,I>::set_element_vector(Vec vec,unsigned int eid, T* e_vec, bool twin, InsertMode mode){

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

        return Error::SUCCESS; // fixme to have a specific error type for other cases
    }


    //******************************************************************************************************************
    template <typename T, typename I>
    par::Error aMat<T,I>::apply_dirichlet(Vec rhs,unsigned int eid,const I** dirichletBMap) {
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
    }


    //******************************************************************************************************************
    template <typename T, typename I>
    par::Error aMat<T,I>::dump_mat(const char* fmat) {

        // write matrix to file
        PetscViewer viewer;
        PetscViewerASCIIOpen(m_comm, fmat, &viewer);
        MatView(m_pMat,viewer);
        PetscViewerDestroy(&viewer);

        return Error::SUCCESS; // fixme to have a specific error type for other cases
    }

    template <typename T, typename I>
    par::Error aMat<T,I>::dump_vec(const char* fvec,Vec vec)
    {
        PetscViewer viewer;
        PetscViewerASCIIOpen(m_comm, fvec, &viewer);
        VecView(vec,viewer);
        PetscViewerDestroy(&viewer);

        return Error::SUCCESS; // fixme to have a specific error type for other cases

    }


    //******************************************************************************************************************
    template <typename T, typename I>
    par::Error aMat<T,I>::petsc_solve(const Vec rhs, Vec out) {

        KSP ksp;     /* linear solver context */
        PC  pc;      /* pre conditioner context */

        KSPCreate(m_comm,&ksp);
        //PCCreate(m_comm,&pc);
        KSPSetOperators(ksp, m_pMat, m_pMat);
        //PCSetFromOptions(pc);
        //KSPSetPC(ksp,pc);
        KSPSetFromOptions(ksp);
        KSPSolve(ksp,rhs,out);

        return Error::SUCCESS; // fixme to have a specific error type for other cases
    }

    //******************************************************************************************************************
    template <typename T, typename I>
    par::Error aMat<T,I>:: exact_sol(Vec exact_sol, unsigned int eid, T* e_sol, InsertMode mode){
        par::ElementType e_type = m_pEtypes[eid];
        unsigned int num_nodes = aMat::nodes_per_element(e_type);
        unsigned int dof = m_uiNumDOFperNode;

        PetscScalar value;
        PetscInt rowId;

        unsigned int index = 0;
        for (unsigned int r = 0; r < num_nodes*dof; ++r) {
            rowId = dof * m_ulpMap[eid][r/dof] + r % dof;
            value = e_sol[index];
            index++;
            VecSetValue(exact_sol, rowId, value, mode);
        }

        return Error::SUCCESS; // fixme to have a specific error type for other cases
    }

}; // end of namespace par
