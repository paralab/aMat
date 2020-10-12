/**
 * @file aMatBased.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Duc Tran     hantran@cs.utah.edu
 *
 * @brief A sparse matrix class for adaptive finite elements: matrix-based approach
 *
 * @version 0.1
 * @date 2020-06-09
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#ifndef ADAPTIVEMATRIX_AMATBASED_H
#define ADAPTIVEMATRIX_AMATBASED_H

#include "aMat.hpp"

namespace par
{

// class aMatBased derived from base class aMat
// DT => type of data stored in matrix (eg: double). GI => size of global index. LI => size of local
// index
template<typename DT, typename GI, typename LI>
class aMatBased : public aMat<aMatBased<DT, GI, LI>, DT, GI, LI>
{
  public:
    using ParentType = aMat<aMatBased<DT, GI, LI>, DT, GI, LI>;

    using ParentType::KfcUcVec;   // KfcUc = Kfc * Uc, used to apply bc for rhs
    using ParentType::m_BcMeth;   // method of applying Dirichlet BC
    using ParentType::m_comm;     // communicator
    using ParentType::m_dtTraceK; // penalty number
    using ParentType::m_maps;     // reference to mesh_maps passed in constructor
    using ParentType::m_matType;
    using ParentType::m_pMat;   // Petsc matrix
    using ParentType::m_uiRank; // my rank id
    using ParentType::m_uiSize; // total number of ranks
    using typename ParentType::EigenMat;

#ifdef AMAT_PROFILER
    using ParentType::timing_aMat;
#endif

  public:
    /**@brief constructor to initialize variables of aMatBased */
    aMatBased(Maps<DT, GI, LI>& mesh_maps, BC_METH bcType = BC_METH::BC_IMATRIX);

    /**@brief destructor of aMatBased */
    ~aMatBased();

    /**@brief return assembled Petsc matrix */
    Mat& get_matrix()
    {
        return m_pMat;
    }

    /**@brief allocate matrix, overidden version of aMat */
    Error allocate_matrix();

    /**@brief update matrix, overidden version of aMat */
    Error update_matrix();

    /**@brief assemble single block of element matrix, overidden version of aMatFree */
    template<typename MatrixType>
    Error set_element_matrix(LI eid, const MatrixType& e_mat, LI block_i, LI block_j, LI blocks_dim)
    {
        petsc_set_element_matrix(eid, e_mat, block_i, block_j, ADD_VALUES);
        return Error::SUCCESS;
    }

    /**@brief, assemble element matrix with all blocks at once, overidden version of aMat */
    template<typename MatrixType>
    Error set_element_matrix(LI eid,
                             LI* index_non_zero_block_i,
                             LI* index_non_zero_block_j,
                             const MatrixType** non_zero_block_mats,
                             LI num_non_zero_blocks);

    /**@brief overidden version of aMat::apply_bc */
    Error apply_bc(Vec rhs)
    {
        apply_bc_rhs(rhs);
        apply_bc_mat();
        return Error::SUCCESS;
    }

    /**@brief calls finalize_begin() and finalize_end() */
    Error finalize()
    {
        finalize_begin();
        finalize_end();
        return Error::SUCCESS;
    }

    /**@brief overidden version, begin assembling matrix */
    Error finalize_begin() const
    {
        MatAssemblyBegin(m_pMat, MAT_FINAL_ASSEMBLY);
        return Error::SUCCESS;
    }

    /**@brief overidden version, complete assembling matrix */
    Error finalize_end()
    {
        MatAssemblyEnd(m_pMat, MAT_FINAL_ASSEMBLY);
        return Error::SUCCESS;
    }

    /**@brief overidden version */
    // Note: can't be 'const' because may call matvec which may need MPI data to be stored...
    Error dump_mat(const char* filename = nullptr);

  protected:
    /**@brief matrix-based version of set_element_matrix */
    template<typename MatrixType>
    Error petsc_set_element_matrix(LI eid,
                                   const MatrixType& e_mat,
                                   LI block_i,
                                   LI block_j,
                                   InsertMode = ADD_VALUES);

    /**@brief apply Dirichlet BCs by modifying the rhs vector, also used for diagonal vector in
     * Jacobi precondition*/
    Error apply_bc_rhs(Vec rhs);

    /**@brief apply Dirichlet BCs by modifying the matrix "m_pMat" */
    Error apply_bc_mat();

}; // class aMatBased

//==============================================================================================================

template<typename DT, typename GI, typename LI>
aMatBased<DT, GI, LI>::aMatBased(Maps<DT, GI, LI>& mesh_maps, BC_METH bcType)
  : ParentType(mesh_maps, bcType)
{
    m_matType = MATRIX_TYPE::MATRIX_BASED;
    // allocate memory holding elemental matrices
    allocate_matrix();

} // constructor

template<typename DT, typename GI, typename LI>
aMatBased<DT, GI, LI>::~aMatBased()
{
    MatDestroy(&m_pMat);
} // destructor

template<typename DT, typename GI, typename LI>
Error aMatBased<DT, GI, LI>::allocate_matrix()
{

    const LI m_uiNumDofs = m_maps.get_NumDofs();

    MatCreate(m_comm, &m_pMat);
    MatSetSizes(m_pMat, m_uiNumDofs, m_uiNumDofs, PETSC_DECIDE, PETSC_DECIDE);
    if (m_uiSize > 1)
    {
        MatSetType(m_pMat, MATMPIAIJ);
        MatMPIAIJSetPreallocation(m_pMat, NNZ, PETSC_NULL, NNZ, PETSC_NULL);
    }
    else
    {
        MatSetType(m_pMat, MATSEQAIJ);
        MatSeqAIJSetPreallocation(m_pMat, NNZ, PETSC_NULL);
    }
    // this will disable on preallocation errors (but not good for performance)
    MatSetOption(m_pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

    return Error::SUCCESS;
} // allocate_matrix

template<typename DT, typename GI, typename LI>
Error aMatBased<DT, GI, LI>::update_matrix()
{

    const LI m_uiNumDofs = m_maps.get_NumDofs();

    // allocate new (larger) matrix of size m_uiNumDofs
    if (m_pMat != nullptr)
    {
        MatDestroy(&m_pMat);
        m_pMat = nullptr;
    }
    MatCreate(m_comm, &m_pMat);
    MatSetSizes(m_pMat, m_uiNumDofs, m_uiNumDofs, PETSC_DECIDE, PETSC_DECIDE);
    if (m_uiSize > 1)
    {
        // initialize matrix
        MatSetType(m_pMat, MATMPIAIJ);
        MatMPIAIJSetPreallocation(m_pMat, 30, PETSC_NULL, 30, PETSC_NULL);
    }
    else
    {
        MatSetType(m_pMat, MATSEQAIJ);
        MatSeqAIJSetPreallocation(m_pMat, 30, PETSC_NULL);
    }

    return Error::SUCCESS;
} // update_matrix

template<typename DT, typename GI, typename LI>
template<typename MatrixType>
Error aMatBased<DT, GI, LI>::set_element_matrix(LI eid,
                                                LI* ind_non_zero_block_i,
                                                LI* ind_non_zero_block_j,
                                                const MatrixType** non_zero_block_mats,
                                                LI num_non_zero_blocks)
{

    for (LI b = 0; b < num_non_zero_blocks; b++)
    {
        const LI block_i = ind_non_zero_block_i[b];
        const LI block_j = ind_non_zero_block_j[b];
        petsc_set_element_matrix(eid, *non_zero_block_mats[b], block_i, block_j, ADD_VALUES);
    }

    return Error::SUCCESS;
}

// use with Eigen, matrix-based, set every row of the matrix (faster than set every term of the
// matrix)
template<typename DT, typename GI, typename LI>
template<typename MatrixType>
Error aMatBased<DT, GI, LI>::petsc_set_element_matrix(LI eid,
                                                      const MatrixType& e_mat,
                                                      LI block_i,
                                                      LI block_j,
                                                      InsertMode mode)
{
    GI** const m_ulpMap = m_maps.get_Map();

#ifdef AMAT_PROFILER
    timing_aMat[static_cast<int>(PROFILER::PETSC_ASS)].start();
#endif

    // this is number of dofs per block:
    const LI num_dofs_per_block = e_mat.rows();
    assert(num_dofs_per_block == e_mat.cols());

    // assemble global matrix (petsc matrix)
    // now set values ...
    std::vector<PetscScalar> values(num_dofs_per_block);
    std::vector<PetscInt> colIndices(num_dofs_per_block);
    PetscInt rowId;
    for (LI r = 0; r < num_dofs_per_block; ++r)
    {
        // this ONLY WORKS with assumption that all blocks have the same number of dofs (that is
        // true for RXFEM ?)
        rowId = m_ulpMap[eid][block_i * num_dofs_per_block + r];
        for (LI c = 0; c < num_dofs_per_block; ++c)
        {
            colIndices[c] = m_ulpMap[eid][block_j * num_dofs_per_block + c];
            values[c]     = e_mat(r, c);
        } // c
        // MatSetValues(m_pMat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())),
        // (&(*values.begin())), mode);
        MatSetValues(m_pMat, 1, &rowId, colIndices.size(), colIndices.data(), values.data(), mode);
    } // r

    // compute the trace of matrix for penalty method
    if (m_BcMeth == BC_METH::BC_PENALTY)
    {
        for (LI r = 0; r < num_dofs_per_block; r++)
            m_dtTraceK += e_mat(r, r);
    }

#ifdef AMAT_PROFILER
    timing_aMat[static_cast<int>(PROFILER::PETSC_ASS)].stop();
#endif

    return Error::SUCCESS;
} // petsc_set_element_matrix

template<typename DT, typename GI, typename LI>
Error aMatBased<DT, GI, LI>::dump_mat(const char* filename /* = nullptr */)
{

    if (m_pMat == nullptr)
    {
        std::cout << "Matrix has not yet been allocated, can't display...\n";
        return Error::SUCCESS;
    }

    PetscBool assembled = PETSC_FALSE;
    MatAssembled(m_pMat, &assembled);
    if (!assembled)
    {
        std::cout << "Matrix has not yet been assembled, can't display...\n";
        return Error::SUCCESS;
    }

    if (filename == nullptr)
    {
        MatView(m_pMat, PETSC_VIEWER_STDOUT_WORLD);
    }
    else
    {
        PetscViewer viewer;
        PetscViewerASCIIOpen(m_comm, filename, &viewer);
        // write to file readable by Matlab (filename must be filename.m in order to execute in
        // Matlab)
        // PetscViewerPushFormat( viewer, PETSC_VIEWER_ASCII_MATLAB );
        MatView(m_pMat, viewer);
        PetscViewerDestroy(&viewer);
    }

    return Error::SUCCESS;
} // dump_mat

// apply Dirichlet bc by modifying matrix, only used in matrix-based approach
template<typename DT, typename GI, typename LI>
Error aMatBased<DT, GI, LI>::apply_bc_mat()
{

    const LI m_uiNumElems            = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem  = m_maps.get_DofsPerElem();
    GI** const m_ulpMap              = m_maps.get_Map();
    unsigned int** const m_uipBdrMap = m_maps.get_BdrMap();

    LI num_dofs_per_elem;
    PetscInt rowId, colId;

    for (LI eid = 0; eid < m_uiNumElems; eid++)
    {
        num_dofs_per_elem = m_uiDofsPerElem[eid];
        if (m_BcMeth == BC_METH::BC_IMATRIX)
        {
            for (LI r = 0; r < num_dofs_per_elem; r++)
            {
                rowId = m_ulpMap[eid][r];
                if (m_uipBdrMap[eid][r] == 1)
                {
                    for (LI c = 0; c < num_dofs_per_elem; c++)
                    {
                        colId = m_ulpMap[eid][c];
                        if (colId == rowId)
                        {
                            MatSetValue(m_pMat, rowId, colId, 1.0, INSERT_VALUES);
                        }
                        else
                        {
                            MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                        }
                    }
                }
                else
                {
                    for (LI c = 0; c < num_dofs_per_elem; c++)
                    {
                        colId = m_ulpMap[eid][c];
                        if (m_uipBdrMap[eid][c] == 1)
                        {
                            MatSetValue(m_pMat, rowId, colId, 0.0, INSERT_VALUES);
                        }
                    }
                }
            }
        }
        else if (m_BcMeth == BC_METH::BC_PENALTY)
        {
            for (LI r = 0; r < num_dofs_per_elem; r++)
            {
                rowId = m_ulpMap[eid][r];
                if (m_uipBdrMap[eid][r] == 1)
                {
                    for (LI c = 0; c < num_dofs_per_elem; c++)
                    {
                        colId = m_ulpMap[eid][c];
                        if (colId == rowId)
                        {
                            MatSetValue(m_pMat,
                                        rowId,
                                        colId,
                                        PENALTY_FACTOR * m_dtTraceK,
                                        INSERT_VALUES);
                        }
                    }
                }
            }
        }
    }

    return Error::SUCCESS;
} // apply_bc_mat

template<typename DT, typename GI, typename LI>
Error aMatBased<DT, GI, LI>::apply_bc_rhs(Vec rhs)
{

    const std::vector<GI>& ownedConstrainedDofs  = m_maps.get_ownedConstrainedDofs();
    const std::vector<DT>& ownedPrescribedValues = m_maps.get_ownedPrescribedValues();
    const std::vector<GI>& ownedFreeDofs         = m_maps.get_ownedFreeDofs();

    // move from set_bdr_map:
    if (m_BcMeth == BC_METH::BC_IMATRIX)
    {
        // allocate KfcUc with size = m_uiNumDofs, this will be subtracted from rhs to apply bc
        // this->petsc_create_vec( KfcUcVec );
        create_vec(m_maps, KfcUcVec);
    }

    // set rows associated with constrained dofs to be equal to Uc
    PetscInt global_Id;
    PetscScalar value, value1, value2;

    // compute KfcUc
    if (m_BcMeth == BC_METH::BC_IMATRIX)
    {
#ifdef AMAT_PROFILER
        timing_aMat[static_cast<int>(PROFILER::PETSC_KfcUc)].start();
#endif
        Vec uVec, vVec;
        // this->petsc_create_vec( uVec );
        // this->petsc_create_vec( vVec );
        create_vec(m_maps, uVec);
        create_vec(m_maps, vVec);

        // [uVec] contains the prescribed values at location of constrained dofs
        for (LI r = 0; r < ownedConstrainedDofs.size(); r++)
        {
            value     = ownedPrescribedValues[r];
            global_Id = ownedConstrainedDofs[r];
            VecSetValue(uVec, global_Id, value, INSERT_VALUES);
        }

        // multiply [K][uVec] = [vVec] where locations of free dofs equal to [Kfc][Uc]
        MatMult(m_pMat, uVec, vVec);

        VecAssemblyBegin(vVec);
        VecAssemblyEnd(vVec);

        // extract the values of [Kfc][Uc] from vVec and set to KfcUcVec
        std::vector<PetscScalar> KfcUc_values(ownedFreeDofs.size());
        std::vector<PetscInt> KfcUc_indices(ownedFreeDofs.size());
        for (LI r = 0; r < ownedFreeDofs.size(); r++)
        {
            KfcUc_indices[r] = ownedFreeDofs[r];
        }
        VecGetValues(vVec, KfcUc_indices.size(), KfcUc_indices.data(), KfcUc_values.data());
        for (LI r = 0; r < ownedFreeDofs.size(); r++)
        {
            KfcUc_values[r] = -1.0 * KfcUc_values[r];
        }

        VecSetValues(KfcUcVec,
                     KfcUc_indices.size(),
                     KfcUc_indices.data(),
                     KfcUc_values.data(),
                     ADD_VALUES);

        // petsc_destroy_vec(uVec);
        // petsc_destroy_vec(vVec);
        VecDestroy(&uVec);
        VecDestroy(&vVec);

#ifdef AMAT_PROFILER
        timing_aMat[static_cast<int>(PROFILER::PETSC_KfcUc)].stop();
#endif
    } // if (m_BcMeth == BC_METH::BC_IMATRIX)

    // modify Fc
    for (LI nid = 0; nid < ownedConstrainedDofs.size(); nid++)
    {
        global_Id = ownedConstrainedDofs[nid];
        if (m_BcMeth == BC_METH::BC_IMATRIX)
        {
            value = ownedPrescribedValues[nid];
            VecSetValue(rhs, global_Id, value, INSERT_VALUES);
        }
        else if (m_BcMeth == BC_METH::BC_PENALTY)
        {
            value = PENALTY_FACTOR * m_dtTraceK * ownedPrescribedValues[nid];
            VecSetValue(rhs, global_Id, value, INSERT_VALUES);
        }
    }

    // modify Ff for the case of BC_IMATRIX
    if (m_BcMeth == BC_METH::BC_IMATRIX)
    {
        // need to finalize vector KfcUcVec before extracting its value
        VecAssemblyBegin(KfcUcVec);
        VecAssemblyEnd(KfcUcVec);

        for (LI nid = 0; nid < ownedFreeDofs.size(); nid++)
        {
            global_Id = ownedFreeDofs[nid];
            VecGetValues(KfcUcVec, 1, &global_Id, &value1);
            VecGetValues(rhs, 1, &global_Id, &value2);
            value = value1 + value2;
            VecSetValue(rhs, global_Id, value, INSERT_VALUES);
        }
        VecDestroy(&KfcUcVec);
    }

    // petsc_destroy_vec(KfcUcVec);

    return Error::SUCCESS;
} // apply_bc_rhs

} // namespace par
#endif // APTIVEMATRIX_AMATBASED_H