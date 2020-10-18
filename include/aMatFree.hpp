/**
 * @file aMatFree.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Duc Tran     hantran@cs.utah.edu
 *
 * @brief A sparse matrix class for adaptive finite elements: matrix-free approach
 *
 * @version 0.1
 * @date 2020-06-09
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#ifndef ADAPTIVEMATRIX_AMATFREE_H
#define ADAPTIVEMATRIX_AMATFREE_H

#include "aMat.hpp"

namespace par
{

// class aMatFree derived from base class aMat
// DT => type of data stored in matrix (eg: double). GI => size of global index. LI => size of local
// index
template<typename DT, typename GI, typename LI>
class aMatFree : public aMat<aMatFree<DT, GI, LI>, DT, GI, LI>
{

  public:
    using ParentType = aMat<aMatFree<DT, GI, LI>, DT, GI, LI>;

    using ParentType::KfcUcVec;   // KfcUc = Kfc * Uc, used to apply bc for rhs
    using ParentType::m_BcMeth;   // method of applying Dirichlet BC
    using ParentType::m_comm;     // communicator
    using ParentType::m_dtTraceK; // penalty number
    using ParentType::m_maps;     // reference to mesh_maps passed in constructor
    using ParentType::m_matType;  // matrix type (aMatFree or aMatBased)
    using ParentType::m_pMat;     // Petsc matrix
    using ParentType::m_uiRank;   // my rank id
    using ParentType::m_uiSize;   // total number of ranks
    using typename ParentType::EigenMat;

#ifdef AMAT_PROFILER
    using ParentType::timing_aMat;
#endif

  protected:
    /**@brief storage of element matrices */
    std::vector<DT*>* m_epMat;

    LI m_uiMaxDofsPerBlock; // max number of DoFs per element

    LI m_uiMaxNumPads; // max number of pads to be added to the end of ve

    int m_iCommTag; // MPI communication tag

    std::vector<AsyncExchangeCtx> m_vAsyncCtx; // ghost exchange context

    std::vector<MatRecord<DT, LI>> m_vMatRec; // matrix record for block jacobi matrix

    DT* m_dpUc; // used to save constrained dofs when applying BCs in matvec

    DT* m_dpVvg; // used in MatMult_mf
    DT* m_dpUug;

    std::function<PetscErrorCode(Mat, Vec, Vec)>*  m_fptr_mat_mult=nullptr;
    
    std::function<PetscErrorCode(Mat, Vec)>* m_fptr_diag=nullptr;

    std::function<PetscErrorCode(Mat, Mat*)>* m_fptr_bdiag=nullptr;

    Mat* m_pMatBJ=nullptr; // block jacobi preconditioner. 



#ifdef HYBRID_PARALLEL
    unsigned int m_uiNumThreads; // max number of omp threads
    DT** m_veBufs;               // elemental vectors used in matvec
    DT** m_ueBufs;
#else
    DT* ve; // elemental vectors used in matvec
    DT* ue;
#endif

  public:
    /**@brief constructor to initialize variables of aMatFree */
    aMatFree(Maps<DT, GI, LI>& mesh_maps, BC_METH bcType = BC_METH::BC_IMATRIX);

    /**@brief destructor of aMatFree */
    ~aMatFree();

    /**@brief return (const) reference to the maps used in the method */
    const Maps<DT, GI, LI>& get_maps()
    {
        return m_maps;
    }

    /**@brief return Petsc matrix shell used for solving */
    Mat& get_matrix();

    /**@brief allocate matrix, overidden version of aMat */
    Error allocate_matrix();

    /**@brief update matrix, overidden version of aMat */
    Error update_matrix();

    /**@brief assemble single block of element matrix, overidden version of aMatFree */
    template<typename MatrixType>
    Error set_element_matrix(LI eid,
                             const MatrixType& e_mat,
                             LI block_i,
                             LI block_j,
                             LI blocks_dim);

    /**@brief, assemble element matrix with all blocks at once, overidden version of aMat */
    template<typename MatrixType>
    Error set_element_matrix(LI eid,
                             LI* ind_non_zero_block_i,
                             LI* ind_non_zero_block_j,
                             const MatrixType** non_zero_block_mats,
                             LI num_non_zero_blocks);

    /**@brief overidden version of aMat::apply_bc */
    Error apply_bc(Vec rhs);

    Error finalize()
    {
        finalize_begin();
        finalize_end();
        return Error::SUCCESS;
    }

    /**@brief not applicable */
    Error finalize_begin() const
    {
        // printf("petsc_init_mat is not applied for matrix-free\n");
        return Error::SUCCESS;
    }

    /**@brief not applicable */
    Error finalize_end()
    {
        // calculates trace, moved from copy_element_matrix
        if (m_BcMeth == BC_METH::BC_PENALTY)
        {
            const LI m_uiNumElems           = m_maps.get_NumElems();
            const LI* const m_uiDofsPerElem = m_maps.get_DofsPerElem();

            LI blocks_dim, num_dofs_per_block, block_row_offset, block_id;
#ifdef VECTORIZED_OPENMP_ALIGNED
            unsigned int nPads = 0;
#endif
            m_dtTraceK = 0.0;
            for (LI eid = 0; eid < m_uiNumElems; eid++)
            {
                LI blocks_dim = (LI)sqrt(m_epMat[eid].size());
                assert((blocks_dim * blocks_dim) == m_epMat[eid].size());
                num_dofs_per_block = m_uiDofsPerElem[eid] / blocks_dim;
#ifdef VECTORIZED_OPENMP_ALIGNED
                if ((num_dofs_per_block % (ALIGNMENT / sizeof(DT))) != 0)
                {
                    nPads =
                      (ALIGNMENT / sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT / sizeof(DT)));
                }
#endif
                for (LI block_i = 0; block_i < blocks_dim; block_i++)
                {
                    block_row_offset = block_i * num_dofs_per_block;
                    block_id         = block_i * blocks_dim + block_i;
                    assert(m_epMat[eid][block_id] != nullptr);
                    for (LI r = 0; r < num_dofs_per_block; r++)
                    {
#ifdef VECTORIZED_OPENMP_ALIGNED
                        m_dtTraceK += m_epMat[eid][block_id][r * (num_dofs_per_block + nPads) + r];
#else
                        m_dtTraceK += m_epMat[eid][block_id][r * num_dofs_per_block + r];
#endif
                    }
                }
            }
        }
        // printf("petsc_finalize_mat is not applied for matrix-free\n");
        return Error::SUCCESS;
    }

    /**@brief display global matrix to filename using matrix free approach */
    Error dump_mat(const char* filename);

    /**@brief pointer function points to MatMult_mt */
    std::function<PetscErrorCode(Mat, Vec, Vec)>* get_MatMult_func()
    {
        
        //std::cout<<"calling matvec func"<<std::endl;
        if(m_fptr_mat_mult!=nullptr)
        {
            // no need to reallocate the function ptr to the same function. 
            // delete m_fptr_mat_mult;
            // m_fptr_mat_mult=nullptr;
            return m_fptr_mat_mult;
        }

        m_fptr_mat_mult =
          new std::function<PetscErrorCode(Mat, Vec, Vec)>();

        (*m_fptr_mat_mult) = [this](Mat A, Vec u, Vec v) {
            this->MatMult_mf(A, u, v);
            return 0;
        };
        return m_fptr_mat_mult;
    }

    /**@brief pointer function points to MatGetDiagonal_mf */
    std::function<PetscErrorCode(Mat, Vec)>* get_MatGetDiagonal_func()
    {
        //std::cout<<"get diag"<<std::endl;
        if(m_fptr_diag!=nullptr)
        {
            // delete m_fptr_diag;
            // m_fptr_diag=nullptr;
            return m_fptr_diag;
        }

        m_fptr_diag = new std::function<PetscErrorCode(Mat, Vec)>();

        (*m_fptr_diag) = [this](Mat A, Vec d) {
            this->MatGetDiagonal_mf(A, d);
            return 0;
        };
        return m_fptr_diag;
    }

    /**@brief pointer function points to MatGetDiagonalBlock_mf */
    std::function<PetscErrorCode(Mat, Mat*)>* get_MatGetDiagonalBlock_func()
    {
        //std::cout<<"get blk diag"<<std::endl;
        if(m_fptr_bdiag!=nullptr)
        {
            // delete m_fptr_bdiag;
            // m_fptr_bdiag=nullptr;
            return m_fptr_bdiag;
        }

        m_fptr_bdiag = new std::function<PetscErrorCode(Mat, Mat*)>();

        (*m_fptr_bdiag) = [this](Mat A, Mat* a) {
            this->MatGetDiagonalBlock_mf_petsc(A, a);
            return 0;
        };
        return m_fptr_bdiag;
    }

  protected:
    /**@brief apply Dirichlet BCs to the rhs vector */
    Error apply_bc_rhs(Vec rhs);

    /**@brief allocate memory for "vec", size includes ghost DoFs if isGhosted=true, initialized by
     * alpha */
    Error create_vec_mf(DT*& vec, bool isGhosted = false, DT alpha = (DT)0.0) const;

    /**@brief free memory allocated for vec and set vec to null */
    Error destroy_vec(DT*& vec);

    /**@brief copy local (size = m_uiNumDofs) to corresponding positions of gVec (size =
     * m_uiNumDofsTotal) */
    Error local_to_ghost(DT* gVec, const DT* local) const;

    /**@brief copy gVec (size = m_uiNumDofsTotal) to local (size = m_uiNumDofs) */
    Error ghost_to_local(DT* local, const DT* gVec) const;

    /**@brief matrix-free version of set_element_matrix: copy element matrix and store in m_pMat */
    template<typename MatrixType>
    Error copy_element_matrix(LI eid,
                              const MatrixType& e_mat,
                              LI block_i,
                              LI block_j,
                              LI blocks_dim);

    /**@brief get diagonal terms of structure matrix by accumulating diagonal of element matrices */
    Error mat_get_diagonal(DT* diag, bool isGhosted = false);

    /**@brief get diagonal terms with ghosted vector diag */
    Error mat_get_diagonal_ghosted(DT* diag);

    /**@brief get diagonal block matrix (sparse matrix) */
    Error mat_get_diagonal_block(std::vector<MatRecord<DT, LI>>& diag_blk);

    /**@brief get max number of DoF per block (over all elements)*/
    // this function is not needed since max_dof_per_block is computed in set_map and it is
    // unchanged later on
    // Error get_max_dof_per_block();

    /**@brief allocate memory for ue and ve used for elemental matrix-vector multiplication */
    Error allocate_ue_ve();

    /**@brief begin: owned DoFs send, ghost DoFs receive, called before matvec() */
    Error ghost_receive_begin(DT* vec);

    /**@brief end: ghost DoFs receive, called before matvec() */
    Error ghost_receive_end(DT* vec);

    /**@brief begin: ghost DoFs send, owned DoFs receive and accumulate to current data, called
     * after matvec() */
    Error ghost_send_begin(DT* vec);

    /**@brief end: ghost DoFs send, owned DoFs receive and accumulate to current data, called after
     * matvec() */
    Error ghost_send_end(DT* vec);

    /**@brief v = K * u (K is not assembled, but directly using elemental K_e's).  v (the result)
     * must be allocated by the caller.
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
    PetscErrorCode MatGetDiagonal_mf(Mat A, Vec d);

    /**@brief matrix-free version of MatGetDiagonalBlock of PETSc (Version 1: This does the communication by aMat) */
    PetscErrorCode MatGetDiagonalBlock_mf(Mat A, Mat* a);

    /**@brief matrix-free version of MatGetDiagonalBlock of PETSc (Version 2: PETSC parallel mat is used. ) */
    PetscErrorCode MatGetDiagonalBlock_mf_petsc(Mat A, Mat* a);

    PetscErrorCode petscSetValuesInMatrix(Mat mat, std::vector<MatRecord<DT,LI>>& records, InsertMode mode);

    /**@brief apply Dirichlet BCs to diagonal vector used for Jacobi preconditioner */
    Error apply_bc_diagonal(Vec rhs);

    /**@brief apply Dirichlet BCs to block diagonal matrix */
    Error apply_bc_blkdiag(Mat* blkdiagMat);

    Error apply_bc_blkdiag_petsc(std::vector<MatRecord<DT,LI>> & records);

    /**@brief allocate an aligned memory */
    DT* create_aligned_array(unsigned int alignment, unsigned int length);

    /**@brief deallocate an aligned memory */
    void delete_algined_array(DT* array);

}; // class aMatFree

//==============================================================================================================
// context for aMat
template<typename DT, typename GI, typename LI>
struct aMatCTX
{
    par::aMatFree<DT, GI, LI>* aMatPtr;
};

// matrix shell to use aMat::MatMult_mf
template<typename DT, typename GI, typename LI>
PetscErrorCode aMat_matvec(Mat A, Vec u, Vec v)
{
    aMatCTX<DT, GI, LI>* pCtx;
    MatShellGetContext(A, &pCtx);

    par::aMatFree<DT, GI, LI>* pLap                 = pCtx->aMatPtr;
    std::function<PetscErrorCode(Mat, Vec, Vec)>* f = pLap->get_MatMult_func();
    (*f)(A, u, v);
    return 0;
}

// matrix shell to use aMat::MatGetDiagonal_mf
template<typename DT, typename GI, typename LI>
PetscErrorCode aMat_matgetdiagonal(Mat A, Vec d)
{
    aMatCTX<DT, GI, LI>* pCtx;
    MatShellGetContext(A, &pCtx);

    par::aMatFree<DT, GI, LI>* pLap            = pCtx->aMatPtr;
    std::function<PetscErrorCode(Mat, Vec)>* f = pLap->get_MatGetDiagonal_func();
    (*f)(A, d);
    return 0;
}

// matrix shell to use aMat::MatGetDiagonalBlock_mf
template<typename DT, typename GI, typename LI>
PetscErrorCode aMat_matgetdiagonalblock(Mat A, Mat* a)
{
    aMatCTX<DT, GI, LI>* pCtx;
    MatShellGetContext(A, &pCtx);

    par::aMatFree<DT, GI, LI>* pLap             = pCtx->aMatPtr;
    std::function<PetscErrorCode(Mat, Mat*)>* f = pLap->get_MatGetDiagonalBlock_func();
    (*f)(A, a);
    return 0;
}

//==============================================================================================================

template<typename DT, typename GI, typename LI>
aMatFree<DT, GI, LI>::aMatFree(Maps<DT, GI, LI>& mesh_maps, BC_METH bcType)
  : ParentType(mesh_maps, bcType)
{
    m_epMat    = nullptr; // element matrices (Eigen matrix), used in matrix-free
    m_iCommTag = 0;       // tag for sends & receives used in matvec and mat_get_diagonal_block_seq
    m_matType  = MATRIX_TYPE::MATRIX_FREE;

    m_dpUc  = nullptr;
    m_dpVvg = nullptr;
    m_dpUug = nullptr;

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
    // allocate memory holding elemental matrices
    allocate_matrix();

} // constructor

template<typename DT, typename GI, typename LI>
aMatFree<DT, GI, LI>::~aMatFree()
{
    // number of elements owned by my rank
    const LI m_uiNumElems = m_maps.get_NumElems();
    // number of constraints owned by myRank
    const LI n_owned_constraints = m_maps.get_n_owned_constraints();

    // free memory allocated for m_epMat storing elemental matrices
    if (m_epMat != nullptr)
    {
        for (LI eid = 0; eid < m_uiNumElems; eid++)
        {
            for (LI bid = 0; bid < m_epMat[eid].size(); bid++)
            {
                if (m_epMat[eid][bid] != nullptr)
                {
                    // delete the block matrix bid
                    delete_algined_array(m_epMat[eid][bid]);
                }
            }
            // clear the content of vector of DT* and resize to 0
            m_epMat[eid].clear();
        }
        // delete the array created by new in set_map
        delete[] m_epMat;
    }

// free memory allocated for storing elemental vectors ue and ve
#ifdef HYBRID_PARALLEL
    for (unsigned int tid = 0; tid < m_uiNumThreads; tid++)
    {
        if (m_ueBufs[tid] != nullptr)
            delete_algined_array(m_ueBufs[tid]);
        if (m_veBufs[tid] != nullptr)
            delete_algined_array(m_veBufs[tid]);
    }
    if (m_ueBufs != nullptr)
    {
        free(m_ueBufs);
    }
    if (m_veBufs != nullptr)
    {
        free(m_veBufs);
    }
#else
    if (ue != nullptr)
    {
        delete_algined_array(ue);
    }
    if (ve != nullptr)
    {
        delete_algined_array(ve);
    }
#endif

    // free memory allocated for Uc
    if (m_dpUc != nullptr)
    {
        delete[] m_dpUc;
        m_dpUc=nullptr;
    }

    // free memory allocated for vvg and uug
    if (m_dpVvg != nullptr)
        delete[] m_dpVvg;
    if (m_dpUug != nullptr)
        delete[] m_dpUug;

    if(m_fptr_mat_mult!=nullptr)
    {
        delete m_fptr_mat_mult;
        m_fptr_mat_mult=nullptr;
    }

    if(m_fptr_diag!=nullptr)
    {
        delete m_fptr_diag;
        m_fptr_diag=nullptr;
    }

    if(m_fptr_bdiag!=nullptr)
    {
        delete m_fptr_bdiag;
        m_fptr_bdiag=nullptr;
    }   

    if(m_pMatBJ!=nullptr)
    {
        MatDestroy(m_pMatBJ);
        m_pMatBJ=nullptr;
    }

    MatDestroy(m_pMat);
    



} // destructor

template<typename DT, typename GI, typename LI>
Mat& aMatFree<DT, GI, LI>::get_matrix()
{

    const LI m_uiNumDofs = m_maps.get_NumDofs();

    // get context to aMatFree (need to allocate in heap so that it is usable in solve which is now
    // outside of aMat)
    aMatCTX<DT, GI, LI>* ctx = new aMatCTX<DT, GI, LI>();

    // point back to aMatFree
    ctx->aMatPtr = this;

    // create matrix shell
    MatCreateShell(m_comm,
                   m_uiNumDofs,
                   m_uiNumDofs,
                   PETSC_DETERMINE,
                   PETSC_DETERMINE,
                   ctx,
                   &m_pMat);

    // set operation for matrix-vector multiplication using aMatFree::MatMult_mf
    MatShellSetOperation(m_pMat, MATOP_MULT, (void (*)(void))aMat_matvec<DT, GI, LI>);

    // set operation for geting matrix diagonal using aMatFree::MatGetDiagonal_mf
    MatShellSetOperation(m_pMat,
                         MATOP_GET_DIAGONAL,
                         (void (*)(void))aMat_matgetdiagonal<DT, GI, LI>);

    // set operation for geting block matrix diagonal using aMatFree::MatGetDiagonalBlock_mf
    MatShellSetOperation(m_pMat,
                         MATOP_GET_DIAGONAL_BLOCK,
                         (void (*)(void))aMat_matgetdiagonalblock<DT, GI, LI>);

    return m_pMat;
} // get_matrix

// this function was at the end of set_map() which moved to Maps class
template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::allocate_matrix()
{

    const LI m_uiNumElems           = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem = m_maps.get_DofsPerElem();
    const LI n_owned_constraints    = m_maps.get_n_owned_constraints();
    const LI m_uiNumDofsTotal       = m_maps.get_NumDofsTotal();

    // clear elemental matrices stored in m_epMat (if it is not empty)
    if (m_epMat != nullptr)
    {
        for (LI eid = 0; eid < m_uiNumElems; eid++)
        {
            for (LI bid = 0; bid < m_epMat[eid].size(); bid++)
            {
                if (m_epMat[eid][bid] != nullptr)
                {
                    free(m_epMat[eid][bid]);
                }
            }
            m_epMat[eid].clear();
        }
        delete[] m_epMat;
        m_epMat = nullptr;
    }

    // allocate m_epMat as an array with size equals number of owned elements
    // we do not know how many blocks and size of blocks for each element at this time
    if (m_uiNumElems > 0)
    {
        m_epMat = new std::vector<DT*>[m_uiNumElems];
    }

    // compute the largest number of dofs per block, to be used for allocation of ue and ve...
    // ASSUME that initially every element has only one block, AND the size of block is unchanged
    // during crack growth
    for (LI eid = 0; eid < m_uiNumElems; eid++)
    {
        if (m_uiMaxDofsPerBlock < m_uiDofsPerElem[eid])
            m_uiMaxDofsPerBlock = m_uiDofsPerElem[eid];
    }

// get number of pads added to ve (where ve = block_matrix * ue)
#ifdef VECTORIZED_OPENMP_ALIGNED
    assert((ALIGNMENT % sizeof(DT)) == 0);
    if ((m_uiMaxDofsPerBlock % (ALIGNMENT / sizeof(DT))) != 0)
    {
        m_uiMaxNumPads =
          (ALIGNMENT / sizeof(DT)) - (m_uiMaxDofsPerBlock % (ALIGNMENT / sizeof(DT)));
    }
    else
    {
        m_uiMaxNumPads = 0;
    }
#endif

    // allocate memory for ue and ve used in elemental matrix-vector multiplication
    allocate_ue_ve();

    // allocate memory for vvg and uug used in MatMult_mf
    m_dpVvg = new DT[m_uiNumDofsTotal];
    m_dpUug = new DT[m_uiNumDofsTotal];

    if (m_dpUc != nullptr)
    {
        delete[] m_dpUc;
        m_dpUc = nullptr;
    }

    m_dpUc = new DT[m_uiNumDofsTotal];

    return Error::SUCCESS;
} // allocate_matrix

// this new function is extracted from the end of update_map() which now belongs to Maps class
template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::update_matrix()
{

    const LI m_uiNumDofsTotal = m_maps.get_NumDofsTotal();

    // re-allocate memory for vvg and uug used in MatMult_mf
    if (m_dpVvg != nullptr)
        delete[] m_dpVvg;
    m_dpVvg = new DT[m_uiNumDofsTotal];

    if (m_dpUug != nullptr)
        delete[] m_dpUug;
    m_dpUug = new DT[m_uiNumDofsTotal];

    return Error::SUCCESS;
} // update_matrix

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::create_vec_mf(DT*& vec,
                                          bool isGhosted /* = false */,
                                          DT alpha /* = 0.0 */) const
{
    const LI m_uiNumDofsTotal = m_maps.get_NumDofsTotal();
    const LI m_uiNumDofs      = m_maps.get_NumDofs();

    if (isGhosted)
    {
        vec = new DT[m_uiNumDofsTotal];
    }
    else
    {
        vec = new DT[m_uiNumDofs];
    }
    // initialize
    if (isGhosted)
    {
        for (LI i = 0; i < m_uiNumDofsTotal; i++)
        {
            vec[i] = alpha;
        }
    }
    else
    {
        for (LI i = 0; i < m_uiNumDofs; i++)
        {
            vec[i] = alpha;
        }
    }

    return Error::SUCCESS;

} // create_vec_mf

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::destroy_vec(DT*& vec)
{
    if (vec != nullptr)
    {
        delete[] vec;
        vec = nullptr;
    }

    return Error::SUCCESS;
} // destroy_vec

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::local_to_ghost(DT* gVec, const DT* local) const
{
    const LI m_uiNumDofsTotal    = m_maps.get_NumDofsTotal();
    const LI m_uiNumPreGhostDofs = m_maps.get_NumPreGhostDofs();
    const LI m_uiNumDofs         = m_maps.get_NumDofs();

    for (LI i = 0; i < m_uiNumDofsTotal; i++)
    {
        if ((i >= m_uiNumPreGhostDofs) && (i < m_uiNumPreGhostDofs + m_uiNumDofs))
        {
            gVec[i] = local[i - m_uiNumPreGhostDofs];
        }
        else
        {
            gVec[i] = 0.0;
        }
    }
    return Error::SUCCESS;
} // local_to_ghost

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::ghost_to_local(DT* local, const DT* gVec) const
{
    const LI m_uiNumPreGhostDofs = m_maps.get_NumPreGhostDofs();
    const LI m_uiNumDofs         = m_maps.get_NumDofs();

    for (LI i = 0; i < m_uiNumDofs; i++)
    {
        local[i] = gVec[i + m_uiNumPreGhostDofs];
    }

    return Error::SUCCESS;
} // ghost_to_local

template<typename DT, typename GI, typename LI>
template<typename MatrixType>
Error aMatFree<DT, GI, LI>::set_element_matrix(LI eid,
                                               const MatrixType& e_mat,
                                               LI block_i,
                                               LI block_j,
                                               LI blocks_dim)
{
    aMatFree<DT, GI, LI>::copy_element_matrix(eid, e_mat, block_i, block_j, blocks_dim);
    return Error::SUCCESS;
} // set_element_matrix

template<typename DT, typename GI, typename LI>
template<typename MatrixType>
Error aMatFree<DT, GI, LI>::set_element_matrix(LI eid,
                                               LI* ind_non_zero_block_i,
                                               LI* ind_non_zero_block_j,
                                               const MatrixType** non_zero_block_mats,
                                               LI num_non_zero_blocks)
{
    for (LI b = 0; b < num_non_zero_blocks; b++)
    {
        const LI block_i = ind_non_zero_block_i[b];
        const LI block_j = ind_non_zero_block_j[b];
        copy_element_matrix(eid, *non_zero_block_mats[b], block_i, block_j, num_non_zero_blocks);
    }

    return Error::SUCCESS;
}

template<typename DT, typename GI, typename LI>
inline Error aMatFree<DT, GI, LI>::apply_bc(Vec rhs)
{
    // Allocate m_dpUc, which will temporarily stores terms related to constrained DoFs
    // (boundary conditions) during MatMult_mf.
    if (m_dpUc != nullptr)
    {
        delete[] m_dpUc;
    }
    auto n_owned_constraints = m_maps.get_n_owned_constraints();
    if (n_owned_constraints > 0)
    {
        m_dpUc = new DT[n_owned_constraints];
    }

    apply_bc_rhs(rhs);
    return Error::SUCCESS;
}

template<typename DT, typename GI, typename LI>
template<typename MatrixType>
Error aMatFree<DT, GI, LI>::copy_element_matrix(LI eid,
                                                const MatrixType& e_mat,
                                                LI block_i,
                                                LI block_j,
                                                LI blocks_dim)
{
    unsigned int nPads = 0;
    
    // resize the vector of blocks for element eid
    if (m_epMat[eid].size() != blocks_dim * blocks_dim)
    {
        for (auto& block : m_epMat[eid])
        {
            delete_algined_array(block);
        }
        m_epMat[eid].resize(blocks_dim * blocks_dim);
        for (auto& block : m_epMat[eid])
        {
            block = nullptr;
        }
    }

    // 1D-index of (block_i, block_j)
    LI index = (block_i * blocks_dim) + block_j;

    auto num_dofs_per_block = e_mat.rows();
    assert(num_dofs_per_block == e_mat.cols());

    const LI* const m_uiDofsPerElem = m_maps.get_DofsPerElem();
    assert((num_dofs_per_block * blocks_dim) == m_uiDofsPerElem[eid]);

    // allocate memory to store e_mat (e_mat is one of blocks of the elemental matrix of element eid)
    if (m_epMat[eid][index] == nullptr)
    {
// allocate memory for elemental matrices
#ifdef VECTORIZED_OPENMP_ALIGNED
        // compute number of paddings appended to each column of elemental block matrix so that each
        // column is aligned with ALIGNMENT
        assert((ALIGNMENT % sizeof(DT)) == 0);
        //unsigned int nPads = 0;
        if ((num_dofs_per_block % (ALIGNMENT / sizeof(DT))) != 0)
        {
            nPads = (ALIGNMENT / sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT / sizeof(DT)));
        }

        // allocate block matrix with added paddings and aligned with ALIGNMENT
        m_epMat[eid][index] =
          create_aligned_array(ALIGNMENT, ((num_dofs_per_block + nPads) * num_dofs_per_block));
#else
        // allocate block matrix as normal
        m_epMat[eid][index] = (DT*)malloc((num_dofs_per_block * num_dofs_per_block) * sizeof(DT));
#endif
    }

// store block matrix in column-major for all methods of vectorization, row-major for non
// vectorization
#if defined(VECTORIZED_AVX512) || defined(VECTORIZED_AVX256) || defined(VECTORIZED_OPENMP)
    for (LI col = 0; col < num_dofs_per_block; col++)
    {
        for (LI row = 0; row < num_dofs_per_block; row++)
        {
            m_epMat[eid][index][(col * num_dofs_per_block) + row] = e_mat(row, col);
            // if (eid == 0) printf("e_mat[%d][%d,%d]= %f\n",eid,r,c,e_mat(r,c));
        }
    }
#elif VECTORIZED_OPENMP_ALIGNED
    for (LI col = 0; col < num_dofs_per_block; col++)
    {
        for (LI row = 0; row < num_dofs_per_block; row++)
        {
            m_epMat[eid][index][(col * (num_dofs_per_block + nPads)) + row] = e_mat(row, col);
        }
    }
#else
    for (LI row = 0; row < num_dofs_per_block; row++)
    {
        for (LI col = 0; col < num_dofs_per_block; col++)
        {
            m_epMat[eid][index][(row * num_dofs_per_block) + col] = e_mat(row, col);
        }
    }
#endif

    return Error::SUCCESS;
} // copy_element_matrix

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::mat_get_diagonal(DT* diag, bool isGhosted)
{
    if (isGhosted)
    {
        mat_get_diagonal_ghosted(diag);
    }
    else
    {
        DT* g_diag;
        create_vec_mf(g_diag, true, 0.0);
        mat_get_diagonal_ghosted(g_diag);
        ghost_to_local(diag, g_diag);
        delete[] g_diag;
    }

    return Error::SUCCESS;
} // mat_get_diagonal

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::mat_get_diagonal_ghosted(DT* diag)
{
    const LI m_uiNumElems           = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem = m_maps.get_DofsPerElem();
    LI** const m_uipLocalMap        = m_maps.get_LocalMap();

    LI rowID;

#ifdef VECTORIZED_OPENMP_ALIGNED
    unsigned int nPads = 0;
#endif

    for (LI eid = 0; eid < m_uiNumElems; eid++)
    {
        // number of blocks in each direction (i.e. blocks_dim)
        LI blocks_dim = (LI)sqrt(m_epMat[eid].size());

        // number of block must be a square of blocks_dim
        assert((blocks_dim * blocks_dim) == m_epMat[eid].size());

        // number of dofs per block, must be the same for all blocks
        const LI num_dofs_per_block = m_uiDofsPerElem[eid] / blocks_dim;

// compute number of paddings appended to each column
#ifdef VECTORIZED_OPENMP_ALIGNED
        if ((num_dofs_per_block % (ALIGNMENT / sizeof(DT))) != 0)
        {
            nPads = (ALIGNMENT / sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT / sizeof(DT)));
        }
#endif

        LI block_row_offset = 0;
        for (LI block_i = 0; block_i < blocks_dim; block_i++)
        {

            // only get diagonals of diagonal blocks
            LI index = block_i * blocks_dim + block_i;

            // diagonal block must be non-zero
            assert(m_epMat[eid][index] != nullptr);

            for (LI r = 0; r < num_dofs_per_block; r++)
            {
                // local (rank) row ID
                rowID = m_uipLocalMap[eid][block_row_offset + r];

// get diagonal of elemental block matrix
#ifdef VECTORIZED_OPENMP_ALIGNED
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

    return Error::SUCCESS;
} // mat_get_diagonal_ghosted

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::mat_get_diagonal_block(std::vector<MatRecord<DT, LI>>& diag_blk)
{

    const LI m_uiNumElems                    = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem          = m_maps.get_DofsPerElem();
    GI** const m_ulpMap                      = m_maps.get_Map();
    LI** const m_uipLocalMap                 = m_maps.get_LocalMap();
    const std::vector<GI>& m_ulvLocalDofScan = m_maps.get_LocalDofScan();
    const LI m_uiDofLocalBegin               = m_maps.get_DofLocalBegin();
    const LI m_uiDofLocalEnd                 = m_maps.get_DofLocalEnd();

    LI blocks_dim;
    GI glo_RowId, glo_ColId;
    LI loc_RowId, loc_ColId;
    LI rowID, colID;
    unsigned int rank_r, rank_c;
    DT value;
    LI ind = 0;

#ifdef VECTORIZED_OPENMP_ALIGNED
    unsigned int nPads = 0;
#endif

    std::vector<MatRecord<DT, LI>> matRec1;
    std::vector<MatRecord<DT, LI>> matRec2;

    MatRecord<DT, LI> matr;

    m_vMatRec.clear();
    diag_blk.clear();

    for (LI eid = 0; eid < m_uiNumElems; eid++)
    {

        // number of blocks in row (or column)
        blocks_dim = (LI)sqrt(m_epMat[eid].size());
        assert(blocks_dim * blocks_dim == m_epMat[eid].size());

        const LI num_dofs_per_block = m_uiDofsPerElem[eid] / blocks_dim;

#ifdef VECTORIZED_OPENMP_ALIGNED
        // compute number of paddings appended to each column
        if ((num_dofs_per_block % (ALIGNMENT / sizeof(DT))) != 0)
        {
            nPads = (ALIGNMENT / sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT / sizeof(DT)));
        }
#endif

        for (LI block_i = 0; block_i < blocks_dim; block_i++)
        {

            LI block_row_offset = block_i * num_dofs_per_block;

            for (LI block_j = 0; block_j < blocks_dim; block_j++)
            {

                LI index            = block_i * blocks_dim + block_j;
                LI block_col_offset = block_j * num_dofs_per_block;

                if (m_epMat[eid][index] != nullptr)
                {

                    for (LI r = 0; r < num_dofs_per_block; r++)
                    {

                        // local row Id (include ghost nodes)
                        rowID = m_uipLocalMap[eid][block_row_offset + r];

                        // global row Id
                        glo_RowId = m_ulpMap[eid][block_row_offset + r];

                        // rank who owns global row Id
                        rank_r = m_maps.globalId_2_rank(
                          glo_RowId); // globalId_2_rank() is moved to Maps class

                        // local ID in that rank (not include ghost nodes)
                        loc_RowId = (glo_RowId - m_ulvLocalDofScan[rank_r]);

                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            // local column Id (include ghost nodes)
                            colID = m_uipLocalMap[eid][block_col_offset + c];

                            // global column Id
                            glo_ColId = m_ulpMap[eid][block_col_offset + c];

                            // rank who owns global column Id
                            rank_c = m_maps.globalId_2_rank(
                              glo_ColId); // globalId_2_rank() is moved to Maps class

                            // local column Id in that rank (not include ghost nodes)
                            loc_ColId = (glo_ColId - m_ulvLocalDofScan[rank_c]);

                            if (rank_r == rank_c)
                            {
                                // put all data in a MatRecord object
                                matr.setRank(rank_r);
                                matr.setRowId(loc_RowId);
                                matr.setColId(loc_ColId);
#if defined(VECTORIZED_AVX512) || defined(VECTORIZED_AVX256) || defined(VECTORIZED_OPENMP)
                                // elemental block matrix stored in column-major
                                matr.setVal(m_epMat[eid][index][(c * num_dofs_per_block) + r]);
#elif VECTORIZED_OPENMP_ALIGNED
                                // paddings are inserted at columns' ends
                                matr.setVal(
                                  m_epMat[eid][index][c * (num_dofs_per_block + nPads) + r]);
#else
                                // elemental block matrix stored in row-major
                                matr.setVal(m_epMat[eid][index][(r * num_dofs_per_block) + c]);
#endif

                                if ((rowID >= m_uiDofLocalBegin) && (rowID < m_uiDofLocalEnd) &&
                                    (colID >= m_uiDofLocalBegin) && (colID < m_uiDofLocalEnd))
                                {
                                    // add to diagonal block of my rank
                                    assert(rank_r == m_uiRank);
                                    matRec1.push_back(matr);
                                }
                                else
                                {
                                    // add to matRec for sending to rank who owns this matrix term
                                    assert(rank_r != m_uiRank);
                                    matRec2.push_back(matr);
                                }
                            }
                        }
                    }
                } // if block is not null

            } // for block_j

        } // for block_i

    } // for (eid = 0:m_uiNumElems)

    // sorting matRec2
    std::sort(matRec2.begin(), matRec2.end());

    // accumulate if 2 components of matRec2 are equal, then reduce the size
    ind = 0;
    while (ind < matRec2.size())
    {
        matr.setRank(matRec2[ind].getRank());
        matr.setRowId(matRec2[ind].getRowId());
        matr.setColId(matRec2[ind].getColId());

        value = matRec2[ind].getVal();
        // since matRec is sorted, we keep increasing i for all members that are equal
        while (((ind + 1) < matRec2.size()) && (matRec2[ind] == matRec2[ind + 1]))
        {
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

    for (unsigned int i = 0; i < m_uiSize; i++)
    {
        sendCounts[i] = 0;
        recvCounts[i] = 0;
    }

    // number of elements sending to each rank
    for (LI i = 0; i < m_vMatRec.size(); i++)
    {
        sendCounts[m_vMatRec[i].getRank()]++;
    }

    // number of elements receiving from each rank
    MPI_Alltoall(sendCounts, 1, MPI_UNSIGNED, recvCounts, 1, MPI_UNSIGNED, m_comm);

    sendOffset[0] = 0;
    recvOffset[0] = 0;
    for (unsigned int i = 1; i < m_uiSize; i++)
    {
        sendOffset[i] = sendCounts[i - 1] + sendOffset[i - 1];
        recvOffset[i] = recvCounts[i - 1] + recvOffset[i - 1];
    }

    // allocate receive buffer
    std::vector<MatRecord<DT, LI>> recv_buff;
    recv_buff.resize(recvCounts[m_uiSize - 1] + recvOffset[m_uiSize - 1]);

    // send to all other ranks
    for (unsigned int i = 0; i < m_uiSize; i++)
    {
        if (sendCounts[i] == 0)
            continue;
        const MPI_Datatype dtype = par::MPI_datatype_matrecord<DT, LI>::value();
        MPI_Send(&m_vMatRec[sendOffset[i]], sendCounts[i], dtype, i, m_iCommTag, m_comm);
    }

    // receive from all other ranks
    for (unsigned int i = 0; i < m_uiSize; i++)
    {
        if (recvCounts[i] == 0)
            continue;
        const MPI_Datatype dtype = par::MPI_datatype_matrecord<DT, LI>::value();
        MPI_Status status;
        MPI_Recv(&recv_buff[recvOffset[i]], recvCounts[i], dtype, i, m_iCommTag, m_comm, &status);
    }

    m_iCommTag++;

    // add the received data to matRec1
    for (LI i = 0; i < recv_buff.size(); i++)
    {
        if (recv_buff[i].getRank() != m_uiRank)
        {
            return Error::WRONG_COMMUNICATION;
        }
        else
        {
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
    while (ind < matRec1.size())
    {
        matr.setRank(matRec1[ind].getRank());
        matr.setRowId(matRec1[ind].getRowId());
        matr.setColId(matRec1[ind].getColId());

        DT val = matRec1[ind].getVal();
        // since matRec1 is sorted, we keep increasing i for all members that are equal
        while (((ind + 1) < matRec1.size()) && (matRec1[ind] == matRec1[ind + 1]))
        {
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

    delete[] sendCounts;
    delete[] recvCounts;
    delete[] sendOffset;
    delete[] recvOffset;

    return Error::SUCCESS;
} // mat_get_diagonal_block

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::allocate_ue_ve()
{
#ifdef HYBRID_PARALLEL
    // ue and ve are local to each thread, they will be allocated after #pragma omp parallel
    // at the moment, we only allocate an array (of size = number of threads) of DT*
    m_ueBufs = (DT**)malloc(m_uiNumThreads * sizeof(DT*));
    m_veBufs = (DT**)malloc(m_uiNumThreads * sizeof(DT*));
    for (unsigned int i = 0; i < m_uiNumThreads; i++)
    {
        m_ueBufs[i] = nullptr;
        m_veBufs[i] = nullptr;
    }

#else
#ifdef VECTORIZED_OPENMP_ALIGNED
    // allocate and align ve = MaxDofsPerBlock + MaxNumPads
    ve = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock + m_uiMaxNumPads);
    // allocate ue as normal
    ue = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
    // ue = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock);
#else
    // allocate ve and ue as normal
    ue = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
    ve = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
#endif
#endif

    return Error::SUCCESS;
} // allocate_ue_ve

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::ghost_receive_begin(DT* vec)
{
    if (m_uiSize == 1)
        return Error::SUCCESS;

    const std::vector<LI>& m_uivSendDofOffset         = m_maps.get_SendDofOffset();
    const std::vector<LI>& m_uivSendDofCounts         = m_maps.get_SendDofCounts();
    const std::vector<LI>& m_uivSendDofIds            = m_maps.get_SendDofIds();
    const std::vector<LI>& m_uivRecvDofOffset         = m_maps.get_RecvDofOffset();
    const std::vector<LI>& m_uivRecvDofCounts         = m_maps.get_RecvDofCounts();
    const std::vector<unsigned int>& m_uivSendRankIds = m_maps.get_SendRankIds();
    const std::vector<unsigned int>& m_uivRecvRankIds = m_maps.get_RecvRankIds();

    // exchange context for vec
    AsyncExchangeCtx ctx((const void*)vec);

    // total number of DoFs to be sent
    const LI total_send = m_uivSendDofOffset[m_uiSize - 1] + m_uivSendDofCounts[m_uiSize - 1];
    assert(total_send == m_uivSendDofIds.size());

    // total number of DoFs to be received
    const LI total_recv = m_uivRecvDofOffset[m_uiSize - 1] + m_uivRecvDofCounts[m_uiSize - 1];

    // send data of owned DoFs to corresponding ghost DoFs in all other ranks
    if (total_send > 0)
    {
        // allocate memory for sending buffer
        ctx.allocateSendBuffer(sizeof(DT) * total_send);
        // get the address of sending buffer
        DT* send_buf = (DT*)ctx.getSendBuffer();
        // put all sending values to buffer
        for (LI i = 0; i < total_send; i++)
        {
            send_buf[i] = vec[m_uivSendDofIds[i]];
        }
        for (unsigned int r = 0; r < m_uivSendRankIds.size(); r++)
        {
            unsigned int i = m_uivSendRankIds[r]; // rank that I will send to
            // send to rank i
            MPI_Request* req = new MPI_Request();
            MPI_Isend(&send_buf[m_uivSendDofOffset[i]],
                      m_uivSendDofCounts[i] * sizeof(DT),
                      MPI_BYTE,
                      i,
                      m_iCommTag,
                      m_comm,
                      req);
            // put output request req of sending into the Request list of ctx
            ctx.getRequestList().push_back(req);
        }
    }

    // received data for ghost DoFs from all other ranks
    if (total_recv > 0)
    {
        ctx.allocateRecvBuffer(sizeof(DT) * total_recv);
        DT* recv_buf = (DT*)ctx.getRecvBuffer();

        for (unsigned int r = 0; r < m_uivRecvRankIds.size(); r++)
        {
            unsigned int i   = m_uivRecvRankIds[r]; // rank that I will receive from
            MPI_Request* req = new MPI_Request();
            MPI_Irecv(&recv_buf[m_uivRecvDofOffset[i]],
                      m_uivRecvDofCounts[i] * sizeof(DT),
                      MPI_BYTE,
                      i,
                      m_iCommTag,
                      m_comm,
                      req);
            // put output request req of receiving into Request list of ctx
            ctx.getRequestList().push_back(req);
        }
    }
    // save the ctx of v for later access
    m_vAsyncCtx.push_back(ctx);
    // get a different value of tag if we have another ghost_exchange for a different vec
    m_iCommTag++;

    return Error::SUCCESS;
} // ghost_receive_begin

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::ghost_receive_end(DT* vec)
{
    if (m_uiSize == 1)
        return Error::SUCCESS;

    const LI m_uiNumPreGhostDofs  = m_maps.get_NumPreGhostDofs();
    const LI m_uiNumPostGhostDofs = m_maps.get_NumPostGhostDofs();
    const LI m_uiNumDofs          = m_maps.get_NumDofs();

    // get the context associated with vec
    unsigned int ctx_index;
    for (unsigned int i = 0; i < m_vAsyncCtx.size(); i++)
    {
        if (vec == (DT*)m_vAsyncCtx[i].getBuffer())
        {
            ctx_index = i;
            break;
        }
    }
    AsyncExchangeCtx ctx = m_vAsyncCtx[ctx_index];

    // wait for all sends and receives finish
    MPI_Status sts;
    // total number of sends and receives have issued
    auto num_req = ctx.getRequestList().size();
    for (std::size_t i = 0; i < num_req; i++)
    {
        MPI_Wait(ctx.getRequestList()[i], &sts);
    }

    // const unsigned  int total_recv = m_uivRecvDofOffset[m_uiSize-1] +
    // m_uivRecvDofCounts[m_uiSize-1];

    DT* recv_buf = (DT*)ctx.getRecvBuffer();

    // received values are now put at pre-ghost and post-ghost positions of vec
    std::memcpy(vec, recv_buf, m_uiNumPreGhostDofs * sizeof(DT));
    std::memcpy(&vec[m_uiNumPreGhostDofs + m_uiNumDofs],
                &recv_buf[m_uiNumPreGhostDofs],
                m_uiNumPostGhostDofs * sizeof(DT));

    // free memory of send and receive buffers of ctx
    ctx.deAllocateRecvBuffer();
    ctx.deAllocateSendBuffer();

    // erase the context associated with ctx in m_vAsyncCtx
    m_vAsyncCtx.erase(m_vAsyncCtx.begin() + ctx_index);

    return Error::SUCCESS;
} // ghost_receive_end

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::ghost_send_begin(DT* vec)
{
    if (m_uiSize == 1)
        return Error::SUCCESS;

    const std::vector<LI>& m_uivSendDofOffset         = m_maps.get_SendDofOffset();
    const std::vector<LI>& m_uivSendDofCounts         = m_maps.get_SendDofCounts();
    const std::vector<LI>& m_uivRecvDofOffset         = m_maps.get_RecvDofOffset();
    const std::vector<LI>& m_uivRecvDofCounts         = m_maps.get_RecvDofCounts();
    const std::vector<unsigned int>& m_uivSendRankIds = m_maps.get_SendRankIds();
    const std::vector<unsigned int>& m_uivRecvRankIds = m_maps.get_RecvRankIds();
    const LI m_uiNumPreGhostDofs                      = m_maps.get_NumPreGhostDofs();
    const LI m_uiNumDofs                              = m_maps.get_NumDofs();
    const LI m_uiNumPostGhostDofs                     = m_maps.get_NumPostGhostDofs();

    AsyncExchangeCtx ctx((const void*)vec);

    // number of owned dofs to be received from other ranks (i.e. number of dofs that I sent out
    // before doing matvec)
    const LI total_recv = m_uivSendDofOffset[m_uiSize - 1] + m_uivSendDofCounts[m_uiSize - 1];
    // number of dofs to be sent back to their owners (i.e. number of dofs that I received before
    // doing matvec)
    const LI total_send = m_uivRecvDofOffset[m_uiSize - 1] + m_uivRecvDofCounts[m_uiSize - 1];

    // receive data
    if (total_recv > 0)
    {
        ctx.allocateRecvBuffer(sizeof(DT) * total_recv);
        DT* recv_buf = (DT*)ctx.getRecvBuffer();
        for (unsigned int r = 0; r < m_uivSendRankIds.size(); r++)
        {
            unsigned int i = m_uivSendRankIds[r];
            // printf("rank %d receives from %d, after matvec\n",m_uiRank,i);
            MPI_Request* req = new MPI_Request();
            MPI_Irecv(&recv_buf[m_uivSendDofOffset[i]],
                      m_uivSendDofCounts[i] * sizeof(DT),
                      MPI_BYTE,
                      i,
                      m_iCommTag,
                      m_comm,
                      req);
            ctx.getRequestList().push_back(req);
        }
    }

    // send data of ghost DoFs to ranks owning the DoFs
    if (total_send > 0)
    {
        ctx.allocateSendBuffer(sizeof(DT) * total_send);
        DT* send_buf = (DT*)ctx.getSendBuffer();

        // pre-ghost DoFs
        for (LI i = 0; i < m_uiNumPreGhostDofs; i++)
        {
            send_buf[i] = vec[i];
        }
        // post-ghost DoFs
        for (LI i = m_uiNumPreGhostDofs + m_uiNumDofs;
             i < m_uiNumPreGhostDofs + m_uiNumDofs + m_uiNumPostGhostDofs;
             i++)
        {
            send_buf[i - m_uiNumDofs] = vec[i];
        }
        for (unsigned int r = 0; r < m_uivRecvRankIds.size(); r++)
        {
            unsigned int i = m_uivRecvRankIds[r];
            // printf("rank %d sends to %d, after matvec\n",m_uiRank,i);
            MPI_Request* req = new MPI_Request();
            MPI_Isend(&send_buf[m_uivRecvDofOffset[i]],
                      m_uivRecvDofCounts[i] * sizeof(DT),
                      MPI_BYTE,
                      i,
                      m_iCommTag,
                      m_comm,
                      req);
            ctx.getRequestList().push_back(req);
        }
    }
    m_vAsyncCtx.push_back(ctx);
    m_iCommTag++; // get a different value if we have another ghost_exchange for a different vec

    return Error::SUCCESS;
} // ghost_send_begin

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::ghost_send_end(DT* vec)
{
    if (m_uiSize == 1)
        return Error::SUCCESS;

    const std::vector<LI>& m_uivSendDofCounts = m_maps.get_SendDofCounts();
    const std::vector<LI>& m_uivSendDofIds    = m_maps.get_SendDofIds();
    const std::vector<LI>& m_uivSendDofOffset = m_maps.get_SendDofOffset();

    unsigned int ctx_index;
    for (unsigned i = 0; i < m_vAsyncCtx.size(); i++)
    {
        if (vec == (DT*)m_vAsyncCtx[i].getBuffer())
        {
            ctx_index = i;
            break;
        }
    }
    AsyncExchangeCtx ctx = m_vAsyncCtx[ctx_index];
    auto num_req         = ctx.getRequestList().size();

    MPI_Status sts;
    for (std::size_t i = 0; i < num_req; i++)
    {
        MPI_Wait(ctx.getRequestList()[i], &sts);
    }

    // const unsigned  int total_recv = m_uivSendDofOffset[m_uiSize-1] +
    // m_uivSendDofCounts[m_uiSize-1];
    DT* recv_buf = (DT*)ctx.getRecvBuffer();

    // accumulate the received data at the positions that I sent out before doing matvec
    // these positions are indicated in m_uivSendDofIds[]
    // note: size of recv_buf[] = size of m_uivSendDofIds[]
    for (unsigned int i = 0; i < m_uiSize; i++)
    {
        for (LI j = 0; j < m_uivSendDofCounts[i]; j++)
        {
            // bug fixed on 2020.05.23:
            // vec[m_uivSendDofIds[m_uivSendDofOffset[i]] + j] += recv_buf[m_uivSendDofOffset[i] +
            // j];
            vec[m_uivSendDofIds[m_uivSendDofOffset[i] + j]] += recv_buf[m_uivSendDofOffset[i] + j];
        }
    }

    // free memory of send and receive buffers of ctx
    ctx.deAllocateRecvBuffer();
    ctx.deAllocateSendBuffer();

    // erase the contex associated wit ctx in m_vAsyncCtx
    m_vAsyncCtx.erase(m_vAsyncCtx.begin() + ctx_index);

    return Error::SUCCESS;
} // ghost_send_end

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::matvec(DT* v, const DT* u, bool isGhosted)
{
#ifdef AMAT_PROFILER
    timing_aMat[static_cast<int>(PROFILER::MATVEC)].start();
#endif
    if (isGhosted)
    {
// std::cout << "GHOSTED MATVEC" << std::endl;
#ifdef HYBRID_PARALLEL
        matvec_ghosted_OMP(v, (DT*)u);

#else
        matvec_ghosted_noOMP(v, (DT*)u);
#endif
    }
    else
    {
        // std::cout << "NON GHOSTED MATVEC" << std::endl;
        DT* gv;
        DT* gu;
        // allocate memory for gv and gu including ghost dof's
        create_vec_mf(gv, true, 0.0);
        create_vec_mf(gu, true, 0.0);
        // copy u to gu
        local_to_ghost(gu, u);

#ifdef HYBRID_PARALLEL
        matvec_ghosted_OMP(gv, (DT*)gu);
#else
        matvec_ghosted_noOMP(gv, (DT*)gu);
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
} // matvec

#ifdef HYBRID_PARALLEL

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::matvec_ghosted_OMP(DT* v, DT* u)
{

    const LI m_uiDofPostGhostEnd    = m_maps.get_DofPostGhostEnd();
    const LI m_uiNumElems           = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem = m_maps.get_DofsPerElem();
    LI** const m_uipLocalMap        = m_maps.get_LocalMap();

    const std::vector<LI>& m_uivIndependentElem = m_maps.get_independentElem();
    const std::vector<LI>& m_uivDependentElem   = m_maps.get_dependentElem();

    // initialize v (size of v = m_uiNodesPostGhostEnd = m_uiNumDofsTotal)
    for (LI i = 0; i < m_uiDofPostGhostEnd; i++)
    {
        v[i] = 0.0;
    }

    // send data from owned nodes to ghost nodes (of other processors) to get ready for computing v
    // = Ku
    ghost_receive_begin(u);

// ========================== multiply [ve] = [ke][ue] for all elements ============================
// while waiting for communication, go ahead for independent elements
#pragma omp parallel
    {
        LI rowID, colID;
        LI blocks_dim, num_dofs_per_block;

        // get thread id
        const unsigned int tId = omp_get_thread_num();

// number of pads used in padding
#ifdef VECTORIZED_OPENMP_ALIGNED
        unsigned int nPads = 0;
#endif

        // allocate private ve and ue (if not allocated yet)
        if (m_veBufs[tId] == nullptr)
        {
#ifdef VECTORIZED_OPENMP_ALIGNED
            // allocate and align ve = MaxDofsPerBlock + MaxNumPads, ue as normal
            m_veBufs[tId] = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock + m_uiMaxNumPads);
            // m_ueBufs[tId] = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock);
            m_ueBufs[tId] = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
#else
            // allocate ve and ue without alignment
            m_veBufs[tId] = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
            m_ueBufs[tId] = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
#endif
        }
        DT* ueLocal = m_ueBufs[tId];
        DT* veLocal = m_veBufs[tId];

#pragma omp for
        for (LI i = 0; i < m_uivIndependentElem.size(); i++)
        {
            // independent element id
            LI eid = m_uivIndependentElem[i];

            // get number of blocks of element eid
            blocks_dim = (LI)sqrt(m_epMat[eid].size());

            // number of dofs per block must be the same for all blocks
            num_dofs_per_block = m_uiDofsPerElem[eid] / blocks_dim;

// compute needed pads added to the end of each column of block matrix
#ifdef VECTORIZED_OPENMP_ALIGNED
            // nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
            if ((num_dofs_per_block % (ALIGNMENT / sizeof(DT))) != 0)
            {
                nPads = (ALIGNMENT / sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT / sizeof(DT)));
            }
#endif

            for (LI block_i = 0; block_i < blocks_dim; block_i++)
            {

                LI block_row_offset = block_i * num_dofs_per_block;

                for (LI block_j = 0; block_j < blocks_dim; block_j++)
                {

                    LI block_ID         = block_i * blocks_dim + block_j;
                    LI block_col_offset = block_j * num_dofs_per_block;

                    if (m_epMat[eid][block_ID] != nullptr)
                    {
                        // extract block-element vector ue from structure vector u, and initialize
                        // ve
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            colID      = m_uipLocalMap[eid][block_col_offset + c];
                            ueLocal[c] = u[colID];
                            veLocal[c] = 0.0;
                        }

// ve = elemental matrix * ue
#ifdef VECTORIZED_AVX512
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            unsigned int remain      = num_dofs_per_block % SIMD_LENGTH;
                            unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                            DT* x          = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                            const DT alpha = ueLocal[c];
                            DT* y          = veLocal;
                            // broadcast alpha to form alpha vector
                            __m512d alphaVec = _mm512_set1_pd(alpha);
                            // vector operation y += alpha * x;
                            for (LI j = 0; j < n_intervals; j++)
                            {
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
                            if (remain != 0)
                            {
                                double* xVec_remain = new double[SIMD_LENGTH];
                                double* yVec_remain = new double[SIMD_LENGTH];
                                for (LI j = 0; j < remain; j++)
                                {
                                    xVec_remain[j] = x[j + n_intervals * SIMD_LENGTH];
                                    yVec_remain[j] = y[j + n_intervals * SIMD_LENGTH];
                                }
                                for (unsigned int j = remain; j < SIMD_LENGTH; j++)
                                {
                                    xVec_remain[j] = 0.0;
                                    yVec_remain[j] = 0.0;
                                }
                                __m512d xVec = _mm512_loadu_pd(&xVec_remain[0]);
                                __m512d yVec = _mm512_loadu_pd(&yVec_remain[0]);
                                __m512d tVec = _mm512_mul_pd(xVec, alphaVec);
                                yVec         = _mm512_add_pd(tVec, yVec);
                                _mm512_storeu_pd(&yVec_remain[0], yVec);
                                for (unsigned int j = 0; j < remain; j++)
                                {
                                    y[j + n_intervals * SIMD_LENGTH] = yVec_remain[j];
                                }
                                delete[] xVec_remain;
                                delete[] yVec_remain;
                            }
                        }

#elif VECTORIZED_AVX256
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            unsigned int remain      = num_dofs_per_block % SIMD_LENGTH;
                            unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                            DT* x          = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                            const DT alpha = ueLocal[c];
                            DT* y          = veLocal;
                            // broadcast alpha to form alpha vector
                            __m256d alphaVec = _mm256_set1_pd(alpha);
                            // vector operation y += alpha * x;
                            for (LI i = 0; i < n_intervals; i++)
                            {
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
                            if (remain != 0)
                            {

                                double* xVec_remain = new double[SIMD_LENGTH];
                                double* yVec_remain = new double[SIMD_LENGTH];
                                for (unsigned int i = 0; i < remain; i++)
                                {
                                    xVec_remain[i] = x[i + n_intervals * SIMD_LENGTH];
                                    yVec_remain[i] = y[i + n_intervals * SIMD_LENGTH];
                                }
                                for (unsigned int i = remain; i < SIMD_LENGTH; i++)
                                {
                                    xVec_remain[i] = 0.0;
                                    yVec_remain[i] = 0.0;
                                }
                                __m256d xVec = _mm256_load_pd(&xVec_remain[0]);
                                __m256d yVec = _mm256_load_pd(&yVec_remain[0]);
                                __m256d tVec = _mm256_mul_pd(xVec, alphaVec);
                                yVec         = _mm256_add_pd(tVec, yVec);
                                _mm256_storeu_pd(&yVec_remain[0], yVec);
                                for (unsigned int i = 0; i < remain; i++)
                                {
                                    y[i + n_intervals * SIMD_LENGTH] = yVec_remain[i];
                                }
                                delete[] xVec_remain;
                                delete[] yVec_remain;
                            }
                        }

#elif VECTORIZED_OPENMP
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            const DT alpha = ueLocal[c];
                            const DT* x    = &m_epMat[eid][block_ID][c * num_dofs_per_block];
#pragma omp simd safelen(512)
                            for (LI r = 0; r < num_dofs_per_block; r++)
                            {
                                veLocal[r] += alpha * x[r];
                            }
                        }

#elif VECTORIZED_OPENMP_ALIGNED
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            const DT alpha = ueLocal[c];
                            const DT* x = &m_epMat[eid][block_ID][c * (num_dofs_per_block + nPads)];
#pragma omp simd aligned(x, veLocal : ALIGNMENT) safelen(512)
                            for (LI r = 0; r < (num_dofs_per_block + nPads); r++)
                            {
                                veLocal[r] += alpha * x[r];
                            }
                        }

#else
                        //#pragma novector noparallel nounroll
                        for (LI r = 0; r < num_dofs_per_block; r++)
                        {
                            //#pragma novector noparallel nounroll
                            for (LI c = 0; c < num_dofs_per_block; c++)
                            {
                                veLocal[r] +=
                                  m_epMat[eid][block_ID][(r * num_dofs_per_block) + c] * ueLocal[c];
                            }
                        }

#endif

                        // accumulate element vector ve to structure vector v
                        for (LI r = 0; r < num_dofs_per_block; r++)
                        {
                            rowID = m_uipLocalMap[eid][block_row_offset + r];
#pragma omp atomic
                            v[rowID] += veLocal[r];
                        }
                    } // if block_ID is not nullptr

                } // loop over blocks j

            } // loop over blocks i
        }     // loop over elements
    }         // pragma omp parallel

    // finishing communication
    ghost_receive_end(u);

// ready to go for dependent elements
#pragma omp parallel
    {
        LI rowID, colID;
        LI blocks_dim, num_dofs_per_block;

        // get thread id
        const unsigned int tId = omp_get_thread_num();

// number of pads used in padding
#ifdef VECTORIZED_OPENMP_ALIGNED
        unsigned int nPads = 0;
#endif

        // allocate private ve and ue (if not allocated yet)
        if (m_veBufs[tId] == nullptr)
        {
#ifdef VECTORIZED_OPENMP_ALIGNED
            // allocate and align ve = MaxDofsPerBlock + MaxNumPads, ue as normal
            m_veBufs[tId] = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock + m_uiMaxNumPads);
            // m_ueBufs[tId] = create_aligned_array(ALIGNMENT, m_uiMaxDofsPerBlock);
            m_ueBufs[tId] = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
#else
            // allocate ve and ue without alignment
            m_veBufs[tId] = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
            m_ueBufs[tId] = (DT*)malloc(m_uiMaxDofsPerBlock * sizeof(DT));
#endif
        }
        DT* ueLocal = m_ueBufs[tId];
        DT* veLocal = m_veBufs[tId];

#pragma omp for
        for (LI i = 0; i < m_uivDependentElem.size(); i++)
        {
            // dependent element id
            LI eid = m_uivDependentElem[i];

            // get number of blocks of element eid
            blocks_dim = (LI)sqrt(m_epMat[eid].size());

            // number of dofs per block must be the same for all blocks
            num_dofs_per_block = m_uiDofsPerElem[eid] / blocks_dim;

// compute needed pads added to the end of each column of block matrix
#ifdef VECTORIZED_OPENMP_ALIGNED
            // nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
            if ((num_dofs_per_block % (ALIGNMENT / sizeof(DT))) != 0)
            {
                nPads = (ALIGNMENT / sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT / sizeof(DT)));
            }
#endif

            for (LI block_i = 0; block_i < blocks_dim; block_i++)
            {

                LI block_row_offset = block_i * num_dofs_per_block;

                for (LI block_j = 0; block_j < blocks_dim; block_j++)
                {

                    LI block_ID         = block_i * blocks_dim + block_j;
                    LI block_col_offset = block_j * num_dofs_per_block;

                    if (m_epMat[eid][block_ID] != nullptr)
                    {
                        // extract block-element vector ue from structure vector u, and initialize
                        // ve
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            colID      = m_uipLocalMap[eid][block_col_offset + c];
                            ueLocal[c] = u[colID];
                            veLocal[c] = 0.0;
                        }

// ve = elemental matrix * ue
#ifdef VECTORIZED_AVX512
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            unsigned int remain      = num_dofs_per_block % SIMD_LENGTH;
                            unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                            DT* x          = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                            const DT alpha = ueLocal[c];
                            DT* y          = veLocal;
                            // broadcast alpha to form alpha vector
                            __m512d alphaVec = _mm512_set1_pd(alpha);
                            // vector operation y += alpha * x;
                            for (LI j = 0; j < n_intervals; j++)
                            {
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
                            if (remain != 0)
                            {
                                double* xVec_remain = new double[SIMD_LENGTH];
                                double* yVec_remain = new double[SIMD_LENGTH];
                                for (LI j = 0; j < remain; j++)
                                {
                                    xVec_remain[j] = x[j + n_intervals * SIMD_LENGTH];
                                    yVec_remain[j] = y[j + n_intervals * SIMD_LENGTH];
                                }
                                for (unsigned int j = remain; j < SIMD_LENGTH; j++)
                                {
                                    xVec_remain[j] = 0.0;
                                    yVec_remain[j] = 0.0;
                                }
                                __m512d xVec = _mm512_loadu_pd(&xVec_remain[0]);
                                __m512d yVec = _mm512_loadu_pd(&yVec_remain[0]);
                                __m512d tVec = _mm512_mul_pd(xVec, alphaVec);
                                yVec         = _mm512_add_pd(tVec, yVec);
                                _mm512_storeu_pd(&yVec_remain[0], yVec);
                                for (unsigned int j = 0; j < remain; j++)
                                {
                                    y[j + n_intervals * SIMD_LENGTH] = yVec_remain[j];
                                }
                                delete[] xVec_remain;
                                delete[] yVec_remain;
                            }
                        }

#elif VECTORIZED_AVX256
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            unsigned int remain      = num_dofs_per_block % SIMD_LENGTH;
                            unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                            DT* x          = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                            const DT alpha = ueLocal[c];
                            DT* y          = veLocal;
                            // broadcast alpha to form alpha vector
                            __m256d alphaVec = _mm256_set1_pd(alpha);
                            // vector operation y += alpha * x;
                            for (LI i = 0; i < n_intervals; i++)
                            {
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
                            if (remain != 0)
                            {

                                double* xVec_remain = new double[SIMD_LENGTH];
                                double* yVec_remain = new double[SIMD_LENGTH];
                                for (unsigned int i = 0; i < remain; i++)
                                {
                                    xVec_remain[i] = x[i + n_intervals * SIMD_LENGTH];
                                    yVec_remain[i] = y[i + n_intervals * SIMD_LENGTH];
                                }
                                for (unsigned int i = remain; i < SIMD_LENGTH; i++)
                                {
                                    xVec_remain[i] = 0.0;
                                    yVec_remain[i] = 0.0;
                                }
                                __m256d xVec = _mm256_load_pd(&xVec_remain[0]);
                                __m256d yVec = _mm256_load_pd(&yVec_remain[0]);
                                __m256d tVec = _mm256_mul_pd(xVec, alphaVec);
                                yVec         = _mm256_add_pd(tVec, yVec);
                                _mm256_storeu_pd(&yVec_remain[0], yVec);
                                for (unsigned int i = 0; i < remain; i++)
                                {
                                    y[i + n_intervals * SIMD_LENGTH] = yVec_remain[i];
                                }
                                delete[] xVec_remain;
                                delete[] yVec_remain;
                            }
                        }

#elif VECTORIZED_OPENMP
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            const DT alpha = ueLocal[c];
                            const DT* x    = &m_epMat[eid][block_ID][c * num_dofs_per_block];
#pragma omp simd safelen(512)
                            for (LI r = 0; r < num_dofs_per_block; r++)
                            {
                                veLocal[r] += alpha * x[r];
                            }
                        }

#elif VECTORIZED_OPENMP_ALIGNED
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            const DT alpha = ueLocal[c];
                            const DT* x = &m_epMat[eid][block_ID][c * (num_dofs_per_block + nPads)];
#pragma omp simd aligned(x, veLocal : ALIGNMENT) safelen(512)
                            for (LI r = 0; r < (num_dofs_per_block + nPads); r++)
                            {
                                veLocal[r] += alpha * x[r];
                            }
                        }

#else
                        //#pragma novector noparallel nounroll
                        for (LI r = 0; r < num_dofs_per_block; r++)
                        {
                            //#pragma novector noparallel nounroll
                            for (LI c = 0; c < num_dofs_per_block; c++)
                            {
                                veLocal[r] +=
                                  m_epMat[eid][block_ID][(r * num_dofs_per_block) + c] * ueLocal[c];
                            }
                        }

#endif

                        // accumulate element vector ve to structure vector v
                        for (LI r = 0; r < num_dofs_per_block; r++)
                        {
                            rowID = m_uipLocalMap[eid][block_row_offset + r];
#pragma omp atomic
                            v[rowID] += veLocal[r];
                        }
                    } // if block_ID is not nullptr

                } // loop over blocks j

            } // loop over blocks i
        }     // loop over elements
    }         // pragma omp parallel

    // send data from ghost nodes back to owned nodes after computing v
    ghost_send_begin(v);
    ghost_send_end(v);

    return Error::SUCCESS;
} // matvec_ghosted_OMP

#else

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::matvec_ghosted_noOMP(DT* v, DT* u)
{

    const LI m_uiDofPostGhostEnd = m_maps.get_DofPostGhostEnd();
    const LI m_uiNumElems = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem = m_maps.get_DofsPerElem();
    LI** const m_uipLocalMap = m_maps.get_LocalMap();

    const std::vector<LI>& m_uivIndependentElem = m_maps.get_independentElem();
    const std::vector<LI>& m_uivDependentElem = m_maps.get_dependentElem();

    LI blocks_dim, num_dofs_per_block;

    // initialize v (size of v = m_uiNodesPostGhostEnd = m_uiNumDofsTotal)
    for (LI i = 0; i < m_uiDofPostGhostEnd; i++)
    {
        v[i] = 0.0;
    }

    LI rowID, colID;
// number of pads used in padding
#ifdef VECTORIZED_OPENMP_ALIGNED
    unsigned int nPads = 0;
#endif

    // send data from owned nodes to ghost nodes (of other processors) to get ready for computing v
    // = Ku
    ghost_receive_begin(u);

    // while waiting for communication, go ahead for independent elements
    // multiply [ve] = [ke][ue] for all elements
    for (LI i = 0; i < m_uivIndependentElem.size(); i++)
    {
        // independent element id
        LI eid = m_uivIndependentElem[i];

        blocks_dim = (LI)sqrt(m_epMat[eid].size());

        // number of dofs per block must be the same for all blocks
        num_dofs_per_block = m_uiDofsPerElem[eid] / blocks_dim;

// compute number of paddings inserted to the end of each column of block matrix
#ifdef VECTORIZED_OPENMP_ALIGNED
        // nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
        if ((num_dofs_per_block % (ALIGNMENT / sizeof(DT))) != 0)
        {
            nPads = (ALIGNMENT / sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT / sizeof(DT)));
        }
#endif

        for (LI block_i = 0; block_i < blocks_dim; block_i++)
        {

            LI block_row_offset = block_i * num_dofs_per_block;

            for (LI block_j = 0; block_j < blocks_dim; block_j++)
            {

                LI block_col_offset = block_j * num_dofs_per_block;
                LI block_ID = block_i * blocks_dim + block_j;

                if (m_epMat[eid][block_ID] != nullptr)
                {
#ifdef AMAT_PROFILER
                    timing_aMat[static_cast<int>(PROFILER::MATVEC_MUL)].start();
#endif
                    // extract block-element vector ue from structure vector u, and initialize ve
                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {
                        colID = m_uipLocalMap[eid][block_col_offset + c];
                        ue[c] = u[colID];
                        ve[c] = 0.0;
                    }

// ve = elemental matrix * ue
#ifdef VECTORIZED_AVX512
                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {
                        unsigned int remain = num_dofs_per_block % SIMD_LENGTH;
                        unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                        DT* x = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                        const DT alpha = ue[c];
                        DT* y = ve;
                        // broadcast alpha to form alpha vector
                        __m512d alphaVec = _mm512_set1_pd(alpha);
                        // vector operation y += alpha * x;
                        for (LI j = 0; j < n_intervals; j++)
                        {
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
                        if (remain != 0)
                        {
                            double* xVec_remain = new double[SIMD_LENGTH];
                            double* yVec_remain = new double[SIMD_LENGTH];
                            for (LI j = 0; j < remain; j++)
                            {
                                xVec_remain[j] = x[j + n_intervals * SIMD_LENGTH];
                                yVec_remain[j] = y[j + n_intervals * SIMD_LENGTH];
                            }
                            for (unsigned int j = remain; j < SIMD_LENGTH; j++)
                            {
                                xVec_remain[j] = 0.0;
                                yVec_remain[j] = 0.0;
                            }
                            __m512d xVec = _mm512_loadu_pd(&xVec_remain[0]);
                            __m512d yVec = _mm512_loadu_pd(&yVec_remain[0]);
                            __m512d tVec = _mm512_mul_pd(xVec, alphaVec);
                            yVec = _mm512_add_pd(tVec, yVec);
                            _mm512_storeu_pd(&yVec_remain[0], yVec);
                            for (unsigned int j = 0; j < remain; j++)
                            {
                                y[j + n_intervals * SIMD_LENGTH] = yVec_remain[j];
                            }
                            delete[] xVec_remain;
                            delete[] yVec_remain;
                        }
                    }

#elif VECTORIZED_AVX256

                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {

                        unsigned int remain      = num_dofs_per_block % SIMD_LENGTH;
                        unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                        DT* x                    = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                        const DT alpha           = ue[c];
                        DT* y                    = ve;
                        // broadcast alpha to form alpha vector
                        __m256d alphaVec = _mm256_set1_pd(alpha);
                        // vector operation y += alpha * x;
                        for (LI i = 0; i < n_intervals; i++)
                        {
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
                        if (remain != 0)
                        {

                            double* xVec_remain = new double[SIMD_LENGTH];
                            double* yVec_remain = new double[SIMD_LENGTH];
                            for (unsigned int i = 0; i < remain; i++)
                            {
                                xVec_remain[i] = x[i + n_intervals * SIMD_LENGTH];
                                yVec_remain[i] = y[i + n_intervals * SIMD_LENGTH];
                            }
                            for (unsigned int i = remain; i < SIMD_LENGTH; i++)
                            {
                                xVec_remain[i] = 0.0;
                                yVec_remain[i] = 0.0;
                            }
                            __m256d xVec = _mm256_load_pd(&xVec_remain[0]);
                            __m256d yVec = _mm256_load_pd(&yVec_remain[0]);
                            __m256d tVec = _mm256_mul_pd(xVec, alphaVec);
                            yVec         = _mm256_add_pd(tVec, yVec);
                            _mm256_storeu_pd(&yVec_remain[0], yVec);
                            for (unsigned int i = 0; i < remain; i++)
                            {
                                y[i + n_intervals * SIMD_LENGTH] = yVec_remain[i];
                            }
                            delete[] xVec_remain;
                            delete[] yVec_remain;
                        }
                    }
#elif VECTORIZED_OPENMP
                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {
                        const DT alpha = ue[c];
                        const DT* x    = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                        DT* y          = ve;
#pragma omp simd safelen(512)
                        for (LI r = 0; r < num_dofs_per_block; r++)
                        {
                            y[r] += alpha * x[r];
                        }
                    }
#elif VECTORIZED_OPENMP_ALIGNED
                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {
                        const DT alpha = ue[c];
                        const DT* x    = &m_epMat[eid][block_ID][c * (num_dofs_per_block + nPads)];
                        DT* y          = ve;
#pragma omp simd aligned(x, y : ALIGNMENT) safelen(512)
                        for (LI r = 0; r < (num_dofs_per_block + nPads); r++)
                        {
                            y[r] += alpha * x[r];
                        }
                    }
#else
                    //#pragma novector noparallel nounroll
                    for (LI r = 0; r < num_dofs_per_block; r++)
                    {
                        //#pragma novector noparallel nounroll
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
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
                    for (LI r = 0; r < num_dofs_per_block; r++)
                    {
                        rowID = m_uipLocalMap[eid][block_row_offset + r];
                        v[rowID] += ve[r];
                    }
#ifdef AMAT_PROFILER
                    timing_aMat[static_cast<int>(PROFILER::MATVEC_ACC)].stop();
#endif
                } // if block_ID is not null_ptr

            } // loop over blocks j
        }     // loop over blocks i
    }         // loop over elements

    // finishing communication
    ghost_receive_end(u);

    // ready to go for dependent elements
    // multiply [ve] = [ke][ue] for all elements
    for (LI i = 0; i < m_uivDependentElem.size(); i++)
    {
        // dependent element id
        LI eid = m_uivDependentElem[i];

        blocks_dim = (LI)sqrt(m_epMat[eid].size());

        // number of dofs per block must be the same for all blocks
        num_dofs_per_block = m_uiDofsPerElem[eid] / blocks_dim;

// compute number of paddings inserted to the end of each column of block matrix
#ifdef VECTORIZED_OPENMP_ALIGNED
        // nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
        if ((num_dofs_per_block % (ALIGNMENT / sizeof(DT))) != 0)
        {
            nPads = (ALIGNMENT / sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT / sizeof(DT)));
        }
#endif

        for (LI block_i = 0; block_i < blocks_dim; block_i++)
        {

            LI block_row_offset = block_i * num_dofs_per_block;

            for (LI block_j = 0; block_j < blocks_dim; block_j++)
            {

                LI block_ID = block_i * blocks_dim + block_j;
                LI block_col_offset = block_j * num_dofs_per_block;

                if (m_epMat[eid][block_ID] != nullptr)
                {
#ifdef AMAT_PROFILER
                    timing_aMat[static_cast<int>(PROFILER::MATVEC_MUL)].start();
#endif
                    // extract block-element vector ue from structure vector u, and initialize ve
                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {
                        colID = m_uipLocalMap[eid][block_col_offset + c];
                        ue[c] = u[colID];
                        ve[c] = 0.0;
                    }

// ve = elemental matrix * ue
#ifdef VECTORIZED_AVX512
                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {
                        unsigned int remain = num_dofs_per_block % SIMD_LENGTH;
                        unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                        DT* x = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                        const DT alpha = ue[c];
                        DT* y = ve;
                        // broadcast alpha to form alpha vector
                        __m512d alphaVec = _mm512_set1_pd(alpha);
                        // vector operation y += alpha * x;
                        for (LI j = 0; j < n_intervals; j++)
                        {
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
                        if (remain != 0)
                        {
                            double* xVec_remain = new double[SIMD_LENGTH];
                            double* yVec_remain = new double[SIMD_LENGTH];
                            for (LI j = 0; j < remain; j++)
                            {
                                xVec_remain[j] = x[j + n_intervals * SIMD_LENGTH];
                                yVec_remain[j] = y[j + n_intervals * SIMD_LENGTH];
                            }
                            for (unsigned int j = remain; j < SIMD_LENGTH; j++)
                            {
                                xVec_remain[j] = 0.0;
                                yVec_remain[j] = 0.0;
                            }
                            __m512d xVec = _mm512_loadu_pd(&xVec_remain[0]);
                            __m512d yVec = _mm512_loadu_pd(&yVec_remain[0]);
                            __m512d tVec = _mm512_mul_pd(xVec, alphaVec);
                            yVec = _mm512_add_pd(tVec, yVec);
                            _mm512_storeu_pd(&yVec_remain[0], yVec);
                            for (unsigned int j = 0; j < remain; j++)
                            {
                                y[j + n_intervals * SIMD_LENGTH] = yVec_remain[j];
                            }
                            delete[] xVec_remain;
                            delete[] yVec_remain;
                        }
                    }

#elif VECTORIZED_AVX256

                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {

                        unsigned int remain      = num_dofs_per_block % SIMD_LENGTH;
                        unsigned int n_intervals = num_dofs_per_block / SIMD_LENGTH;
                        DT* x                    = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                        const DT alpha           = ue[c];
                        DT* y                    = ve;
                        // broadcast alpha to form alpha vector
                        __m256d alphaVec = _mm256_set1_pd(alpha);
                        // vector operation y += alpha * x;
                        for (LI i = 0; i < n_intervals; i++)
                        {
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
                        if (remain != 0)
                        {

                            double* xVec_remain = new double[SIMD_LENGTH];
                            double* yVec_remain = new double[SIMD_LENGTH];
                            for (unsigned int i = 0; i < remain; i++)
                            {
                                xVec_remain[i] = x[i + n_intervals * SIMD_LENGTH];
                                yVec_remain[i] = y[i + n_intervals * SIMD_LENGTH];
                            }
                            for (unsigned int i = remain; i < SIMD_LENGTH; i++)
                            {
                                xVec_remain[i] = 0.0;
                                yVec_remain[i] = 0.0;
                            }
                            __m256d xVec = _mm256_load_pd(&xVec_remain[0]);
                            __m256d yVec = _mm256_load_pd(&yVec_remain[0]);
                            __m256d tVec = _mm256_mul_pd(xVec, alphaVec);
                            yVec         = _mm256_add_pd(tVec, yVec);
                            _mm256_storeu_pd(&yVec_remain[0], yVec);
                            for (unsigned int i = 0; i < remain; i++)
                            {
                                y[i + n_intervals * SIMD_LENGTH] = yVec_remain[i];
                            }
                            delete[] xVec_remain;
                            delete[] yVec_remain;
                        }
                    }
#elif VECTORIZED_OPENMP
                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {
                        const DT alpha = ue[c];
                        const DT* x    = &m_epMat[eid][block_ID][c * num_dofs_per_block];
                        DT* y          = ve;
#pragma omp simd safelen(512)
                        for (LI r = 0; r < num_dofs_per_block; r++)
                        {
                            y[r] += alpha * x[r];
                        }
                    }
#elif VECTORIZED_OPENMP_ALIGNED
                    for (LI c = 0; c < num_dofs_per_block; c++)
                    {
                        const DT alpha = ue[c];
                        const DT* x    = &m_epMat[eid][block_ID][c * (num_dofs_per_block + nPads)];
                        DT* y          = ve;
#pragma omp simd aligned(x, y : ALIGNMENT) safelen(512)
                        for (LI r = 0; r < (num_dofs_per_block + nPads); r++)
                        {
                            y[r] += alpha * x[r];
                        }
                    }
#else
                    //#pragma novector noparallel nounroll
                    for (LI r = 0; r < num_dofs_per_block; r++)
                    {
                        //#pragma novector noparallel nounroll
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
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
                    for (LI r = 0; r < num_dofs_per_block; r++)
                    {
                        rowID = m_uipLocalMap[eid][block_row_offset + r];
                        v[rowID] += ve[r];
                    }
#ifdef AMAT_PROFILER
                    timing_aMat[static_cast<int>(PROFILER::MATVEC_ACC)].stop();
#endif
                } // if block_ID is not null_ptr

            } // loop over blocks j

        } // loop over blocks i
    }     // loop over elements

    // send data from ghost nodes back to owned nodes after computing v
    ghost_send_begin(v);
    ghost_send_end(v);

    return Error::SUCCESS;
} // matvec_ghosted_noOMP

#endif

template<typename DT, typename GI, typename LI>
PetscErrorCode aMatFree<DT, GI, LI>::MatMult_mf(Mat A, Vec u, Vec v)
{

    const std::vector<GI>& ownedConstrainedDofs = m_maps.get_ownedConstrainedDofs();
    const std::vector<GI>& m_ulvLocalDofScan    = m_maps.get_LocalDofScan();
    const LI m_uiNumPreGhostDofs                = m_maps.get_NumPreGhostDofs();

    PetscScalar* vv; // this allows vv to be considered as regular vector
    PetscScalar* uu;

    LI local_Id;
    // VecZeroEntries(v);

    VecGetArray(v, &vv);
    VecGetArrayRead(u, (const PetscScalar**)&uu);

    DT* vvg = m_dpVvg;
    DT* uug = m_dpUug;

    // copy data of uu (not-ghosted) to uug
    local_to_ghost(uug, uu);

    // apply BC: save value of U_c, then make U_c = 0
    const auto numConstraints = ownedConstrainedDofs.size();
    DT* Uc                    = m_dpUc;
    for (LI nid = 0; nid < numConstraints; nid++)
    {
        local_Id = ownedConstrainedDofs[nid] - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;
        Uc[nid]  = uug[local_Id];
        uug[local_Id] = 0.0;
    }
    // end of apply BC

    // vvg = K * uug
    matvec(vvg, uug, true); // this gives V_f = (K_ff * U_f) + (K_fc * 0) = K_ff * U_f

    // apply BC: now set V_c = U_c which was saved in U'_c
    for (LI nid = 0; nid < numConstraints; nid++)
    {
        local_Id = ownedConstrainedDofs[nid] - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;
        vvg[local_Id] = Uc[nid];
    }
    // end of apply BC

    ghost_to_local(vv, vvg);

    VecRestoreArray(v, &vv);

    return 0;
} // MatMult_mf

template<typename DT, typename GI, typename LI>
PetscErrorCode aMatFree<DT, GI, LI>::MatGetDiagonal_mf(Mat A, Vec d)
{
    // point to data of PETSc vector d
    PetscScalar* dd;
    VecGetArray(d, &dd);

    // allocate regular vector used for get_diagonal() in aMatFree
    double* ddg;
    create_vec_mf(ddg, true, 0);

    // get diagonal of matrix and put into ddg
    mat_get_diagonal(ddg, true);

    // copy ddg (ghosted) into (non-ghosted) dd
    ghost_to_local(dd, ddg);

    // deallocate ddg
    destroy_vec(ddg);

    // update data of PETSc vector d
    VecRestoreArray(d, &dd);

    // apply Dirichlet boundary condition
    apply_bc_diagonal(d);

    VecAssemblyBegin(d);
    VecAssemblyEnd(d);

    return 0;
} // MatGetDiagonal_mf

template<typename DT, typename GI, typename LI>
PetscErrorCode aMatFree<DT, GI, LI>::MatGetDiagonalBlock_mf(Mat A, Mat* a)
{

    const LI m_uiNumDofs = m_maps.get_NumDofs();

    // sparse block diagonal matrix
    std::vector<MatRecord<DT, LI>> ddg;
    mat_get_diagonal_block(ddg);

    if(m_pMatBJ==nullptr)
    {
        m_pMatBJ = new Mat();
        MatCreateSeqAIJ(PETSC_COMM_SELF,
                    m_uiNumDofs,
                    m_uiNumDofs,
                    NNZ,
                    PETSC_NULL,
                    m_pMatBJ);

    }

    *a=*m_pMatBJ;
    
    // set values ...
    std::vector<PetscScalar> values;
    std::vector<PetscInt> colIndices;
    PetscInt rowID;

    std::sort(ddg.begin(), ddg.end());

    LI ind = 0;
    while (ind < ddg.size())
    {
        assert(ddg[ind].getRank() == m_uiRank);
        rowID = ddg[ind].getRowId();

        // clear data used for previous rowID
        colIndices.clear();
        values.clear();

        // push the first value
        colIndices.push_back(ddg[ind].getColId());
        values.push_back(ddg[ind].getVal());

        // push other values having same rowID
        while (((ind + 1) < ddg.size()) && (ddg[ind].getRowId() == ddg[ind + 1].getRowId()))
        {
            colIndices.push_back(ddg[ind + 1].getColId());
            values.push_back(ddg[ind + 1].getVal());
            ind++;
        }

        // set values for rowID
        MatSetValues((*a),
                     1,
                     &rowID,
                     static_cast<PetscInt>(colIndices.size()),
                     colIndices.data(),
                     values.data(),
                     INSERT_VALUES);

        // move to next rowID
        ind++;
    }

    // apply boundary condition for block diagonal matrix
    apply_bc_blkdiag(a);

    MatAssemblyBegin((*a), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd((*a), MAT_FINAL_ASSEMBLY);

    return 0;
} // MatGetDiagonalBlock_mf

template<typename DT, typename GI, typename LI>
PetscErrorCode aMatFree<DT, GI, LI>::petscSetValuesInMatrix(Mat mat, std::vector<MatRecord<DT,LI>>& records, InsertMode mode)
{

    if(records.empty())
        return 0;

    // assembly code based on Dendro4
    std::vector<PetscScalar > values;
    std::vector<PetscInt> colIndices;
    
    //Can make it more efficient later.
    if (!records.empty())
    {
        GI* const m_ulpLocal2Global =m_maps.get_Local2Global();
        // GI** const m_ulpMap                      = m_maps.get_Map();
        // LI** const m_uipLocalMap                 = m_maps.get_LocalMap();

        ///TODO: change this to more efficient one later. 
        std::vector<MatRecord<DT,LI>> tmpRecords;
        
        for(LI i=0;i<records.size();i++)
            tmpRecords.push_back(records[i]);
        
        std::swap(tmpRecords,records);
        tmpRecords.clear();
        
        //Sort Order: row first, col next, val last
        std::sort(records.begin(), records.end());
        
        LI currRecord = 0;

        while (currRecord < (records.size() - 1))
        {
            values.push_back(records[currRecord].getVal());
            //colIndices.push_back(static_cast<PetscInt>((records[currRecord].getColDim()) + dof * m_uiLocalToGlobalNodalMap[records[currRecord].getColID()]));
            colIndices.push_back(static_cast<PetscInt>(m_ulpLocal2Global[records[currRecord].getColId()]));
            if ((records[currRecord].getRowId() != records[currRecord + 1].getRowId()))
            {
                PetscInt rowId = static_cast<PetscInt>(m_ulpLocal2Global[records[currRecord].getRowId()]);
                MatSetValues(mat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())),  (&(*values.begin())), mode);

                colIndices.clear();
                values.clear();
            }
            currRecord++;
        } //end while


        PetscInt rowId = static_cast<PetscInt>(m_ulpLocal2Global[records[currRecord].getRowId()]);
        if (values.empty())
        {
            //Last row is different from the previous row
            PetscInt colId = static_cast<PetscInt>(m_ulpLocal2Global[records[currRecord].getColId()]);
            PetscScalar value = records[currRecord].getVal();
            MatSetValues(mat, 1, &rowId, 1, &colId, &value, mode);
        }
        else
        {
            //Last row is same as the previous row
            values.push_back(records[currRecord].getVal());
            colIndices.push_back(static_cast<PetscInt>(m_ulpLocal2Global[records[currRecord].getColId()]));
            MatSetValues(mat, 1, &rowId, colIndices.size(), (&(*colIndices.begin())), (&(*values.begin())), mode);
            colIndices.clear();
            values.clear();
        }
        records.clear();
    } // records not empty

    return 0;

}

template<typename DT, typename GI, typename LI>
PetscErrorCode aMatFree<DT, GI, LI>::MatGetDiagonalBlock_mf_petsc(Mat A, Mat* a)
{

    const LI m_uiNumDofs = m_maps.get_NumDofs();
    const LI m_uiNumElems                    = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem          = m_maps.get_DofsPerElem();
    GI** const m_ulpMap                      = m_maps.get_Map();
    LI** const m_uipLocalMap                 = m_maps.get_LocalMap();
    GI* const m_ulpLocal2Global              = m_maps.get_Local2Global();
    const std::vector<GI>& m_ulvLocalDofScan = m_maps.get_LocalDofScan();
    const LI m_uiDofLocalBegin               = m_maps.get_DofLocalBegin();
    const LI m_uiDofLocalEnd                 = m_maps.get_DofLocalEnd();
    const std::vector<LI> & m_uivIndependentElem = m_maps.get_independentElem();
    const std::vector<LI> & m_uivDependentElem = m_maps.get_dependentElem();
    
    if(!m_uiRank)
        std::cout<<"[aMat] : calling func : "<<__func__<<std::endl;
    
    if(m_pMatBJ==nullptr)
    {
        m_pMatBJ = new Mat();
        MatCreate(m_comm, m_pMatBJ);
        MatSetSizes(*m_pMatBJ, m_uiNumDofs, m_uiNumDofs, PETSC_DECIDE, PETSC_DECIDE);
        if (m_uiSize > 1)
        {
            MatSetType(*m_pMatBJ, MATMPIAIJ);
            MatMPIAIJSetPreallocation(*m_pMatBJ, NNZ, PETSC_NULL, NNZ, PETSC_NULL);
        }
        else
        {
            MatSetType(*m_pMatBJ, MATSEQAIJ);
            MatSeqAIJSetPreallocation(*m_pMatBJ, NNZ, PETSC_NULL);
        }
        // this will disable on preallocation errors (but not good for performance)
        MatSetOption(*m_pMatBJ, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
        MatSetOption(*m_pMatBJ, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);

    }
    
    MatZeroEntries(*m_pMatBJ);
    std::vector<MatRecord<DT,LI>> bj_mat_records;

    for (LI i = 0; i < m_uivIndependentElem.size(); i++)
    {
        // independent element id
        const LI eid = m_uivIndependentElem[i];
        const LI blocks_dim = (LI)sqrt(m_epMat[eid].size());

        // number of dofs per block must be the same for all blocks
        assert(blocks_dim>0);
        const LI num_dofs_per_block = m_uiDofsPerElem[eid] / blocks_dim;

        for (LI block_i = 0; block_i < blocks_dim; block_i++)
        {

            const LI block_row_offset = block_i * num_dofs_per_block;

            for (LI block_j = 0; block_j < blocks_dim; block_j++)
            {
                const LI index            = block_i * blocks_dim + block_j;
                const LI block_col_offset = block_j * num_dofs_per_block;
                const LI block_ID = block_i * blocks_dim + block_j;

                if (m_epMat[eid][block_ID] != nullptr)
                {

                    for (LI r = 0; r < num_dofs_per_block; r++)
                    {
                        const LI row_lid = m_uipLocalMap[eid][block_row_offset + r];
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            const LI col_lid = m_uipLocalMap[eid][block_col_offset + c];
                            bj_mat_records.push_back(MatRecord<DT,LI>(m_uiRank,row_lid,col_lid,m_epMat[eid][index][(r * num_dofs_per_block) + c]));
                        }
                    }
                    
                }
            }   
        }
        
        if(bj_mat_records.size()>500)
        {
            this->petscSetValuesInMatrix(*m_pMatBJ,bj_mat_records,ADD_VALUES);
            bj_mat_records.clear();
        }

    }

    if(!bj_mat_records.empty())
    {
        this->petscSetValuesInMatrix(*m_pMatBJ,bj_mat_records,ADD_VALUES);
        bj_mat_records.clear();
    }

    LI tmp_dof_per_block;
    for (LI i = 0; i < m_uivDependentElem.size(); i++)
    {
        // independent element id
        const LI eid = m_uivDependentElem[i];
        const LI blocks_dim = (LI)sqrt(m_epMat[eid].size());

        // number of dofs per block must be the same for all blocks
        assert(blocks_dim>0);
        const LI num_dofs_per_block = m_uiDofsPerElem[eid] / blocks_dim;
        tmp_dof_per_block=num_dofs_per_block;

        for (LI block_i = 0; block_i < blocks_dim; block_i++)
        {

            const LI block_row_offset = block_i * num_dofs_per_block;
            

            for (LI block_j = 0; block_j < blocks_dim; block_j++)
            {
                const LI index            = block_i * blocks_dim + block_j;
                const LI block_col_offset = block_j * num_dofs_per_block;
                const LI block_ID = block_i * blocks_dim + block_j;

                if (m_epMat[eid][block_ID] != nullptr)
                {

                    for (LI r = 0; r < num_dofs_per_block; r++)
                    {
                        const LI row_lid = m_uipLocalMap[eid][block_row_offset + r];
                        const LI row_rank = m_maps.globalId_2_rank(m_ulpLocal2Global[row_lid]);
                        for (LI c = 0; c < num_dofs_per_block; c++)
                        {
                            const LI col_lid = m_uipLocalMap[eid][block_col_offset + c];
                            const LI col_rank = m_maps.globalId_2_rank(m_ulpLocal2Global[col_lid]);
                            if(col_rank==row_rank)
                             bj_mat_records.push_back(MatRecord<DT,LI>(m_uiRank,row_lid,col_lid,m_epMat[eid][index][(r * num_dofs_per_block) + c]));
                        }
                    }
                    
                }
            }   
        }
        
        if(bj_mat_records.size()>500)
        {
            // std::cout<<"dep dec: "<<bj_mat_records.size() <<" outof :  "<<m_uivDependentElem.size()*(tmp_dof_per_block*tmp_dof_per_block)<<std::endl;
            this->petscSetValuesInMatrix(*m_pMatBJ,bj_mat_records,ADD_VALUES);
            bj_mat_records.clear();
        }

    }

    
    if(!bj_mat_records.empty())
    {
        // std::cout<<"dep dec: "<<bj_mat_records.size() <<" outof :  "<<m_uivDependentElem.size()*(tmp_dof_per_block*tmp_dof_per_block)<<std::endl;
        this->petscSetValuesInMatrix(*m_pMatBJ,bj_mat_records,ADD_VALUES);
        bj_mat_records.clear();
    }

    MatAssemblyBegin((*m_pMatBJ), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd((*m_pMatBJ), MAT_FINAL_ASSEMBLY);

    // apply_bc_blkdiag_petsc(bj_mat_records);
    // if(!bj_mat_records.empty())
    // {
    //     //std::cout<<"dep dec: "<<bj_mat_records.size() <<" outof :  "<<m_uivDependentElem.size()*(tmp_dof_per_block*tmp_dof_per_block)<<std::endl;
    //     this->petscSetValuesInMatrix(*m_pMatBJ,bj_mat_records,INSERT_VALUES);
    //     bj_mat_records.clear();
    // }

    // MatAssemblyBegin((*m_pMatBJ), MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd((*m_pMatBJ), MAT_FINAL_ASSEMBLY);
    
    MatGetDiagonalBlock(*m_pMatBJ, a);

    return 0;

} // MatGetDiagonalBlock_mf

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::apply_bc_diagonal(Vec diag)
{

    const LI m_uiNumElems            = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem  = m_maps.get_DofsPerElem();
    unsigned int** const m_uipBdrMap = m_maps.get_BdrMap();
    GI** const m_ulpMap              = m_maps.get_Map();

    PetscInt rowId;

    for (LI eid = 0; eid < m_uiNumElems; eid++)
    {
        for (LI r = 0; r < m_uiDofsPerElem[eid]; r++)
        {
            if (m_uipBdrMap[eid][r] == 1)
            {
                // global row ID
                rowId = m_ulpMap[eid][r];
                // 05/01/2020: add the case of penalty method for apply bc
                if (m_BcMeth == BC_METH::BC_IMATRIX)
                {
                    VecSetValue(diag, rowId, 1.0, INSERT_VALUES);
                }
                else if (m_BcMeth == BC_METH::BC_PENALTY)
                {
                    VecSetValue(diag, rowId, (PENALTY_FACTOR * m_dtTraceK), INSERT_VALUES);
                }
            }
        }
    }

    return Error::SUCCESS;
} // apply_bc_diagonal

// apply Dirichlet BC to block diagonal matrix
template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::apply_bc_blkdiag(Mat* blkdiagMat)
{

    const LI m_uiNumElems            = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem  = m_maps.get_DofsPerElem();
    LI** const m_uipLocalMap         = m_maps.get_LocalMap();
    auto m_uiDofLocalBegin           = static_cast<PetscInt>(m_maps.get_DofLocalBegin());
    auto m_uiDofLocalEnd             = static_cast<PetscInt>(m_maps.get_DofLocalEnd());
    unsigned int** const m_uipBdrMap = m_maps.get_BdrMap();
    const LI m_uiNumPreGhostDofs     = m_maps.get_NumPreGhostDofs();

    LI num_dofs_per_elem;
    PetscInt loc_rowId, loc_colId;
    for (LI eid = 0; eid < m_uiNumElems; eid++)
    {
        // total number of dofs per element eid
        num_dofs_per_elem = m_uiDofsPerElem[eid];

        // loop on all dofs of element
        for (LI r = 0; r < num_dofs_per_elem; r++)
        {
            loc_rowId = m_uipLocalMap[eid][r];
            // 05.21.20: bug loc_rowId <= m_uiDofLocalEnd is fixed
            if ((loc_rowId >= m_uiDofLocalBegin) && (loc_rowId < m_uiDofLocalEnd))
            {
                if (m_uipBdrMap[eid][r] == 1)
                {
                    for (LI c = 0; c < num_dofs_per_elem; c++)
                    {
                        loc_colId = m_uipLocalMap[eid][c];
                        // 05.21.20: bug loc_rowId <= m_uiDofLocalEnd is fixed
                        if ((loc_colId >= m_uiDofLocalBegin) && (loc_colId < m_uiDofLocalEnd))
                        {
                            if (loc_rowId == loc_colId)
                            {
                                // 05/01/2020: add the case of penalty method for apply bc
                                if (m_BcMeth == BC_METH::BC_IMATRIX)
                                {
                                    MatSetValue(*blkdiagMat,
                                                (loc_rowId - m_uiNumPreGhostDofs),
                                                (loc_colId - m_uiNumPreGhostDofs),
                                                1.0,
                                                INSERT_VALUES);
                                }
                                else if (m_BcMeth == BC_METH::BC_PENALTY)
                                {
                                    MatSetValue(*blkdiagMat,
                                                (loc_rowId - m_uiNumPreGhostDofs),
                                                (loc_colId - m_uiNumPreGhostDofs),
                                                (PENALTY_FACTOR * m_dtTraceK),
                                                INSERT_VALUES);
                                }
                            }
                            else
                            {
                                // 05/01/2020: only for identity-matrix method, not for penalty
                                // method
                                if (m_BcMeth == BC_METH::BC_IMATRIX)
                                {
                                    MatSetValue(*blkdiagMat,
                                                (loc_rowId - m_uiNumPreGhostDofs),
                                                (loc_colId - m_uiNumPreGhostDofs),
                                                0.0,
                                                INSERT_VALUES);
                                }
                            }
                        }
                    }
                }
                // 10/11/2020: BC with NS IOWA.
                // else
                // {
                //     for (LI c = 0; c < num_dofs_per_elem; c++)
                //     {
                //         loc_colId = m_uipLocalMap[eid][c];
                //         // 05.21.20: bug loc_rowId <= m_uiDofLocalEnd is fixed
                //         if ((loc_colId >= m_uiDofLocalBegin) && (loc_colId < m_uiDofLocalEnd))
                //         {
                //             if (m_uipBdrMap[eid][c] == 1)
                //             {
                //                 // 05/01/2020: only for identity-matrix method, not for penalty
                //                 // method
                //                 if (m_BcMeth == BC_METH::BC_IMATRIX)
                //                 {
                //                     MatSetValue(*blkdiagMat,
                //                                 (loc_rowId - m_uiNumPreGhostDofs),
                //                                 (loc_colId - m_uiNumPreGhostDofs),
                //                                 0.0,
                //                                 INSERT_VALUES);
                //                 }
                //             }
                //         }
                //     }
                // }
            }
        }
    }

    return Error::SUCCESS;
} // apply_bc_blkdiag

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::apply_bc_blkdiag_petsc(std::vector<MatRecord<DT,LI>>& records)
{
    const LI m_uiNumElems            = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem  = m_maps.get_DofsPerElem();
    LI** const m_uipLocalMap         = m_maps.get_LocalMap();
    auto m_uiDofLocalBegin           = static_cast<PetscInt>(m_maps.get_DofLocalBegin());
    auto m_uiDofLocalEnd             = static_cast<PetscInt>(m_maps.get_DofLocalEnd());
    unsigned int** const m_uipBdrMap = m_maps.get_BdrMap();
    const LI m_uiNumPreGhostDofs     = m_maps.get_NumPreGhostDofs();

    LI num_dofs_per_elem;
    PetscInt loc_rowId, loc_colId;
    for (LI eid = 0; eid < m_uiNumElems; eid++)
    {
        // total number of dofs per element eid
        num_dofs_per_elem = m_uiDofsPerElem[eid];

        // loop on all dofs of element
        for (LI r = 0; r < num_dofs_per_elem; r++)
        {
            loc_rowId = m_uipLocalMap[eid][r];
            // 05.21.20: bug loc_rowId <= m_uiDofLocalEnd is fixed
            if ((loc_rowId >= m_uiDofLocalBegin) && (loc_rowId < m_uiDofLocalEnd))
            {
                if (m_uipBdrMap[eid][r] == 1)
                {
                    for (LI c = 0; c < num_dofs_per_elem; c++)
                    {
                        loc_colId = m_uipLocalMap[eid][c];
                        // 05.21.20: bug loc_rowId <= m_uiDofLocalEnd is fixed
                        if ((loc_colId >= m_uiDofLocalBegin) && (loc_colId < m_uiDofLocalEnd))
                        {
                            if (loc_rowId == loc_colId)
                            {
                                // 05/01/2020: add the case of penalty method for apply bc
                                if (m_BcMeth == BC_METH::BC_IMATRIX)
                                {
                                    records.push_back(MatRecord<DT,LI>(m_uiRank,loc_rowId,loc_colId,1));
                                }
                                else if (m_BcMeth == BC_METH::BC_PENALTY)
                                {
                                    records.push_back(MatRecord<DT,LI>(m_uiRank,loc_rowId,loc_colId,PENALTY_FACTOR * m_dtTraceK));
                                }
                            }
                            else
                            {
                                // 05/01/2020: only for identity-matrix method, not for penalty
                                // method
                                if (m_BcMeth == BC_METH::BC_IMATRIX)
                                {
                                    records.push_back(MatRecord<DT,LI>(m_uiRank,loc_rowId,loc_colId,0));
                                }
                            }
                        }
                    }
                }
                //10/11/2020: BC with NS IOWA.
                // else
                // {
                //     for (LI c = 0; c < num_dofs_per_elem; c++)
                //     {
                //         loc_colId = m_uipLocalMap[eid][c];
                //         // 05.21.20: bug loc_rowId <= m_uiDofLocalEnd is fixed
                //         if ((loc_colId >= m_uiDofLocalBegin) && (loc_colId < m_uiDofLocalEnd))
                //         {
                //             if (m_uipBdrMap[eid][c] == 1)
                //             {
                //                 // 05/01/2020: only for identity-matrix method, not for penalty
                //                 // method
                //                 if (m_BcMeth == BC_METH::BC_IMATRIX)
                //                 {
                //                     // MatSetValue(*blkdiagMat,
                //                     //             (loc_rowId - m_uiNumPreGhostDofs),
                //                     //             (loc_colId - m_uiNumPreGhostDofs),
                //                     //             0.0,
                //                     //             INSERT_VALUES);
                //                     records.push_back(MatRecord<DT,LI>(m_uiRank,loc_rowId,loc_colId,0));
                //                 }
                //             }
                //         }
                //     }
                // }
            }
        }
    }

}

// rhs[i] = Uc_i if i is on boundary of Dirichlet condition, Uc_i is the prescribed value on
// boundary rhs[i] = rhs[i] - sum_{j=1}^{nc}{K_ij * Uc_j} if i is a free dof
//          where nc is the total number of boundary dofs and K_ij is stiffness matrix
template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::apply_bc_rhs(Vec rhs)
{

    const LI m_uiNumElems                        = m_maps.get_NumElems();
    const LI* const m_uiDofsPerElem              = m_maps.get_DofsPerElem();
    unsigned int** const m_uipBdrMap             = m_maps.get_BdrMap();
    GI** const m_ulpMap                          = m_maps.get_Map();
    DT** const m_dtPresValMap                    = m_maps.get_PresValMap();
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
        LI block_dims;
        LI num_dofs_per_block;
        LI block_index;
        std::vector<PetscScalar> KfcUc_elem;
        std::vector<PetscInt> row_Indices_KfcUc_elem;
        PetscInt rowId;
        PetscScalar temp;
        bool bdrFlag, rowFlag;

#ifdef VECTORIZED_OPENMP_ALIGNED
        unsigned int nPads = 0;
#endif

        for (LI eid = 0; eid < m_uiNumElems; eid++)
        {
            block_dims = (LI)sqrt(m_epMat[eid].size());
            assert((block_dims * block_dims) == m_epMat[eid].size());
            num_dofs_per_block = m_uiDofsPerElem[eid] / block_dims;

#ifdef VECTORIZED_OPENMP_ALIGNED
            // nPads = get_column_paddings(ALIGNMENT, num_dofs_per_block);
            if ((num_dofs_per_block % (ALIGNMENT / sizeof(DT))) != 0)
            {
                nPads = (ALIGNMENT / sizeof(DT)) - (num_dofs_per_block % (ALIGNMENT / sizeof(DT)));
            }
#endif

            // clear the vectors storing values of KfcUc for element eid
            KfcUc_elem.clear();
            row_Indices_KfcUc_elem.clear();

            for (LI block_i = 0; block_i < block_dims; block_i++)
            {
                LI block_row_offset = block_i * num_dofs_per_block;
                for (LI block_j = 0; block_j < block_dims; block_j++)
                {
                    block_index         = block_i * block_dims + block_j;
                    LI block_col_offset = block_j * num_dofs_per_block;
                    // continue if block_index is not nullptr
                    if (m_epMat[eid][block_index] != nullptr)
                    {
                        for (LI r = 0; r < num_dofs_per_block; r++)
                        {
                            rowFlag = false;
                            // continue if row is associated with a free dof
                            if (m_uipBdrMap[eid][block_i * num_dofs_per_block + r] == 0)
                            {
                                rowId = m_ulpMap[eid][block_i * num_dofs_per_block + r];
                                temp  = 0;
                                // loop over columns of the element matrix (block)
                                for (LI c = 0; c < num_dofs_per_block; c++)
                                {
                                    // continue if column is associated with a constrained dof
                                    if (m_uipBdrMap[eid][block_j * num_dofs_per_block + c] == 1)
                                    {
// accumulate Kfc[r,c]*Uc[c]
#if defined(VECTORIZED_AVX512) || defined(VECTORIZED_AVX256) || defined(VECTORIZED_OPENMP)
                                        // block m_epMat[eid][block_index] is stored in column-major
                                        temp +=
                                          m_epMat[eid][block_index][(c * num_dofs_per_block) + r] *
                                          m_dtPresValMap[eid][block_j * num_dofs_per_block + c];
#elif VECTORIZED_OPENMP_ALIGNED
                                        // block m_epMat[eid][block_index] is stored in column-major
                                        // with nPads appended to columns
                                        temp +=
                                          m_epMat[eid][block_index]
                                                 [c * (num_dofs_per_block + nPads) + r] *
                                          m_dtPresValMap[eid][block_j * num_dofs_per_block + c];
#else
                                        // block m_epMat[eid][block_index] is stored in row-major
                                        temp +=
                                          m_epMat[eid][block_index][(r * num_dofs_per_block) + c] *
                                          m_dtPresValMap[eid][block_j * num_dofs_per_block + c];
#endif
                                        rowFlag = true; // this rowId has constrained column dof
                                        bdrFlag = true; // this element matrix has KfcUc
                                    }
                                }
                                if (rowFlag)
                                {
                                    row_Indices_KfcUc_elem.push_back(rowId);
                                    KfcUc_elem.push_back(-1.0 * temp);
                                }
                            }
                        }
                        if (bdrFlag)
                        {
                            VecSetValues(KfcUcVec,
                                         static_cast<PetscInt>(row_Indices_KfcUc_elem.size()),
                                         row_Indices_KfcUc_elem.data(),
                                         KfcUc_elem.data(),
                                         ADD_VALUES);
                        }
                    } // m_epMat[eid][index] != nullptr
                }     // for block_j
            }         // for block_i
        }             // for eid
    }                 // if (m_BcMeth == BC_METH::BC_IMATRIX)

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

template<typename DT, typename GI, typename LI>
DT* aMatFree<DT, GI, LI>::create_aligned_array(unsigned int alignment, unsigned int length)
{

    DT* array;

#ifdef USE_WINDOWS
    array = (DT*)_aligned_malloc(length * sizeof(DT), alignment);
#else
    int err;
    err = posix_memalign((void**)&array, alignment, length * sizeof(DT));
    if (err)
    {
        return nullptr;
    }
    // supported (and recommended) by Intel compiler:
    // array = (DT*)_mm_malloc(length * sizeof(DT), alignment);
#endif

    return array;
} // create_aligned_array

template<typename DT, typename GI, typename LI>
inline void aMatFree<DT, GI, LI>::delete_algined_array(DT* array)
{
#ifdef USE_WINDOWS
    _aligned_free(array);
#else
    free(array);
#endif
} // delete_aligned_array

template<typename DT, typename GI, typename LI>
Error aMatFree<DT, GI, LI>::dump_mat(const char* filename)
{
    // this prints out the global matrix by using matvec(u,v) in oder to test the implementation of
    // matvec(u,v) algorithm: u0 = [1 0 0 ... 0] --> matvec(v0,u0) will give v0 is the first column
    // of the global matrix u1 = [0 1 0 ... 0] --> matvec(v1,u1) will give v1 is the second column
    // of the global matrix u2 = [0 0 1 ... 0] --> matvec(v2,u2) will give v2 is the third column of
    // the global matrix
    // ... up to the last column of the global matrix

    const LI m_uiNumDofs                     = m_maps.get_NumDofs();
    const LI m_uiNumDofsTotal                = m_maps.get_NumDofsTotal();
    const GI m_ulNumDofsGlobal               = m_maps.get_NumDofsGlobal();
    const std::vector<GI>& m_ulvLocalDofScan = m_maps.get_LocalDofScan();
    const LI m_uiNumPreGhostDofs             = m_maps.get_NumPreGhostDofs();
    GI* const m_ulpLocal2Global              = m_maps.get_Local2Global();

    // create matrix computed by matrix-free
    Mat m_pMatFree;
    MatCreate(m_comm, &m_pMatFree);
    MatSetSizes(m_pMatFree, m_uiNumDofs, m_uiNumDofs, PETSC_DECIDE, PETSC_DECIDE);
    if (m_uiSize > 1)
    {
        MatSetType(m_pMatFree, MATMPIAIJ);
        MatMPIAIJSetPreallocation(m_pMatFree, NNZ, PETSC_NULL, NNZ, PETSC_NULL);
    }
    else
    {
        MatSetType(m_pMatFree, MATSEQAIJ);
        MatSeqAIJSetPreallocation(m_pMatFree, NNZ, PETSC_NULL);
    }
    MatSetOption(m_pMatFree, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

    // create vectors ui and vi preparing for multiple matrix-vector multiplications
    DT* ui;
    DT* vi;
    ui = new DT[m_uiNumDofsTotal];
    vi = new DT[m_uiNumDofsTotal];
    LI localId;
    PetscScalar value;
    PetscInt rowId, colId;

    // loop over all dofs of the global vector
    for (GI globalId = 0; globalId < m_ulNumDofsGlobal; globalId++)
    {
        // initialize input ui and output vi
        for (LI i = 0; i < m_uiNumDofsTotal; i++)
        {
            ui[i] = 0.0;
            vi[i] = 0.0;
        }

        // check if globalId is owned by me --> set input ui = [0...1...0] (if not ui = [0...0])
        if ((globalId >= m_ulvLocalDofScan[m_uiRank]) &&
            (globalId < (m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)))
        {
            localId     = m_uiNumPreGhostDofs + (globalId - m_ulvLocalDofScan[m_uiRank]);
            ui[localId] = 1.0;
        }

        // doing vi = matrix * ui
        matvec(vi, ui, true);

        // set vi to (globalId)th column of the matrix
        colId = globalId;
        for (LI r = 0; r < m_uiNumDofs; r++)
        {
            rowId = m_ulpLocal2Global[r + m_uiNumPreGhostDofs];
            value = vi[r + m_uiNumPreGhostDofs];
            if (fabs(value) > 1e-16)
            {
                MatSetValue(m_pMatFree, rowId, colId, value, ADD_VALUES);
            }
        }
    }

    MatAssemblyBegin(m_pMatFree, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_pMatFree, MAT_FINAL_ASSEMBLY);

    // display matrix
    if (filename == nullptr)
    {
        MatView(m_pMatFree, PETSC_VIEWER_STDOUT_WORLD);
    }
    else
    {
        PetscViewer viewer;
        PetscViewerASCIIOpen(m_comm, filename, &viewer);
        // write to file readable by Matlab (filename must be filename.m in order to execute in
        // Matlab)
        // PetscViewerPushFormat( viewer, PETSC_VIEWER_ASCII_MATLAB );
        MatView(m_pMatFree, viewer);
        PetscViewerDestroy(&viewer);
    }

    delete[] ui;
    delete[] vi;
    MatDestroy(&m_pMatFree);

    return Error::SUCCESS;
} // dump_mat

} // namespace par

#endif // APTIVEMATRIX_AMATFREE_H
