/**
 * @file aMat.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Duc Tran     hantran@cs.utah.edu
 *
 * @brief A sparse matrix class for adaptive finite elements.
 *
 * @version 0.1
 * @date 2018-11-07
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#ifndef ADAPTIVEMATRIX_AMAT_H
#define ADAPTIVEMATRIX_AMAT_H

#include "aVec.hpp"
#include "asyncExchangeCtx.hpp"
#include "enums.hpp"
#include "maps.hpp"
#include "matRecord.hpp"
#include "profiler.hpp"
#include "ke_matrix.hpp"

#include "lapack_extern.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <petsc.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifdef USE_GPU
    #include "aMatGpu.hpp"
    //#include "aMatGpu_1.hpp"
    #include <cuda.h>
    #include "magma_v2.h"
    #include "magma_lapack.h"
#endif
// alternatives for vectorization, alignment = cacheline = vector register
#ifdef VECTORIZED_AVX512
#define SIMD_LENGTH (512 / (sizeof(DT) * 8)) // length of vector register = 512 bytes
#define ALIGNMENT   64
#elif VECTORIZED_AVX256
#define SIMD_LENGTH (256 / (sizeof(DT) * 8)) // length of vector register = 256 bytes
#define ALIGNMENT   64
#elif VECTORIZED_OPENMP_ALIGNED
#define ALIGNMENT 64
#endif

// number of nonzero terms in the matrix (used in matrix-base and block Jacobi preconditioning)
// e.g. in a structure mesh, eight of 20-node quadratic elements sharing the node
// --> 81 nodes (3 dofs/node) constitue one row of the stiffness matrix
#define NNZ (81 * 4) // for example08 with unstructured mesh, we need NNZ = 4 * 81 (instead of 3*81 for structured mesh)

// weight factor for penalty method in applying BC
#define PENALTY_FACTOR 100

namespace par
{

// Class aMat
// DT => type of data stored in matrix (eg: double). GI => size of global index. LI => size of local index
template<typename Derived, typename DT, typename GI, typename LI>
class aMat {

  public:
    typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> EigenMat;
    typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatRowMajor;

    typedef DT DTType;
    typedef GI GIType;
    typedef LI LIType;

    protected:
    MPI_Comm m_comm;       // communicator
    unsigned int m_uiRank; // my rank id
    unsigned int m_uiSize; // total number of ranks

    Mat m_pMat; // Petsc matrix

    Maps<DT, GI, LI>& m_maps; // reference to mesh_maps passed in constructor

    BC_METH m_BcMeth; // method of applying Dirichlet BC
    Vec KfcUcVec;     // KfcUc = Kfc * Uc, used to apply bc for rhs
    DT m_dtTraceK;    // penalty number

    MATRIX_TYPE m_matType; // matrix type (aMatFree or aMatBased)
    MATFREE_TYPE m_freeType; // hybrid or free for matrix-free method

    // this is a function pointer pointing the function to compute element matrix
    //void (*m_eleMatFunc)(LI, const GI*, unsigned int, DT*, DT*) = nullptr;
    void (*m_eleMatFunc)(LI, DT*, DT*) = nullptr;
    
    public:
    aMat(Maps<DT, GI, LI>& mesh_maps, BC_METH bcType = BC_METH::BC_IMATRIX);

    ~aMat() {}

    /* void set_element_matrix_function(void (*eMat)(LI, const GI*, unsigned int, DT*, DT*)){
        m_eleMatFunc = eMat;
    } */
    void set_element_matrix_function(void (*eMat)(LI, DT*, DT*)){
        m_eleMatFunc = eMat;
    }
    
    /**@brief get communicator */
    inline MPI_Comm get_comm() {
        return m_comm;
    }

    /**@brief return method (matrix free / matrix based) used for analysis */
    inline MATRIX_TYPE get_matrix_type() {
        return m_matType;
    }

    /**@brief aMatBased returns Petsc assembled matrix, aMatFree returns Petsc matrix shell */
    inline Mat& get_matrix() {
        return static_cast<Derived*>(this)->get_matrix();
    }

    /**@brief assemble single block of element matrix to global matrix */
    template<typename T>
    inline Error set_element_matrix(LI eid, const T& e_mat, LI block_i, LI block_j, LI blocks_dim) {
        return static_cast<Derived*>(this)->set_element_matrix(eid,
                                                               e_mat,
                                                               block_i,
                                                               block_j,
                                                               blocks_dim);
    }

    /**@brief assemble element matrix, all blocks at once */
    template<typename T>
    inline Error set_element_matrix(LI eid,
                                    LI* ind_non_zero_block_i,
                                    LI* ind_non_zero_block_j,
                                    const T** non_zero_block_mats,
                                    LI num_non_zero_blocks) {
        return static_cast<Derived*>(this)->set_element_matrix(eid,
                                                               ind_non_zero_block_i,
                                                               ind_non_zero_block_j,
                                                               non_zero_block_mats,
                                                               num_non_zero_blocks);
    }

    /**@brief apply Dirichlet bc: matrix-free --> apply bc on rhs, matrix-based --> apply bc on rhs
     * and matrix */
    inline Error apply_bc(Vec rhs) {
        return static_cast<Derived*>(this)->apply_bc(rhs);
    }

    /**@brief calls finalize_begin() and finalize_end() */
    inline Error finalize() {
        return static_cast<Derived*>(this)->finalize();
    }

    /**@brief begin assembling the matrix */
    inline Error finalize_begin(MatAssemblyType mode) const {
        return static_cast<const Derived*>(this)->finalize_begin();
    }

    /**@brief complete assembling the matrix */
    inline Error finalize_end() {
        return static_cast<Derived*>(this)->finalize_end();
    }

    /**@brief write global matrix to filename "fvec" */
    inline Error dump_mat(const char* filename = nullptr) {
        return static_cast<Derived*>(this)->dump_mat(filename);
    }

    /**@brief set matrix-free type */
    Error set_matfree_type(MATFREE_TYPE type) {
        m_freeType = type;
        return Error::SUCCESS;
    }

    /**@brief set number of streams if GPU is used */
    #ifdef USE_GPU
    Error set_num_streams(LI nStreams) {
        return static_cast<Derived*>(this)->set_num_streams(nStreams);
    }
    Error get_timer(long double *scatterTime, long double *gatherTime, long double *mvTime, long double *mvTotalTime) {
        return static_cast<Derived*>(this)->get_timer(scatterTime, gatherTime, mvTime, mvTotalTime);
    }
    #endif

    /**@brief v = aMat_matrix * u, using quasi matrix free */
    Error matvec(DT* v, const DT* u, bool isGhosted) {
        return static_cast<Derived*>(this)->matvec(v, u, isGhosted);
    }

    /**@brief v = aMat_matrix * u, using Petsc */
    Error matmult(Vec v, Vec u) {
        return static_cast<Derived*>(this)->matmult(v, u);
    }

    #ifdef AMAT_PROFILER
    public:
    /**@brief list of profilers for timing different tasks */
    std::vector<profiler_t> timing_aMat = std::vector<profiler_t>(static_cast<int>(PROFILER::LAST));

    /**@brief reset variables for timing*/
    void reset_profile_counters() {
        for (unsigned int i = 0; i < timing_aMat.size(); i++) {
            // printf("i= %d\n",i);
            timing_aMat[i].clear();
            timing_aMat[i].start();
        }
    }

    /**@brief print out timing */
    void profile_dump(std::ostream& s) {

        long double t_rank, t_max;

        // get the time of task
        t_rank = timing_aMat[static_cast<int>(PROFILER::MATVEC)].seconds;

        // get the max time among all ranks
        MPI_Reduce(&t_rank, &t_max, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, m_comm);

        // display the time
        if (m_uiRank == 0)
        {
            s << "time of matvec: = " << t_max << "\n";
        }
    }
    #endif

}; // class aMat

//==============================================================================================================
// aMat constructor, also reference m_maps to mesh_maps
template<typename Derived, typename DT, typename GI, typename LI>
aMat<Derived, DT, GI, LI>::aMat(Maps<DT, GI, LI>& mesh_maps, BC_METH bcType)
  : m_maps(mesh_maps) {
    m_comm = mesh_maps.get_comm();
    MPI_Comm_rank(m_comm, (int*)&m_uiRank);
    MPI_Comm_size(m_comm, (int*)&m_uiSize);

    m_BcMeth   = bcType; // method to apply bc
    m_dtTraceK = 0.0;    // penalty number
} // constructor

} // end of namespace par

#endif // APTIVEMATRIX_AMAT_H