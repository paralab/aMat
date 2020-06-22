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

#include "asyncExchangeCtx.hpp"
#include "enums.hpp"
#include "maps.hpp"
#include "aVec.hpp"
#include "matRecord.hpp"

#include "profiler.hpp"

// alternatives for vectorization, alignment = cacheline = vector register
#ifdef VECTORIZED_AVX512
    #define SIMD_LENGTH (512/(sizeof(DT) * 8)) // length of vector register = 512 bytes
    #define ALIGNMENT 64
#elif VECTORIZED_AVX256
    #define SIMD_LENGTH (256/(sizeof(DT) * 8)) // length of vector register = 256 bytes
    #define ALIGNMENT 64
#elif VECTORIZED_OPENMP_ALIGNED
    #define ALIGNMENT 64
#endif

// number of nonzero terms in the matrix (used in matrix-base and block Jacobi preconditioning)
// e.g. in a structure mesh, eight of 20-node quadratic elements sharing the node
// --> 81 nodes (3 dofs/node) constitue one row of the stiffness matrix
#define NNZ (81*3)

// weight factor for penalty method in applying BC
#define PENALTY_FACTOR 100

namespace par {

    // Class aMat
    // DT => type of data stored in matrix (eg: double). GI => size of global index. LI => size of local index
    template <typename DT, typename GI, typename LI>
    class aMat {

        public:
        typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> EigenMat;

        typedef DT DTType;
        typedef GI GIType;
        typedef LI LIType;

        protected:
        MPI_Comm m_comm;                        // communicator
        unsigned int m_uiRank;                  // my rank id
        unsigned int m_uiSize;                  // total number of ranks

        Mat m_pMat;                             // Petsc matrix

        Maps<DT, GI, LI>& m_maps;               // reference to mesh_maps passed in constructor

        BC_METH m_BcMeth;                       // method of applying Dirichlet BC
        Vec KfcUcVec;                           // KfcUc = Kfc * Uc, used to apply bc for rhs
        DT m_dtTraceK;                          // penalty number

        MATRIX_TYPE m_matType;                  // matrix type (aMatFree or aMatBased)

        public:
        aMat( Maps<DT, GI, LI> &mesh_maps, BC_METH bcType = BC_METH::BC_IMATRIX );

        ~aMat();

        /**@brief get communicator */
        MPI_Comm get_comm() { 
            return m_comm; 
        }

        /**@brief return method (matrix free / matrix based) used for analysis */
        MATRIX_TYPE get_matrix_type() { 
            return m_matType; 
        }

        /**@brief aMatBased returns Petsc assembled matrix, aMatFree returns Petsc matrix shell */
        virtual Mat& get_matrix() = 0;
        
        /**@brief assemble element matrix to global matrix */
        virtual Error set_element_matrix( LI eid, EigenMat e_mat, LI block_i, LI block_j, LI blocks_dim ) = 0;

        /**@brief apply Dirichlet bc: matrix-free --> apply bc on rhs, matrix-based --> apply bc on rhs and matrix */
        virtual Error apply_bc( Vec rhs ) = 0;

        /**@brief begin assembling the matrix, we need this for aMatBased */
        virtual Error petsc_init_mat( MatAssemblyType mode ) const = 0;

        /**@brief complete assembling the matrix, we need this for aMatBased */
        virtual Error petsc_finalize_mat( MatAssemblyType mode ) const = 0;

        /**@brief write global matrix to filename "fvec" */
        virtual Error dump_mat( const char* filename = nullptr ) = 0;

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

    }; // class aMat


    //==============================================================================================================
    // aMat constructor, also reference m_maps to mesh_maps
    template <typename DT, typename GI, typename LI>
    aMat<DT, GI, LI>::aMat( Maps<DT, GI, LI> &mesh_maps, BC_METH bcType ) : m_maps(mesh_maps) {
        m_comm     = mesh_maps.get_comm();
        MPI_Comm_rank(m_comm, (int*)&m_uiRank);
        MPI_Comm_size(m_comm, (int*)&m_uiSize);

        m_BcMeth   = bcType;              // method to apply bc
        m_dtTraceK = 0.0;                 // penalty number
    } // constructor


    template <typename DT, typename GI, typename LI>
    aMat<DT, GI, LI>::~aMat() {
        
    } // destructor

} // end of namespace par

#endif// APTIVEMATRIX_AMAT_H