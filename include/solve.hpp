/**
 * @file solve.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Tran         hantran@cs.utah.edu
 *
 * @brief Hold functions to solve Ax = b
 * 
 * @version
 * @date   2020.06.19
 * 
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 * 
 */

#ifndef ADAPTIVEMATRIX_SOLVE_H
#define ADAPTIVEMATRIX_SOLVE_H

#include <Eigen/Dense>
#include "enums.hpp"

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include "aMatBased.hpp"
#include "aMatFree.hpp"

namespace par {

    template<typename MatrixType>
    static Error solve(MatrixType& matrix, Vec rhs, Vec out ) {

        MPI_Comm m_comm = matrix.get_comm();

        Mat m_pMat = matrix.get_matrix();

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

        KSPSetTolerances(ksp, 1E-12, 1E-12, PETSC_DEFAULT, 20000);

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

        if (matrix.get_matrix_type() == MATRIX_TYPE::MATRIX_FREE){
            void* ctx = nullptr;
            MatShellGetContext(m_pMat, &ctx);
            delete ctx;
            ctx = nullptr;
        }

        return Error::SUCCESS;
    } // solve
} // namespace par
#endif// APTIVEMATRIX_SOLVE_H