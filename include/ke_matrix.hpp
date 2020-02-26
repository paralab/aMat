/**
 * @file ke_matrix.hpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief header file of ke_matrix.cpp
 *
 * @version 0.1
 * @date 2018-12-07
 */
#ifndef ADAPTIVEMATRIX_KE_MATRIX_H
#define ADAPTIVEMATRIX_KE_MATRIX_H

#include <Eigen/Dense>

/**
* @brief: element stiffness matrix of hex 8-node element of potential problem
* @param[in] double xe[8*3] physical coordinates of element
* @param[out] double ke_hex8[8*8] element stiffness matrix
* */
void ke_hex8(double* ke, const double* xe);

/**
* @brief: element stiffness matrix of hex 8-node element of potential problem, using Eigen matrix
* @param[in] double xe(8,3) physical coordinates of element
* @param[out] double ke_hex8_eig(8,8) element stiffness matrix
* */
void ke_hex8_eig(Eigen::Matrix<double,8,8> &ke, double* xe);

/** @brief element stiffness matrix of quad 4-node element of potential problem */
void ke_quad4_eig(Eigen::Matrix<double,4,4> &ke, double* xe);

/** @brief element stiffness matrix of quad 4-node element of isotropic elastic problem */
void ke_quad4_iso(Eigen::Matrix<double,8,8> &ke, double* xe, const double E, const double nu);

/** @brief element stiffness matrix of hex 8-node element of isotropic elastic problem */
void ke_hex8_iso(Eigen::Matrix<double,24,24> &ke, double* xe, double E, double nu);

#endif //ADAPTIVEMATRIX_KE_MATRIX_H
