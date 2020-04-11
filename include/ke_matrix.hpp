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
#include <iostream>
#include <math.h>
#include "shapeFunc.hpp"

/**
* @brief: element stiffness matrix of hex 8-node element of potential problem
* @param[in] xe[8*3] physical coordinates of element
* @param[in] xw[2*NGT] coordinates and weights of Gauss points
* @param[in] NGT number of Gauss points in each direction
* @param[out] ke[8*8] element stiffness matrix
* */
void ke_hex8(double* ke, const double* xe, const double* xw, const unsigned int NGT);


/**
* @brief: element stiffness matrix of hex 8-node element of potential problem, using Eigen matrix
* @param[in] xe[3*8] physical coordinates of element
* @param[in] xw[2*NGT] coordinates and weights of Gauss points
* @param[in] NGT number of Gauss points in each direction
* @param[out] ke(8,8) element stiffness matrix
* */
void ke_hex8_eig(Eigen::Matrix<double,8,8> &ke, double* xe, const double* xw, const unsigned int NGT);


/**
 * @brief element stiffness matrix of quad 4-node element of potential problem
 * @param[in] xe[3*4] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(4,4) element stiffness matrix
 * */
void ke_quad4_eig(Eigen::Matrix<double,4,4> &ke, double* xe, const double* xw, const unsigned int NGT);

/**
 * @brief element stiffness matrix of quad 4-node element of isotropic elastic problem
 * @param[in] xe[2*4] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(8,8) element stiffness matrix
 * */
void ke_quad4_iso(Eigen::Matrix<double,8,8> &ke, double* xe, const double E, const double nu,
                const double* xw, const unsigned int NGT);

/**
 * @brief element stiffness matrix of hex 8-node element of isotropic elastic problem
 * @param[in] xe[3*8] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(24,24) element stiffness matrix
 * */
void ke_hex8_iso(Eigen::Matrix<double,24,24> &ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);

/**
 * @brief element stiffness matrix of hex 20-node element of isotropic elastic problem
 * @param[in] xe[3*20] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(60,60) element stiffness matrix
 * */
void ke_hex20_iso(Eigen::Matrix<double,60,60> &ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);

#endif //ADAPTIVEMATRIX_KE_MATRIX_H
