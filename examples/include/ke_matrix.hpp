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
* @brief: element stiffness matrix of hex 4-node tetrahedron element of potential problem
* @param[in] xe[8*3] physical coordinates of element
* @param[in] xw[2*NGT] coordinates and weights of Gauss points
* @param[in] NGT number of Gauss points in each direction
* @param[out] ke[8*8] element stiffness matrix
* */
void ke_tet4(Eigen::Matrix<double, 4, 4> &ke, const double* xe, const double* xw, const unsigned int NGT);
void ke_tet4(double* ke, const double* xe, const double* xw, const unsigned int NGT);


/**
* @brief: element stiffness matrix of hex 10-node tetrahedron element of potential problem
* @param[in] xe[8*3] physical coordinates of element
* @param[in] xw[2*NGT] coordinates and weights of Gauss points
* @param[in] NGT number of Gauss points in each direction
* @param[out] ke[8*8] element stiffness matrix
* */
void ke_tet10(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &ke, const double* xe, const double* xw, const unsigned int NGT);
void ke_tet10(double* ke, const double* xe, const double* xw, const unsigned int NGT);


/**
* @brief: element stiffness matrix of hex 8-node element of potential problem, using Eigen matrix
* @param[in] xe[3*8] physical coordinates of element
* @param[in] xw[2*NGT] coordinates and weights of Gauss points
* @param[in] NGT number of Gauss points in each direction
* @param[out] ke(8,8) element stiffness matrix
* */
void ke_hex8_eig(Eigen::Matrix<double, 8, 8> &ke, double* xe, const double* xw, const unsigned int NGT);
void ke_hex8(double* ke, const double* xe, const double* xw, const unsigned int NGT);


/**
 * @brief element stiffness matrix of 2D quad 4-node element of potential problem
 * @param[in] xe[3*4] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(4,4) element stiffness matrix
 * */
void ke_quad4_eig(Eigen::Matrix<double, 4, 4> &ke, double* xe, const double* xw, const unsigned int NGT);
void ke_quad4(double* ke, double* xe, const double* xw, const unsigned int NGT);


/**
 * @brief element stiffness matrix of 2D quad 4-node element of isotropic elastic problem
 * @param[in] xe[2*4] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(8,8) element stiffness matrix
 * */
void ke_quad4_iso(Eigen::Matrix<double, 8, 8> &ke, double* xe, const double E, const double nu,
                const double* xw, const unsigned int NGT);


/**
 * @brief element stiffness matrix of tet 4-node element of isotropic elastic problem
 * @param[in] xe[3*20] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(12,12) element stiffness matrix
 * */
void ke_tet4_iso(Eigen::Matrix<double, 12, 12> &ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);
void ke_tet4_iso(double* ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);


/**
 * @brief element stiffness matrix of tet 4-node element of isotropic elastic problem
 * @param[in] xe[3*20] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(30,30) element stiffness matrix
 * */
void ke_tet10_iso(Eigen::Matrix<double, 30, 30> &ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);
void ke_tet10_iso(double* ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);


/**
 * @brief element stiffness matrix of hex 8-node element of isotropic elastic problem
 * @param[in] xe[3*8] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(24,24) element stiffness matrix
 * */
void ke_hex8_iso(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> &ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);
void ke_hex8_iso(Eigen::Matrix<double, 24, 24> &ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);
void ke_hex8_iso(double* ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);


/**
 * @brief element stiffness matrix of hex 20-node element of isotropic elastic problem
 * @param[in] xe[3*20] physical coordinates of element
 * @param[in] xw[2*NGT] coordinates and weights of Gauss points
 * @param[in] NGT number of Gauss points in each direction
 * @param[out] ke(60,60) element stiffness matrix
 * */
void ke_hex20_iso(Eigen::Matrix<double, 60, 60> &ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);
void ke_hex20_iso(double* ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);


/**
* @brief: element stiffness matrix of hex 8-node element of potential problem with crack, block 01 (off diagonal)
* @brief: ad-hoc assumption: crack plane is zeta = 0
* */
void ke_hex8_iso_crack_01(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> &ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);

/* 2022.09.01: copy from aMat_dev: quadratic 27-node element for elasticity */
void ke_hex27_iso(double* ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);
void ke_hex27_iso(Eigen::Matrix<double,81,81> &ke, double* xe, double E, double nu,
                const double* xw, const unsigned int NGT);
                
#endif //ADAPTIVEMATRIX_KE_MATRIX_H
