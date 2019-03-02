//
// Created by Han Tran on 12/7/18.
//



#ifndef ADAPTIVEMATRIX_KE_MATRIX_H
#define ADAPTIVEMATRIX_KE_MATRIX_H

#include "Dense"

/**
* @brief: element stiffness matrix of hex 8-node element of potential problem
* @param[in] double xe[8*3] physical coordinates of element
* @param[out] double ke_hex8[8*8] element stiffness matrix
* */
void ke_hex8(double *ke, const double *xe);


/**
* @brief: element stiffness matrix of hex 8-node element of potential problem, using Eigen matrix
* @param[in] double xe(8,3) physical coordinates of element
* @param[out] double ke_hex8_eig(8,8) element stiffness matrix
* */
Eigen::Matrix<double,8,8> ke_hex8_eig(Eigen::Matrix<double,8,3> xe);


#endif //ADAPTIVEMATRIX_KE_MATRIX_H
