/**
 * @file fe_vector.hpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief header file of fe_matrix.cpp
 *
 * @version 0.1
 * @date 2018-12-07
 */

#include <Eigen/Dense>

#ifndef ADAPTIVEMATRIX_FE_MATRIX_H
#define ADAPTIVEMATRIX_FE_MATRIX_H

/**
 * @brief: element load vector of 8-node hex element for Poisson equation
 * @param[in] xe nodal coordinates
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_hex8(double* fe, const double* xe);

/**
 * @brief: element load vector of 8-node hex element for Poisson equation, use Eigen vector
 * @param[in] xe nodal coordinates
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_hex8_eig(Eigen::Matrix<double,8,1> &fe, const double* xe);

/**
 * @brief: element load vector of 8-node hex element for 3D elasticity, due to body force
 * @param[in] xe nodal coordinates
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_hex8_iso(Eigen::Matrix<double,24,1> &fe, const double* xe);

/**
 * @brief: element load vector of 8-node hex element for 3D elasticity, due to surface traction
 * @param[in] xe nodal coordinates
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_hex8_iso_surface(Eigen::Matrix<double,24,1> &fe, const double* xe, const unsigned int faceId, const double* faceTrac);

#endif //ADAPTIVEMATRIX_KE_MATRIX_H
