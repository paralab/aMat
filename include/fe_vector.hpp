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

void fe_hex8(double* fe, const double* xe);
void fe_hex8_eig(Eigen::Matrix<double,8,1> &fe, const double* xe);

#endif //ADAPTIVEMATRIX_KE_MATRIX_H
