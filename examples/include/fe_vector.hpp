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


#ifndef ADAPTIVEMATRIX_FE_MATRIX_H
#define ADAPTIVEMATRIX_FE_MATRIX_H

#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include "shapeFunc.hpp"

/**
 * @brief: element load vector of 4-node tet element for Poisson equation, use Eigen vector
 * @param[in] xe nodal coordinates
 * @param[in] be nodal body force
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_tet4(Eigen::Matrix<double,4,1> &fe, const double* xe, const double* be, const double* xw, const unsigned int NGT);

/**
 * @brief: element load vector of 10-node tet element for Poisson equation, use Eigen vector
 * @param[in] xe nodal coordinates
 * @param[in] be nodal body force
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_tet10(Eigen::Matrix<double, Eigen::Dynamic, 1> &fe, const double* xe, const double* be, const double* xw, const unsigned int NGT);

/**
 * @brief: element load vector of 8-node hex element for Poisson equation
 * @param[in] xe nodal coordinates
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_hex8(double* fe, const double* xe, const double* xw, const unsigned int NGT);

/**
 * @brief: element load vector of 8-node hex element for Poisson equation, use Eigen vector
 * @param[in] xe nodal coordinates
 * @param[in] be nodal body force
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_hex8_eig(Eigen::Matrix<double,8,1> &fe, const double* xe, const double* be, const double* xw, const unsigned int NGT);

/**
 * @brief: element load vector of 8-node hex element for 3D elasticity, due to body force
 * @param[in] xe nodal coordinates
 * @param[in] bN nodal values of body force
 * @param[in] GaussPoints coordinates and weights of Gauss points
 * @param[in] nGauss number of Gauss points in each direction
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_hex8_iso(Eigen::Matrix<double,24,1> &fe, const double* xe, const double* bN,
                const double* GaussPoints, const unsigned int nGauss);

/**
 * @brief: element load vector of 8-node hex element for 3D elasticity, due to surface traction
 * @param[in] xe nodal coordinates
 * @param[in] tN nodal values of traction
 * @param[in] GaussPoints coordinates and weights of Gauss points
 * @param[in] nGauss number of Gauss points in each direction
 * @param[out] feT element load vector corresponding to nodes on the surface where traction applied
 * @author Han Tran
* */
void feT_hex8_iso(Eigen::Matrix<double,12,1> &feT, const double* xe, const double* tN,
                const double* GaussPoints, const unsigned int nGauss);


/**
 * @brief: element load vector of 20-node hex element for 3D elasticity, due to body force
 * @param[in] xe nodal coordinates
 * @param[in] bN nodal values of body force
 * @param[in] GaussPoints coordinates and weights of Gauss points
 * @param[in] nGauss number of Gauss points in each direction
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_hex20_iso(Eigen::Matrix<double,60,1> &fe, const double* xe, const double* bN,
                const double* GaussPoints, const unsigned int nGauss);


/**
 * @brief: element load vector of 20-node hex element for 3D elasticity, due to surface traction
 * @param[in] xe nodal coordinates
 * @param[in] tN nodal values of traction
 * @param[in] GaussPoints coordinates and weights of Gauss points
 * @param[in] nGauss number of Gauss points in each direction
 * @param[out] feT element load vector corresponding to nodes on the surface where traction applied
 * @author Han Tran
* */
void feT_hex20_iso(Eigen::Matrix<double,24,1> &feT, const double* xe, const double* tN,
                const double* GaussPoints, const unsigned int nGauss);


/**
 * @brief: element load vector of 4-node tet element for 3D elasticity, due to body force
 * @param[in] xe nodal coordinates
 * @param[in] bN nodal values of body force
 * @param[in] GaussPoints coordinates and weights of Gauss points
 * @param[in] nGauss number of Gauss points in each direction
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_tet4_iso(Eigen::Matrix<double,12,1> &fe, const double* xe, const double* bN,
                const double* GaussPoints, const unsigned int nGauss);


/**
 * @brief: element load vector of 4-node tet element for 3D elasticity, due to surface traction
 * @param[in] xe nodal coordinates
 * @param[in] tN nodal values of traction
 * @param[in] GaussPoints coordinates and weights of Gauss points
 * @param[in] nGauss number of Gauss points in each direction
 * @param[out] feT element load vector corresponding to nodes on the surface where traction applied
 * @author Han Tran
* */
void feT_tet4_iso(Eigen::Matrix<double,9,1> &feT, const double* xe, const double* tN,
                const double* GaussPoints, const unsigned int nGauss);


/**
 * @brief: element load vector of 10-node tet element for 3D elasticity, due to body force
 * @param[in] xe nodal coordinates
 * @param[in] bN nodal values of body force
 * @param[in] GaussPoints coordinates and weights of Gauss points
 * @param[in] nGauss number of Gauss points in each direction
 * @param[out] fe element load vector
 * @author Han Tran
* */
void fe_tet10_iso(Eigen::Matrix<double,30,1> &fe, const double* xe, const double* bN,
                const double* GaussPoints, const unsigned int nGauss);


/**
 * @brief: element load vector of 10-node tet element for 3D elasticity, due to surface traction
 * @param[in] xe nodal coordinates
 * @param[in] tN nodal values of traction
 * @param[in] GaussPoints coordinates and weights of Gauss points
 * @param[in] nGauss number of Gauss points in each direction
 * @param[out] feT element load vector corresponding to nodes on the surface where traction applied
 * @author Han Tran
* */
void feT_tet10_iso(Eigen::Matrix<double,18,1> &feT, const double* xe, const double* tN,
                const double* GaussPoints, const unsigned int nGauss);

// element load vector for 27-node hex element for 3D elasticity, due to body force
void fe_hex27_iso(double* fe, const double* xe, const double* bN,
                const double* GaussPoints, const unsigned int nGauss);
void fe_hex27_iso(Eigen::Matrix<double,81,1> &fe, const double* xe, const double* bN,
                const double* GaussPoints, const unsigned int nGauss);

#endif //ADAPTIVEMATRIX_KE_MATRIX_H
