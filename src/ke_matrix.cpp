/**
 * @file ke_matrix.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief functions to compute element stiffness matrices
 *
 * @version 0.1
 * @date 2018-12-07
 */
#include <iostream>

#include <math.h>

#include <Eigen/Dense>

#include "shfunction.hpp"
#include "ke_matrix.hpp"

using Eigen::Matrix;

/**
* @brief: element stiffness matrix of hex 8-node element of potential problem
* @param[in] double xe[8*3] physical coordinates of element
* @param[out] double ke_hex8[8*8] element stiffness matrix
* */
void ke_hex8(double* ke, const double* xe)
{
    const int NGT = 2; // number of Gauss points in each direction
    double x[3], w[3]; // x=[xi,eta,zeta], w=[weight_xi, weight_eta,weight_zeta]
    double *xw;
    double *dN;
    double dxds[3], dyds[3], dzds[3];
    double dxids[3], detads[3], dzetads[3];
    double jaco;
    double B0[8], B1[8], B2[8];
    int idx;

    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 8; j++){
            ke[i*8 + j] = 0.0;
        }
    }

    xw = gauss(NGT); // coordinates and weights of Gauss points

    for (int i = 0; i < NGT; i++){
        x[0] = xw[i*2];                     // xi
        w[0] = xw[i*2 + 1];                 // weight_xi
        for (int j = 0; j < NGT; j++){
            x[1] = xw[j*2];                 // eta
            w[1] = xw[j*2 + 1];             // weight_eta
            for (int k = 0; k < NGT; k++){
                x[2] = xw[k*2];             // zeta
                w[2] = xw[k*2 + 1];         // weight_zeta
                dN = dfbasis_hex8(x);

                // initialize
                for (int p = 0; p < 3; p++){
                    dxds[p] = 0.0;          //[dx/dxi, dx/deta, dx/dzeta]
                    dyds[p] = 0.0;          //[dy/dxi, dy/deta, dy/dzeta]
                    dzds[p] = 0.0;          //[dz/dxi, dz/deta, dz/dzeta]
                }
                for (int p = 0; p < 3; p++){
                    for (int m = 0; m < 8; m++){
                        dxds[p] += xe[3*m]*dN[3*m + p];
                        dyds[p] += xe[3*m + 1]*dN[3*m + p];
                        dzds[p] += xe[3*m + 2]*dN[3*m + p];
                    }
                }

                // jacobian
                jaco =dxds[0]*(dyds[1]*dzds[2] - dzds[1]*dyds[2]) + dyds[0]*(dzds[1]*dxds[2] - dxds[1]*dzds[2]) +
                        dzds[0]*(dxds[1]*dyds[2] - dyds[1]*dxds[2]);
                if (jaco < 0.0) {
                    printf(" Jacobian is negative!");
                    exit(0);
                }
                //jaco = fabs(jaco);

                // dxi/dx, dxi/dy, dxi/dz
                dxids[0] = (dzds[2]*dyds[1] - dyds[2]*dzds[1])/jaco;
                dxids[1] = (dxds[2]*dzds[1] - dzds[2]*dxds[1])/jaco;
                dxids[2] = (dyds[2]*dxds[1] - dxds[2]*dyds[1])/jaco;

                // deta/dx, deta/dy, deta/dz
                detads[0] = (dyds[2]*dzds[0] - dzds[2]*dyds[0])/jaco;
                detads[1] = (dzds[2]*dxds[0] - dxds[2]*dzds[0])/jaco;
                detads[2] = (dxds[2]*dyds[0] - dyds[2]*dxds[0])/jaco;

                // dzeta/dx, dzeta/dy, dzeta/dz
                dzetads[0] = (dzds[1]*dyds[0] - dyds[1]*dzds[0])/jaco;
                dzetads[1] = (dxds[1]*dzds[0] - dzds[1]*dxds[0])/jaco;
                dzetads[2] = (dyds[1]*dxds[0] - dxds[1]*dyds[0])/jaco;

                // B-matrix
                for (int m = 0; m < 8; m++){
                    // first row of [B]
                    B0[m] = dN[m*3]*dxids[0] + dN[m*3 + 1]*detads[0] + dN[m*3 + 2]*dzetads[0];
                    // second row of [B]
                    B1[m] = dN[m*3]*dxids[1] + dN[m*3 + 1]*detads[1] + dN[m*3 + 2]*dzetads[1];
                    // third row of [B]
                    B2[m] = dN[m*3]*dxids[2] + dN[m*3 + 1]*detads[2] + dN[m*3 + 2]*dzetads[2];
                }

                // ke matrix
                for (int row = 0; row < 8; row++){
                    for (int col = 0; col < 8; col++){
                        idx = (row*8) + col;
                        ke[idx] += (B0[row]*B0[col] + B1[row]*B1[col] + B2[row]*B2[col]) * jaco * w[2] * w[1] * w[0];
                    }
                }
            } // k integration
        } // j integration
    } // i integration

}

// for testing only ke = 1
void ke_hex8_test(double* ke, const double* xe){
    unsigned int idx;
    for (unsigned int i = 0; i < 8; i++){
        for (unsigned int j = 0; j < 8; j++){
            idx = (i*8) + j;
            ke[idx] = 1.0;
        }
    }
}

/**
* @brief: element stiffness matrix of hex 8-node element of potential problem, using Eigen matrix
* @param[in] double xe(8,3) physical coordinates of element
* @param[out] double ke_hex8_eig(8,8) element stiffness matrix
* */
void ke_hex8_eig(Matrix<double,8,8> &ke, double *xe){

    const int NGT = 2; // number of Gauss points in each direction
    double x[3], w[3]; // x=[xi,eta,zeta], w=[weight_xi, weight_eta,weight_zeta]
    double *xw;
    double *dN;
    double dxds[3], dyds[3], dzds[3];
    double dxids[3], detads[3], dzetads[3];
    double jaco;
    double B0[8], B1[8], B2[8];

    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 8; j++){
            ke(i,j) = 0.0;
        }
    }

    xw = gauss(NGT); // coordinates and weights of Gauss points

    for (int i = 0; i < NGT; i++){
        x[0] = xw[i*2];                     // xi
        w[0] = xw[i*2 + 1];                 // weight_xi
        for (int j = 0; j < NGT; j++){
            x[1] = xw[j*2];                 // eta
            w[1] = xw[j*2 + 1];             // weight_eta
            for (int k = 0; k < NGT; k++){
                x[2] = xw[k*2];             // zeta
                w[2] = xw[k*2 + 1];         // weight_zeta
                dN = dfbasis_hex8(x);

                // initialize
                for (int p = 0; p < 3; p++){
                    dxds[p] = 0.0;          //[dx/dxi, dx/deta, dx/dzeta]
                    dyds[p] = 0.0;          //[dy/dxi, dy/deta, dy/dzeta]
                    dzds[p] = 0.0;          //[dz/dxi, dz/deta, dz/dzeta]
                }
                for (int p = 0; p < 3; p++){
                    for (int m = 0; m < 8; m++){
                        dxds[p] += xe[3*m+0] * dN[3*m + p];
                        dyds[p] += xe[3*m+1] * dN[3*m + p];
                        dzds[p] += xe[3*m+2] * dN[3*m + p];
                    }
                }

                // jacobian
                jaco =dxds[0]*(dyds[1]*dzds[2] - dzds[1]*dyds[2]) + dyds[0]*(dzds[1]*dxds[2] - dxds[1]*dzds[2]) +
                      dzds[0]*(dxds[1]*dyds[2] - dyds[1]*dxds[2]);
                assert(jaco >= 0.0);
                if (jaco <= 0.0) {
                    printf(" Jacobian is negative!");
                    exit(0);
                }

                // dxi/dx, dxi/dy, dxi/dz
                dxids[0] = (dzds[2]*dyds[1] - dyds[2]*dzds[1])/jaco;
                dxids[1] = (dxds[2]*dzds[1] - dzds[2]*dxds[1])/jaco;
                dxids[2] = (dyds[2]*dxds[1] - dxds[2]*dyds[1])/jaco;

                // deta/dx, deta/dy, deta/dz
                detads[0] = (dyds[2]*dzds[0] - dzds[2]*dyds[0])/jaco;
                detads[1] = (dzds[2]*dxds[0] - dxds[2]*dzds[0])/jaco;
                detads[2] = (dxds[2]*dyds[0] - dyds[2]*dxds[0])/jaco;

                // dzeta/dx, dzeta/dy, dzeta/dz
                dzetads[0] = (dzds[1]*dyds[0] - dyds[1]*dzds[0])/jaco;
                dzetads[1] = (dxds[1]*dzds[0] - dzds[1]*dxds[0])/jaco;
                dzetads[2] = (dyds[1]*dxds[0] - dxds[1]*dyds[0])/jaco;

                // B-matrix
                for (int m = 0; m < 8; m++){
                    // first row of [B]
                    B0[m] = dN[m*3]*dxids[0] + dN[m*3 + 1]*detads[0] + dN[m*3 + 2]*dzetads[0];
                    // second row of [B]
                    B1[m] = dN[m*3]*dxids[1] + dN[m*3 + 1]*detads[1] + dN[m*3 + 2]*dzetads[1];
                    // third row of [B]
                    B2[m] = dN[m*3]*dxids[2] + dN[m*3 + 1]*detads[2] + dN[m*3 + 2]*dzetads[2];
                }

                // ke matrix
                for (int row = 0; row < 8; row++){
                    for (int col = 0; col < 8; col++){
                        ke(row,col) += (B0[row]*B0[col] + B1[row]*B1[col] + B2[row]*B2[col]) * jaco * w[2] * w[1] * w[0];
                    }
                }
                delete [] dN;
            } // k integration
        } // j integration
    } // i integration
    delete [] xw;
}

void ke_hex8_eig_test(Eigen::Matrix<double,8,8> &ke, double* xe){
    for (unsigned int i = 0; i < 8; i++){
        for (unsigned int j = 0; j < 8; j++){
            ke(i,j) = 1.0;
        }
    }
}

void ke_quad4_eig(Eigen::Matrix<double,4,4> &ke, double* xe){
    const int NGT = 2; // number of Gauss points in each direction
    double x[2], w[2]; // x=[xi,eta], w=[weight_xi, weight_eta]
    double *xw;
    double *dN;
    double dxds[2], dyds[2];
    double dxids[2], detads[2];
    double jaco;
    double B0[4], B1[4];

    for (unsigned int i = 0; i < 4; i++) {
        for (unsigned int j = 0; j < 4; j++){
            ke(i,j) = 0.0;
        }
    }

    xw = gauss(NGT); // coordinates and weights of Gauss points

    for (unsigned int i = 0; i < NGT; i++){
        x[0] = xw[i*2];                     // xi
        w[0] = xw[i*2 + 1];                 // weight_xi
        for (unsigned int j = 0; j < NGT; j++){
            x[1] = xw[j*2];                 // eta
            w[1] = xw[j*2 + 1];             // weight_eta

            // dN[0] = dN0/dxi; dN[1] = dN0/deta; dN[2] = dN1/dxi; dN[3] = dN1/deta ...
            dN = dfbasis_quad4(x);

            // initialize [dx/dix, dx/deta] and [dy/dxi, dy/deta]
            for (int p = 0; p < 2; p++){
                dxds[p] = 0.0;          //[dx/dxi, dx/deta]
                dyds[p] = 0.0;          //[dy/dxi, dy/deta]
            }
            for (int p = 0; p < 2; p++){
                for (int m = 0; m < 4; m++){
                    dxds[p] += xe[2 * m] * dN[2 * m + p];
                    dyds[p] += xe[2 * m + 1] * dN[2 * m + p];
                }
            }

            // jacobian
            jaco = dxds[0] * dyds[1] - dxds[1] * dyds[0];
            assert(jaco > 0.0);

            // dxi/dx, dxi/dy (inverse of jacobian matrix)
            dxids[0] = dyds[1]/jaco;
            dxids[1] = -dxds[1]/jaco;

            // deta/dx, deta/dy
            detads[0] = -dyds[0]/jaco;
            detads[1] = dxds[0]/jaco;

            // B-matrix
            for (int m = 0; m < 4; m++){
                // first row of [B]
                B0[m] = dN[m*2]*dxids[0] + dN[(m*2) + 1]*detads[0];
                // second row of [B]
                B1[m] = dN[m*2]*dxids[1] + dN[(m*2) + 1]*detads[1];
            }

            // ke matrix: K_ij = dNi/dx * dNj/dx + dNi/dy * dNj/dy;
            for (int row = 0; row < 4; row++){
                for (int col = 0; col < 4; col++){
                    ke(row,col) += (B0[row]*B0[col] + B1[row]*B1[col]) * jaco * w[1] * w[0];
                }
            }

            // since dfbasis_quad4() allocated memory for dN --> free memory here!
            delete [] dN;
        } // j integration
    } // i integration

    // since gauss() allocated memory for xw --> free memory here!
    delete [] xw;
}