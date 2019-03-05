/**
 * @file fe_matrix.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief functions to compute element load vectors
 *
 * @version 0.1
 * @date 2018-12-07
 */
#include <iostream>
#include "../include/shfunction.hpp"
#include "fe_vector.hpp"
#include <math.h>


/**
 * @brief: element load vector of 8-node hex element for Poisson equation
 * @param[in] xe nodal coordinates
 * @param[out] fe element load vector
 * @author Han Tran
* */

void fe_hex8(double* fe,const double* xe) {

    const int NGT = 2; // number of Gauss points in each direction
    double x[3], w[3]; // x=[xi,eta,zeta], w=[weight_xi, weight_eta,weight_zeta]
    double *xw;
    double *N;
    double *dN;
    double dxds[3], dyds[3], dzds[3];
    double jaco;

    double xp, yp, zp, force;

    for (int i = 0; i < 8; i++) {
        fe[i] = 0.0;
    }

    xw = gauss(NGT); // coordinates and weights of Gauss points, e.g. xw[1]: point 1, xw[2]: weight of point 1

    for (int i = 0; i < NGT; i++){
        x[0] = xw[i*2];                     // xi
        w[0] = xw[i*2 + 1];                 // weight_xi
        for (int j = 0; j < NGT; j++){
            x[1] = xw[j*2];                 // eta
            w[1] = xw[j*2 + 1];             // weight_eta
            for (int k = 0; k < NGT; k++){
                x[2] = xw[k*2];             // zeta
                w[2] = xw[k*2 + 1];         // weight_zeta

                N = basis_hex8(x);
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

                // physical coordinates of Gauss points
                xp = 0.0;
                yp = 0.0;
                zp = 0.0;
                for (unsigned int n = 0; n < 8; n++){
                    xp += xe[n*3] * N[n];
                    yp += xe[(n*3) + 1] * N[n];
                    zp += xe[(n*3) + 2] * N[n];
                }
                // loading function (USER-DEFINED, ad-hoc at the moment, fixme)
                force = sin(2*M_PI*xp) * sin(2*M_PI*yp) * sin(2*M_PI*zp);

                // fe vector
                for (unsigned int i = 0; i < 8; i++){
                    fe[i] += force * N[i] * jaco * w[2] * w[1] * w[0];
                }
            } // k integration
        } // j integration
    } // i integration
}
