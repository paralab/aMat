//
// Created by Han Tran on 12/7/18.
//
#include <iostream>
#include "../include/shfunction.hpp"
#include "me_matrix.hpp"
#include <math.h>

double *me_hex8(double *xe)
//****************************************************************************80
/*
Purpose: element mass matrix of 8-node hex
Author : Han Tran
Input  : double xe[8*3], nodal coordinates of the element
Output : double me_hex8[8*8], element mass matrix
*/
{
    const int NGT = 2; // number of Gauss points in each direction
    double x[3], w[3];
    double *xw;
    double *N;
    double *dN;
    double dxds[3], dyds[3], dzds[3];
    double jaco;
    int idx;

    double *me = new double[8*8];
    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 8; j++){
            me[i*8 + j] = 0.0;
        }
    }

    xw = gauss(NGT); // coordinates and weights of Gauss points

    for (int i = 0; i < NGT; i++){
        x[0] = xw[i*2];                     // xi
        w[0] = xw[i*2 + 1];                 // wxi
        for (int j = 0; j < NGT; j++){
            x[1] = xw[j*2];                 // eta
            w[1] = xw[j*2 + 1];             // weta
            for (int k = 0; k < NGT; k++){
                x[2] = xw[k*2];             // zeta
                w[2] = xw[k*2 + 1];         // wzeta
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

                // me matrix
                for (int i = 0; i < 8; i++){
                    for (int j = 0; j < 8; j++){
                        idx = (i*8) + j;
                        me[idx] += N[i] * N[j] * jaco * w[2] * w[1] * w[0];
                    }
                }

            } // k integration
        } // j integration
    } // i integration

    return me;
}
