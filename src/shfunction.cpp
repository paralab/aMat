/**
 * @file shfunction.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief functions to compute element stiffness matrices
 *
 * @version 0.1
 * @date 2018-12-07
 */

#include <iostream>
#include "shfunction.hpp"
#include <math.h>


/**
 * @brief: generate Gauss points
 * @brief: This function is extracted from FADD3D, a symmetric Galerkin boundary element program
 * @brief: for crack analysis in general anisotropic elastic media
 * @param[in] n number of Gauss points
 * @param[out] g coordinates and weights of the points
 * @authors Lin Xiao, Mark Mear (University of Texas at Austin), Han Tran translated to C++
* */

double* gauss(int n)
{
    const double EPS = 3.0e-14;
    const double x1 = -1.0;
    const double x2 = 1.0;
    const int m = (n+1)/2;
    const double xm = 0.5*(x2 + x1);
    const double xl = 0.5*(x2 - x1);
    double z, z1, p1, p2, p3, pp;
    const double PI = acos(-1.0);

    double * g;
    g = new double[2*n];

    for (int i = 1; i <= m; i++){
        z = cos(PI*((double)i - 0.25)/((double)n + 0.5));
        do {
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 1; j <= n; j++){
                p3 = p2;
                p2 = p1;
                p1 = ((2.0*(double)j - 1.0)*z*p2 - ((double)j-1.0)*p3)/(double)j;
            }
            pp = (double)n*(z*p1 - p2)/(z*z - 1.0);
            z1 = z;
            z = z1 - (p1/pp);
        } while (abs(z-z1) >= EPS);
        g[2*(i-1)] = xm - xl*z;                     // coordinate of point i (i = 1,...,n)
        g[2*(i-1) + 1] = (2.0*xl)/((1.0-z*z)*pp*pp);// weight of point i
        g[2*(n-i)] = xm + xl*z;                     // weight of point n+1-i
        g[2*(n-i) + 1] = g[2*(i-1) + 1];            // weight of point n+1-i
    }
    return g;
}


/**
 * @brief: compute shape functions of 8-node hex element
 * @param[in] x natural coordinates
 * @param[out] N values of shape functions at x
 * @author Han Tran
* */
double *basis_hex8 (double xi[])

{
    double *N = new double[8];

    N[0] = (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]) / 8.0;
    N[1] = (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]) / 8.0;
    N[2] = (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]) / 8.0;
    N[3] = (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]) / 8.0;
    N[4] = (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]) / 8.0;
    N[5] = (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]) / 8.0;
    N[6] = (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]) / 8.0;
    N[7] = (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]) / 8.0;

    return N;
}


/**
 * @brief: compute derivatives of shape functions of 8-node hex element
 * @param[in] x natural coordinates
 * @param[out] dN values of derivatives of shape functions at x
 * @author Han Tran
* */
double *dfbasis_hex8 (double xi[])
/*
Purpose: compute derivatives of shape functions of 8-node linear element
Input  : double xi[3] natural coordinates
Output : double *dN[24] values of the derivatives of shape functions
*/
{
    double *dN = new double[24];

    // dN[0]/dxi, dN[0]/deta, dN[0]/dzeta
    dN[0] = (1.0 - xi[1]) * (1.0 - xi[2]) / (-8.0);
    dN[1] = (1.0 - xi[0]) * (1.0 - xi[2]) / (-8.0);
    dN[2] = (1.0 - xi[0]) * (1.0 - xi[1]) / (-8.0);

    // dN[1]/dxi, dN[1]/deta, dN[1]/dzeta
    dN[3] = (1.0 - xi[1]) * (1.0 - xi[2]) / (8.0);
    dN[4] = (1.0 + xi[0]) * (1.0 - xi[2]) / (-8.0);
    dN[5] = (1.0 + xi[0]) * (1.0 - xi[1]) / (-8.0);

    // dN[2]/dxi, dN[2]/deta, dN[2]/dzeta
    dN[6] = (1.0 + xi[1]) * (1.0 - xi[2]) / (8.0);
    dN[7] = (1.0 + xi[0]) * (1.0 - xi[2]) / (8.0);
    dN[8] = (1.0 + xi[0]) * (1.0 + xi[1]) / (-8.0);

    // dN[3]/dxi, dN[3]/deta, dN[3]/dzeta
    dN[9] = (1.0 + xi[1]) * (1.0 - xi[2]) / (-8.0);
    dN[10] = (1.0 - xi[0]) * (1.0 - xi[2]) / (8.0);
    dN[11] = (1.0 - xi[0]) * (1.0 + xi[1]) / (-8.0);

    // dN[4]/dxi, dN[4]/deta, dN[4]/dzeta
    dN[12] = (1.0 - xi[1]) * (1.0 + xi[2]) / (-8.0);
    dN[13] = (1.0 - xi[0]) * (1.0 + xi[2]) / (-8.0);
    dN[14] = (1.0 - xi[0]) * (1.0 - xi[1]) / (8.0);

    // dN[5]/dxi, dN[5]/deta, dN[5]/dzeta
    dN[15] = (1.0 - xi[1]) * (1.0 + xi[2]) / (8.0);
    dN[16] = (1.0 + xi[0]) * (1.0 + xi[2]) / (-8.0);
    dN[17] = (1.0 + xi[0]) * (1.0 - xi[1]) / (8.0);

    // dN[6]/dxi, dN[6]/deta, dN[6]/dzeta
    dN[18] = (1.0 + xi[1]) * (1.0 + xi[2]) / (8.0);
    dN[19] = (1.0 + xi[0]) * (1.0 + xi[2]) / (8.0);
    dN[20] = (1.0 + xi[0]) * (1.0 + xi[1]) / (8.0);

    // dN[7]/dxi, dN[7]/deta, dN[7]/dzeta
    dN[21] = (1.0 + xi[1]) * (1.0 + xi[2]) / (-8.0);
    dN[22] = (1.0 - xi[0]) * (1.0 + xi[2]) / (8.0);
    dN[23] = (1.0 - xi[0]) * (1.0 + xi[1]) / (8.0);

    return dN;
}

/**
 * @brief: compute shape functions of 4-node quad element
 * @brief: allocate memory for dN --> NEED TO FREE MEMORY AFTER N IS USED
 * @param[in] x natural coordinates
 * @param[out] N values of shape functions at x
 * @author Han Tran
* */
double* basis_quad4(double xi[]) {
    // allocate memory for N --> NEED TO FREE MEMORY AFTER N IS USED
    double* N = new double[4];

    N[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]);
    N[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]);
    N[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]);
    N[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]);

    // return the address of the memory stored value of N
    return N;
}

/**
 * @brief: compute derivatives of shape functions of 4-node quad element
 * @brief: allocate memory for dN --> NEED TO FREE MEMORY AFTER dN IS USED
 * @param[in] x natural coordinates
 * @param[out] dN values of derivatives of shape functions at x
 * @author Han Tran
* */
double* dfbasis_quad4(double xi[])
{
    // allocate memory for dN --> NEED TO FREE MEMORY AFTER dN IS USED
    double* dN = new double[8];

    // dN[0]/dxi, dN[0]/deta
    dN[0] = 0.25 * (xi[1] - 1.0);
    dN[1] = 0.25 * (xi[0] - 1.0);

    // dN[1]/dxi, dN[1]/deta
    dN[2] = 0.25 * (1.0 - xi[1]);
    dN[3] = (-0.25) * (1.0 + xi[0]);

    // dN[2]/dxi, dN[2]/deta
    dN[4] = 0.25 * (xi[1] + 1.0);
    dN[5] = 0.25 * (xi[0] + 1.0);

    // dN[3]/dxi, dN[3]/deta
    dN[6] = (-0.25) * (1.0 + xi[1]);
    dN[7] = 0.25 * (1.0 - xi[0]);

    // return the address of the memory stored value of dN
    return dN;
}
