/**
 * @file integration.hpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief class for Gauss points and weights
 *
 * @version 0.1
 * @date 2020-02-29
 */

#ifndef ADAPTIVEMATRIX_INTEGRATION_H
#define ADAPTIVEMATRIX_INTEGRATION_H

#include <math.h>

template <typename DT>
class integration {

public:
    DT*	Pts_n_Wts;  // coordinates and weights of Gauss points

private:
    unsigned int nPoints; // number of Gauss points in one direction

public:
    integration(unsigned int N);
    ~integration();

private:
    void gauss();

}; // class integration

template <typename DT>
integration<DT>::integration(const unsigned int N){
    nPoints = N;
    Pts_n_Wts = new DT [2 * nPoints];
    gauss();
}

template <typename DT>
integration<DT>::~integration(){
    if (Pts_n_Wts != nullptr) delete [] Pts_n_Wts;
}

template <typename DT>
void integration<DT>::gauss(){

    const double EPS = 3.0e-14;
    const double x1 = -1.0;
    const double x2 = 1.0;

    const unsigned int m = (nPoints + 1)/2;
    
    const double xm = 0.5*(x2 + x1);
    const double xl = 0.5*(x2 - x1);
    double z, z1, p1, p2, p3, pp;

    for (unsigned int i = 1; i <= m; i++){
        z = cos(M_PI*((double)i - 0.25)/((double)nPoints + 0.5));
        do {
            p1 = 1.0;
            p2 = 0.0;
            for (unsigned int j = 1; j <= nPoints; j++){
                p3 = p2;
                p2 = p1;
                p1 = ((2.0*(double)j - 1.0)*z*p2 - ((double)j-1.0)*p3)/(double)j;
            }
            pp = (double)nPoints * (z * p1 - p2)/(z * z - 1.0);
            z1 = z;
            z = z1 - (p1/pp);
        } while (fabs(z-z1) >= EPS);
        Pts_n_Wts[2*(i-1)] = xm - xl*z;                     // coordinate of point i (i = 1,...,n)
        Pts_n_Wts[2*(i-1) + 1] = (2.0*xl)/((1.0-z*z)*pp*pp);// weight of point i
        Pts_n_Wts[2*(nPoints - i)] = xm + xl*z;                     // coordinate of point n+1-i
        Pts_n_Wts[2*(nPoints - i) + 1] = Pts_n_Wts[2*(i-1) + 1];    // weight of point n+1-i
    }
}

#endif// DAPTIVEMATRIX_INTEGRATION_H
