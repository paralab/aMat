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
    DT*	Pts_n_Wts = nullptr;  // coordinates and weights of Gauss points
    DT* Pts_nWts_tri = nullptr; // coordinates and weights of Gauss points in unit triangle
private:
    unsigned int nPoints; // number of Gauss points in one direction
    bool useTriangle = false; // indicator if using Gauss points/weights for triangle
public:
    integration(){};
    integration(unsigned int N);
    integration(unsigned int N, bool triangle);
    ~integration();

private:
    void gauss();
    void gauss_tri();
}; // class integration


// second constructor, just generate 1D points
template <typename DT>
integration<DT>::integration(const unsigned int N){
    nPoints = N;
    Pts_n_Wts = new DT [2 * nPoints];
    gauss();
}


// third constructor, with indicator of using triangle or not
template <typename DT>
integration<DT>::integration(const unsigned int N, bool triangle){
    nPoints = N;
    
    if (triangle){
        // Pts_n_Wts[3*i] and Pts_n_Wts[3*i + 1] are xi and eta of ith point
        // Pts_n_Wts[3*i + 2] is weight of ith point
        Pts_n_Wts = new DT [3 * nPoints];

        // generate Gauss points and weights for triangle
        gauss_tri();

    } else {
        // Pts_n_Wts[2*i] = xi of ith point
        // Pts_n_Wts[2*i + 1] = weight of ith point
        Pts_n_Wts = new DT [2 * nPoints];

        // generate Gauss points and weights for 1D
        gauss();
    }
}


// destructor: deallocate space storing points and weights
template <typename DT>
integration<DT>::~integration(){
    if (Pts_n_Wts != nullptr) {
        delete [] Pts_n_Wts;
        Pts_n_Wts = nullptr;
    }
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


template <typename DT>
void integration<DT>::gauss_tri(){
    // this code is refered to setint.f90 written by Lin Xiao and Mark Mear (UT Austin)
    switch (nPoints) {
        case 3:
            /* 3-point formula */
            Pts_n_Wts[0]=1./6.;
            Pts_n_Wts[1]= Pts_n_Wts[0];

            Pts_n_Wts[3]=2./3.;
            Pts_n_Wts[4]=Pts_n_Wts[0];

            Pts_n_Wts[6]=Pts_n_Wts[0];
            Pts_n_Wts[7]=Pts_n_Wts[3];

            Pts_n_Wts[2]=1./6.;
            Pts_n_Wts[5]=Pts_n_Wts[2];
            Pts_n_Wts[8]=Pts_n_Wts[2];
            break;
        case 4:
            /* 4-point formula */
            Pts_n_Wts[0]=1./3.;
            Pts_n_Wts[1]=Pts_n_Wts[0];

            Pts_n_Wts[3]=.6;
            Pts_n_Wts[4]=.2;

            Pts_n_Wts[6]=Pts_n_Wts[4];
            Pts_n_Wts[7]=Pts_n_Wts[4];

            Pts_n_Wts[9]=Pts_n_Wts[4];
            Pts_n_Wts[10]=Pts_n_Wts[3];

            Pts_n_Wts[2]=-27./96.;
            Pts_n_Wts[5]=25./96.;
            Pts_n_Wts[8]=Pts_n_Wts[5];
            Pts_n_Wts[11]=Pts_n_Wts[5];
            break;
        case 7:
            /* 7-point formula */
            Pts_n_Wts[0] = .1012865073235;
            Pts_n_Wts[1] = Pts_n_Wts[0];

            Pts_n_Wts[3] = .7974269853531;
            Pts_n_Wts[4] = Pts_n_Wts[0];

            Pts_n_Wts[6] = Pts_n_Wts[0];
            Pts_n_Wts[7] = Pts_n_Wts[3];

            Pts_n_Wts[9] = .4701420641051;
            Pts_n_Wts[10] = .0597158717898;

            Pts_n_Wts[12] = Pts_n_Wts[9];
            Pts_n_Wts[13] = Pts_n_Wts[9];

            Pts_n_Wts[15] = Pts_n_Wts[10];
            Pts_n_Wts[16] = Pts_n_Wts[9];

            Pts_n_Wts[18] = 1./3.;
            Pts_n_Wts[19] = Pts_n_Wts[18];

            Pts_n_Wts[2] = .1259391805448/2.;
            Pts_n_Wts[5] = Pts_n_Wts[2];
            Pts_n_Wts[8] = Pts_n_Wts[2];
            Pts_n_Wts[11] = .1323941527885/2.;
            Pts_n_Wts[14] = Pts_n_Wts[11];
            Pts_n_Wts[17] = Pts_n_Wts[11];
            Pts_n_Wts[20] = .225/2.;
            break;
        case 12:
            /* 12-point formula */
            Pts_n_Wts[0]= 0.063089014491502;
            Pts_n_Wts[1]= 0.063089014491502;

            Pts_n_Wts[3]=1.-2.* Pts_n_Wts[0];
            Pts_n_Wts[4]=Pts_n_Wts[0];

            Pts_n_Wts[6]=Pts_n_Wts[0];
            Pts_n_Wts[7]=1.-2.*Pts_n_Wts[0];

            Pts_n_Wts[9]=0.249286745170910;
            Pts_n_Wts[10]=0.249286745170910;

            Pts_n_Wts[12]=1.-2.*Pts_n_Wts[9];
            Pts_n_Wts[13]=Pts_n_Wts[9];

            Pts_n_Wts[15]=Pts_n_Wts[9];
            Pts_n_Wts[16]=1.-2.*Pts_n_Wts[9];

            Pts_n_Wts[18]=0.310352451033785;
            Pts_n_Wts[19]=0.053145049844816;

            Pts_n_Wts[21]=Pts_n_Wts[19];
            Pts_n_Wts[22]=Pts_n_Wts[18];

            Pts_n_Wts[24]=1.-(Pts_n_Wts[18]+Pts_n_Wts[19]);
            Pts_n_Wts[25]=Pts_n_Wts[18];

            Pts_n_Wts[27]=1.-(Pts_n_Wts[18]+Pts_n_Wts[19]);
            Pts_n_Wts[28]=Pts_n_Wts[19];

            Pts_n_Wts[30]=Pts_n_Wts[18];
            Pts_n_Wts[31]=1.-(Pts_n_Wts[18]+Pts_n_Wts[19]);

            Pts_n_Wts[33]=Pts_n_Wts[19];
            Pts_n_Wts[34]=1.-(Pts_n_Wts[18]+Pts_n_Wts[19]);

            Pts_n_Wts[2]=0.025422453185103;
            Pts_n_Wts[5]=Pts_n_Wts[2];
            Pts_n_Wts[8]=Pts_n_Wts[2];
            Pts_n_Wts[11]=0.058393137863189;
            Pts_n_Wts[14]=Pts_n_Wts[11];
            Pts_n_Wts[17]=Pts_n_Wts[11];
            Pts_n_Wts[20]=0.041425537809187;
            Pts_n_Wts[23]=Pts_n_Wts[20];
            Pts_n_Wts[26]=Pts_n_Wts[20];
            Pts_n_Wts[29]=Pts_n_Wts[20];
            Pts_n_Wts[32]=Pts_n_Wts[20];
            Pts_n_Wts[35]=Pts_n_Wts[20];
            break;
        case 13:
            /* 13-point formula */
            Pts_n_Wts[0] = .0651301029022;
            Pts_n_Wts[1] = Pts_n_Wts[0];
            Pts_n_Wts[3] = .8697397941956;
            Pts_n_Wts[4] = Pts_n_Wts[0];
            Pts_n_Wts[6] = Pts_n_Wts[0];
            Pts_n_Wts[7] = Pts_n_Wts[3];
            Pts_n_Wts[9] = .3128654960049;
            Pts_n_Wts[10] = .0486903154253;
            Pts_n_Wts[12] = .6384441885698;
            Pts_n_Wts[13] = Pts_n_Wts[9];
            Pts_n_Wts[15] = Pts_n_Wts[10];
            Pts_n_Wts[16] = Pts_n_Wts[12];
            Pts_n_Wts[18] = Pts_n_Wts[12];
            Pts_n_Wts[19] = Pts_n_Wts[15];
            Pts_n_Wts[21] = Pts_n_Wts[9];
            Pts_n_Wts[22] = Pts_n_Wts[12];
            Pts_n_Wts[24] = Pts_n_Wts[15];
            Pts_n_Wts[25] = Pts_n_Wts[9];
            Pts_n_Wts[27] = .2603459660790;
            Pts_n_Wts[28] = Pts_n_Wts[27];
            Pts_n_Wts[30] = .4793080678419;
            Pts_n_Wts[31] = Pts_n_Wts[27];
            Pts_n_Wts[33] = Pts_n_Wts[27];
            Pts_n_Wts[34] = Pts_n_Wts[30];
            Pts_n_Wts[36] = 1./3.;
            Pts_n_Wts[37] = Pts_n_Wts[36];

            Pts_n_Wts[2] = .0533472356088/2.;
            Pts_n_Wts[5] = Pts_n_Wts[2];
            Pts_n_Wts[8] = Pts_n_Wts[2];
            Pts_n_Wts[11] = .0771137608903/2.;
            Pts_n_Wts[14] = Pts_n_Wts[11];
            Pts_n_Wts[17] = Pts_n_Wts[11];
            Pts_n_Wts[20] = Pts_n_Wts[11];
            Pts_n_Wts[23] = Pts_n_Wts[11];
            Pts_n_Wts[26] = Pts_n_Wts[11];
            Pts_n_Wts[29] = .1756152574332/2.;
            Pts_n_Wts[32] = Pts_n_Wts[29];
            Pts_n_Wts[35] = Pts_n_Wts[29];
            Pts_n_Wts[38] = -.1495700444677/2.;
            break;
        case 25:
            /* 25-point formua */
            Pts_n_Wts[0]= 0.333333333333332982;
            Pts_n_Wts[1]= 0.333333333333332982;
            Pts_n_Wts[3]= 0.485577633383657004;
            Pts_n_Wts[4]= 0.485577633383657004;
            Pts_n_Wts[6]= 0.485577633383657004;
            Pts_n_Wts[7]= 0.288447332326850006E-01;
            Pts_n_Wts[9]= 0.288447332326850006E-01;
            Pts_n_Wts[10]= 0.485577633383657004;
            Pts_n_Wts[12]= 0.109481575485036994;
            Pts_n_Wts[13]= 0.109481575485036994;
            Pts_n_Wts[15]= 0.109481575485036994;
            Pts_n_Wts[16]= 0.781036849029925984;
            Pts_n_Wts[18]= 0.781036849029925984;
            Pts_n_Wts[19]= 0.109481575485036994;
            Pts_n_Wts[21]= 0.307939838764121010;
            Pts_n_Wts[22]= 0.550352941820999031;
            Pts_n_Wts[24]= 0.550352941820999031;
            Pts_n_Wts[25]= 0.141707219414879987;
            Pts_n_Wts[27]= 0.141707219414879987;
            Pts_n_Wts[28]= 0.307939838764121010;
            Pts_n_Wts[30]= 0.307939838764121010;
            Pts_n_Wts[31]= 0.141707219414879987;
            Pts_n_Wts[33]= 0.550352941820999031;
            Pts_n_Wts[34]= 0.307939838764121010;
            Pts_n_Wts[36]= 0.141707219414879987;
            Pts_n_Wts[37]= 0.550352941820999031;
            Pts_n_Wts[39]= 0.246672560639903005;
            Pts_n_Wts[40]= 0.728323904597411032;
            Pts_n_Wts[42]= 0.728323904597411032;
            Pts_n_Wts[43]= 0.250035347626859986E-01;
            Pts_n_Wts[45]= 0.250035347626859986E-01;
            Pts_n_Wts[46]= 0.246672560639903005;
            Pts_n_Wts[48]= 0.246672560639903005;
            Pts_n_Wts[49]= 0.250035347626859986E-01;
            Pts_n_Wts[51]= 0.728323904597411032;
            Pts_n_Wts[52]= 0.246672560639903005;
            Pts_n_Wts[54]= 0.250035347626859986E-01;
            Pts_n_Wts[55]= 0.728323904597411032;
            Pts_n_Wts[57]= 0.668032510121999989E-01;
            Pts_n_Wts[58]= 0.923655933587499978;
            Pts_n_Wts[60]= 0.923655933587499978;
            Pts_n_Wts[61]= 0.954081540029899991E-02;
            Pts_n_Wts[63]= 0.954081540029899991E-02;
            Pts_n_Wts[64]= 0.668032510121999989E-01;
            Pts_n_Wts[66]= 0.668032510121999989E-01;
            Pts_n_Wts[67]= 0.954081540029899991E-02;
            Pts_n_Wts[69]= 0.923655933587499978;
            Pts_n_Wts[70]= 0.668032510121999989E-01;
            Pts_n_Wts[72]= 0.954081540029899991E-02;
            Pts_n_Wts[73]= 0.923655933587499978;

            Pts_n_Wts[2]= 0.908179903827539964E-01/2.;
            Pts_n_Wts[5]= 0.367259577564669967E-01/2.;
            Pts_n_Wts[8]= 0.367259577564669967E-01/2.;
            Pts_n_Wts[11]= 0.367259577564669967E-01/2.;
            Pts_n_Wts[14]= 0.453210594355279994E-01/2.;
            Pts_n_Wts[17]= 0.453210594355279994E-01/2.;
            Pts_n_Wts[20]= 0.453210594355279994E-01/2.;
            Pts_n_Wts[23]= 0.727579168454200037E-01/2.;
            Pts_n_Wts[26]= 0.727579168454200037E-01/2.;
            Pts_n_Wts[29]= 0.727579168454200037E-01/2.;
            Pts_n_Wts[32]= 0.727579168454200037E-01/2.;
            Pts_n_Wts[35]= 0.727579168454200037E-01/2.;
            Pts_n_Wts[38]= 0.727579168454200037E-01/2.;
            Pts_n_Wts[41]= 0.283272425310569995E-01/2.;
            Pts_n_Wts[44]= 0.283272425310569995E-01/2.;
            Pts_n_Wts[47]= 0.283272425310569995E-01/2.;
            Pts_n_Wts[50]= 0.283272425310569995E-01/2.;
            Pts_n_Wts[53]= 0.283272425310569995E-01/2.;
            Pts_n_Wts[56]= 0.283272425310569995E-01/2.;
            Pts_n_Wts[59]= 0.942166696373300007E-02/2.;
            Pts_n_Wts[62]= 0.942166696373300007E-02/2.;
            Pts_n_Wts[65]= 0.942166696373300007E-02/2.;
            Pts_n_Wts[68]= 0.942166696373300007E-02/2.;
            Pts_n_Wts[71]= 0.942166696373300007E-02/2.;
            Pts_n_Wts[74]= 0.942166696373300007E-02/2.;
            break;
        default:
            printf("No formulation for triangle Gauss points with n = %d\n", nPoints);
            exit(0);
    }
}

#endif// DAPTIVEMATRIX_INTEGRATION_H
