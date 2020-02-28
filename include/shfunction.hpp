/**
 * @file shfunction.hpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief header file of shfunction.cpp
 *
 * @version 0.1
 * @date 2018-12-05
 */

#ifndef ADAPTIVEMATRIX_SHFUNCTION_H
#define ADAPTIVEMATRIX_SHFUNCTION_H

double *gauss(unsigned int n);
double *basis_hex8 (double p[]);
double *dfbasis_hex8 (double p[]);

double* basis_quad4(double p[]);
double* dfbasis_quad4(double p[]);

#endif //ADAPTIVEMATRIX_SHFUNCTION_H
