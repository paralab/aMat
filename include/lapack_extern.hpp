/**
 * @file lapack_extern.hpp
 * @brief : Lapack header definitions. 
 * @version 0.1
 * @date 2021-01-18
 * 
 * @copyright Copyright (c) 2021
 * 
 */



    extern "C" void dgesv_( int* n, int* nrhs, double* a, int* lda, int* ipiv,double* b, int* ldb, int* info );
    
    extern "C" void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda,double* w, double* work, int* lwork, int* info);

    // LU decomoposition of a general matrix
    extern "C" void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    extern "C" void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);

    // generic mat-mat multiplications
    extern "C" void dgemm_(char * transa, char * transb, int * m, int * n, int * k, double * alpha, double * A, int * lda, double * B, int * ldb, double * beta, double * C, int * ldc);

    // generic matrix vector multiplication. 
    extern "C" void dgemv_(char * trans, int * m, int *n, double *alpha, double * A, int *lda, double * x, int *incx, double *beta, double* y, int* incy);


