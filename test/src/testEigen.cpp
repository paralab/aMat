//
// Created by Han Tran on 2/20/19.
//

#include <iostream>
#include <Dense>
#include "aMat.h" // it is not aMat.hpp of
#include "ke_matrix.hpp"
#include "iostream"
using namespace std;
using Eigen::MatrixXd;
using Eigen::Matrix;

// function using Eigen matrix
Matrix<double,2,3> square_matrix(Matrix<double,2,3> m){
    Matrix<double,2,3> k;
    for (unsigned i = 0; i < 2; i++){
        for (unsigned j = 0; j < 3; j++){
            k(i,j) = m(i,j) * m(i,j);
        }
    }
    return k;
}

int main()
{
    /* ==================== BASIS of Eigen matrix ============================== */

    // matrix of double type, fixed size 2x2 -> cannot change size later, data in stack
    Eigen::Matrix2d a;

    // matrix of type double, not defined dimension yet -> can change size (using q.resize(r,c)) -> put in heap
    MatrixXd q; // this is the same as Eigen::Matrix<double,Eigen::dynamic,Eigen::dynamic> q;

    // matrix of type double, constructor with size
    MatrixXd m(2,2), k(3,3), n(2,3);

    // matrix of type float, specify dimension at declaration
    Eigen::MatrixXf p(2,3);

    // matrix of type double, dynamic dimension (will be defined later, see below)
    Matrix<double,Eigen::Dynamic,Eigen::Dynamic> b, c;

    // pointer to matrix of type double, undefined dimension
    MatrixXd* mpt;
    MatrixXd* npt;

    // array of matrix (pointer to matrix) of type double, undefined dimension
    MatrixXd* arraypt = new MatrixXd[2];

    // set values for matrix
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);

    n(0,0) = 1;
    n(0,1) = 2;
    n(0,2) = 3;
    n(1,0) = 4;
    n(1,1) = 5;
    n(1,2) = 6;

    // call function
    n = square_matrix(n);

    // print out matrix
    std::cout << "m = \n" << m << std::endl;
    std::cout << "n = \n" << n << std::endl;

    // try to print out undefined matrix q, it will do nothing
    std::cout << "before assign, q = \n" << q << std::endl;

    // assign undefined matrix
    q = m;

    // now print q
    std::cout << "after assign q = m = \n" << q << std::endl;

    // assign b to m, then b has dimension of m
    b = m;

    // specify dimension of c
    c.resize(3,4);

    // get the size of matrix after resizing
    std::cout << "size of c = " << c.size() << std::endl;

    // assign address of matrix to pointer
    mpt = &m; // this is given to us
    npt = &n;

    // get the size of pointer to matrix
    std::cout << "row size of npt = " << npt->rows() << std::endl;
    std::cout << "column size of npt = " << npt->cols() << std::endl;


    // each member of the array can have different dimension
    arraypt[0] = *mpt; // mpt points to m which is (2,2)
    arraypt[1] = k;    // k is (3,3)


    /* ==================== Using Eigen matrix in aMat class ============================== */

    // constructor 1: using dimension
    aMat<double, unsigned long> emat1(3,4);
    emat1.print_matrix();

    // constructor 2: using pre-defined matrix, p must be the same type as template type (here is float)
    p(0,0) = 1; p(0,1) = 2; p(0,2) = 3;
    p(1,0) = 1; p(1,1) = 2; p(1,2) = 3;
    aMat<float, unsigned> emat(p);
    emat.print_matrix();

    // default constructor: nothing is specified, thus the function print will do nothing
    aMat<double, unsigned > emat2;
    emat2.print_matrix();

    /* ==================== test element stiffness matrix ============================== */
    Matrix<double,8,8> ke;
    double* xe = new double[24];
    double L = 1.0;

    xe[0] = 0.0;
    xe[1] = 0.0;
    xe[2] = 0.0;
    xe[3] = L;
    xe[4] = 0.0;
    xe[5] = 0.0;
    xe[6] = L;
    xe[7] = L;
    xe[8] = 0.0;
    xe[9] = 0.0;
    xe[10] = L;
    xe[11] = 0.0;

    xe[12] = 0.0;
    xe[13] = 0.0;
    xe[14] = L;
    xe[15] = L;
    xe[16] = 0.0;
    xe[17] = L;
    xe[18] = L;
    xe[19] = L;
    xe[20] = L;
    xe[21] = 0.0;
    xe[22] = L;
    xe[23] = L;

    ke_hex8_eig(ke,xe);

    std::cout << "ke matrix =\n" << ke << std::endl;

}