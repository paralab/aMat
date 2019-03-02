//
// Created by Han Tran on 2/18/19.
//
#include <iostream>
#include "Dense"

#ifndef TEST_EIGEN_AMAT_H
#define TEST_EIGEN_AMAT_H

enum Error {SUCCESS, INDEX_OUT_OF_BOUNDS, UNKNOWN_ELEMENT_TYPE, UNKNOWN_ELEMENT_STATUS};

template <typename T, typename I>
class aMat{

    typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> EigenMat;

public:

    EigenMat mat;

    // default constructor
    aMat()
    {

    }

    // constructor1: specify matrix by dimension
    aMat(I rows, I cols){
        mat.resize(rows,cols);
        for (I i = 0; i < rows; i++){
            for (I j = 0; j < cols; j++){
                mat(i,j) = 1;
            }
        }
    }

    // constructor 2: specify matrix by pre-defined matrix
    aMat(EigenMat pMat){
        mat = pMat;
        I rows = mat.rows();
        I cols = mat.cols();
        /*for (I i = 0; i < rows; i++){
            for (I j = 0; j < cols; j++){
                mat(i,j) = 0;
            }
        }*/
    }

    // function to print matrix
    Error print_matrix(){
        std::cout << "print from aMat: matrix = \n" << mat << std::endl;
        return SUCCESS;
    }
};


#endif //TEST_EIGEN_AMAT_H
