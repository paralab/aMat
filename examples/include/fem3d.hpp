#pragma once
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#ifdef BUILD_WITH_PETSC
#include <petsc.h>
#endif

#include "aMat.hpp"
#include "aMatBased.hpp"
#include "aMatFree.hpp"
#include "aVec.hpp"
#include "constraintRecord.hpp"
#include "enums.hpp"
#include "fe_vector.hpp"
#include "integration.hpp"
#include "ke_matrix.hpp"
#include "maps.hpp"
#include "solve.hpp"

#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct AppData
{
    double L;
    unsigned int Nex = 10, Ney = 10, Nez = 10;
    double hx, hy, hz;

    unsigned int NGT;
    integration<double>* intData = nullptr;
    
    unsigned int NNODE_PER_ELEM = 8;
    unsigned int NDOF_PER_NODE = 1;
    
    unsigned long ** globalMap = nullptr;
};

extern AppData fem3dAppData;