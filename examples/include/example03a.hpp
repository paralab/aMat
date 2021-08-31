#pragma once
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <functional>

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

struct AppData {
    unsigned int Nex, Ney;
    double hx, hy, hz;

    unsigned int NGT;
    integration<double>* intData;
    
    unsigned int NNODE_PER_ELEM;
    
    unsigned long ** globalMap;
};

extern AppData example03aAppData;