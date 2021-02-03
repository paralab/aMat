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

#include "Eigen/Dense"
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

using Eigen::Matrix;

struct AppData {
    double Lx, Ly;
    unsigned int Nex, Ney;
    double hx, hy;

    unsigned int NGT;
    integration<double>* intData = nullptr;
    
    unsigned int NNODE_PER_ELEM = 4;
    unsigned int NDOF_PER_NODE = 1;
    
    unsigned long ** globalMap = nullptr;
};

extern AppData fem2dAppData;