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

#include "Eigen/Dense"
#include "aMat.hpp"
#include "aMatBased.hpp"
#include "aMatFree.hpp"
#include "aVec.hpp"
#include "constraintRecord.hpp"
#include "enums.hpp"

#include "integration.hpp"
#include "ke_matrix.hpp"
#include "maps.hpp"
#include "solve.hpp"

using Eigen::Matrix;

struct AppData {
    unsigned int Nex;
    double hx, hy;
    unsigned int NDOF_PER_ELEM;
    unsigned long ** globalMap;
    unsigned int NGT;
    integration<double>* intData;
};

extern AppData example02aAppData;