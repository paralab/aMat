#pragma once
#include <fstream>
#include <iostream>
#include <mpi.h>

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
#include "profiler.hpp"
#include "solve.hpp"

#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct AppData
{
    double E;
    double nu;
    double rho;
    double g;
    double Lx, Ly, Lz;
    double hx, hy, hz;
    unsigned int NGT;
    integration<double>* intData = nullptr;
    unsigned int NNODE_PER_ELEM = 8;
    unsigned int NDOF_PER_NODE = 3;
    unsigned int Nex = 10, Ney = 10, Nez = 10;
    unsigned long ** ElementToGIDNode = nullptr;
};

extern AppData example06aAppData; 