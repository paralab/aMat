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
#include "GmshMesh.hpp"

#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct AppData
{
    unsigned int NGT;
    integration<double>* intData = nullptr;
    GmshMesh * p_mesh;
};

extern AppData example05aAppData;