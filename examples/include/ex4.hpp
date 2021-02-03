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

// using Eigen::Matrix;

template<typename DT, typename LI>
class nodeData {
  private:
    LI nodeId;
    DT x;
    DT y;
    DT z;

  public:
    nodeData() {
        nodeId = 0;
        x      = 0.0;
        y      = 0.0;
        z      = 0.0;
    }

    inline LI get_nodeId() const{
        return nodeId;
    }
    inline DT get_x() const {
        return x;
    }
    inline DT get_y() const {
        return y;
    }
    inline DT get_z() const {
        return z;
    }

    inline void set_nodeId(LI id) {
        nodeId = id;
    }
    inline void set_x(DT value) {
        x = value;
    }
    inline void set_y(DT value) {
        y = value;
    }
    inline void set_z(DT value) {
        z = value;
    }

    bool operator==(nodeData const& other) const {
        return (nodeId == other.get_nodeId());
    }
    bool operator<(nodeData const& other) const {
        if (nodeId < other.get_nodeId())
            return true;
        else
            return false;
    }
    bool operator<=(nodeData const& other) const {
        return (((*this) < other) || ((*this) == other));
    }

    ~nodeData() {}
}; // class nodeData

struct AppData
{
    double E;
    double nu;
    double rho;
    double g;
    double Lx, Ly, Lz;
    unsigned int Nex = 10, Ney = 10, Nez = 10;
    double hx, hy, hz;

    unsigned int NGT;
    integration<double>* intData = nullptr;
    
    unsigned int NNODE_PER_ELEM = 20;
    unsigned int NDOF_PER_NODE = 3;
    
    unsigned int ** localMap = nullptr;
    nodeData<double, unsigned int>* localNodes;
};

extern AppData Ex4AppData; 