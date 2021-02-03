/**
 * @file ex6.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Example: A square plate under uniform traction tz = 1 on top surface, and tz = -1 on bottom surface
 * @brief this is the next step of ex6a, update local map when crack appears
 * 
 * @brief Partition of elements in z direction: owned elements in z direction ~ Nez/(number of ranks)
 * @brief Size of the domain: Lx = Ly = Lz = 1.0
 * 
 * @version 0.1
 * @date 2018-11-30
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#ifdef BUILD_WITH_PETSC
#    include <petsc.h>
#endif

#include <Eigen/Dense>
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

// number of cracks allowed in 1 element
#define MAX_CRACK_LEVEL 1

// max number of block dimensions in one cracked element
//#define MAX_BLOCKS_PER_ELEMENT (1u<<MAX_CRACK_LEVEL)
#define MAX_BLOCKS_PER_ELEMENT 4
#define MAX_BLOCKS_PER_DIM 2

//////////////////////////////////////////////////////////////////////////////////////////////////////

void usage() {
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  ex6 <Nex> <Ney> <Nez> <matrix based/free> <bc-method>\n";
    std::cout << "\n";
    std::cout << "     Nex: Number of elements in X\n";
    std::cout << "     Ney: Number of elements in y\n";
    std::cout << "     Nez: Number of elements in z\n";
    std::cout << "     use matrix-free: 1 => yes.  0 => matrix-based method. \n";
    std::cout << "     use identity-matrix: 0    use penalty method: 1 \n";
    exit(0);
}

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

    inline LI get_nodeId() const {
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
};
//////////////////////////////////////////////////////////////////////////////////////////////////////

int main( int argc, char *argv[] ) {
    // User provides: Nex - number of elements (global) in x direction
    //                Ney - number of elements (global) in y direction
    //                Nez - number of elements (global) in z direction
    //                flag - 1 matrix-free method, 0 matrix-based method
    //                bcMethod = 0 --> identity matrix method; 1 --> penalty method
    if( argc < 6 ) {
        usage();
    }

    int rc;
    double zmin, zmax;

    double x, y, z;
    double hx, hy, hz;

    unsigned  int emin = 0, emax = 0;

    const unsigned int NDOF_PER_NODE = 3;       // number of dofs per node
    const unsigned int NDIM = 3;                // number of dimension
    const unsigned int NNODE_PER_ELEM = 8;      // number of nodes per element

    // material properties
    const double E = 1.0;
    const double nu = 0.3;

    // number of (global) elements in x, y and z directions
    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]);
    const unsigned int Nez = atoi(argv[3]);

    //05.19.20 only use Eigen matrix
    const bool useEigen = true;

    const unsigned int matType = atoi(argv[4]); // use matrix-free method
    const unsigned int bcMethod = atoi(argv[5]); // method of applying BC

    // domain sizes: Lx, Ly, Lz - length of the (global) domain in x/y/z direction
    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

    // element sizes
    hx = Lx/double(Nex);// element size in x direction
    hy = Ly/double(Ney);// element size in y direction
    hz = Lz/double(Nez);// element size in z direction

    const double zero_number = 1E-12;

    PetscInitialize(&argc, &argv, NULL, NULL);

    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Status Stat;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // element matrix and vector
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EigenMat;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> EigenVec;

    EigenMat** kee;
    EigenVec** fee;

    kee = new EigenMat* [MAX_BLOCKS_PER_ELEMENT];
    fee = new EigenVec* [MAX_BLOCKS_PER_ELEMENT];
    for (unsigned int i = 1; i < MAX_BLOCKS_PER_ELEMENT; i++){
        kee[i] = nullptr;
        fee[i] = nullptr;
    }

    // at beginning, only 1 block per element (no crack)
    kee[0] = new EigenMat;
    fee[0] = new EigenVec;

    // nodal coordinates of element
    double* xe = new double[NDIM * NNODE_PER_ELEM];

    // timing variables
    profiler_t aMat_time;
    profiler_t petsc_time;
    profiler_t total_time;
    if (matType != 0){
        aMat_time.clear();
    } else {
        petsc_time.clear();
    }
    total_time.clear();

    if (!rank){
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNex : "<< Nex << " Ney: " << Ney << " Nez: " << Nez << "\n";
        std::cout << "\t\tLx : "<< Lx << " Ly: " << Ly << " Lz: " << Lz << "\n";
        std::cout << "\t\tMethod (0 = matrix based; 1 = matrix free) = " << matType << "\n";
        std::cout << "\t\tBC method (0 = 'identity-matrix'; 1 = penalty): " << bcMethod << "\n";
    }
    
    #ifdef VECTORIZED_AVX512
    if (!rank) {std::cout << "\t\tVectorization using AVX_512\n";}
    #elif VECTORIZED_AVX256
    if (!rank) {std::cout << "\t\tVectorization using AVX_256\n";}
    #elif VECTORIZED_OPENMP
    if (!rank) {std::cout << "\t\tVectorization using OpenMP\n";}
    #elif VECTORIZED_OPENMP_ALIGNED
    if (!rank) {std::cout << "\t\tVectorization using OpenMP with aligned memory\n";}
    #else
    if (!rank) {std::cout << "\t\tNo vectorization\n";}
    #endif

    #ifdef HYBRID_PARALLEL
    if (!rank) {
        std::cout << "\t\tHybrid parallel OpenMP + MPI\n";
        std::cout << "\t\tMax number of threads: "<< omp_get_max_threads() << "\n";
        std::cout << "\t\tNumber of MPI processes: "<< size << "\n";
    }
    #else
    if (!rank) {
        std::cout << "\t\tOnly MPI parallel\n";
        std::cout << "\t\tNumber of MPI processes: "<< size << "\n";
    }
    #endif
    
    if (rank == 0) {
        if (size > Nez) {
            printf("The number of processes must be less than or equal Nez, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // partition number of elements in z direction
    unsigned int nelem_z;
    // minimum number of elements in z-dir for each rank
    unsigned int nzmin = Nez/size;
    // remaining
    unsigned int nRemain = Nez % size;
    // distribute nRemain uniformly from rank = 0 up to rank = nRemain - 1
    if (rank < nRemain){
        nelem_z = nzmin + 1;
    } else {
        nelem_z = nzmin;
    }
    if (rank < nRemain){
        emin = rank * nzmin + rank;
    } else {
        emin = rank * nzmin + nRemain;
    }
    emax = emin + nelem_z - 1;

    // number of owned elements
    unsigned int nelem_y = Ney;
    unsigned int nelem_x = Nex;
    unsigned int nelem = (nelem_x) * (nelem_y) * (nelem_z);

    // origin of each rank
    double origin[3] = { 0.0 };
    origin[2]        = emin * hz;

    // generate nodes...
    std::vector<nodeData<double, unsigned int>> localNodes;
    nodeData<double, unsigned int> node;
    unsigned int nid = 0;
    for (unsigned int k = 0; k < nelem_z + 1; k++){
        z = k * hz + origin[2];
        for (unsigned int j = 0; j < nelem_y + 1; j++){
            y = j * hy + origin[1];
            for (unsigned int i = 0; i < nelem_x + 1; i++){
                x = i * hx + origin[0];
                node.set_nodeId(nid);
                node.set_x(x);
                node.set_y(y);
                node.set_z(z);
                localNodes.push_back(node);
                nid++;
            }
        }
    }
    // total number of local nodes
    unsigned int numLocalNodes = localNodes.size();

    // number of local dofs
    unsigned int numLocalDofs = numLocalNodes * NDOF_PER_NODE;

    // build local nodal map:
    unsigned int* * localMap;
    localMap = new unsigned int* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localMap[eid] = new unsigned int[MAX_BLOCKS_PER_DIM * NNODE_PER_ELEM];
    }
    for (unsigned k = 0; k < nelem_z; k++){
        for (unsigned int j = 0; j < nelem_y; j++){
            for (unsigned int i = 0; i < nelem_x; i++){
                unsigned int elemID = nelem_x * nelem_y * k + nelem_x * j + i;
                localMap[elemID][0] = i + j * (Nex + 1) + k * (Nex + 1) * (Ney + 1);
                localMap[elemID][1] = localMap[elemID][0] + 1;
                localMap[elemID][3] = localMap[elemID][0] + (Nex + 1);
                localMap[elemID][2] = localMap[elemID][3] + 1;
                localMap[elemID][4] = localMap[elemID][0] + (Nex + 1) * (Ney + 1);
                localMap[elemID][5] = localMap[elemID][4] + 1;
                localMap[elemID][7] = localMap[elemID][4] + (Nex + 1);
                localMap[elemID][6] = localMap[elemID][7] + 1;
            }
        }
    }
    // build local dof map:
    unsigned int** localDofMap;
    localDofMap = new unsigned int *[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localDofMap[eid] = new unsigned int[MAX_BLOCKS_PER_DIM * NNODE_PER_ELEM * NDOF_PER_NODE];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                localDofMap[eid][(nid * NDOF_PER_NODE) + did] = (localMap[eid][nid] * NDOF_PER_NODE) + did;
            }
        }
    }

    // number of own nodes
    unsigned int nnode;
    if (rank == 0) {
        nnode = numLocalNodes;
    } else {
        nnode = numLocalNodes - (Nex + 1) * (Ney + 1);
    }

    // gather number of own nodes across ranks
    unsigned int* nnodeCount = new unsigned int[size];
    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);
    /* if (rank == 0){
        for (unsigned int i = 0; i < size; i++){
            printf("[rank %d] nnodeCount[%d]= %d\n", rank, i, nnodeCount[i]);
        }
    } */

    // offset of nnodeCount
    unsigned int* nnodeOffset = new unsigned int[size];
    nnodeOffset[0]            = 0;
    for (unsigned int i = 1; i < size; i++) {
        nnodeOffset[i] = nnodeOffset[i - 1] + nnodeCount[i - 1];
    }
    // total number of nodes for all ranks
    unsigned long int nnode_total, ndofs_total;
    nnode_total = nnodeOffset[size - 1] + nnodeCount[size - 1];
    ndofs_total = nnode_total * NDOF_PER_NODE;
    if (rank == 0)
        printf("Total dofs = %d\n", ndofs_total);

    // build global nodal map:
    unsigned long gNodeId;
    unsigned long int** globalMap;
    globalMap = new unsigned long int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalMap[eid] = new unsigned long int[MAX_BLOCKS_PER_DIM * NNODE_PER_ELEM];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            if (rank == 0) {
                globalMap[eid][nid] = localMap[eid][nid];
            } else {
                globalMap[eid][nid] = localMap[eid][nid] + nnodeOffset[rank] -
                                      ((Nex + 1) * (Ney + 1));
            }
        }
    }
    /* for (unsigned int eid = 0; eid < nelem; eid ++){
        printf("rank %d, elem %d, globalMap= %d, %d, %d, %d, %d, %d, %d, %d\n", rank, eid, globalMap[eid][0],globalMap[eid][1],
        globalMap[eid][2],globalMap[eid][3],globalMap[eid][4],globalMap[eid][5],globalMap[eid][6],globalMap[eid][7]);
    } */

    // map from elemental dof to global dof
    unsigned long int** globalDofMap;
    globalDofMap = new unsigned long int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        globalDofMap[eid] =
          new unsigned long int[MAX_BLOCKS_PER_DIM * NNODE_PER_ELEM * NDOF_PER_NODE];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                globalDofMap[eid][(nid * NDOF_PER_NODE) + did] =
                  (globalMap[eid][nid] * NDOF_PER_NODE) + did;
            }
        }
    }

    // build local2Global node map:
    unsigned long* local2GlobalMap = new unsigned long[numLocalNodes];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            gNodeId = globalMap[eid][nid];
            local2GlobalMap[localMap[eid][nid]] = gNodeId;
        }
    }
    /* for (unsigned int nid = 0; nid < numLocalNodes; nid++){
        printf("rank %d, local2Global[%d]= %d\n",rank, nid, local2GlobalMap[nid]);
    } */

    // build local2Global dof map:
    unsigned long* local2GlobalDofMap = new unsigned long[numLocalNodes * NDOF_PER_NODE];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                local2GlobalDofMap[localDofMap[eid][(nid * NDOF_PER_NODE) + did]] =
                  globalDofMap[eid][(nid * NDOF_PER_NODE) + did];
            }
        }
    }

    // start and end (inclusive) global nodes owned by my rank
    unsigned long start_global_node, end_global_node;
    start_global_node = nnodeOffset[rank];
    end_global_node   = start_global_node + (nnode - 1);

    // start and end (inclusive) global dofs owned by my rank
    unsigned long start_global_dof, end_global_dof;
    start_global_dof = start_global_node * NDOF_PER_NODE;
    end_global_dof   = (end_global_node * NDOF_PER_NODE) + (NDOF_PER_NODE - 1);

    // number of dofs per element
    unsigned int* ndofs_per_element = new unsigned int[nelem];
    for (unsigned eid = 0; eid < nelem; eid++) {
        ndofs_per_element[eid] = NNODE_PER_ELEM * NDOF_PER_NODE;
    }

    // elemental boundary dofs and prescribed value
    unsigned int* * bound_dofs = new unsigned int* [nelem];
    double* * bound_values = new double* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        bound_dofs[eid] = new unsigned int[NNODE_PER_ELEM * NDOF_PER_NODE];
        bound_values[eid] = new double [NNODE_PER_ELEM * NDOF_PER_NODE];
    }

    // construct elemental constrained DoFs and prescribed values
    unsigned int lNodeId;
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {

            lNodeId = localMap[eid][nid];
            // get nodal coordinates
            x = localNodes[lNodeId].get_x();
            y = localNodes[lNodeId].get_y();
            z = localNodes[lNodeId].get_z();

            // fix rigid body motions:
            if ((fabs(x) < zero_number) && (fabs(y) < zero_number) && (fabs(z) < zero_number)){
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 2] = 1; // constrained dof
                bound_dofs[eid][nid * NDOF_PER_NODE] = 1;
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 1] = 1;
                bound_values[eid][(nid * NDOF_PER_NODE) + 2] = 0.0; // prescribed value
                bound_values[eid][nid * NDOF_PER_NODE] = 0.0;
                bound_values[eid][(nid * NDOF_PER_NODE) + 1] = 0.0;
            } else {
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                    bound_dofs[eid][(nid * NDOF_PER_NODE) + did] = 0; // free dof
                    bound_values[eid][(nid * NDOF_PER_NODE) + did] = -1000000;
                }
            }
            if ((fabs(x - Lx) < zero_number) && (fabs(y - Ly) < zero_number) && (fabs(z) < zero_number)){
                // node at (Lx,Ly,0) --> fix in x and z
                bound_dofs[eid][nid * NDOF_PER_NODE] = 1;    //x
                bound_dofs[eid][nid * NDOF_PER_NODE + 2] = 1;
                bound_values[eid][nid * NDOF_PER_NODE] = 0.0;
                bound_values[eid][nid * NDOF_PER_NODE + 2] = 0.0;
            }
            if ((fabs(x - Lx) < zero_number) && (fabs(y) < zero_number) && (fabs(z) < zero_number)){
                // node at (Lx,0,0) --> fix in z
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 2] = 1;
                bound_values[eid][(nid * NDOF_PER_NODE) + 2] = 0.0;
            }
            /* if ((fabs(x) < zero_number) && (fabs(y - Ly) < zero_number) && (fabs(z) < zero_number)){
                // node at (0, Ly, 0) --> fix in z
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 2] = 1;
                bound_values[eid][(nid * NDOF_PER_NODE) + 2] = 0.0;
            } */
        }
    }

    // create lists of constrained dofs
    std::vector< par::ConstraintRecord<double, unsigned long int> > list_of_constraints;
    par::ConstraintRecord<double, unsigned long int> cdof;
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                if (bound_dofs[eid][(nid * NDOF_PER_NODE) + did] == 1) {
                    // save the global id of constrained dof
                    cdof.set_dofId( globalDofMap[eid][(nid * NDOF_PER_NODE) + did] );
                    cdof.set_preVal( bound_values[eid][(nid * NDOF_PER_NODE) + did] );
                    list_of_constraints.push_back(cdof);
                }
            }
        }
    }

    // sort to prepare for deleting repeated constrained dofs in the list
    std::sort(list_of_constraints.begin(), list_of_constraints.end());
    list_of_constraints.erase(std::unique(list_of_constraints.begin(),list_of_constraints.end()),list_of_constraints.end());

    // transform vector data to pointer (to be conformed with the aMat interface)
    unsigned long int* constrainedDofs_ptr;
    double* prescribedValues_ptr;
    constrainedDofs_ptr = new unsigned long int [list_of_constraints.size()];
    prescribedValues_ptr = new double [list_of_constraints.size()];
    for (unsigned int i = 0; i < list_of_constraints.size(); i++) {
        constrainedDofs_ptr[i] = list_of_constraints[i].get_dofId();
        prescribedValues_ptr[i] = list_of_constraints[i].get_preVal();
    }

    // ad-hoc: elemental traction vector
    Matrix<double, Eigen::Dynamic, 1> * elem_trac;
    elem_trac = new Matrix<double, Eigen::Dynamic, 1> [nelem];

    double area, x0, y0, x2, y2, x4, y4, x6, y6;
    for (unsigned int eid = 0; eid < nelem; eid++){
        bool traction_top = false;
        bool traction_bot = false;
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            gNodeId = globalMap[eid][nid];
            z = (double) (gNodeId / ((Nex + 1) * (Ney + 1))) * hz;
            if (fabs(z - Lz) < zero_number){
                // element eid has one face is the top surface with applied traction
                traction_top = true;
            } else if (fabs(z) < zero_number){
                // element eid has one face is the bot surface with applied traction
                traction_bot = true;
            }
        }
        
        if (traction_top || traction_bot) {
            elem_trac[eid].resize(NNODE_PER_ELEM * NDOF_PER_NODE, 1);
            for (unsigned int did = 0; did < NNODE_PER_ELEM * NDOF_PER_NODE; did++){
                elem_trac[eid][did] = 0.0;
            }
        }
        if (traction_top){
            // ad-hoc setting force vector due to uniform traction applied on face 4-5-6-7 of the element
            // compute area of the top face
            gNodeId = globalMap[eid][4];
            x4 = (double)(gNodeId % (Nex + 1)) * hx;
            y4 = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            gNodeId = globalMap[eid][6];
            x6 = (double)(gNodeId % (Nex + 1)) * hx;
            y6 = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            area = ((x6 - x4)*(y6 - y4));
            assert(area > zero_number);
            
            // put into load vector
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                    if (((nid == 4) || (nid == 5) || (nid == 6) || (nid == 7)) && (did == (NDOF_PER_NODE - 1))){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = area/4;
                    }
                }
            }
        }
        
        if (traction_bot){
            // ad-hoc setting force vector due to uniform traction applied on face 0-1-2-3 of the element
            // compute area of the bottom face
            gNodeId = globalMap[eid][0];
            x0 = (double)(gNodeId % (Nex + 1)) * hx;
            y0 = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            gNodeId = globalMap[eid][2];
            x2 = (double)(gNodeId % (Nex + 1)) * hx;
            y2 = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            area = (x2 - x0)*(y2 - y0);
            assert(area > zero_number);

            // put into load vector
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                    if (((nid == 0) || (nid == 1) || (nid == 2) || (nid == 3)) && (did == (NDOF_PER_NODE - 1))){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = -area/4;
                    }
                }
            }
        }
    }

    total_time.start();

    // declare Maps object  =================================
    par::Maps<double, unsigned long, unsigned int> meshMaps(comm);
    meshMaps.set_map(nelem, localDofMap, ndofs_per_element, numLocalDofs, local2GlobalDofMap, start_global_dof, 
                    end_global_dof, ndofs_total);
    if (matType == 1) {
        meshMaps.buildScatterMap();
    }
    meshMaps.set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, list_of_constraints.size());

    /// declare aMat object =================================
    typedef par::aMat<par::aMatBased<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> aMatBased; // aMat type taking aMatBased as derived class
    typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> aMatFree; // aMat type taking aMatBased as derived class

    aMatBased* stMatBased; // pointer of aMat taking aMatBased as derived
    aMatFree* stMatFree; // pointer of aMat taking aMatFree as derived

    if (matType == 0){
        // assign stMatBased to the derived class aMatBased
        stMatBased =
          new par::aMatBased<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
    } else {
        // assign stMatFree to the derived class aMatFree
        stMatFree =
          new par::aMatFree<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
    }

    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact, error;
    par::create_vec(meshMaps, rhs);
    par::create_vec(meshMaps, out);
    par::create_vec(meshMaps, sol_exact);
    par::create_vec(meshMaps, error);

    // index of non-zero blocks
    unsigned int** non_zero_block_i;
    unsigned int** non_zero_block_j;
    non_zero_block_i = new unsigned int* [nelem];
    non_zero_block_j = new unsigned int* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        non_zero_block_i[eid] = new unsigned int [MAX_BLOCKS_PER_ELEMENT];
        non_zero_block_j[eid] = new unsigned int [MAX_BLOCKS_PER_ELEMENT];
    }

    // compute element stiffness matrix and assemble global stiffness matrix and load vector
    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            lNodeId = localMap[eid][nid];
            // get node coordinates
            x = localNodes[lNodeId].get_x();
            y = localNodes[lNodeId].get_y();
            z = localNodes[lNodeId].get_z();

            xe[nid * 3] = x;
            xe[(nid * 3) + 1] = y;
            xe[(nid * 3) + 2] = z;
        }

        /* if (eid == 21){
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
                printf("node %d, {x,y,z}= {%f,%f,%f}\n",nid,xe[nid*3],xe[nid*3 + 1],xe[nid*3+2]);
            }
        } */

        // compute element stiffness matrix
        if (useEigen) {
            ke_hex8_iso(*kee[0], xe, E, nu, intData.Pts_n_Wts, NGT);
        } else {
            printf("Error: not yet implement element stiffness matrix which is not Eigen matrix format\n");
        }

        // suppose element 21 (for case 4 x 4 x 4, sequential) is cracked
        if (eid == 21){
            // suppose crack go through center, block_01 = block_10 = 0, block_11 = block_00
            kee[1] = new EigenMat;
            fee[1] = new EigenVec;
            ke_hex8_iso(*kee[1], xe, E, nu, intData.Pts_n_Wts, NGT);
        }
        /* if (eid == 21){
            for (unsigned int i = 1; i < MAX_BLOCKS_PER_ELEMENT; i++){
                kee[i] = new EigenMat;
                fee[i] = new EigenVec;
            }
            // block 01
            ke_hex8_iso_crack_01(*kee[1], xe, E, nu, intData.Pts_n_Wts, NGT);
            // block 10
            //printf("ke has %d rows and %d columns\n",(*kee[1]).rows(),(*kee[1]).cols());
            (*kee[2]).resize(24,24);
            for (unsigned int r = 0; r < (*kee[1]).rows(); r++){
                for (unsigned int c = 0; c < (*kee[1]).cols(); c++){
                    (*kee[2])(c,r) = (*kee[1])(r,c);
                    printf("[%d,%d]= %f, ",r,c,(*kee[1])(r,c));
                }
                printf("\n");
            }
            // block 11
            (*kee[3]).resize(24,24);
            for (unsigned int r = 0; r < (*kee[0]).rows(); r++){
                for (unsigned int c = 0; c < (*kee[0]).cols(); c++){
                    (*kee[3])(r,c) = (*kee[0])(r,c);
                }
            }
        } */

        // assemble element stiffness matrix to global K
        non_zero_block_i[eid][0] = 0; // i-index of non-zero block 0
        non_zero_block_j[eid][0] = 0; // j-index of non-zero block 0
        if (eid == 21){
            non_zero_block_i[eid][1] = 1; // i-index of non-zero block 1
            non_zero_block_j[eid][0] = 1; // j-index of non-zero block 1
        }
        if (matType == 0){
            petsc_time.start();
            if (eid == 21){
                stMatBased->set_element_matrix(eid,non_zero_block_i[eid],non_zero_block_j[eid],(const EigenMat**)kee, 2u);    
            } else {
                stMatBased->set_element_matrix(eid,non_zero_block_i[eid],non_zero_block_j[eid],(const EigenMat**)kee, 1u);
            }
            petsc_time.stop();
        } else {
            aMat_time.start();
            if (eid == 21){
                stMatFree->set_element_matrix(eid,non_zero_block_i[eid],non_zero_block_j[eid],(const EigenMat**)kee, 2u);
            } else {
                stMatFree->set_element_matrix(eid,non_zero_block_i[eid],non_zero_block_j[eid],(const EigenMat**)kee, 1u);
            }
            aMat_time.stop();
        }

        // assemble element load vector to global F
        if (elem_trac[eid].size() != 0){
            if (matType == 0)
                petsc_time.start();
            else
                aMat_time.start();
            par::set_element_vec(meshMaps, rhs, eid, elem_trac[eid], 0u, ADD_VALUES);
            //stMat->petsc_set_element_vec(rhs, eid, elem_trac[eid], 0, ADD_VALUES);
            if (matType == 0)
                petsc_time.stop();
            else
                aMat_time.stop();
        }
        
    }
    delete [] xe;

    // Pestc begins and completes assembling the global stiffness matrix
    /* if (matType == 0){
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
        //stMat.dump_mat("matrix.dat");
    } */
    if (matType == 0) {
        petsc_time.start();
        stMatBased->finalize();
        petsc_time.stop();
    } else {
        aMat_time.start();
        stMatFree->finalize(); // compute trace of matrix when using penalty method
        aMat_time.stop();
    }
    // These are needed because we used ADD_VALUES for rhs when assembling
    // now we are going to use INSERT_VALUE for Fc in apply_bc_rhs
    if (matType != 0)
        aMat_time.start();
    else
        petsc_time.start();
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);
    // apply bc for rhs: this must be done before applying bc for the matrix
    // because we use the original matrix to compute KfcUc in matrix-based method
    //stMat->apply_bc(rhs);
    if (matType == 0){
        stMatBased->apply_bc(rhs);
    } else {
        stMatFree->apply_bc(rhs);
    }
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);
    if (matType != 0)
        aMat_time.stop();
    else
        petsc_time.stop();

    //char fname[256];
    //sprintf(fname,"rhsVec_beforeBC_%d.dat",size);
    //stMat->dump_vec(rhs,fname);

    // apply dirichlet BCs to the matrix
    if (matType == 0){
        petsc_time.start();
        stMatBased->finalize();
        petsc_time.stop();
        // sprintf(fname,"matrix_%d.dat",size);
        // stMat->dump_mat(fname);
    }
    /* if (matType == 0){
        //stMat.apply_bc_mat();
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
        //sprintf(fname,"matrix_ex1a_%d.dat",size);    
        //stMat.dump_mat(fname);
    } */

    //sprintf(fname,"rhsVec_%d.dat",size);
    //stMat.dump_vec(rhs,fname);

    // solve
    //stMat->petsc_solve((const Vec) rhs, out);
    if (matType == 0){
        petsc_time.start();
        par::solve(*stMatBased, (const Vec)rhs, out);
        petsc_time.stop();
    } else {
        aMat_time.start();
        par::solve(*stMatFree, (const Vec)rhs, out);
        aMat_time.stop();
    }

    total_time.stop();

    // display timing
    if (matType != 0) {
        if (size > 1) {
            long double aMat_maxTime;
            MPI_Reduce(&aMat_time.seconds, &aMat_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                std::cout << "aMat time = " << aMat_maxTime << "\n";
            }
        } else {
            if (rank == 0) {
                std::cout << "aMat time = " << aMat_time.seconds << "\n";
            }
        }
    } else {
        if (size > 1) {
            long double petsc_maxTime;
            MPI_Reduce(&petsc_time.seconds, &petsc_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0)
            {
                std::cout << "PETSC time = " << petsc_maxTime << "\n";
            }
        } else {
            if (rank == 0) {
                std::cout << "PETSC time = " << petsc_time.seconds << "\n";
            }
        }
    }
    if (size > 1) {
        long double total_time_max;
        MPI_Reduce(&total_time.seconds, &total_time_max, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            std::cout << "total time = " << total_time_max << "\n";
        }
    } else {
        if (rank == 0) {
            std::cout << "total time = " << total_time.seconds << "\n";
        }
    }

    // these could be not nessary:
    //VecAssemblyBegin(out);
    //VecAssemblyEnd(out);
    
    //stMat->dump_vec(out);

    PetscScalar norm, alpha = -1.0;
    VecNorm(out, NORM_2, &norm);
    if (rank == 0){
        printf("L2 norm of computed solution = %20.10f\n",norm);
    }

    // exact solution
    Matrix< double, NDOF_PER_NODE * NNODE_PER_ELEM, 1 > e_exact;
    double disp [3];
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            gNodeId = globalMap[eid][nid];
            x = (double)(gNodeId % (Nex + 1)) * hx;
            y = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(gNodeId / ((Nex + 1)*(Ney + 1))) * hz;

            disp[0] = -nu*x/E + nu/E*(Lx/Ly)*y;
            disp[1] = -nu*y/E - nu/E*(Lx/Ly)*x;
            disp[2] = z/E;
            
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                e_exact[(nid * NDOF_PER_NODE) + did] = disp[did];
            }
        }
        //stMat->petsc_set_element_vec(sol_exact, eid, e_exact, 0, INSERT_VALUES);
        par::set_element_vec(meshMaps, sol_exact, eid, e_exact, 0u, INSERT_VALUES);
    }

    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

    VecNorm(sol_exact, NORM_2, &norm);
    if (rank == 0){
        printf("L2 norm of exact solution = %20.10f\n",norm);
    }

    // compute the error vector
    VecCopy(sol_exact, error);
    // subtract error = error(=sol_exact) - out
    VecAXPY(error, alpha, out);
    VecNorm(error, NORM_INFINITY, &norm);
    if (rank == 0){
        printf("Inf norm of error = %20.10f\n", norm);
    }

    // export ParaView
    /* std::ofstream myfile;
    if (!rank){
        myfile.open("ex6a.vtk");
        myfile << "# vtk DataFile Version 2.0 " << std::endl;
        myfile << "Stress field" << std::endl;
        myfile << "ASCII" << std::endl;
        myfile << "DATASET UNSTRUCTURED_GRID" << std::endl;
        myfile << "POINTS " << numLocalNodes << " float" << std::endl;
        for (unsigned int nid = 0; nid < numLocalNodes; nid++){
            x = localNodes[nid].get_x();
            y = localNodes[nid].get_y();
            z = localNodes[nid].get_z();
            // transformed coordinates
            x = x - Lx/2;
            y = y - Ly/2;
            myfile << x << "  " << y << "  " << z << std::endl;
        }
        unsigned int size_cell_list = nelem * 9;
        myfile << "CELLS " << nelem << " " << size_cell_list << std::endl;
        for (unsigned int eid = 0; eid < nelem; eid++){
            myfile << "8 " << localMap[eid][0] << " " << localMap[eid][1] << " "
            << localMap[eid][2] << " " << localMap[eid][3] << " "
            << localMap[eid][4] << " " << localMap[eid][5] << " "
            << localMap[eid][6] << " " << localMap[eid][7] << " " << std::endl;
        }
        myfile << "CELL_TYPES " << nelem << std::endl;
        for (unsigned int eid = 0; eid < nelem; eid++){
            myfile << "12" << std::endl;
        }
        // myfile << "POINT_DATA " << numLocalNodes << std::endl;
        // myfile << "VECTORS " << "displacement " << "float " << std::endl;
        // std::vector<PetscInt> indices (NDOF_PER_NODE);
        // std::vector<PetscScalar> values (NDOF_PER_NODE);
        // for (unsigned int nid = 0; nid < numLocalNodes; nid++){
        //     gNodeId = local2GlobalMap[nid];
        //     for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
        //         indices[did] = gNodeId * NDOF_PER_NODE + did;
        //     }
        //     VecGetValues(out, NDOF_PER_NODE, indices.data(), values.data());
        //     myfile << values[0] << " " << values[1] << " " << values[2] << std::endl;
        // }
        myfile.close();
    } */

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete[] localMap[eid];
    }
    delete[] localMap;

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete[] localDofMap[eid];
    }
    delete[] localDofMap;
    
    for (unsigned int eid = 0; eid < nelem; eid++){
        delete[] globalMap[eid];
    }
    delete[] globalMap;
    
    for (unsigned int eid = 0; eid < nelem; eid++){
        delete[] globalDofMap[eid];
    }
    delete[] globalDofMap;
    
    delete[] local2GlobalMap;

    PetscFinalize();
    return 0;
}