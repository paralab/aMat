/**
 * @file ex4.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Example: Stretching of a prismatic bar by its own weight (Timoshenko page 246)
 * @brief          using 20-node quadratic element
 * @brief Exact solution (origin at centroid of bottom face)
 * @brief    uniform stress s_zz = rho * g * z
 * @brief    displacement u = -(nu * rho * g/E) * x * z
 * @brief    displacement v = -(nu * rho * g/E) * y * z
 * @brief    displacement w = (rho * g/2/E)(z^2 - Lz^2) + (nu * rho * g)/2/E(x^2 + y^2)
 * @brief Boundary condition: traction tz = rho * g * Lz applied on top surface + blocking rigid motions
 * @brief Partition of elements in z direction: owned elements in z direction ~ Nez/(number of ranks)
 * @brief Size of the domain: Lx = Ly = 1; Lz = 1.0
 * 
 * @version 0.1
 * @date 2020-02-26
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#include <iostream>
#include <fstream>

#include <mpi.h>

#ifdef BUILD_WITH_PETSC
#    include <petsc.h>
#endif

#include <Eigen/Dense>
#include "ke_matrix.hpp"
#include "fe_vector.hpp"
#include "aMat.hpp"
#include "integration.hpp"

#include "profiler.hpp"

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;

// number of cracks allowed in 1 element
#define AMAT_MAX_CRACK_LEVEL 0

// max number of block dimensions in one cracked element
#define AMAT_MAX_BLOCKSDIM_PER_ELEMENT (1u<<AMAT_MAX_CRACK_LEVEL)

//////////////////////////////////////////////////////////////////////////////////////////////////////

void usage()
{
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  ex4 <Nex> <Ney> <Nez> <matrix based/free> <bc-method>\n";
    std::cout << "\n";
    std::cout << "     Nex: Number of elements in X\n";
    std::cout << "     Ney: Number of elements in y\n";
    std::cout << "     Nez: Number of elements in z\n";
    std::cout << "     use matrix-free: 1 => yes.  0 => matrix-based method. \n";
    std::cout << "     use identity-matrix: 0    use penalty method: 1 \n";
    exit(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DT, typename LI>
class nodeData {
    private:
    LI nodeId;
    DT x;
    DT y;
    DT z;

    public:
    nodeData() {
        nodeId = 0;
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    inline LI get_nodeId() const { return nodeId; }
    inline DT get_x() const { return x; }
    inline DT get_y() const { return y; }
    inline DT get_z() const { return z; }

    inline void set_nodeId(LI id) { nodeId = id; }
    inline void set_x(DT value) { x = value; }
    inline void set_y(DT value) { y = value; }
    inline void set_z(DT value) { z = value; }

    bool operator == (nodeData const &other) const {
        return (nodeId == other.get_nodeId());
    }
    bool operator < (nodeData const &other) const {
        if (nodeId < other.get_nodeId()) return true;
        else return false;
    }
    bool operator <= (nodeData const &other) const {
        return (((*this) < other) || ((*this) == other));
    }

    ~nodeData() {}
};

int main( int argc, char *argv[] ) {
    // User provides: Nex - number of elements (global) in x direction
    //                Ney - number of elements (global) in y direction
    //                Nez - number of elements (global) in z direction
    //                flag1 - 1 matrix-free method, 0 matrix-based method
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
    const unsigned int NNODE_PER_ELEM = 20;     // number of nodes per element

    // material properties of alumina
    //const double E = 300.0; // GPa
    const double E = 1.0E6;
    //const double nu = 0.2;
    const double nu = 0.3;
    //const double rho = 3950;// kg.m^-3
    const double rho = 1.0;
    //const double g = 9.8;   // m.s^-2
    const double g = 1.0;

    // number of (global) elements in x, y and z directions
    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]);
    const unsigned int Nez = atoi(argv[3]);

    //05.19.20 only use Eigen matrix
    const bool useEigen = true;

    const unsigned int matType = atoi(argv[4]);  // approach (matrix based/free)
    const unsigned int bcMethod = atoi(argv[5]); // method of applying BC

    // domain sizes: Lx, Ly, Lz - length of the (global) domain in x/y/z direction
    const double Lx = 1.0, Ly = 1.0, Lz = 100.0;

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

    // element matrix (contains multiple matrix blocks)
    std::vector< Matrix< double, NDOF_PER_NODE * NNODE_PER_ELEM, NDOF_PER_NODE * NNODE_PER_ELEM > > kee;
    kee.resize(AMAT_MAX_BLOCKSDIM_PER_ELEMENT * AMAT_MAX_BLOCKSDIM_PER_ELEMENT);

    // element force vector
    std::vector<Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1>> fee;
    fee.resize(AMAT_MAX_BLOCKSDIM_PER_ELEMENT);

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

    if(!rank) {
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
    #elif VECTORIZED_OPENMP_PADDING
    if (!rank) {std::cout << "\t\tVectorization using OpenMP with paddings\n";}
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

    if( rank == 0 ) {
        if (size > Nez) {
            printf("The number of processes must be less than or equal Nez, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // for fixing rigid motions at centroid of the top/bottom face
    // number of elements in x and y directions must be even numbers
    if ((Nex % 2 != 0) || (Ney % 2 != 0)){
        if (!rank){
            printf("Number of elements in x and y must be even numbers, program stops.\n");
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
    
    double origin [3] = {0.0};
    origin[2] = emin * hz;
    //printf("[%d] emin= %d, emax= %d\n", rank,emin,emax);
    //printf("[%d] origin = {%f,%f,%f}\n", rank, origin[0], origin[1], origin[2]);

    // generate nodes...
    std::vector<nodeData<double, unsigned int>> localNodes;
    nodeData<double, unsigned int> node;
    unsigned int nid = 0;
    for (unsigned int k = 0; k < (2*nelem_z + 1); k++){
        z = k*(hz/2) + origin[2];
        for (unsigned int j = 0; j < (2*nelem_y + 1); j++){
            y = j*(hy/2) + origin[1];
            for (unsigned int i = 0; i < (2*nelem_x + 1); i++){
                x = i*(hx/2) + origin[0];
                if ( !(((i%2 == 1) && (j%2 == 1)) || ((i%2 == 1) && (k%2 ==1)) || ((j%2 ==1) && (k%2 ==1))) ){
                    node.set_nodeId(nid);
                    node.set_x(x);
                    node.set_y(y);
                    node.set_z(z);
                    localNodes.push_back(node);
                    nid++;
                }
            }
        }
    }
    // total number of local nodes
    unsigned int numLocalNodes = localNodes.size();

    // number of local dofs
    unsigned int numLocalDofs = numLocalNodes * NDOF_PER_NODE;

    /* for (unsigned int i = 0; i < localNodes.size(); i++){
        printf("[rank %d] node %d, {x,y,z}= %4.3f,%4.3f,%4.3f\n",rank,localNodes[i].get_nodeId(), localNodes[i].get_x(), localNodes[i].get_y(), localNodes[i].get_z());
    } */

    // map from elemental nodes to local nodes
    unsigned int* * localMap;
    localMap = new unsigned int* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localMap[eid] = new unsigned int[AMAT_MAX_BLOCKSDIM_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned int k = 0; k < nelem_z; k++){
        for (unsigned int j = 0; j < nelem_y; j++){
            for (unsigned int i = 0; i < nelem_x; i++){
                unsigned int elemID = nelem_x * nelem_y * k + nelem_x * j + i;
                localMap[elemID][0] = (2*i) + j*(3*Nex + 2) + k*((2*Nex + 1)*(2*Ney + 1) - (Nex * Ney) + (Nex + 1)*(Ney + 1));
                localMap[elemID][1] = localMap[elemID][0] + 2;
                localMap[elemID][3] = localMap[elemID][0] + (3*Nex + 2);
                localMap[elemID][2] = localMap[elemID][3] + 2;
                localMap[elemID][4] = localMap[elemID][0] + ((2*Nex + 1)*(2*Ney + 1) - (Nex * Ney) + (Nex + 1)*(Ney + 1));
                localMap[elemID][5] = localMap[elemID][4] + 2;
                localMap[elemID][7] = localMap[elemID][4] + (3*Nex + 2);
                localMap[elemID][6] = localMap[elemID][7] + 2;

                localMap[elemID][8] = localMap[elemID][0] + 1;
                localMap[elemID][10] = localMap[elemID][3] + 1;
                localMap[elemID][11] = localMap[elemID][0] + (2*Nex + 1) - i;
                localMap[elemID][9] = localMap[elemID][11] + 1;

                localMap[elemID][12] = localMap[elemID][4] + 1;
                localMap[elemID][14] = localMap[elemID][7] + 1;
                localMap[elemID][15] = localMap[elemID][4] + (2*Nex + 1) - i;
                localMap[elemID][13] = localMap[elemID][15] + 1;
                
                localMap[elemID][16] = localMap[elemID][0] + ((2*Nex + 1)*(2*Ney + 1) - (Nex * Ney) - i) - j*(2*Nex + 1);
                localMap[elemID][17] = localMap[elemID][16] + 1;
                localMap[elemID][19] = localMap[elemID][16] + (Nex + 1);
                localMap[elemID][18] = localMap[elemID][19] + 1;
            }
        }
    }

    // map from local dof to global dof
    unsigned int* * localDofMap;
    localDofMap = new unsigned int* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localDofMap[eid] = new unsigned int [AMAT_MAX_BLOCKSDIM_PER_ELEMENT * NNODE_PER_ELEM * NDOF_PER_NODE];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                localDofMap[eid][(nid * NDOF_PER_NODE) + did] = (localMap[eid][nid] * NDOF_PER_NODE) + did;
            }
        }
    }

    // number of owned nodes
    unsigned int nnode;
    if (rank == 0){
        nnode = numLocalNodes;
    } else {
        nnode = numLocalNodes - ((2*Nex + 1)*(2*Ney + 1) - (Nex * Ney));
    }

    // gather number of own nodes across ranks
    unsigned int* nnodeCount = new unsigned int [size];
    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);
    /* if (rank == 0){
        for (unsigned int i = 0; i < size; i++){
            printf("[rank %d] nnodeCount[%d]= %d\n", rank, i, nnodeCount[i]);
        }
    } */
    // offset of nnodeCount
    unsigned int* nnodeOffset = new unsigned int [size];
    nnodeOffset[0] = 0;
    for (unsigned int i = 1; i < size; i++){
        nnodeOffset[i] = nnodeOffset[i-1] + nnodeCount[i-1];
    }
    // total number of nodes for all ranks
    unsigned long int nnode_total, ndofs_total;
    nnode_total = nnodeOffset[size-1] + nnodeCount[size-1];
    ndofs_total = nnode_total * NDOF_PER_NODE;
    if (rank == 0) printf("Total dofs = %d\n",ndofs_total);
    
    
    // build global map from local map
    unsigned long gNodeId;
    unsigned long int* * globalMap;
    globalMap = new unsigned long int *[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalMap[eid] = new unsigned long int[AMAT_MAX_BLOCKSDIM_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            if (rank == 0){
                globalMap[eid][nid] = localMap[eid][nid];
            } else {
                globalMap[eid][nid] = localMap[eid][nid] + nnodeOffset[rank] - ((2*Nex + 1)*(2*Ney + 1) - (Nex * Ney));
            }
        }
    }
    

    // map from elemental dof to global dof
    unsigned long int* * globalDofMap;
    globalDofMap = new unsigned long int *[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalDofMap[eid] = new unsigned long int[AMAT_MAX_BLOCKSDIM_PER_ELEMENT * NNODE_PER_ELEM * NDOF_PER_NODE];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                globalDofMap[eid][(nid * NDOF_PER_NODE) + did] = (globalMap[eid][nid] * NDOF_PER_NODE) + did;
            }
        }
    }

    // local node to global node map
    unsigned long * local2GlobalMap = new unsigned long[numLocalNodes];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            gNodeId = globalMap[eid][nid];
            local2GlobalMap[localMap[eid][nid]] = gNodeId;
        }
    }
    /* for (unsigned int nid = 0; nid < numLocalNodes; nid++){
        printf("local2Global[%d]= %d\n",nid,local2GlobalMap[nid]);
    } */

    // local dof to global dof map
    unsigned long* local2GlobalDofMap = new unsigned long[numLocalNodes * NDOF_PER_NODE];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                local2GlobalDofMap[localDofMap[eid][(nid * NDOF_PER_NODE) + did]] = globalDofMap[eid][(nid * NDOF_PER_NODE) + did];
            }
        }
    }

    // start and end (inclusive) global nodes owned by my rank
    unsigned long start_global_node, end_global_node;
    start_global_node = nnodeOffset[rank];
    end_global_node = start_global_node + (nnode - 1);

    // start and end (inclusive) global dofs owned by my rank
    unsigned long start_global_dof, end_global_dof;
    start_global_dof = start_global_node * NDOF_PER_NODE;
    end_global_dof = (end_global_node * NDOF_PER_NODE) + (NDOF_PER_NODE - 1);

    // number of dofs per element
    unsigned int* ndofs_per_element = new unsigned int[nelem];
    for (unsigned eid = 0; eid < nelem; eid ++){
        ndofs_per_element[eid] = NNODE_PER_ELEM * NDOF_PER_NODE;
    }

    // elemental boundary dofs and prescribed value
    unsigned int* * bound_dofs = new unsigned int* [nelem];
    double* * bound_values = new double* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        bound_dofs[eid] = new unsigned int[ndofs_per_element[eid]];
        bound_values[eid] = new double [ndofs_per_element[eid]];
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

            // translate origin to center of bottom face
            x = x - Lx/2;
            y = y - Ly/2;

            // node at centroid of top face: fix all directions
            if ((fabs(x) < zero_number) && (fabs(y) < zero_number) && (fabs(z-Lz) < zero_number)) {
                bound_dofs[eid][(nid * NDOF_PER_NODE)] = 1;
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 1] = 1;
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 2] = 1;
                
                bound_values[eid][(nid * NDOF_PER_NODE)] = 0.0;
                bound_values[eid][(nid * NDOF_PER_NODE) + 1] = 0.0;
                bound_values[eid][(nid * NDOF_PER_NODE) + 2] = 0.0;
            } else {
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                    bound_dofs[eid][(nid * NDOF_PER_NODE) + did] = 0; // free dof
                    bound_values[eid][(nid * NDOF_PER_NODE) + did] = -1000000;
                }
            }

            // node at centroid of bottom surface: fix in x and y
            if ((fabs(x) < zero_number) && (fabs(y) < zero_number) && (fabs(z) < zero_number)){
                bound_dofs[eid][nid * NDOF_PER_NODE] = 1;
                bound_dofs[eid][nid * NDOF_PER_NODE + 1] = 1;

                bound_values[eid][nid * NDOF_PER_NODE] = 0.0;
                bound_values[eid][nid * NDOF_PER_NODE + 1] = 0.0;
            }

            // node at center of right edge of bottom surface: fix in y
            if ((fabs(x - Lx/2) < zero_number) && (fabs(y) < zero_number) && (fabs(z) < zero_number)){
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 1] = 1;
                bound_values[eid][(nid * NDOF_PER_NODE) + 1] = 0.0;
            }
        }
    }
    
    // create lists of constrained dofs
    std::vector< par::ConstrainedRecord<double, unsigned long int> > list_of_constraints;
    par::ConstrainedRecord<double, unsigned long int> cdof;
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
    /* for (unsigned int i = 0; i < list_of_constraints.size(); i++){
        printf("[rank %d], constraint[%d] = %d, value = %f\n",rank,i, constrainedDofs_ptr[i],prescribedValues_ptr[i]);
    } */


    // elemental traction vector
    Matrix<double, Eigen::Dynamic, 1> * elem_trac;
    elem_trac = new Matrix<double, Eigen::Dynamic, 1> [nelem];

    // nodal traction of tractioned face
    double nodalTraction [24] = {0.0};

    // nodal coordinates of tractioned face
    double xeSt [24];
    
    // force vector due to traction
    Matrix<double, 24, 1> feT;

    // Gauss points and weights
    const unsigned int NGT = 4;
    integration<double> intData(NGT);

    for (unsigned int eid = 0; eid < nelem; eid++){
        bool traction = false;
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            lNodeId = localMap[eid][nid];
            z = localNodes[lNodeId].get_z();
            // top face is subjected to traction t3 = sigma_33 = -rho * g
            if (fabs(z - Lz) < zero_number){
                // element eid has one face is the top surface with applied traction
                traction = true;
                break;
            }
        }
        if (traction) {
            // get coordinates of all nodes
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
                lNodeId = localMap[eid][nid];
                // get node coordinates
                x = localNodes[lNodeId].get_x();
                y = localNodes[lNodeId].get_y();
                z = localNodes[lNodeId].get_z();

                // tranlation origin
                x = x - Lx/2;
                y = y - Ly/2;

                xe[nid * NDOF_PER_NODE] = x;
                xe[(nid * NDOF_PER_NODE) + 1] = y;
                xe[(nid * NDOF_PER_NODE) + 2] = z;
            }

            // get coordinates of nodes belonging to the face where traction is applied
            // traction applied on face 4-5-6-7-12-13-14-15 ==> nodes [4,5,6,7] corresponds to nodes [0,1,2,3,4,5,6,7] of 2D element
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                // node 0 of 2D element <-- node 4 of 3D element
                xeSt[did] = xe[4*NDOF_PER_NODE + did];

                // node 1 of 2D element <-- node 5 of 3D element
                xeSt[NDOF_PER_NODE + did] = xe[5*NDOF_PER_NODE + did];

                // node 2 of 2D element <-- node 6 of 3D element
                xeSt[2*NDOF_PER_NODE + did] = xe[6*NDOF_PER_NODE + did];

                // node 3 of 2D element <-- node 7 of 3D element
                xeSt[3*NDOF_PER_NODE + did] = xe[7*NDOF_PER_NODE + did];

                // node 4 of 2D element <-- node 12 of 3D element
                xeSt[4*NDOF_PER_NODE + did] = xe[12*NDOF_PER_NODE + did];

                // node 5 of 2D element <-- node 13 of 3D element
                xeSt[5*NDOF_PER_NODE + did] = xe[13*NDOF_PER_NODE + did];

                // node 6 of 2D element <-- node 14 of 3D element
                xeSt[6*NDOF_PER_NODE + did] = xe[14*NDOF_PER_NODE + did];

                // node 7 of 2D element <-- node 15 of 3D element
                xeSt[7*NDOF_PER_NODE + did] = xe[15*NDOF_PER_NODE + did];
            }

            // get nodal traction of face where traction is applied (uniform traction t3 = rho*g*Lz applied on top surface)
            for (unsigned int nid = 0; nid < 8; nid++){
                nodalTraction[nid*NDOF_PER_NODE + 2] = rho * g * Lz;
            }
            /* for (unsigned int nid = 0; nid < 8; nid++){
                printf("[e %d][n %d] nodalTraction = %f, %f, %f\n", eid, nid, nodalTraction[nid*NDOF_PER_NODE],
                nodalTraction[nid*NDOF_PER_NODE+1],nodalTraction[nid*NDOF_PER_NODE+2]);
            } */

            // compute force vector due traction applied on one face of element
            feT_hex20_iso(feT, xeSt, nodalTraction, intData.Pts_n_Wts, NGT);
            /* for (unsigned int nid = 0; nid < 8; nid++){
                printf("[e %d][n %d] feT = %f, %f, %f\n", eid, nid, feT(nid*NDOF_PER_NODE),
                feT(nid*NDOF_PER_NODE + 1),feT(nid*NDOF_PER_NODE + 2));
            } */

            // put traction force vector into element force vector
            elem_trac[eid].resize(NNODE_PER_ELEM * NDOF_PER_NODE, 1);
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                    // nodes [4,5,6,7] of 3D element are nodes [0,1,2,3] of 2D element where traction applied
                    if (nid == 4){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[did];
                    } else if (nid == 5){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[NDOF_PER_NODE + did];
                    } else if (nid == 6){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[2*NDOF_PER_NODE + did];
                    } else if (nid == 7){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[3*NDOF_PER_NODE + did];
                    // nodes [12,13,14,15] of 3D element are nodes [4,5,6,7] of 2D element where traction applied
                    } else if (nid == 12){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[4*NDOF_PER_NODE + did];
                    } else if (nid == 13){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[5*NDOF_PER_NODE + did];
                    } else if (nid == 14){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[6*NDOF_PER_NODE + did];
                    } else if (nid == 15){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[7*NDOF_PER_NODE + did];
                    } else {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = 0.0;
                    }
                }
            }
        }
    }
    /* for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            if (elem_trac[eid].size() > 0){
                printf("[e %d][n %d] elem_trac = %f, %f, %f\n", eid, nid, elem_trac[eid](nid*NDOF_PER_NODE),
                elem_trac[eid](nid*NDOF_PER_NODE + 1),elem_trac[eid](nid*NDOF_PER_NODE + 2));
            }
        }
    } */

    total_time.start();

    /// declare aMat object =================================
    par::aMat<double, unsigned long, unsigned int> * stMat;
    if (matType == 0){
        stMat = new par::aMatBased<double, unsigned long, unsigned int>((par::BC_METH)bcMethod);
    } else {
        stMat = new par::aMatFree<double, unsigned long, unsigned int>((par::BC_METH)bcMethod);
    }

    // set communicator
    stMat->set_comm(comm);

    // set global dof map
    stMat->set_map(nelem, localDofMap, ndofs_per_element, numLocalDofs, local2GlobalDofMap, start_global_dof,
                  end_global_dof, ndofs_total);

    // set boundary map
    stMat->set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, list_of_constraints.size());

    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact, error;
    stMat->petsc_create_vec(rhs);
    stMat->petsc_create_vec(out);
    stMat->petsc_create_vec(sol_exact);
    stMat->petsc_create_vec(error);

    // compute element stiffness matrix and force vector, then assemble
    // nodal value of body force
    double beN [60] = {0.0};

    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            lNodeId = localMap[eid][nid];
            // get node coordinates
            x = localNodes[lNodeId].get_x();
            y = localNodes[lNodeId].get_y();
            z = localNodes[lNodeId].get_z();

            // translate origin
            x = x - Lx/2;
            y = y - Ly/2;

            xe[nid * NDOF_PER_NODE] = x;
            xe[(nid * NDOF_PER_NODE) + 1] = y;
            xe[(nid * NDOF_PER_NODE) + 2] = z;

            // const body force in z direction
            beN[(nid * NDOF_PER_NODE)] = 0.0;
            beN[(nid * NDOF_PER_NODE) + 1] = 0.0;
            beN[(nid * NDOF_PER_NODE) + 2] = -rho * g;
        }

        // compute element stiffness matrix
        if (useEigen) {
            ke_hex20_iso(kee[0], xe, E, nu, intData.Pts_n_Wts, NGT);
        } else {
            printf("Error: not yet implement element stiffness matrix which is not Eigen matrix format\n");
        }

        // assemble element stiffness matrix to global K
        if (matType == 0) petsc_time.start();
        else aMat_time.start();
        stMat->set_element_matrix(eid, kee[0], 0, 0, 1);
        if (matType == 0) petsc_time.stop();
        else aMat_time.stop();
        
        // compute element force vector due to body force
        fe_hex20_iso(fee[0], xe, beN, intData.Pts_n_Wts, NGT);

        // assemble element load vector due to body force
        if (matType == 0) petsc_time.start();
        else aMat_time.start();
        stMat->petsc_set_element_vec(rhs, eid, fee[0], 0, ADD_VALUES);
        if (matType == 0) petsc_time.stop();
        else aMat_time.stop();

        // assemble element load vector due to traction
        if (elem_trac[eid].size() != 0){
            if (matType == 0) petsc_time.start();
            else aMat_time.start();
            stMat->petsc_set_element_vec(rhs, eid, elem_trac[eid], 0, ADD_VALUES);
            if (matType == 0) petsc_time.stop();
            else aMat_time.stop();
        }
    }
    delete [] xe;

    // Pestc begins and completes assembling the global stiffness matrix
    if (matType == 0){
        petsc_time.start();
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
        petsc_time.stop();
    }

    // char fname[256];
    // sprintf(fname,"matrix_%d.dat",matType);
    // stMat->petsc_dump_mat(fname);

    // Pestc begins and completes assembling the global load vector
    // These are needed because we used ADD_VALUES for rhs when assembling
    // now we are going to use INSERT_VALUE for Fc in apply_bc_rhs
    if (matType != 0) aMat_time.start();
    else petsc_time.start();

    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);
    // apply bc for rhs: this must be done before applying bc for the matrix
    // because we use the original matrix to compute KfcUc in matrix-based method
    stMat->apply_bc(rhs);
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    if (matType != 0) aMat_time.stop();
    else petsc_time.stop();

    // this is needed because the matrix is applied bc in apply_bc(rhs)
    if (matType == 0){
        petsc_time.start();
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
        petsc_time.stop();
    }
    
    // solve
    if (matType != 0) aMat_time.start();
    else petsc_time.start();
    stMat->petsc_solve((const Vec) rhs, out);
    if (matType != 0) aMat_time.stop();
    else petsc_time.stop();

    total_time.stop();

    // display timing
    if (matType != 0){
        if (size > 1){
            long double aMat_maxTime;
            MPI_Reduce(&aMat_time.seconds, &aMat_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0){
                std::cout << "aMat time = " << aMat_maxTime << "\n";
            }
        } else {
            if (rank == 0){
                std::cout << "aMat time = " << aMat_time.seconds << "\n";
            }
        }
        
    } else {
        if (size > 1){
            long double petsc_maxTime;
            MPI_Reduce(&petsc_time.seconds, &petsc_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0){
                std::cout << "PETSC time = " << petsc_maxTime << "\n";
            }
        } else {
            if (rank == 0){
                std::cout << "PETSC time = " << petsc_time.seconds << "\n";
            }
        }
    }
    if (size > 1){
        long double total_time_max;
        MPI_Reduce(&total_time.seconds, &total_time_max, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0){
            std::cout << "total time = " << total_time_max << "\n";
        }
    } else {
        if (rank == 0){
            std::cout << "total time = " << total_time.seconds << "\n";
        }
    }
    
    //sprintf(fname,"outVec_%d.dat",size);
    //stMat.dump_vec(out,fname);

    PetscScalar norm, alpha = -1.0;

    // compute norm of solution
    VecNorm(out, NORM_2, &norm);
    if (rank == 0){
        printf("L2 norm of computed solution = %20.10f\n",norm);
    }

    // exact solution
    Matrix< double, NDOF_PER_NODE * NNODE_PER_ELEM, 1 > e_exact;
    double disp [3];
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            lNodeId = localMap[eid][nid];
            // get node coordinates
            x = localNodes[lNodeId].get_x();
            y = localNodes[lNodeId].get_y();
            z = localNodes[lNodeId].get_z();
            
            // transformed coordinates
            x = x - Lx/2;
            y = y - Ly/2;

            disp[0] = (-nu*rho*g/E) * x * z;
            disp[1] = (-nu*rho*g/E) * y * z;
            disp[2] = (rho*g/2/E)*(z*z - Lz*Lz) + (nu*rho*g/2/E)*(x*x + y*y);
            
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                e_exact[(nid * NDOF_PER_NODE) + did] = disp[did];
            }
        }
        stMat->petsc_set_element_vec(sol_exact, eid, e_exact, 0, INSERT_VALUES);
    }

    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

    //sprintf(fname,"exactVec_%d.dat",size);
    //stMat.dump_vec(sol_exact,fname);

    // compute norm of exact solution
    VecNorm(sol_exact, NORM_2, &norm);
    if (rank == 0){
        printf("L2 norm of exact solution = %20.10f\n",norm);
    }

    // compute the error vector
    VecCopy(sol_exact, error);

    // subtract error = error(=sol_exact) - out
    VecAXPY(error, alpha, out);

    // compute norm of error
    //VecNorm(sol_exact, NORM_INFINITY, &norm);
    VecNorm(error, NORM_INFINITY, &norm);
    if (rank == 0){
        printf("Inf norm of error = %20.10f\n", norm);
    }

    // export ParaView
    /* std::ofstream myfile;
    if (!rank){
        myfile.open("ex4.vtk");
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
        unsigned int size_cell_list = nelem * 21;
        myfile << "CELLS " << nelem << " " << size_cell_list << std::endl;
        for (unsigned int eid = 0; eid < nelem; eid++){
            myfile << "20 " << localMap[eid][0] << " " << localMap[eid][1] << " "
             << localMap[eid][2] << " " << localMap[eid][3] << " "
              << localMap[eid][4] << " " << localMap[eid][5] << " "
               << localMap[eid][6] << " " << localMap[eid][7] << " "
               << localMap[eid][8] << " " << localMap[eid][9] << " "
               << localMap[eid][10] << " " << localMap[eid][11] << " "
               << localMap[eid][12] << " " << localMap[eid][13] << " "
               << localMap[eid][14] << " " << localMap[eid][15] << " "
               << localMap[eid][16] << " " << localMap[eid][17] << " "
               << localMap[eid][18] << " " << localMap[eid][19] << " " << std::endl;
        }
        myfile << "CELL_TYPES " << nelem << std::endl;
        for (unsigned int eid = 0; eid < nelem; eid++){
            myfile << "25" << std::endl;
        }
        myfile << "POINT_DATA " << nnode << std::endl;
        myfile << "VECTORS " << "displacement " << "float " << std::endl;
        std::vector<PetscInt> indices (NDOF_PER_NODE);
        std::vector<PetscScalar> values (NDOF_PER_NODE);
        for (unsigned int nid = 0; nid < numLocalNodes; nid++){
            gNodeId = local2GlobalMap[nid];
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                indices[did] = gNodeId * NDOF_PER_NODE + did;
            }
            VecGetValues(out, NDOF_PER_NODE, indices.data(), values.data());
            myfile << values[0] << " " << values[1] << " " << values[2] << std::endl;
        }
        myfile.close();
    } */
    
    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete [] globalMap[eid];
        delete [] globalDofMap[eid];
    }
    delete [] globalMap;
    delete [] globalDofMap;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete [] localMap[eid];
    }
    delete [] localMap;
    delete [] nnodeCount;
    delete [] nnodeOffset;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete [] localDofMap[eid];
    }
    delete [] localDofMap;
    delete [] local2GlobalMap;
    delete [] local2GlobalDofMap;

    delete [] ndofs_per_element;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete [] bound_dofs[eid];
        delete [] bound_values[eid];
    }
    delete [] bound_dofs;
    delete [] bound_values;

    delete [] constrainedDofs_ptr;
    delete [] prescribedValues_ptr;

    delete [] elem_trac;

    VecDestroy(&out);
    VecDestroy(&sol_exact);
    VecDestroy(&rhs);

    PetscFinalize();
    return 0;
}