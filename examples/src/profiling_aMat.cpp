//
// Created by Han Tran on 2020-01-20.
//

#include <iostream>

#include <math.h>
#include <stdio.h>
#include <time.h>

#include <omp.h>
#include <mpi.h>

#ifdef BUILD_WITH_PETSC
#    include <petsc.h>
#endif

#include <Dense>

#include "ke_matrix.hpp"
#include "fe_vector.hpp"
#include "aMat.hpp"
#include "integration.hpp"
#include "profiler.hpp"

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;

using namespace std;

// number of cracks allowed in 1 element
#define AMAT_MAX_CRACK_LEVEL 0

// max number of block dimensions in one cracked element
#define AMAT_MAX_BLOCKSDIM_PER_ELEMENT (1u<<AMAT_MAX_CRACK_LEVEL)

//////////////////////////////////////////////////////////////////////////////////////////////////////

void
usage()
{
    cout << "\n";
    cout << "Usage:\n";
    cout << "  fem3d <Nex> <Ney> <Nez> <use eigen> <use matrix-free>\n";
    cout << "\n";
    cout << "     Nex: Number of elements in X\n";
    cout << "     Ney: Number of elements in y\n";
    cout << "     Nez: Number of elements in z\n";
    cout << "     use eigen: 1 => yes\n";
    cout << "     use matrix-free: 1 => yes.  0 => matrix-based method. \n";
    exit( 0) ;
}

void compute_nodal_body_force(double* xe, unsigned int nnode, double* be){
    double x, y, z;
    for (unsigned int nid = 0; nid < nnode; nid++){
        x = xe[nid * 3];
        y = xe[nid * 3 + 1];
        z = xe[nid * 3 + 2];
        be[nid] = sin(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * z);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

int main( int argc, char *argv[] ) {
    // User provides: Nex - number of elements (global) in x direction
    //                Ney - number of elements (global) in y direction
    //                Nez - number of elements (global) in z direction
    //                flag1 - 1 use Eigen, 0 not use Eigen
    //                flag2 - 1 matrix-free method, 0 matrix-based method

    if( argc < 5 ) {
        usage();
    }

    int rc;
    double zmin, zmax;

    double x, y, z;
    double hx, hy, hz;

    unsigned  int emin = 0, emax = 0;
    unsigned long nid, eid;
    const unsigned int nDofPerNode = 1;         // number of dofs per node
    const unsigned int nDim = 3;                // number of dimension
    const unsigned int nNodePerElem = 8;        // number of nodes per element

    // number of (global) elements in x, y and z directions
    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]);
    const unsigned int Nez = atoi(argv[3]);

    //05.19.20 only use Eigen matrix
    const bool useEigen = true;

    const bool matType = atoi(argv[4]); // use matrix-free method
    const unsigned int bcMethod = atoi(argv[5]); // method of applying BC

    std::vector<Matrix<double,8,8>> kee;
    kee.resize(AMAT_MAX_BLOCKSDIM_PER_ELEMENT);

    double* xe = new double[nDim * nNodePerElem];
    double* be = new double[nNodePerElem];
    double* ke = new double[(nDofPerNode * nNodePerElem) * (nDofPerNode * nNodePerElem)];

    Matrix<double,8,1> fe;

    // domain sizes: Lx, Ly, Lz - length of the (global) domain in x, y, z direction
    const double Lx = 2.0, Ly = 2.0, Lz = 2.0;

    // element sizes
    hx = Lx/double(Nex);// element size in x direction
    hy = Ly/double(Ney);// element size in y direction
    hz = Lz/double(Nez);// element size in z direction

    const double tol = 0.001;

    PetscInitialize(&argc, &argv, NULL, NULL);

    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Status Stat;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(!rank) {
        std::cout<<"============ parameters read  =======================\n";
        std::cout<<"\t\tNex : "<<Nex<<" Ney: "<<Ney<<" Nez: "<<Nez<< "\n";
        std::cout<<"\t\tRunning with: "<<size << " ranks \n";
    }
    if(!rank && useEigen) { std::cout<<"\t\tuseEigen: "<<useEigen << "\n"; }
    if(!rank && matType)  { std::cout<<"\t\tmatrix free: "<<matType << "\n"; }

    #ifdef AVX_512
    if (!rank) {std::cout << "\t\tRun with AVX_512\n";}
    #elif AVX_256
    if (!rank) {std::cout << "\t\tRun with AVX_256\n";}
    #elif OMP_SIMD
    if (!rank) {std::cout << "\t\tRun with OMP_SIMD\n";}
    #else
    if (!rank) {std::cout << "\t\tRun with no vectorization\n";}
    #endif
    //if(!rank) { std::cout<<"=====================================================\n"; }

    

    if( rank == 0 ) {
        if (size > Nez) {
            printf("The number of processes must be less than or equal Nez, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // find min & max element index in z direction
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

    /* double d = (Nez)/(double)(size);
    zmin = (rank*d);
    if (rank == 0) zmin = zmin - 0.0001;
    zmax = zmin + d;
    if (rank == size) zmax = zmax + 0.0001;
    for (unsigned int i = 0; i < Nez; i++) {
        if (i >= zmin) {
            emin = i;
            break;
        }
    }
    for (unsigned int i = (Nez-1); i >= 0; i--) {
        if (i < zmax) {
            emax = i;
            break;
        }
    } */

    // number of owned elements
    unsigned int nelem_y = Ney;
    unsigned int nelem_x = Nex;
    unsigned int nelem = (nelem_x) * (nelem_y) * (nelem_z);

    // number of nodes, specify what nodes I own based
    unsigned int nnode_z;
    if (rank == 0) {
        nnode_z = nelem_z + 1;
    } else {
        nnode_z = nelem_z;
    }


    unsigned int nnode_y = nelem_y + 1;
    unsigned int nnode_x = nelem_x + 1;
    unsigned int nnode = (nnode_x) * (nnode_y) * (nnode_z);


    // map from local nodes to global nodes
    unsigned long int** globalMap;
    globalMap = new unsigned long int *[nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        globalMap[e] = new unsigned long int[AMAT_MAX_BLOCKSDIM_PER_ELEMENT*nNodePerElem];
    }
    for (unsigned k = 0; k < nelem_z; k++){
        for (unsigned j = 0; j < nelem_y; j++){
            for (unsigned i = 0; i < nelem_x; i++){
                eid = nelem_x * nelem_y * k + nelem_x * j + i;
                globalMap[eid][0] = (emin*(Nex + 1)*(Ney + 1) + i) + j*(Nex + 1) + k*(Nex + 1)*(Ney + 1);
                globalMap[eid][1] = globalMap[eid][0] + 1;
                globalMap[eid][3] = globalMap[eid][0] + (Nex + 1);
                globalMap[eid][2] = globalMap[eid][3] + 1;
                globalMap[eid][4] = globalMap[eid][0] + (Nex + 1)*(Ney + 1);
                globalMap[eid][5] = globalMap[eid][4] + 1;
                globalMap[eid][7] = globalMap[eid][4] + (Nex + 1);
                globalMap[eid][6] = globalMap[eid][7] + 1;
            }
        }
    }

    // build localMap from globalMap
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalNodes;
    unsigned long gNodeId;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;

    unsigned int* nnodeCount = new unsigned int [size];
    unsigned int* nnodeOffset = new unsigned int [size];

    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    nnodeOffset[0] = 0;
    for (unsigned int i = 1; i < size; i++){
        nnodeOffset[i] = nnodeOffset[i-1] + nnodeCount[i-1];
    }
    unsigned int nnode_total;
    nnode_total = nnodeOffset[size-1] + nnodeCount[size-1];

    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < 8; nid++){
            gNodeId = globalMap[eid][nid];
            if (gNodeId < nnodeOffset[rank]){
                preGhostGIds.push_back(gNodeId);
            } else if (gNodeId >= nnodeOffset[rank] + nnode){
                postGhostGIds.push_back(gNodeId);
            }
        }
    }
    // sort in ascending order
    std::sort(preGhostGIds.begin(), preGhostGIds.end());
    std::sort(postGhostGIds.begin(), postGhostGIds.end());

    // remove consecutive duplicates and erase all after .end()
    preGhostGIds.erase(std::unique(preGhostGIds.begin(), preGhostGIds.end()), preGhostGIds.end());
    postGhostGIds.erase(std::unique(postGhostGIds.begin(), postGhostGIds.end()), postGhostGIds.end());

    numPreGhostNodes = preGhostGIds.size();
    numPostGhostNodes = postGhostGIds.size();
    numLocalNodes = numPreGhostNodes + nnode + numPostGhostNodes;

    unsigned int** localMap;
    localMap = new unsigned int* [nelem];
    for (unsigned int e = 0; e < nelem; e++){
        localMap[e] = new unsigned int[8];
    }

    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int i = 0; i < 8; i++){
            gNodeId = globalMap[eid][i];
            if (gNodeId >= nnodeOffset[rank] &&
                gNodeId < (nnodeOffset[rank] + nnode)) {
                // nid is owned by me
                localMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
            } else if (gNodeId < nnodeOffset[rank]){
                // nid is owned by someone before me
                const unsigned int lookUp = std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) - preGhostGIds.begin();
                localMap[eid][i] = lookUp;
            } else if (gNodeId >= (nnodeOffset[rank] + nnode)){
                // nid is owned by someone after me
                const unsigned int lookUp = std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), gNodeId) - postGhostGIds.begin();
                localMap[eid][i] =  numPreGhostNodes + nnode + lookUp;
            }
        }
    }

    // local2GlobalMap map
    unsigned long * local2GlobalMap = new unsigned long[numLocalNodes];
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < 8; nid++){
            gNodeId = globalMap[eid][nid];
            local2GlobalMap[localMap[eid][nid]] = gNodeId;
        }
    }

    unsigned long start_global_node, end_global_node;
    start_global_node = nnodeOffset[rank];
    end_global_node = start_global_node + (nnode - 1);

    // boundary nodes: bound
    unsigned int** bound_nodes = new unsigned int *[nelem];
    double** bound_values = new double* [nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        bound_nodes[e] = new unsigned int[nNodePerElem];
        bound_values[e] = new double [nNodePerElem];
    }


    unsigned int* nodes_per_element = new unsigned int[nelem];
    for (unsigned e = 0; e < nelem; e ++){
        nodes_per_element[e] = 8;
    }

    // declare aMat object =================================
    par::aMat<double, unsigned long, unsigned int> * stMat;
    if (matType == 0){
        stMat = new par::aMatBased<double, unsigned long, unsigned int>((par::BC_METH)bcMethod);
    } else {
        stMat = new par::aMatFree<double, unsigned long, unsigned int>((par::BC_METH)bcMethod);
    }

    // set communicator
    stMat->set_comm(comm);

    // set globalMap
    stMat->set_map(nelem, localMap, nodes_per_element, numLocalNodes, local2GlobalMap, start_global_node,
                  end_global_node, nnode_total);


    // construct element maps of constrained DoFs and prescribed values
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int n = 0; n < nNodePerElem; n++) {
            nid = globalMap[eid][n];

            // get node coordinates
            x = (double) (nid % (Nex + 1)) * hx;
            y = (double) ((nid % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
            z = (double) (nid / ((Nex + 1) * (Ney + 1))) * hz;

            // specify boundary nodes
            if ((fabs(x) < tol) || (fabs(x - Lx) < tol) ||
                (fabs(y) < tol) || (fabs(y - Ly) < tol) ||
                (fabs(z) < tol) || (fabs(z - Lz) < tol)) {
                bound_nodes[eid][n] = 1; // boundary
                bound_values[eid][n] = 0; // prescribed value
            } else {
                bound_nodes[eid][n] = 0; // interior
                bound_values[eid][n] = -1000000; // for testing
            }
        }
    }

    // create lists of constrained dofs (for new interface of set_bdr_map)
    std::vector<unsigned long> constrainedDofs;
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nNodePerElem; nid++){
            if (bound_nodes[eid][nid] == 1){
                constrainedDofs.push_back(globalMap[eid][nid]);
            }
        }
    }
    std::sort(constrainedDofs.begin(),constrainedDofs.end());
    constrainedDofs.erase(std::unique(constrainedDofs.begin(),constrainedDofs.end()),constrainedDofs.end());

    unsigned long * constrainedDofs_ptr;
    double * prescribedValues_ptr;
    constrainedDofs_ptr = new unsigned long [constrainedDofs.size()];
    prescribedValues_ptr = new double [constrainedDofs.size()];
    for (unsigned int i = 0; i < constrainedDofs.size(); i++){
        constrainedDofs_ptr[i] = constrainedDofs[i];
        prescribedValues_ptr[i] = 0.0;
    }
    // set boundary maps
    stMat->set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, constrainedDofs.size());


    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact;
    stMat->petsc_create_vec(rhs);
    stMat->petsc_create_vec(out);
    stMat->petsc_create_vec(sol_exact);

    // variables for timing
    profiler_t petsc_assembly_time, petsc_matvec_profiler, aMat_matvec_profiler;
    petsc_assembly_time.clear();
    petsc_matvec_profiler.clear();
    aMat_matvec_profiler.clear();

    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);

    // compute element stiffness matrix and assemble global stiffness matrix and load vector
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int n = 0; n < nNodePerElem; n++){
            nid = globalMap[eid][n];

            // get node coordinates
            x = (double)(nid % (Nex + 1)) * hx;
            y = (double)((nid % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(nid / ((Nex + 1)*(Ney + 1))) * hz;
            xe[n * 3] = x;
            xe[(n * 3) + 1] = y;
            xe[(n * 3) + 2] = z;
        }

        // compute element stiffness matrix
        if (useEigen) {
            ke_hex8_eig(kee[0], xe, intData.Pts_n_Wts, NGT);
        } else {
            ke_hex8(ke, xe, intData.Pts_n_Wts, NGT);
        }

        // compute nodal values of body force
        compute_nodal_body_force(xe, 8, be);

        // compute element force vector
        fe_hex8_eig(fe, xe, be, intData.Pts_n_Wts, NGT);

        // assemble element stiffness matrix to global K
        if (useEigen){
            if (matType) {
                stMat->set_element_matrix(eid, kee[0], 0, 0, 1);
            } else {
                petsc_assembly_time.start();
                stMat->set_element_matrix(eid, kee[0], 0, 0, 1);
                petsc_assembly_time.stop();
            }
        } else {
            if (matType){
                std::cout<<"Error: matrix free only works for Eigen matrix"<<std::endl;
            } else {
                petsc_assembly_time.start();
                stMat->set_element_matrix(eid, kee[0], 0, 0, 1);
                petsc_assembly_time.stop();
            }
        }
        // assemble element load vector to global F
        stMat->petsc_set_element_vec(rhs, eid, fe, 0, ADD_VALUES); // 0 is block_i which is only one block for this case
    }

    delete [] ke;
    delete [] xe;
    delete [] be;

    // Pestc begins and completes assembling the global stiffness matrix
    if (!matType){
        petsc_assembly_time.start();
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
        petsc_assembly_time.stop();
    }

    // get the max assembly-time among ranks
    long double petsc_assembly_maxTime;
    MPI_Reduce(&petsc_assembly_time.seconds, &petsc_assembly_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) std::cout << "time for petsc assembly = " << petsc_assembly_maxTime << "\n";

    // These are needed because we used ADD_VALUES for rhs when assembling
    // now we are going to use INSERT_VALUE for Fc in apply_bc_rhs
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // apply bc for rhs: this must be done before applying bc for the matrix
    // because we use the original matrix to compute KfcUc in matrix-based method
    stMat->apply_bc(rhs); // this includes applying bc for matrix in matrix-based approach
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // apply bc to the matrix
    if (matType == 0){
        //stMat.apply_bc_mat();
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }

    // ====================== profiling matvec ====================================
    // generate random vector
    double* X = (double*) malloc(sizeof(double) * nnode);
    for (unsigned int i = 0; i < nnode; i++){
        X[i] = (double)std::rand()/(double)(RAND_MAX/5.0);
    }
    // result vector Y = [K] * X
    double* Y = (double*) malloc(sizeof(double) * nnode);

    // total number of matvec's we want to profile
    const unsigned int num_matvecs = 100;

    /* #ifdef AMAT_PROFILER
    // reset all counters
    stMat.reset_profile_counters();
    #endif */

    // do matvec()...
    if (matType){
        double* temp;
        for (unsigned int i = 0; i < num_matvecs; i++){

            aMat_matvec_profiler.start();
            stMat->matvec(Y, X, false);
            aMat_matvec_profiler.stop();

            // swap X and Y
            temp = X;
            X = Y;
            Y = temp;
        }
    } else {
        Vec petsc_X, petsc_Y;
        stMat.petsc_create_vec(petsc_X);
        stMat.petsc_create_vec(petsc_Y);
        for (unsigned int i = 0; i < nnode; i++){
            PetscInt local_rowID = i + numPreGhostNodes;
            PetscInt global_rowID = local2GlobalMap[local_rowID];
            PetscScalar value;
            value = X[i];
            VecSetValue(petsc_X, global_rowID, value, INSERT_VALUES);
        }

        stMat.petsc_init_vec(petsc_X);
        stMat.petsc_finalize_vec(petsc_X);

        for (unsigned int i = 0; i < num_matvecs; i++){

            petsc_matvec_profiler.start();
            stMat.petsc_matmult(petsc_X, petsc_Y);
            //stMat.petsc_init_vec(petsc_X);
            //stMat.petsc_finalize_vec(petsc_X);
            petsc_matvec_profiler.stop();

            VecSwap(petsc_Y, petsc_X);
        }
        stMat.petsc_destroy_vec(petsc_X);
        stMat.petsc_destroy_vec(petsc_Y);
    }

    // get max matvec-time among ranks
    long double petsc_matvec_maxTime;
    long double aMat_matvec_maxTime;
    if (matType){
        MPI_Reduce(&aMat_matvec_profiler.seconds, &aMat_matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0){
            aMat_matvec_maxTime = aMat_matvec_maxTime/num_matvecs;
            std::cout << "time for aMat matvec = " << aMat_matvec_maxTime << "\n";
        }
    } else {
        MPI_Reduce(&petsc_matvec_profiler.seconds, &petsc_matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0){
            petsc_matvec_maxTime = petsc_matvec_maxTime/num_matvecs;
            std::cout << "time for petsc matvec = " << petsc_matvec_maxTime << "\n";
        }
    }

    /* #ifdef AMAT_PROFILER
    stMat.profile_dump(std::cout);
    #endif */

    free(X);
    free(Y);
    

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] globalMap[eid];
        delete [] bound_nodes[eid];
        delete [] bound_values[eid];
    }
    delete [] globalMap;
    delete [] bound_nodes;
    delete [] bound_values;

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] localMap[eid];
    }
    delete [] localMap;

    delete [] nnodeCount;
    delete [] nnodeOffset;

    delete [] local2GlobalMap;
    delete [] nodes_per_element;

    delete [] constrainedDofs_ptr;
    delete [] prescribedValues_ptr;

    // clean up Pestc vectors
    VecDestroy(&out);
    VecDestroy(&sol_exact);
    VecDestroy(&rhs);
    PetscFinalize();

    return 0;
}