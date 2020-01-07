/**
 * @file fem3d.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 * @author Milinda Fernando milinda@cs.utah.edu
 *
 * @brief Example of solving 3D Poisson equation by FEM, in parallel, using Petsc
 *
 * @version 0.1
 * @date 2018-11-30
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#include <iostream>

#include <math.h>
#include <stdio.h>
#include <time.h>

#include <omp.h>
#include <mpi.h>

#ifdef BUILD_WITH_PETSC
#    include <petsc.h>
#endif

#include <Eigen/Dense>

#include "shfunction.hpp"
#include "ke_matrix.hpp"
#include "me_matrix.hpp"
#include "fe_vector.hpp"
#include "aMat.hpp"

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;

using namespace std;

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

    const bool useEigen = atoi(argv[4]); // use Eigen matrix
    const bool matFree = atoi(argv[5]);

    //Matrix<double,8,8>* kee;
    //kee = new Matrix<double,8,8>[AMAT_MAX_EMAT_PER_ELEMENT];
    std::vector<Matrix<double,8,8>> kee;
    kee.resize(AMAT_MAX_EMAT_PER_ELEMENT);

    double* xe = new double[nDim * nNodePerElem];
    double* ke = new double[(nDofPerNode * nNodePerElem) * (nDofPerNode * nNodePerElem)];
    //double* fe = new double[nDofPerNode * nNodePerElem];
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
    }
    if(!rank && useEigen) { std::cout<<"\t\tuseEigen: "<<useEigen << "\n"; }
    if(!rank && matFree)  { std::cout<<"\t\tmatrix free: "<<matFree << "\n"; }
    if(!rank) { std::cout<<"=====================================================\n"; }

    if( rank == 0 ) {
        if (size > Nez) {
            printf("The number of processes must be less than or equal Nez, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // find min & max element index in x direction =================================
    // number of elements in z direction
    double d = (Nez)/(double)(size);
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
    }

    // number of elements (partition of processes is only in z direction)
    unsigned int nelem_z = (emax - emin + 1);
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
        globalMap[e] = new unsigned long int[AMAT_MAX_EMAT_PER_ELEMENT*nNodePerElem];
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
    par::AMAT_TYPE matType;
    if (matFree){
        matType = par::AMAT_TYPE::MAT_FREE;
    } else {
        matType = par::AMAT_TYPE::PETSC_SPARSE;
    }

    par::aMat<double, unsigned long, unsigned int> stMat(matType);

    // set communicator
    stMat.set_comm(comm);

    // set globalMap
    stMat.set_map(nelem, localMap, nodes_per_element, numLocalNodes, local2GlobalMap, start_global_node,
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
            if ((std::fabs(x) < tol) || (std::fabs(x - Lx) < tol) ||
                (std::fabs(y) < tol) || (std::fabs(y - Ly) < tol) ||
                (std::fabs(z) < tol) || (std::fabs(z - Lz) < tol)) {
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
    stMat.set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, constrainedDofs.size());


    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact;
    stMat.petsc_create_vec(rhs);
    stMat.petsc_create_vec(out);
    stMat.petsc_create_vec(sol_exact);

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

            // specify boundary nodes
            /*if ((std::fabs(x) < tol) || (std::fabs(x - Lx) < tol) ||
                (std::fabs(y) < tol) || (std::fabs(y - Ly) < tol) ||
                (std::fabs(z) < tol) || (std::fabs(z - Lz) < tol)) {
                bound_nodes[eid][n] = 1; // boundary
                bound_values[eid][n] = 0; // prescribed value
            } else {
                bound_nodes[eid][n] = 0; // interior
                bound_values[eid][n] = -1000000; // for testing
            }*/
        }

        // compute element stiffness matrix
        if (useEigen) {
            ke_hex8_eig(kee[0], xe);
            //ke_hex8_eig_test(kee[0], xe);
        } else {
            ke_hex8(ke, xe);
        }

        // compute element force vector
        fe_hex8_eig(fe,xe);

        // assemble element stiffness matrix to global K
        if (useEigen){
            if (matFree) {
                // copy element matrix to store in m_epMat[eid]
                stMat.copy_element_matrix(eid, kee[0], 0, 0, 1);

            } else {
                stMat.petsc_set_element_matrix(eid, kee[0], 0, 0, ADD_VALUES);
            }
        } else {
            if (matFree){
                std::cout<<"Error: matrix free only works for Eigen matrix"<<std::endl;
            } else {
                stMat.petsc_set_element_matrix(eid, ke, ADD_VALUES);
            }
        }
        // assemble element load vector to global F
        stMat.petsc_set_element_vec(rhs, eid, fe, 0, ADD_VALUES); // 0 is block_i which is only one block for this case
    }


    // set boundary globalMap
    //stMat.set_bdr_map_old(bound_nodes, bound_values);

    delete [] ke;
    //delete [] fe;
    delete [] xe;

    // Pestc begins and completes assembling the global stiffness matrix
    if (!matFree){
        stMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }

    // Pestc begins and completes assembling the global load vector
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);

    //stMat.dump_vec("rhs_vec.dat", rhs);
    //stMat.dump_vec("out_vec.dat", out);

    // ---------------------------------------------------------------------------------------------
    // matrix-free test:
    /*if (matFree) {
        double *dv;
        double *du;

        const bool ghosted = false; // not include ghost nodes

        // write assembled matrix for comparison
        stMat.dump_mat("mat_as.m");

        // build scatter globalMap
        stMat.buildScatterMap();

        // set local to global globalMap
        stMat.set_Local2Global(local2GlobalMap);

        // create m_pMat_matvec and initialize all to zero
        stMat.petsc_create_matrix_matvec();

        // create vectors dv and du with size = nnode and initialize all components to zero
        stMat.create_vec(dv, ghosted);
        stMat.create_vec(du, ghosted);

        unsigned int local_dof_one; // local dof that has value 1

        for (unsigned int n = 0; n < nnode_total; n++) {
            //printf("rank= %d, n= %d\n", rank, n);
            if ((n >= nnode_scan) && (n < (nnode_scan + nnode))) {
                local_dof_one = n - nnode_scan;
                for (unsigned int j = 0; j < nnode; j++) {
                    if (j == local_dof_one) {
                        du[j] = 1.0;
                    } else {
                        du[j] = 0.0;
                    }
                }
            } else {
                for (unsigned int j = 0; j < nnode; j++) {
                    du[j] = 0.0;
                }
            }
            stMat.matvec(dv, du);
            stMat.petsc_set_matrix_matvec(dv, n, ADD_VALUES);
        }
        // std::cout << "Finished setting cols" << std::endl;

        stMat.petsc_init_mat_matvec(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat_matvec(MAT_FINAL_ASSEMBLY);
        stMat.dump_mat_matvec("mat_mf.m");

        //stMat.petsc_compare_matrix();
        //stMat.petsc_norm_matrix_difference();

        // do matvec using Petsc
        Vec petsc_dv, petsc_du, petsc_compare;
        stMat.petsc_create_vec(petsc_dv, 0.0);
        stMat.petsc_create_vec(petsc_du, 1.0);
        stMat.petsc_matmult(petsc_du, petsc_dv); //petsc_du = matrix*pets_dv
        //stMat.dump_vec("petsc_dv.dat", petsc_dv);//print to file of petsc_dv

        // transform dv to pestc vector for easy to compare
        stMat.petsc_create_vec(petsc_compare, 0.0);
        stMat.transform_to_petsc_vector(dv, petsc_compare, ghosted); // transform dv to pestc-type vector petsc_compare
        //stMat.dump_vec("petsc_compare.dat", petsc_compare); // print to file petsc_compare

        // subtract two vectors: petsc_compare = petsc_compare - petsc_dv
        PetscScalar norm1, alpha1 = -1.0;
        VecAXPY(petsc_compare, alpha1, petsc_dv);

        // compute the norm of petsc_compare
        VecNorm(petsc_compare, NORM_INFINITY, &norm1);

        if (rank == 0){
            printf("petsc_compare norm= %20.10f\n", norm1);
        }

    }*/

    // testing the get_diagonal
    /*if (matFree){
        double* diag;
        Vec petsc_diag, diag_to_petsc;

        stMat.create_vec(diag, false);
        stMat.get_diagonal(diag, false);

        //stMat.print_vector(diag, false);
        stMat.petsc_create_vec(petsc_diag);
        stMat.petsc_create_vec(diag_to_petsc);

        stMat.transform_to_petsc_vector(diag, diag_to_petsc, false);
        stMat.petsc_get_diagonal(petsc_diag);

        stMat.dump_vec("diag_to_petsc.dat",diag_to_petsc);
        stMat.dump_vec("petsc_diag.dat",petsc_diag);
    }*/

    // modifying stiffness matrix and load vector to apply dirichlet BCs
    if (!matFree){
        stMat.apply_bc_mat();
        stMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }
    //stMat.apply_bc_rhs_old(rhs);
    stMat.apply_bc_rhs(rhs);
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);

    // write results to files
    /*if (!matFree){
        stMat.dump_mat("matrix.dat");
        stMat.dump_vec(rhs,"rhs.dat");
    } else {
        stMat.print_mepMat();
    }*/

    // solve
    stMat.petsc_solve((const Vec) rhs, out);

    stMat.petsc_init_vec(out);
    stMat.petsc_finalize_vec(out);

    // compute exact solution for comparison
    //double* e_exact = new double[nNodePerElem]; // only for 1 DOF/node
    Matrix<double,8,1> e_exact;

    for (unsigned int e = 0; e < nelem; e++) {
        for (unsigned int n = 0; n < nNodePerElem; n++) {
            // global node ID
            nid = globalMap[e][n];
            // nodal coordinates
            x = (double)(nid % (Nex + 1)) * hx;
            y = (double)((nid % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(nid / ((Nex + 1)*(Ney + 1))) * hz;

            // exact solution at node n (and apply BCs)
            if ((std::abs(x) < tol) || (std::abs(x - Lx) < tol) ||
                (std::abs(y) < tol) || (std::abs(y - Ly) < tol) ||
                (std::abs(z) < tol) || (std::abs(z - Lz) < tol)) {
                e_exact(n) = 0.0; // boundary
            } else {
                e_exact(n) = (1.0 / (12.0 * M_PI * M_PI)) * sin(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * z);
            }
        }
        // set exact solution to Pestc vector
        stMat.petsc_set_element_vec(sol_exact, e, e_exact, 0, INSERT_VALUES);
    }

    //delete [] e_exact;

    // Pestc begins and completes assembling the exact solution
    stMat.petsc_init_vec(sol_exact);
    stMat.petsc_finalize_vec(sol_exact);

    //stMat.dump_vec("exact_vec.dat", sol_exact);

    // subtract out from sol_exact
    PetscScalar norm, alpha = -1.0;
    VecAXPY(sol_exact, alpha, out);

    // compute the norm of sol_exact
    VecNorm(sol_exact, NORM_INFINITY, &norm);

    if (rank == 0){
        printf("L_inf norm= %20.10f\n", norm);
    }

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

    //delete [] etype;
    //delete [] kee;
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
