/**
 * @file fem1d.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Example of solving 1D FEM problem, in parallel using aMat, Petsc and Eigen
 * @brief d^u/dx^2 = 0 with BCs u(0) = 0, u(L) = 1
 *
 * @version 0.1
 * @date 2020-01-03
 *
 * @copyright Copyright (c) 2020 School of Computing, University of Utah
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

#include "Eigen/Dense"

#include "ke_matrix.hpp"
#include "fe_vector.hpp"
#include "aMat.hpp"

using Eigen::Matrix;

void usage()
{
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  fem2d <Nex> <use matrix-free> <bc method>\n";
    std::cout << "\n";
    std::cout << "     Nex: Number of elements in X\n";
    std::cout << "     use matrix-free: 1 => yes.  0 => matrix-based method.\n";
    std::cout << "     use identity-matrix: 0    use penalty method: 1 \n";
    std::cout << "\n";
    std::exit( 0 ) ;
}

int main(int argc, char *argv[]){
    // User provides: Ne = number of elements
    //                flag = 1 --> matrix-free method; 0 --> matrix-based method
    if( argc < 3 ) {
        usage();
    }
    const unsigned int Ne = atoi(argv[1]);
    const bool matFree = atoi(argv[2]);
    const unsigned int bcMethod = atoi(argv[3]); // method of applying BC

    // element matrix and force vector
    Matrix<double,2,2> ke;
    Matrix<double,2,1> fe;

    // domain size
    const double L = 1.0;

    // element size
    const double h = L/double(Ne);

    const double tol = 0.0001;

    // MPI initialize
    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(!rank) {
        std::cout<<"============ parameters read  =======================\n";
        std::cout << "\t\tNumber of elements Ne = " << Ne << "\n";
        std::cout << "\t\tMethod (0 = matrix based; 1 = matrix free) = " << matFree << "\n";
        std::cout << "\t\tL : "<< L << "\n";
        std::cout<<"\t\tRunning with: "<< size << " ranks \n";
        std::cout<<"\t\tNumber of threads: "<< omp_get_max_threads() << "\n";
    }
    #ifdef AVX_512
    if (!rank) {std::cout << "\t\tRun with AVX_512\n";}
    #elif AVX_256
    if (!rank) {std::cout << "\t\tRun with AVX_256\n";}
    #elif OMP_SIMD
    if (!rank) {std::cout << "\t\tRun with OMP_SIMD\n";}
    #else
    if (!rank) {std::cout << "\t\tRun with no vectorization\n";}
    #endif

    int rc;
    if (size > Ne){
        if (!rank){
            std::cout << "Number of ranks must be <= Ne, program stops..." << "\n";
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // partition in x direction...
    unsigned int emin = 0, emax = 0;
    unsigned int nelem;
    // minimum number of elements in x-dir for each rank
    unsigned int nxmin = Ne/size;
    // remaining
    unsigned int nRemain = Ne % size;
    // distribute nRemain uniformly from rank = 0 up to rank = nRemain - 1
    if (rank < nRemain){
        nelem = nxmin + 1;
    } else {
        nelem = nxmin;
    }
    if (rank < nRemain){
        emin = rank * nxmin + rank;
    } else {
        emin = rank * nxmin + nRemain;
    }
    emax = emin + nelem - 1;

    // determine number of elements owned my rank
    /* double d = Ne/(double)(size);
    double xmin = rank * d;
    if (rank == 0){
        xmin -= tol;
    }
    double xmax = xmin + d;
    if (rank == size){
        xmax += tol;
    }
    // begin and end element count
    unsigned int emin = 0, emax = 0;
    for (unsigned int i = 0; i < Ne; i++){
        if (i >= xmin){
            emin = i;
            break;
        }
    }
    for (unsigned int i = (Ne - 1); i >= 0; i--){
        if (i < xmax){
            emax = i;
            break;
        }
    } */

    // number of nodes owned by my rank (rank 0 owns 2 boundary nodes, other ranks own right boundary node)
    unsigned int nnode;
    if (rank == 0){
        nnode = nelem + 1;
    } else {
        nnode = nelem;
    }

    // determine globalMap
    unsigned int* nnode_per_elem = new unsigned int [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        nnode_per_elem[eid] = 2; //linear 2-node element
    }

    unsigned long int** globalMap;
    globalMap = new unsigned long* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        globalMap[eid] = new unsigned long [nnode_per_elem[eid]];
    }
    // todo hard-code 2 nodes per element:
    for (unsigned int eid = 0; eid < nelem; eid++){
        globalMap[eid][0] = (emin + eid);
        globalMap[eid][1] = globalMap[eid][0] + 1;
    }


    // build localMap from globalMap (to adapt the interface of bsamxx)
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
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
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
    for (unsigned int eid = 0; eid < nelem; eid++){
        localMap[eid] = new unsigned int[nnode_per_elem[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int i = 0; i < nnode_per_elem[eid]; i++){
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
    // build local2GlobalMap map (to adapt the interface of bsamxx)
    unsigned long * local2GlobalMap = new unsigned long[numLocalNodes];
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            gNodeId = globalMap[eid][nid];
            local2GlobalMap[localMap[eid][nid]] = gNodeId;
        }
    }

    // compute constrained map
    unsigned int** bound_nodes = new unsigned int* [nelem];
    double** bound_values = new double* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        bound_nodes[eid] = new unsigned int [nnode_per_elem[eid]];
        bound_values[eid] = new double [nnode_per_elem[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            unsigned long global_Id = globalMap[eid][nid];
            double x = (double)(global_Id * h);
            if ((fabs(x) < 0.000001) || (fabs(x - L) < 0.000001)){
            //if ((fabs(x) < 0.000001) || (fabs(x - L) < 0.000001) || (fabs(x - 0.5*L) < 0.000001)){
                bound_nodes[eid][nid] = 1;
            } else {
                bound_nodes[eid][nid] = 0;
            }
            if (fabs(x) < 0.000001){
                // left end
                bound_values[eid][nid] = 0.0;
            } else if (fabs(x - L) < 0.000001){
                // right end
                bound_values[eid][nid] = 1.0;
                //bound_values[eid][nid] = 20.0;
            //} else if (fabs(x - 0.5*L) < 0.000001){
            //    bound_values[eid][nid] = 0.7;
            } else {
                // free dofs
                bound_values[eid][nid] = -1000000;
            }
        }
    }
    // create lists of constrained dofs (for new interface of set_bdr_map)
    std::vector<unsigned long> constrainedDofs;
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
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
        unsigned long global_Id = constrainedDofs[i];
        double x = (double)(global_Id * h);
        constrainedDofs_ptr[i] = global_Id;
        if (fabs(x - L) < 0.000001){
            prescribedValues_ptr[i] = 1.0;
            //prescribedValues_ptr[i] = 20.0;
        //} else if (fabs(x - 0.5*L) < 0.000001){
        //    prescribedValues_ptr[i] = 0.7;
        } else if (fabs(x) < 0.000001){
            prescribedValues_ptr[i] = 0.0;
        } else {
            std::cout << "something wrong with the list of constrained dofs...";
            return 0;
        }
    }

    unsigned long start_global_node, end_global_node;
    start_global_node = nnodeOffset[rank];
    end_global_node = start_global_node + (nnode - 1);

    // declare aMat object =================================
    par::AMAT_TYPE matType;
    if (matFree){
        matType = par::AMAT_TYPE::MAT_FREE;
    } else {
        matType = par::AMAT_TYPE::PETSC_SPARSE;
    }

    par::aMat<double, unsigned long, unsigned int> stMat(matType, (par::BC_METH)bcMethod);
    stMat.set_comm(comm);
    stMat.set_map(nelem, localMap, nnode_per_elem, numLocalNodes, local2GlobalMap, start_global_node, end_global_node, nnode_total);
    stMat.set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, constrainedDofs.size());

    Vec rhs, out;
    stMat.petsc_create_vec(rhs);
    stMat.petsc_create_vec(out);

    // element stiffness matrix and assembly
    for (unsigned int eid = 0; eid < nelem; eid++){
        ke(0,0) = 1.0/h;
        ke(0,1) = -1.0/h;
        ke(1,0) = -1.0/h;
        ke(1,1) = 1.0/h;
        if (matFree){
            stMat.copy_element_matrix(eid, ke, 0, 0, 1);
        } else {
            stMat.petsc_set_element_matrix(eid, ke, 0, 0, ADD_VALUES);
        }
        fe(0) = 0.0;
        fe(1) = 0.0;
        stMat.petsc_set_element_vec(rhs, eid, fe, 0, ADD_VALUES);
    }

    // Pestc begins and completes assembling the global stiffness matrix
    if (!matFree){
        stMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }

    // These are needed because we used ADD_VALUES for rhs when assembling
    // now we are going to use INSERT_VALUE for Fc in apply_bc_rhs
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);

    // apply bc for rhs: this must be done before applying bc for the matrix
    // because we use the original matrix to compute KfcUc in matrix-based method
    stMat.apply_bc_rhs(rhs);
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);

    // apply bc for matrix
    if (!matFree){
        stMat.apply_bc_mat();
        stMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }

    // solve
    stMat.petsc_solve((const Vec) rhs, out);

    // display solution on screen
    if (!rank) std::cout << "Computed solution = \n";
    stMat.dump_vec(out);

    // following exact solution is only for the problem of u(0) = 0 and u(L) = 1
    // exact solution: u(x) = (1/L)*x
    Vec sol_exact;
    PetscInt rowId;
    stMat.petsc_create_vec(sol_exact);
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            rowId = globalMap[eid][nid];
            double x = globalMap[eid][nid] * h;
            PetscScalar u = 1/L*x;
            VecSetValue(sol_exact,rowId,u,INSERT_VALUES);
        }
    }
    stMat.petsc_init_vec(sol_exact);
    stMat.petsc_finalize_vec(sol_exact);

    // display exact solution on screen
    //stMat.dump_vec(sol_exact);

    // compute the norm of error
    PetscScalar norm, alpha = -1.0;
    VecAXPY(sol_exact, alpha, out);
    VecNorm(sol_exact, NORM_INFINITY, &norm);

    if (!rank){
        printf("Inf norm of error = %20.16f\n", norm);
    }

    // free allocated memory...
    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] bound_nodes[eid];
        delete [] bound_values[eid];
    }
    delete [] bound_nodes;
    delete [] bound_values;

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] globalMap[eid];
    }
    delete [] globalMap;

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] localMap[eid];
    }
    delete [] localMap;
    delete [] nnodeCount;
    delete [] nnodeOffset;
    delete [] local2GlobalMap;
    delete [] nnode_per_elem;
    VecDestroy(&out);
    //VecDestroy(&sol_exact);
    VecDestroy(&rhs);


    PetscFinalize();
    return 0;
}