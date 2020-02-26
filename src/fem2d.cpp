//
// Created by Han Tran on 1/5/20.
//

/**
 * @file fem2d.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Example of solving 2D FEM problem, in parallel using aMat, Petsc and Eigen
 * @brief (d^2)u/d(x^2) + (d^2)u/d(y^2) = 0
 * @brief BCs u(0,y) = u(1,y) = 0; u(x,0) = sin(pi*x); u(x,1) = sin(pi*x)exp(-pi)
 *
 * @version 0.1
 * @date 2020-01-05
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

#include "shfunction.hpp"
#include "ke_matrix.hpp"
#include "me_matrix.hpp"
#include "fe_vector.hpp"
#include "aMat.hpp"

using Eigen::Matrix;

int main(int argc, char *argv[]){

    // User provides: Nex = number of elements in x direction
    //                Ney = number of elements in y direction
    //                flag = 1 --> matrix-free method; 0 --> matrix-based method
    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]);
    const bool matFree = atoi(argv[3]);

    // element matrix and force vector
    Matrix<double,4,4> ke;
    Matrix<double,4,1> fe;

    // element nodal coordinates
    double* xe = new double [8];

    // domain size
    const double Lx = 1.0;
    const double Ly = 1.0;

    // element size
    const double hx = Lx/double(Nex);
    const double hy = Ly/double(Ney);

    // MPI initialize
    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(!rank) {
        std::cout<<"============ parameters read  =======================\n";
        std::cout << "\t\tNumber of elements Nex = " << Nex << "; Ney = " << Ney << "\n";
        std::cout << "\t\tMethod (0 = matrix based; 1 = matrix free) = " << matFree << "\n";
        std::cout << "=====================================================\n";
    }

    // partition in y direction...
    int rc;
    if (size > Ney){
        if (!rank){
            std::cout << "Number of ranks must be <= Ney, program stops..." << "\n";
            MPI_Abort(comm, rc);
            exit(0);
        }
    }
    // determine number of elements owned my rank
    const double tol = 0.0001;
    double d = Ney/(double)(size);
    double ymin = rank * d;
    if (rank == 0){
        ymin -= tol;
    }
    double ymax = ymin + d;
    if (rank == size){
        ymax += tol;
    }
    // begin and end element count
    unsigned int emin = 0, emax = 0;
    for (unsigned int i = 0; i < Ney; i++){
        if (i >= ymin){
            emin = i;
            break;
        }
    }
    for (unsigned int i = (Ney - 1); i >= 0; i--){
        if (i < ymax){
            emax = i;
            break;
        }
    }
    // number of elements owned by my rank
    unsigned int nelem_y = (emax - emin) + 1;
    unsigned int nelem_x = Nex;
    unsigned int nelem = nelem_x * nelem_y;
    //printf("rank %d, emin %d, emax %d, nelem %d\n",rank,emin,emax,nelem);

    // number of nodes owned by my rank (rank 0 owns 2 boundary nodes, other ranks own right boundary node)
    unsigned int nnode, nnode_y, nnode_x;
    if (rank == 0){
        nnode_y = nelem_y + 1;
    } else {
        nnode_y = nelem_y;
    }
    nnode_x = Nex + 1;
    nnode = nnode_x * nnode_y;
    //printf("rank %d, owned elements= %d, owned nodes= %d\n",rank, nelem,nnode);

    // determine globalMap
    unsigned int* nnode_per_elem = new unsigned int [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        nnode_per_elem[eid] = 4; //linear 4-node element
    }

    unsigned long int** globalMap;
    globalMap = new unsigned long* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        globalMap[eid] = new unsigned long [nnode_per_elem[eid]];
    }
    // todo hard-code 4 nodes per element:
    for (unsigned j = 0; j < nelem_y; j++){
        for (unsigned i = 0; i < nelem_x; i++){
            unsigned int eid = nelem_x * j + i;
            globalMap[eid][0] = (emin*(Nex + 1) + i) + j*(Nex + 1);
            globalMap[eid][1] = globalMap[eid][0] + 1;
            globalMap[eid][3] = globalMap[eid][0] + (Nex + 1);
            globalMap[eid][2] = globalMap[eid][3] + 1;
        }
    }

    /*for (unsigned int eid = 0; eid < nelem; eid++){
        printf("rank %d, eid %d, globalMap = [%d,%d,%d,%d]\n", rank, eid, globalMap[eid][0], globalMap[eid][1],globalMap[eid][2], globalMap[eid][3]);
    }*/

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
    /*for (unsigned int eid = 0; eid < nelem; eid++){
        printf("rank %d, eid %d, localMap= [%d,%d,%d,%d]\n",rank,eid,localMap[eid][0],localMap[eid][1],localMap[eid][2],localMap[eid][3]);
    }*/
    /*for (unsigned int nid = 0; nid < numLocalNodes; nid++){
        printf("rank %d, local node %d --> global node %d\n",rank,nid,local2GlobalMap[nid]);
    }*/

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
            double x = (double)(global_Id % (Nex + 1)) * hx;
            double y = (double)(global_Id / (Nex + 1)) * hy;
            if ((std::fabs(x) < 0.00000001) || (std::fabs(x - Lx) < 0.00000001)){
                // left or right boundary
                bound_nodes[eid][nid] = 1;
                bound_values[eid][nid] = 0.0;
            } else if (std::fabs(y) < 0.00000001){
                // bottom boundary
                bound_nodes[eid][nid] = 1;
                bound_values[eid][nid] = sin(M_PI * x);
            } else if (std::fabs(y - Ly) < 0.00000001){
                // top boundary
                bound_nodes[eid][nid] = 1;
                bound_values[eid][nid] = sin(M_PI * x) * exp(-M_PI);
            } else {
                // iterior
                bound_nodes[eid][nid] = 0;
                bound_values[eid][nid] = -1000000;
            }
        }
    }
    // create lists of constrained dofs (for new interface of set_bdr_map), including not-owned constraints
    std::vector<unsigned long> constrainedDofs;
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            if (bound_nodes[eid][nid] == 1){
                constrainedDofs.push_back(globalMap[eid][nid]);
            }
        }
    }
    std::sort(constrainedDofs.begin(), constrainedDofs.end());
    constrainedDofs.erase(std::unique(constrainedDofs.begin(), constrainedDofs.end()), constrainedDofs.end());

    unsigned long * constrainedDofs_ptr;
    double * prescribedValues_ptr;
    constrainedDofs_ptr = new unsigned long [constrainedDofs.size()];
    prescribedValues_ptr = new double [constrainedDofs.size()];
    for (unsigned int i = 0; i < constrainedDofs.size(); i++){
        unsigned long global_Id = constrainedDofs[i];
        double x = (double)(global_Id % (Nex + 1)) * hx;
        double y = (double)(global_Id / (Nex + 1)) * hy;
        constrainedDofs_ptr[i] = global_Id;
        if ((std::fabs(x) < 0.00000001) || (std::fabs(x - Lx) < 0.00000001)){
            // left or right boundary
            prescribedValues_ptr[i] = 0.0;
        } else if (std::fabs(y) < 0.00000001){
            // bottom boundary
            prescribedValues_ptr[i] = sin(M_PI * x);
        } else if (std::fabs(y - Ly) < 0.00000001){
            // top boundary
            prescribedValues_ptr[i] = sin(M_PI * x) * exp(-M_PI);
        } else {
            // interior
            prescribedValues_ptr[i] = -1000000; //todo could be a "non-sense" value
        }
    }
    /*printf("rank %d, number of constraints %d\n",rank,constrainedDofs.size());
    for (unsigned int i = 0; i < constrainedDofs.size(); i++){
        printf("rank %d constraint %d, global ID %d, prescribed value %f\n",rank,i,constrainedDofs_ptr[i],prescribedValues_ptr[i]);
    }*/

    unsigned long start_global_node, end_global_node;
    start_global_node = nnodeOffset[rank];
    end_global_node = start_global_node + (nnode - 1);


    // declare aMat object
    par::AMAT_TYPE matType;
    if (matFree){
        matType = par::AMAT_TYPE::MAT_FREE;
    } else {
        matType = par::AMAT_TYPE::PETSC_SPARSE;
    }

    par::aMat<double, unsigned long, unsigned int> stMat(matType);
    stMat.set_comm(comm);
    stMat.set_map(nelem, localMap, nnode_per_elem, numLocalNodes, local2GlobalMap, start_global_node, end_global_node, nnode_total);
    stMat.set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, constrainedDofs.size());

    Vec rhs, out;
    stMat.petsc_create_vec(rhs);
    stMat.petsc_create_vec(out);

    // element stiffness matrix, force vector, and assembly...
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            unsigned long global_Id = globalMap[eid][nid];
            xe[nid * 2] = (double)(global_Id % (Nex + 1)) * hx;
            xe[(nid * 2) + 1] = (double)(global_Id / (Nex + 1)) * hy;
        }
        ke_quad4_eig(ke,xe);
        // for this example, no force vector
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            fe(nid) = 0.0;
        }
        // assemble ke
        if (matFree){
            stMat.copy_element_matrix(eid, ke, 0, 0, 1);
        } else {
            stMat.petsc_set_element_matrix(eid, ke, 0, 0, ADD_VALUES);
        }
        // assemble fe
        stMat.petsc_set_element_vec(rhs, eid, fe, 0, ADD_VALUES);
        //printf("rank %d, xe[%d] = [%f,%f,%f,%f,%f,%f,%f,%f]\n",rank,eid,xe[0],xe[1],xe[2],xe[3],xe[4],xe[5],xe[6],xe[7]);
    }
    delete [] xe;

    PetscScalar norm, alpha = -1.0;

    // Pestc begins and completes assembling the global stiffness matrix
    if (!matFree){
        stMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }

    // Pestc begins and completes assembling the global load vector
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);

    /* if (!matFree) {
        stMat.dump_mat("matrix_before_bc.dat");
        stMat.dump_vec(rhs, "rhs_before_bc.dat");
    } */

    // modifying stiffness matrix and load vector to apply dirichlet BCs
    if (!matFree){
        stMat.apply_bc_mat();
        stMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }

    stMat.apply_bc_rhs(rhs);
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);
    //stMat.dump_vec(rhs);

    /*if (!matFree) {
        stMat.dump_mat("matrix_after_bc.dat");
        stMat.dump_vec(rhs, "rhs_after_bc.dat");
    }*/



    // write results to files
    /*if (!matFree){
        stMat.dump_mat("matrix.dat");
        stMat.dump_vec(rhs,"rhs.dat");
    } else {
        stMat.print_mepMat();
    }*/

    // solve
    stMat.petsc_solve((const Vec) rhs, out);

    VecNorm(out, NORM_2, &norm);
    if (!rank){
        printf("L2 norm of computed solution = %f\n",norm);
    }
    //stMat.dump_vec(out);

    // exact solution...
    Vec sol_exact;
    PetscInt rowId;
    stMat.petsc_create_vec(sol_exact);
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            rowId = globalMap[eid][nid];
            double x = (double)(rowId % (Nex + 1)) * hx;
            double y = (double)(rowId / (Nex + 1)) * hy;
            PetscScalar u = sin(M_PI * x) * exp(-M_PI * y);
            VecSetValue(sol_exact, rowId, u, INSERT_VALUES);
        }
    }
    stMat.petsc_init_vec(sol_exact);
    stMat.petsc_finalize_vec(sol_exact);

    // display exact solution on screen
    //stMat.dump_vec(sol_exact);
    VecNorm(sol_exact, NORM_2, &norm);
    if (!rank){
        printf("L2 norm of exact solution = %f\n",norm);
    }

    // compute the norm of error
    VecAXPY(sol_exact, alpha, out);
    VecNorm(sol_exact, NORM_INFINITY, &norm);

    if (!rank){
        printf("Inf norm of error = %20.10f\n", norm);
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


    PetscFinalize();
    return 0;
}