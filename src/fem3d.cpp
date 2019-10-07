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
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifdef BUILD_WITH_PETSC
    #include "petsc.h"
#endif

#include "shfunction.hpp"
#include "ke_matrix.hpp"
#include "me_matrix.hpp"
#include "fe_vector.hpp"
#include "aMat.hpp"
#include "Dense"

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;

int main(int argc, char *argv[]) {
    // User provides: Nex - number of elements (global) in x direction
    //                Ney - number of elements (global) in y direction
    //                Nez - number of elements (global) in z direction
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

    Matrix<double,8,8>* kee;
    kee = new Matrix<double,8,8>[AMAT_MAX_EMAT_PER_ELEMENT];// max of twining elements is set to 8
    unsigned int twin_level = 0; // for this example: no element is twinned

    double* xe = new double[nDim * nNodePerElem];
    double* ke = new double[(nDofPerNode * nNodePerElem) * (nDofPerNode * nNodePerElem)];
    double* fe = new double[nDofPerNode * nNodePerElem];

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

    if(!rank)std::cout<<"============ parameters read  ======================="<<std::endl;
    if(!rank)std::cout<<"\t\tNex : "<<Nex<<" Ney: "<<Ney<<" Nez: "<<Nez<<std::endl;
    if(!rank && useEigen)std::cout<<"\t\tuseEigen: "<<useEigen<<std::endl;
    if(!rank && matFree)std::cout<<"\t\tmatrix free: "<<matFree<<std::endl;
    if(!rank)std::cout<<"====================================================="<<std::endl;

    if (rank == 0) {
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
    unsigned long int** map;
    map = new unsigned long int *[nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        map[e] = new unsigned long int[AMAT_MAX_EMAT_PER_ELEMENT*nNodePerElem];
    }
    for (unsigned k = 0; k < nelem_z; k++){
        for (unsigned j = 0; j < nelem_y; j++){
            for (unsigned i = 0; i < nelem_x; i++){
                eid = nelem_x * nelem_y * k + nelem_x * j + i;
                map[eid][0] = (emin*(Nex + 1)*(Ney + 1) + i) + j*(Nex + 1) + k*(Nex + 1)*(Ney + 1);
                map[eid][1] = map[eid][0] + 1;
                map[eid][3] = map[eid][0] + (Nex + 1);
                map[eid][2] = map[eid][3] + 1;
                map[eid][4] = map[eid][0] + (Nex + 1)*(Ney + 1);
                map[eid][5] = map[eid][4] + 1;
                map[eid][7] = map[eid][4] + (Nex + 1);
                map[eid][6] = map[eid][7] + 1;
            }
        }
    }

    // local_to_global map
    unsigned long * local_to_global = new unsigned long[nnode];
    unsigned long start_global_node;

    printf("rank= %d, {emin, emax}= %d, %d\n", rank, emin, emax);
    if (rank != 0){
        start_global_node = nnode_x * nnode_y * (emin+1);
    } else {
        start_global_node = nnode_x * nnode_y * emin;
    }
    printf("rank= %d, start_global_node= %d\n", rank, start_global_node);

    for (unsigned int n = 0; n < nnode; n++){
        local_to_global[n] = start_global_node + n;
    }


    // boundary nodes: bound
    unsigned int** bound_nodes = new unsigned int *[nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        bound_nodes[e] = new unsigned int[nNodePerElem];
    }

    // exclusive scan to get the shift for global node/element id =================================
    unsigned int nelem_scan = 0;
    unsigned int nnode_scan = 0;
    unsigned int nnode_total;
    MPI_Exscan(&nnode, &nnode_scan, 1, MPI_INT, MPI_SUM, comm);
    MPI_Exscan(&nelem, &nelem_scan, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&nnode, &nnode_total, 1, MPI_INT, MPI_SUM, comm);
    printf("rank= %d, nnode= %d, nnode_scan= %d, nnode_total= %d\n", rank, nnode, nnode_scan, nnode_total);

    // type of elements =================================
    //par::ElementType *etype = new par::ElementType[nelem];
    /*for (unsigned e = 0; e < nelem; e ++){
        etype[e] = par::ElementType::HEX;
    }*/
    unsigned int* nodes_per_element = new unsigned int[nelem];
    for (unsigned e = 0; e < nelem; e ++){
        nodes_per_element[e] = 8;
    }

    // declare aMat =================================
    //par::aMat<double,unsigned long> stMat(matFree, nelem, etype, nnode, comm);
    par::AMAT_TYPE matType;
    if (matFree){
        matType = par::AMAT_TYPE::MAT_FREE;
    } else {
        matType = par::AMAT_TYPE::PETSC_SPARSE;
    }
    par::aMat<double, unsigned long, unsigned int> stMat(matType);

    // set communicator
    stMat.set_comm(comm);

    // set map
    stMat.set_map(map, nodes_per_element, nelem, nnode, nnode_total);


    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact;
    stMat.petsc_create_vec(rhs);
    stMat.petsc_create_vec(out);
    stMat.petsc_create_vec(sol_exact);


    // compute element stiffness matrix and assemble global stiffness matrix and load vector
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int n = 0; n < nNodePerElem; n++){
            nid = map[eid][n];

            // get node coordinates
            x = (double)(nid % (Nex + 1)) * hx;
            y = (double)((nid % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(nid / ((Nex + 1)*(Ney + 1))) * hz;
            xe[n * 3] = x;
            xe[(n * 3) + 1] = y;
            xe[(n * 3) + 2] = z;

            // specify boundary nodes
            if ((std::fabs(x) < tol) || (std::fabs(x - Lx) < tol) ||
                (std::fabs(y) < tol) || (std::fabs(y - Ly) < tol) ||
                (std::fabs(z) < tol) || (std::fabs(z - Lz) < tol)) {
                bound_nodes[eid][n] = 1; // boundary
            } else {
                bound_nodes[eid][n] = 0; // interior
            }
        }

        // compute element stiffness matrix
        if (useEigen) {
            ke_hex8_eig(kee[0], xe);
            //ke_hex8_eig_test(kee[0], xe);
        } else {
            ke_hex8(ke, xe);
        }

        // compute element force vector
        fe_hex8(fe, xe);

        // assemble element stiffness matrix to global K
        if (useEigen){
            if (matFree) {
                // copy element matrix to store in m_epMat[eid]
                stMat.copy_element_matrix(eid, kee[0]);
            } else {
                stMat.petsc_set_element_matrix(eid, kee[0], ADD_VALUES);
            }
        } else {
            if (matFree){
                std::cout<<"Error: matrix free only works for Eigen matrix"<<std::endl;
            } else {
                stMat.petsc_set_element_matrix(eid, ke, ADD_VALUES);
            }
        }
        // assemble element load vector to global F
        stMat.petsc_set_element_vec(rhs, eid, fe, ADD_VALUES);
    }


    // set boundary map
    stMat.set_bdr_map(bound_nodes);

    //stMat.print_matrix();

    delete [] ke;
    delete [] fe;
    delete [] xe;


    // Pestc begins and completes assembling the global stiffness matrix
    if (!matFree){
        stMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }



    // Pestc begins and completes assembling the global load vector
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);

    // write results to files

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

        // build scatter map
        stMat.buildScatterMap();

        // set local to global map
        stMat.set_Local2Global(local_to_global);

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
//        Vec petsc_dv, petsc_du, petsc_compare;
//        stMat.petsc_create_vec(petsc_dv, 0.0);
//        stMat.petsc_create_vec(petsc_du, 1.0);
//        stMat.petsc_matmult(petsc_du, petsc_dv); //petsc_du = matrix*pets_dv
//        //stMat.dump_vec("petsc_dv.dat", petsc_dv);//print to file of petsc_dv
//
//        // transform dv to pestc vector for easy to compare
//        stMat.petsc_create_vec(petsc_compare, 0.0);
//        stMat.transform_to_petsc_vector(dv, petsc_compare, ghosted); // transform dv to pestc-type vector petsc_compare
//        //stMat.dump_vec("petsc_compare.dat", petsc_compare); // print to file petsc_compare
//
//        // subtract two vectors: petsc_compare = petsc_compare - petsc_dv
//        PetscScalar norm1, alpha1 = -1.0;
//        VecAXPY(petsc_compare, alpha1, petsc_dv);
//
//        // compute the norm of petsc_compare
//        VecNorm(petsc_compare, NORM_INFINITY, &norm1);
//
//        if (rank == 0){
//            printf("petsc_compare norm= %20.10f\n", norm1);
//        }

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
    stMat.apply_bc_rhs(rhs);
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);


    // solve
    stMat.petsc_solve((const Vec) rhs, out);


    // Pestc begins and completes assembling the global load vector
    stMat.petsc_init_vec(out);
    stMat.petsc_finalize_vec(out);


    // compute exact solution for comparison
    double* e_exact = new double[nNodePerElem]; // only for 1 DOF/node

    for (unsigned int e = 0; e < nelem; e++) {
        for (unsigned int n = 0; n < nNodePerElem; n++) {
            // global node ID
            nid = map[e][n];
            // nodal coordinates
            x = (double)(nid % (Nex + 1)) * hx;
            y = (double)((nid % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(nid / ((Nex + 1)*(Ney + 1))) * hz;

            // exact solution at node n (and apply BCs)
            if ((std::abs(x) < tol) || (std::abs(x - Lx) < tol) ||
                (std::abs(y) < tol) || (std::abs(y - Ly) < tol) ||
                (std::abs(z) < tol) || (std::abs(z - Lz) < tol)) {
                e_exact[n] = 0.0; // boundary
            } else {
                e_exact[n] = (1.0 / (12.0 * M_PI * M_PI)) * sin(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * z);
            }
        }
        // set exact solution to Pestc vector
        stMat.petsc_set_element_vec(sol_exact, e, e_exact, INSERT_VALUES);
    }

    delete [] e_exact;

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
        delete [] map[eid];
        delete [] bound_nodes[eid];
    }

    delete [] map;
    delete [] bound_nodes;
    //delete [] etype;
    delete [] kee;
    delete [] local_to_global;

    // clean up Pestc vectors
    VecDestroy(&out);
    VecDestroy(&sol_exact);
    VecDestroy(&rhs);
    PetscFinalize();

    return 0;
}