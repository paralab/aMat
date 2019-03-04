// Uses simple partition: arbitrary p will be partitioned in x direction only
// Created by Han Tran on 11/30/18.
//
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

int main(int argc, char *argv[]) {
    // User provides: Nex - number of elements (global) in x direction
    //                Ney - number of elements (global) in y direction
    //                Nez - number of elements (global) in z direction
    int rc;
    double xmin, xmax;
    unsigned  int emin = 0, emax = 0;
    unsigned long nid, eid;
    const unsigned int nDofPerNode = 1;         // number of dofs per node
    const unsigned int nDim = 3;                // number of dimension
    const unsigned int nNodePerElem = 8;        // number of nodes per element

    // number of (global) elements in x, y and z directions
    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]), Nez = atoi(argv[3]);

    // nodal coordinates of an element
    double* xe = new double[nDim * nNodePerElem];

    // element stiffness matrix
    double* ke = new double[(nDofPerNode * nNodePerElem) * (nDofPerNode * nNodePerElem)];

    // element load vector
    double* fe = new double[nDofPerNode * nNodePerElem];

    // domain sizes:
    // Lx - length of the (global) domain in x direction
    // Ly - length of the (global) domain in y direction
    // Lz - length of the (global) domain in z direction
    const double Lx = 0.5, Ly = 0.5, Lz = 0.5;

    // element sizes
    const double hx = Lx/double(Nex);// element size in x direction
    const double hy = Ly/double(Ney);// element size in y direction
    const double hz = Lz/double(Nez);// element size in z direction

    const double tol = 0.001;

    PetscInitialize(&argc,&argv,NULL,NULL);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Status Stat;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("Nex = %d, Ney = %d, Nez = %d\n", Nex, Ney, Nez);
        printf("Lx = %f, Ly = %f, Lz = %f\n", Lx, Ly, Lz);
        if (size > Nex) {
            printf("The number of processes must be less than or equal Nex, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // find min & max element index in x direction =================================
    double d = (Nex)/(double)(size);// number of elements in x direction
    xmin = (rank*d);
    if (rank == 0) xmin = xmin - 0.0001;
    xmax = xmin + d;
    if (rank == size) xmax = xmax + 0.0001;
    for (unsigned int i = 0; i < Nex; i++) {
        if (i >= xmin) {
            emin = i;
            break;
        }
    }
    for (unsigned int i = (Nex-1); i >= 0; i--) {
        if (i < xmax) {
            emax = i;
            break;
        }
    }

    // number of elements (partition of processes is only in x direction)
    unsigned int nelem_x = (emax - emin + 1);
    unsigned int nelem_y = Ney;
    unsigned int nelem_z = Nez;
    unsigned int nelem = (nelem_x) * (nelem_y) * (nelem_z);

    // number of nodes in each direction
    unsigned int nnode_x; // x direction, rank 0 gets nodes on left and right faces, rank 1... gets nodes on left face
    if (rank == 0) {
        nnode_x = nelem_x + 1;
    } else {
        nnode_x = nelem_x;
    }
    unsigned int nnode_y = nelem_y + 1; // y direction
    unsigned int nnode_z = nelem_z + 1; // z direction
    unsigned int nnode = (nnode_x) * (nnode_y) * (nnode_z);

    // map from local nodes to global nodes
    unsigned long int** map = new unsigned long int *[nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        map[e] = new unsigned long int[nNodePerElem];
    }
    for (unsigned k = 0; k < nelem_z; k++){
        for (unsigned j = 0; j < nelem_y; j++){
            for (unsigned i = 0; i < nelem_x; i++){
                eid = nelem_x * nelem_y * k + nelem_x * j + i;
                map[eid][0] = (emin + i) + j*(Nex + 1) + k*(Nex + 1)*(Ney + 1);
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


    /*for (unsigned int e = 0; e < nelem; e++)
    {
        std::cout<<"e: "<<e<<" nodes: ";
        for(unsigned int k=0;k<8;k++)
            std::cout<<", "<<map[e][k];
        std::cout<<std::endl;
    }*/


    // boundary nodes: bound
    unsigned long** bound_nodes = new unsigned long *[nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        bound_nodes[e] = new unsigned long[nNodePerElem];
    }

    // exclusive scan to get the shift for global node/element id =================================
    unsigned int nelem_scan = 0;
    unsigned int nnode_scan = 0;
    MPI_Exscan(&nnode, &nnode_scan, 1, MPI_INT, MPI_SUM, comm);
    MPI_Exscan(&nelem, &nelem_scan, 1, MPI_INT, MPI_SUM, comm);



    // type of elements =================================
    par::ElementType *etype = new par::ElementType[nelem];
    for (unsigned e = 0; e < nelem; e ++){
        etype[e] = par::ElementType::HEX;
    }

    // declare aMat =================================
    par::aMat<double,unsigned long> stiffnessMat(nnode,nDofPerNode,comm);
    // set map
    stiffnessMat.set_map(map);
    // set element types
    stiffnessMat.set_elem_types(nelem, etype);



    // create vec rhs and solution vector
    Vec rhs, out, sol_exact;
    stiffnessMat.create_vec(rhs);
    stiffnessMat.create_vec(out);
    stiffnessMat.create_vec(sol_exact);

    // compute element stiffness matrix and assemble global stiffness matrix and load vector
    for (unsigned int e = 0; e < nelem; e++){
        for (unsigned int n = 0; n < nNodePerElem; n++){
            // global node ID
            nid = map[e][n];
            // nodal coordinates
            xe[n * 3] = (double)(nid % (Nex + 1)) * hx;                             // x
            xe[(n * 3) + 1] = (double)((nid % ((Nex+1)*(Ney+1))) / (Nex + 1)) * hy; // y
            xe[(n * 3) + 2] = (double)(nid / ((Nex+1)*(Ney+1))) * hz;               // z

            // specify boundary nodes
            if ((std::abs(xe[n * 3]) < tol) || (std::abs(xe[n * 3] - Lx) < tol) ||
                    (std::abs(xe[(n * 3) + 1]) < tol) || (std::abs(xe[(n * 3) +1] - Ly) < tol) ||
                    (std::abs(xe[(n * 3) + 2]) < tol) || (std::abs(xe[(n * 3) + 2] - Lz) < tol)) {
                bound_nodes[e][n] = 1; // boundary
            } else {
                bound_nodes[e][n] = 0; // interior
            }
        }

        // element stiffness matrix and force vector
        ke_hex8(ke,xe);
        fe_hex8(fe,xe);

        // assemble to global ones
        stiffnessMat.set_element_matrix(e, ke, false, ADD_VALUES);
        stiffnessMat.set_element_vector(rhs,e, fe, false, ADD_VALUES);

    }

    delete [] ke;
    delete [] fe;
    delete [] xe;

    // let Pestc begin and complete assembling the global stiffness matrix
    stiffnessMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
    stiffnessMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);


    // let Pestc begin and complete assembling the global load vector
    stiffnessMat.petsc_init_vec(rhs);
    stiffnessMat.petsc_finalize_vec(rhs);


    // apply boundary conditions by modifying stiffness matrix and load vector
    for (unsigned e = 0; e < nelem; e++) {
        stiffnessMat.apply_dirichlet(rhs,e,(const unsigned long**)bound_nodes);
    }

    // Pestc begins and completes assembling the global stiffness matrix
    stiffnessMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
    stiffnessMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);

    // Pestc begins and completes assembling the global load vector
    stiffnessMat.petsc_init_vec(rhs);
    stiffnessMat.petsc_finalize_vec(rhs);

    // solve the system
    stiffnessMat.petsc_solve((const Vec) rhs, out);

    // Pestc begins and completes assembling the global load vector
    stiffnessMat.petsc_init_vec(out);
    stiffnessMat.petsc_finalize_vec(out);

    // write results to files
    /*stiffnessMat.dump_mat("stiff_mat.dat");
    stiffnessMat.dump_vec("rhs_vec.dat", rhs);
    stiffnessMat.dump_vec("out_vec.dat", out);*/

    // nodal coordinates of an element
    double* e_exact = new double[nNodePerElem]; // only for 1 DOF/node

    double x, y, z;

    for (unsigned int e = 0; e < nelem; e++) {
        for (unsigned int n = 0; n < nNodePerElem; n++) {
            // global node ID
            nid = map[e][n];
            // nodal coordinates
            x = (double)(nid % (Nex + 1)) * hx;
            y = (double)((nid % ((Nex+1)*(Ney+1))) / (Nex + 1)) * hy;
            z = (double)(nid / ((Nex+1)*(Ney+1))) * hz;

            // exact solution at node n (and apply BCs)
            if ((std::abs(x) < tol) || (std::abs(x - Lx) < tol) ||
                (std::abs(y) < tol) || (std::abs(y - Ly) < tol) ||
                (std::abs(z) < tol) || (std::abs(z - Lz) < tol)) {
                e_exact[n] = 0.0; // boundary
            } else {
                e_exact[n] = (1.0 / (12.0 * M_PI * M_PI)) * sin(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * z);
            }
            //printf("e= %d, nid= %d, {x,y,z}= %f, %f, %f, exact= %f \n", e, nid, x,y,z,e_exact[n]);
        }
        // set exact solution to pestc vector
        stiffnessMat.set_vector(sol_exact, e, e_exact);
    }


    delete [] e_exact;

    // Pestc begins and completes assembling the exact solution
    stiffnessMat.petsc_init_vec(sol_exact);
    stiffnessMat.petsc_finalize_vec(sol_exact);

    //stiffnessMat.dump_vec("exact_vec.dat", sol_exact);


    // subtract out from sol_exact
    PetscScalar norm, alpha = -1.0;
    VecAXPY(sol_exact, alpha, out);

    // compute the norm of sol_exact
    VecNorm(sol_exact, NORM_INFINITY, &norm);

    if (rank == 0){
        printf("L_inf norm= %f\n", norm);
    }

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        delete [] map[eid];
        delete [] bound_nodes[eid];
    }


    delete [] map;
    delete [] bound_nodes;
    delete [] etype;

    PetscFinalize();
    return 0;
}