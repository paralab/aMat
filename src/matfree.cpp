/**
 * @file fem3d.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
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

/*#ifdef BUILD_WITH_PETSC
    #include "petsc.h"
#endif*/

#include "shfunction.hpp"
#include "ke_matrix.hpp"
#include "me_matrix.hpp"
#include "fe_vector.hpp"
#include "aMat.hpp"
#include "Dense"

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;

// global variables
unsigned int Nex, Ney, Nez;
double hx, hy, hz;
unsigned long int** map;

void get_node_coor(double &x, double &y, double &z, const unsigned long nid){
    x = (double)(nid % (::Nex + 1)) * ::hx;
    y = (double)((nid % ((::Nex+1)*(::Ney+1))) / (::Nex + 1)) * ::hy;
    z = (double)(nid / ((::Nex+1)*(::Ney+1))) * ::hz;
}

void get_elem_coor(double* xe, unsigned int eid) {
    unsigned long nid;
    double x, y, z;
    for (unsigned int n = 0; n < 8; n++){
        nid = ::map[eid][n];
        get_node_coor(x, y, z, nid);
        xe[n * 3] = x;
        xe[(n * 3) + 1] = y;
        xe[(n * 3) + 2] = z;
    }
}

void elem_matvec(double* v_e, const double* u_e, unsigned int eid){

    //todo: remove memory allocation outside the function later.
    double* xe = new double[3 * 8];
    double* ke = new double[8 * 8];

    // get node coordinates
    get_elem_coor(xe, eid);

    // compute stiffness matrix
    //ke_hex8(ke, xe);
    ke_hex8_test(ke, xe); // ke = [1]

    // compute v_e = k_e * u_e
    for (unsigned int i = 0; i < 8; i++){
        v_e[i] = 0.0;
        for (unsigned int j = 0; j < 8; j++) {
            v_e[i] += ke[i*8 + j] * u_e[j];
        }
    }

    delete [] xe;
    delete [] ke;
}

int main(int argc, char *argv[]) {
    // User provides: Nex - number of elements (global) in x direction
    //                Ney - number of elements (global) in y direction
    //                Nez - number of elements (global) in z direction
    int rc;
    double zmin, zmax;

    double x, y, z;

    unsigned  int emin = 0, emax = 0;
    unsigned long nid, eid;
    const unsigned int nDofPerNode = 1;         // number of dofs per node
    const unsigned int nDim = 3;                // number of dimension
    const unsigned int nNodePerElem = 8;        // number of nodes per element

    // number of (global) elements in x, y and z directions
    ::Nex = atoi(argv[1]);
    ::Ney = atoi(argv[2]);
    ::Nez = atoi(argv[3]);

    const bool useEigen = atoi(argv[4]);

    Matrix<double,8,8>* kee;
    kee = new Matrix<double,8,8>[AMAT_MAX_EMAT_PER_ELEMENT];// max of twining elements is set to 8
    unsigned int twin_level = 0; // for this example: no element is twinned

    double* xe = new double[nDim * nNodePerElem];
    double* ke = new double[(nDofPerNode * nNodePerElem) * (nDofPerNode * nNodePerElem)];
    double* fe = new double[nDofPerNode * nNodePerElem];

    // domain sizes:
    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

    // element sizes
    ::hx = Lx/double(::Nex);
    ::hy = Ly/double(::Ney);
    ::hz = Lz/double(::Nez);

    const double tol = 0.001;

    PetscInitialize(&argc, &argv, NULL, NULL);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Status Stat;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(!rank)std::cout<<"============ parameters read  ======================="<<std::endl;
    if(!rank)std::cout<<"\t\tNex : "<<::Nex<<" Ney: "<<::Ney<<" Nez: "<<::Nez<<std::endl;
    if(!rank)std::cout<<"\t\tuseEigen: "<<useEigen<<std::endl;
    if(!rank)std::cout<<"====================================================="<<std::endl;

    if (rank == 0) {
        if (size > ::Nez) {
            printf("The number of processes must be less than or equal Nez, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // find min & max element index in z direction =================================
    double d = (::Nez)/(double)(size);// number of elements in z direction
    zmin = (rank*d);
    if (rank == 0) zmin = zmin - 0.0001;
    zmax = zmin + d;
    if (rank == size) zmax = zmax + 0.0001;
    for (unsigned int i = 0; i < ::Nez; i++) {
        if (i >= zmin) {
            emin = i;
            break;
        }
    }
    for (unsigned int i = (::Nez-1); i >= 0; i--) {
        if (i < zmax) {
            emax = i;
            break;
        }
    }

    // number of elements (partition of processes is only in z direction)
    unsigned int nelem_z = (emax - emin + 1);
    unsigned int nelem_y = ::Ney;
    unsigned int nelem_x = ::Nex;
    unsigned int nelem = (nelem_x) * (nelem_y) * (nelem_z);

    // number of nodes in each direction
    unsigned int nnode_z; // z direction, rank 1 gets nodes on left and right faces, rank 1... gets nodes on left face
    if (rank == 1) {
        nnode_z = nelem_z + 1;
    } else {
        nnode_z = nelem_z;
    }
    unsigned int nnode_y = nelem_y + 1; // y direction
    unsigned int nnode_x = nelem_x + 1; // x direction
    unsigned int nnode = (nnode_x) * (nnode_y) * (nnode_z);

    // map from local nodes to global nodes
    ::map = new unsigned long int *[nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        ::map[e] = new unsigned long int[AMAT_MAX_EMAT_PER_ELEMENT*nNodePerElem];
    }
    for (unsigned k = 0; k < nelem_z; k++){
        for (unsigned j = 0; j < nelem_y; j++){
            for (unsigned i = 0; i < nelem_x; i++){
                eid = nelem_x * nelem_y * k + nelem_x * j + i;
                ::map[eid][0] = (emin*(::Nex + 1)*(::Ney + 1) + i) + j*(::Nex + 1) + k*(::Nex + 1)*(::Ney + 1);
                ::map[eid][1] = ::map[eid][0] + 1;
                ::map[eid][3] = ::map[eid][0] + (::Nex + 1);
                ::map[eid][2] = ::map[eid][3] + 1;
                ::map[eid][4] = ::map[eid][0] + (::Nex + 1)*(::Ney + 1);
                ::map[eid][5] = ::map[eid][4] + 1;
                ::map[eid][7] = ::map[eid][4] + (::Nex + 1);
                ::map[eid][6] = ::map[eid][7] + 1;
            }
        }
    }
    
    for (unsigned int e = 0; e < nelem; e++)
    {
        printf("rank = %d, element = %d, nodes = %d,%d,%d,%d,%d,%d,%d,%d,\n",rank,e,map[e][0],map[e][1],map[e][2],map[e][3],map[e][4],map[e][5],map[e][6],map[e][7]);
    }
    printf("rank = %d, nelem = %d, nnode = %d\n",rank,nelem,nnode);

    // boundary nodes: bound
    unsigned long** bound_nodes = new unsigned long *[nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        bound_nodes[e] = new unsigned long[nNodePerElem];
    }

    // exclusive scan to get the shift for global node/element id =================================
    //unsigned int nelem_scan = 0;
    //unsigned int nnode_scan = 0;
    //MPI_Exscan(&nnode, &nnode_scan, 1, MPI_INT, MPI_SUM, comm);
    //MPI_Exscan(&nelem, &nelem_scan, 1, MPI_INT, MPI_SUM, comm);

    // type of elements =================================
    par::ElementType *etype = new par::ElementType[nelem];
    for (unsigned e = 0; e < nelem; e ++){
        etype[e] = par::ElementType::HEX;
    }

    // declare aMat =================================
    par::aMat<double,unsigned long> stiffnessMat(nelem, etype, nnode, comm);

    // set map
    stiffnessMat.set_map(map);

    // build scatter map
    stiffnessMat.buildScatterMap();

    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact;
    stiffnessMat.petsc_create_vec(rhs);
    stiffnessMat.petsc_create_vec(out);
    stiffnessMat.petsc_create_vec(sol_exact);

    // compute element stiffness matrix and assemble global stiffness matrix and load vector
    for (unsigned int eid = 0; eid < nelem; eid++){
        // get node coordinates
        get_elem_coor(xe, eid);
        // specify boundary nodes
        for (unsigned int n = 0; n < nNodePerElem; n++) {
            // global node ID
            nid = ::map[eid][n];
            // nodal coordinates
            get_node_coor(x, y, z, nid);
            if ((std::abs(x) < tol) || (std::abs(x - Lx) < tol) ||
                    (std::abs(y) < tol) || (std::abs(y - Ly) < tol) ||
                    (std::abs(z) < tol) || (std::abs(z - Lz) < tol)) {
                bound_nodes[eid][n] = 1; // boundary
            } else {
                bound_nodes[eid][n] = 0; // interior
            }
        }
        // compute element stiffness matrix
        if (useEigen) {
            ke_hex8_eig(kee[0], xe);
        } else {
            //ke_hex8(ke, xe);
            ke_hex8_test(ke, xe);
        }
        // compute element force vector
        fe_hex8(fe, xe);
        // assemble element stiffness matrix to global K
        if (useEigen){
            stiffnessMat.set_element_matrices(eid, (MatrixXd*)kee, twin_level, ADD_VALUES);
        } else {
            stiffnessMat.set_element_matrix(eid, ke, ADD_VALUES); //fixme: need to implement twinning
        }
        // assemble element load vector to global F
        stiffnessMat.petsc_set_element_vector(rhs, eid, fe, ADD_VALUES);
    }

    delete [] ke;
    delete [] fe;
    delete [] xe;

    // Pestc begins and completes assembling the global stiffness matrix
    stiffnessMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
    stiffnessMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);

    // Pestc begins and completes assembling the global load vector
    stiffnessMat.petsc_init_vec(rhs);
    stiffnessMat.petsc_finalize_vec(rhs);

    // apply boundary conditions by modifying stiffness matrix and load vector
    /*for (unsigned e = 0; e < nelem; e++) {
        stiffnessMat.apply_dirichlet(rhs,e,(const unsigned long**)bound_nodes);
    }*/

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

    // create vectors used to test matrix-free
    Vec petsc_dv, petsc_du, petsc_compare;
    stiffnessMat.petsc_create_vec(petsc_dv, 0.0);
    stiffnessMat.petsc_create_vec(petsc_du, 1.0);
    stiffnessMat.petsc_create_vec(petsc_compare, 0.0);
    //VecZeroEntries(petsc_dv);
    stiffnessMat.petsc_matmult(petsc_du, petsc_dv);
    stiffnessMat.dump_vec("petsc_du.dat", petsc_du);
    stiffnessMat.dump_vec("petsc_dv.dat", petsc_dv);
    stiffnessMat.dump_mat("petsc_mat.dat");


    // my own vectors
    double* dv;
    double* du;
    stiffnessMat.buildScatterMap();
    stiffnessMat.create_vec(dv, true);
    stiffnessMat.create_vec(du, true, 1.0);
    // compute dv = K * du
    //stiffnessMat.matvec(dv, du, elem_matvec);
    stiffnessMat.matvec(dv, du);

    // apply Diriclet bc on dv by setting zero to all boundary nodes
    /*for (unsigned int e = 0; e < nelem; e++){
        stiffnessMat.set_vector_bc(dv, e, (const unsigned long**)bound_nodes);
    }*/

    // display dv
    //stiffnessMat.print_vector(dv);

    // transform dv to petsc vector
    stiffnessMat.transform_to_petsc_vector(dv, petsc_compare);

    // subtract out from petsc_dv
    PetscScalar norm_compare, alpha_compare = -1.0;

    // compute petsc_compare = petsc_compare + alpha * pestc_dv
    VecAXPY(petsc_compare, alpha_compare, petsc_dv);

    // compute the norm of petsc_compare
    VecNorm(petsc_compare, NORM_INFINITY, &norm_compare);
    if (rank == 0){
        printf("norm_compare= %f\n", norm_compare);
    }
    //stiffnessMat.dump_vec("petsc_compare", petsc_compare);

    // nodal coordinates of an element
    double* e_exact = new double[nNodePerElem]; // only for 1 DOF/node
    for (unsigned int e = 0; e < nelem; e++) {
        for (unsigned int n = 0; n < nNodePerElem; n++) {
            // global node ID
            nid = map[e][n];
            // nodal coordinates
            x = (double)(nid % (::Nex + 1)) * ::hx;
            y = (double)((nid % ((::Nex+1)*(::Ney + 1))) / (::Nex + 1)) * ::hy;
            z = (double)(nid / ((::Nex + 1)*(::Ney + 1))) * ::hz;

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
        stiffnessMat.petsc_set_vector(sol_exact, e, e_exact);
    }

    delete [] e_exact;

    // Pestc begins and completes assembling the exact solution
    stiffnessMat.petsc_init_vec(sol_exact);
    stiffnessMat.petsc_finalize_vec(sol_exact);

    //stiffnessMat.dump_vec("exact_vec.dat", sol_exact);

    // subtract out from sol_exact
    PetscScalar norm, alpha = -1.0;

    // compute sol_exact = sol_exact + alpha * out
    VecAXPY(sol_exact, alpha, out);

    // compute the norm of sol_exact
    VecNorm(sol_exact, NORM_INFINITY, &norm);

    /*if (rank == 0){
        printf("L_inf norm= %f\n", norm);
    }*/

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        delete [] map[eid];
        delete [] bound_nodes[eid];
    }

    delete [] map;
    delete [] bound_nodes;
    delete [] etype;
    delete [] kee;

    PetscFinalize();
    return 0;
}