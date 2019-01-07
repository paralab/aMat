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

#include "../include/shfunction.hpp"
#include "../include/ke_matrix.hpp"
#include "../include/aMat.hpp"


int main(int argc, char *argv[]) {
// User provides: Nex - number of elements (global) in x direction
//                Ney - number of elements (global) in y direction
//                Nez - number of elements (global) in z direction
//                Lx - length of the (global) domain in x direction
//                Ly - length of the (global) domain in y direction
//                Lz - length of the (global) domain in z direction
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Status Stat;
    // proc ID on the left and right (-1; no neighbor)
    int left = -1, right = -1;
/*#ifdef BUILD_WITH_PETSC
    Vec vec;
    Mat mat;
#endif*/
    int rc;
    double xmin, xmax;
    int emin = 0, emax = 0;
    int nid, eid;
    const unsigned int nDofPerNode = 1;         // number of dofs per node
    const unsigned int nDim = 3;                // space dimension
    const unsigned int nNodePerElem = 8;        // number of nodes per element

    // number of (global) elements in x, y and z directions
    const int Nex = atoi(argv[1]);
    const int Ney = 2, Nez = 2;

    // nodal coordinates of an element
    double *xe = new double[nDim * nNodePerElem];
    // element stiffness matrix
    double *ke = new double[(nDofPerNode * nNodePerElem) * (nDofPerNode * nNodePerElem)];


    // (global) length in x, y and z directions
    const int Lx = 4, Ly = 2, Lz = 2;
    const double hx = double(Lx)/double(Nex);// element size in x direction
    const double hy = double(Ly)/double(Ney);// element size in y direction
    const double hz = double(Lz)/double(Nez);// element size in z direction

    PetscInitialize(&argc,&argv,NULL,NULL);

    //MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("Nex = %d, Ney = %d, Nez = %d\n", Nex, Ney, Nez);
        if (size > Nex) {
            printf("The number of processes must be less than or equal Nex, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }
    // find min & max element index in x direction =================================
    double d = (Nex)/(double)(size);// number of elements in x direction
    xmin = (double)(rank*d);
    if (rank == 0) xmin = xmin - 0.0001;
    xmax = xmin + d;
    if (rank == size) xmax = xmax + 0.0001;
    //printf("rank %d, xmin= %f, xmax= %f\n", rank, xmin, xmax);
    for (int i = 0; i < Nex; i++) {
        if (i >= xmin) {
            emin = i;
            break;
        }
    }
    for (int i = (Nex-1); i >= 0; i--) {
        if (i < xmax) {
            emax = i;
            break;
        }
    }
    //printf("rank= %d, emin= %d, emax= %d\n", rank, emin, emax);

    // number of elements in each direction
    unsigned int nelem_x = (emax - emin + 1); // x direction
    unsigned int nelem_y = Ney; // y direction
    unsigned int nelem_z = Nez; // z direction
    // total number of (local) elements
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
    // total number of (local) nodes
    unsigned int nnode = (nnode_x) * (nnode_y) * (nnode_z);

    //printf("rank= %d, nnode_x= %d, nnode_y= %d, nnode_z= %d\n", rank, nnode_x,nnode_y,nnode_z);

    // map from local nodes to global nodes
    unsigned long int ** map = new unsigned long int *[nelem];
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
    /*
    for (unsigned e = 0; e < nelem; e++){
        printf("rank= %d, element= %d, nodes= %d %d %d %d %d %d %d %d\n", rank, e, map[e][0],map[e][1],map[e][2],map[e][3],
               map[e][4],map[e][5],map[e][6],map[e][7]);
    }
    */


    // exclusive scan to get the shift for global node/element id
    unsigned int nelem_scan = 0;
    unsigned int nnode_scan = 0;
    MPI_Exscan(&nnode, &nnode_scan, 1, MPI_INT, MPI_SUM, comm);
    MPI_Exscan(&nelem, &nelem_scan, 1, MPI_INT, MPI_SUM, comm);
    /*printf("rank= %d, nnode= %d, nnode_scan= %d\n", rank, nnode, nnode_scan);
    printf("rank= %d, nelem= %d, nelem_scan= %d\n", rank, nelem, nelem_scan);*/

    // process ID on the left and right
    if (rank > 0 && rank < (size-1)) {
        right = rank +1;
        left = rank - 1;
    }
    if (rank == 0) right = rank +1;
    if (rank == (size-1)) left = rank -1;

    // type of elements
    par::ElementType *etype = new par::ElementType[nelem];
    for (unsigned e = 0; e < nelem; e ++){
        etype[e] = par::ElementType::HEX;
    }
    // declare aMat
    par::aMat<double,unsigned long> stiffnessMat(nnode,nDofPerNode,comm);
    // set map
    stiffnessMat.set_map(map);
    // set element types
    stiffnessMat.set_elem_types(nelem, etype);

    // compute element stiffness matrix and assemble global stiffness matrix
    for (unsigned int e = 0; e < nelem; e++){
        for (unsigned int n = 0; n < nNodePerElem; n++){
            // global node ID
            nid = map[e][n];
            // coordinates
            xe[n * 3] = (double)(nid % (Nex + 1)) * hx;                             // x
            xe[(n * 3) + 1] = (double)((nid % ((Nex+1)*(Ney+1))) / (Nex + 1)) * hy; // y
            xe[(n * 3) + 2] = (double)(nid / ((Nex+1)*(Ney+1))) * hz;               // z
            //printf("rank= %d, elem= %d, node= %d, nid= %d, {x,y,z}= %f, %f, %f\n", rank, e, n, nid, x, y, z);
        }
        ke = ke_hex8(xe);
        /*
        for (int j = 0; j < nNodePerElem; j++){
            printf("rank= %d, elem= %d, row= %d: , (0,%f), (1,%f), (2,%f), (3,%f), (4,%f), (5,%f), (6,%f), (7,%f)\n",
                    rank, e, j, ke[j*8], ke[j*8+1], ke[j*8+2], ke[j*8+3], ke[j*8+4], ke[j*8+5], ke[j*8+6], ke[j*8+7]);
        }
        */
        // assemble to global stiffness matrix
        stiffnessMat.set_element_matrix(e, ke, false, ADD_VALUES);
    }

    std::vector<double> u, w;
    u.resize(nnode*nDofPerNode);
    u.resize(nnode*nDofPerNode);
    for (unsigned i = 0; i < nnode*nDofPerNode; i++) u[i] = 1;
    //stiffnessMat.matvec(u,u,w,w);

    stiffnessMat.print_data();

    for (unsigned int eid = 0; eid < nelem; eid++)
        delete [] map[eid];

    delete [] map;

    PetscFinalize();
    return 0;
}