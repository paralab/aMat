/**
 * @file ex3.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Example: Stretching of a prismatic bar by its own weight (Timoshenko page 246)
 * @brief Exact solution (origin at centroid of bottom face)
 * @brief    uniform stress s_zz = rho * g * z
 * @brief    displacement u = -(nu * rho * g/E) * x * z
 * @brief    displacement v = -(nu * rho * g/E) * y * z
 * @brief    displacement w = (rho * g/2/E)(z^2 - Lz^2) + (nu * rho * g)/2/E(x^2 + y^2)
 * @brief Boundary condition: traction tz = rho * g * Lz applied on top surface + blocking rigid motions
 * @brief Partition of elements in z direction: owned elements in z direction ~ Nez/(number of ranks)
 * @brief Size of the domain: Lx = Ly = 1; Lz = 4.0
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

    const unsigned int NDOF_PER_NODE = 3;       // number of dofs per node
    const unsigned int NDIM = 3;                // number of dimension
    const unsigned int NNODE_PER_ELEM = 8;      // number of nodes per element

    // material properties of alumina
    //const double E = 300.0; // GPa
    const double E = 1.0;
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

    const bool useEigen = atoi(argv[4]); // use Eigen matrix
    const bool matFree = atoi(argv[5]); // use matrix-free method

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

    // element matrix (contains multiple matrix blocks)
    std::vector< Matrix< double, NDOF_PER_NODE * NNODE_PER_ELEM, NDOF_PER_NODE * NNODE_PER_ELEM > > kee;
    kee.resize(AMAT_MAX_BLOCKSDIM_PER_ELEMENT * AMAT_MAX_BLOCKSDIM_PER_ELEMENT);

    // element force vector
    std::vector<Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1>> fee;
    fee.resize(AMAT_MAX_BLOCKSDIM_PER_ELEMENT);

    // nodal coordinates of element
    double* xe = new double[NDIM * NNODE_PER_ELEM];


    if(!rank) {
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNex : "<< Nex << " Ney: " << Ney << " Nez: " << Nez << "\n";
    }
    if(!rank && useEigen) { std::cout << "\t\tuseEigen: " << useEigen << "\n"; }
    if(!rank && matFree)  { std::cout << "\t\tmatrix free: " << matFree << "\n"; }
    if(!rank) { std::cout<<"=====================================================\n"; }

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
    zmin = (rank * d);
    if (rank == 0) zmin = zmin - 0.01*(hz);
    zmax = zmin + d;
    if (rank == size) zmax = zmax + 0.01*(hz);
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

    // number of owned nodes
    unsigned int nnode_z;
    if (rank == 0) {
        nnode_z = nelem_z + 1;
    } else {
        nnode_z = nelem_z;
    }

    unsigned int nnode_y = nelem_y + 1;
    unsigned int nnode_x = nelem_x + 1;
    unsigned int nnode = (nnode_x) * (nnode_y) * (nnode_z);

    // map from elemental node to global nodes
    unsigned long gNodeId;
    unsigned long int* * globalMap;
    globalMap = new unsigned long int *[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalMap[eid] = new unsigned long int[AMAT_MAX_BLOCKSDIM_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned k = 0; k < nelem_z; k++) {
        for (unsigned j = 0; j < nelem_y; j++) {
            for (unsigned i = 0; i < nelem_x; i++) {
                unsigned int elemID = nelem_x * nelem_y * k + nelem_x * j + i;
                globalMap[elemID][0] = (emin*(Nex + 1)*(Ney + 1) + i) + j*(Nex + 1) + k*(Nex + 1)*(Ney + 1);
                globalMap[elemID][1] = globalMap[elemID][0] + 1;
                globalMap[elemID][3] = globalMap[elemID][0] + (Nex + 1);
                globalMap[elemID][2] = globalMap[elemID][3] + 1;
                globalMap[elemID][4] = globalMap[elemID][0] + (Nex + 1)*(Ney + 1);
                globalMap[elemID][5] = globalMap[elemID][4] + 1;
                globalMap[elemID][7] = globalMap[elemID][4] + (Nex + 1);
                globalMap[elemID][6] = globalMap[elemID][7] + 1;
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

    // build localMap from globalMap (this is just to conform with aMat interface used for bsamxx)
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalNodes;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;

    // counts of owned nodes: nnodeCount[0] = nnode0, nnodeCount[1] = nnode1, ...
    unsigned int* nnodeCount = new unsigned int [size];
    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

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

    // determine ghost nodes based on:
    // rank 0 owns [0,...,nnode0-1], rank 1 owns [nnode0,..., nnode0 + nnode1 - 1]...
    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            gNodeId = globalMap[eid][nid];
            if (gNodeId < nnodeOffset[rank]) {
                preGhostGIds.push_back(gNodeId);
            } else if (gNodeId >= nnodeOffset[rank] + nnode) {
                postGhostGIds.push_back(gNodeId);
            }
        }
    }
    // sort (in ascending order) to prepare for deleting repeated nodes in preGhostGIds and postGhostGIds
    std::sort(preGhostGIds.begin(), preGhostGIds.end());
    std::sort(postGhostGIds.begin(), postGhostGIds.end());

    // remove consecutive duplicates and erase all after .end()
    preGhostGIds.erase(std::unique(preGhostGIds.begin(), preGhostGIds.end()), preGhostGIds.end());
    postGhostGIds.erase(std::unique(postGhostGIds.begin(), postGhostGIds.end()), postGhostGIds.end());

    // number of pre and post ghost nodes of my rank
    numPreGhostNodes = preGhostGIds.size();
    numPostGhostNodes = postGhostGIds.size();

    // number of local nodes
    numLocalNodes = numPreGhostNodes + nnode + numPostGhostNodes;

    // number of local dofs
    unsigned int numLocalDofs = numLocalNodes * NDOF_PER_NODE;

    // map from elemental nodes to local nodes
    unsigned int* * localMap;
    localMap = new unsigned int* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localMap[eid] = new unsigned int[AMAT_MAX_BLOCKSDIM_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int i = 0; i < NNODE_PER_ELEM; i++) {
            gNodeId = globalMap[eid][i];
            if ((gNodeId >= nnodeOffset[rank]) && (gNodeId < (nnodeOffset[rank] + nnode))) {
                // nid is owned by me
                localMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
            } else if (gNodeId < nnodeOffset[rank]) {
                // nid is owned by someone before me
                const unsigned int lookUp = std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) - preGhostGIds.begin();
                localMap[eid][i] = lookUp;
            } else if (gNodeId >= (nnodeOffset[rank] + nnode)) {
                // nid is owned by someone after me
                const unsigned int lookUp = std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), gNodeId) - postGhostGIds.begin();
                localMap[eid][i] =  numPreGhostNodes + nnode + lookUp;
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

    // local node to global node map
    unsigned long * local2GlobalMap = new unsigned long[numLocalNodes];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            gNodeId = globalMap[eid][nid];
            local2GlobalMap[localMap[eid][nid]] = gNodeId;
        }
    }

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
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {

            // global node id of elemental node n
            gNodeId = globalMap[eid][nid];

            // compute nodal coordinate
            x = (double)(gNodeId % (Nex + 1)) * hx;
            y = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            z = (double) (gNodeId / ((Nex + 1) * (Ney + 1))) * hz;

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

    // elemental traction vector
    Matrix<double, Eigen::Dynamic, 1> * elem_trac;
    elem_trac = new Matrix<double, Eigen::Dynamic, 1> [nelem];

    // nodal traction of tractioned face
    double nodalTraction [12] = {0.0};
    // nodal coordinates of tractioned face
    double xeSt [12];
    // force vector due to traction
    Matrix<double, 12, 1> feT;

    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);

    for (unsigned int eid = 0; eid < nelem; eid++){
        bool traction = false;
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            gNodeId = globalMap[eid][nid];
            z = (double) (gNodeId / ((Nex + 1) * (Ney + 1))) * hz;
            if (fabs(z - Lz) < zero_number){
                // element eid has one face is the top surface with applied traction
                traction = true;
                break;
            }
        }
        if (traction) {
            // get coordinates of all nodes
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
                gNodeId = globalMap[eid][nid];
                // get node coordinates
                x = (double)(gNodeId % (Nex + 1)) * hx;
                y = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
                z = (double)(gNodeId / ((Nex + 1)*(Ney + 1))) * hz;

                x = x - Lx/2;
                y = y - Ly/2;

                xe[nid * NDOF_PER_NODE] = x;
                xe[(nid * NDOF_PER_NODE) + 1] = y;
                xe[(nid * NDOF_PER_NODE) + 2] = z;
            }

            // get coordinates of nodes belonging to the face where traction is applied
            // traction applied on face 4-5-6-7 ==> nodes [4,5,6,7] corresponds to nodes [0,1,2,3] of 2D element
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                xeSt[did] = xe[4*NDOF_PER_NODE + did];
                xeSt[NDOF_PER_NODE + did] = xe[5*NDOF_PER_NODE + did];
                xeSt[2*NDOF_PER_NODE + did] = xe[6*NDOF_PER_NODE + did];
                xeSt[3*NDOF_PER_NODE + did] = xe[7*NDOF_PER_NODE + did];
            }

            // get nodal traction of face where traction is applied (uniform traction t3 = rho*g*Lz applied on top surface)
            for (unsigned int nid = 0; nid < 4; nid++){
                nodalTraction[nid*NDOF_PER_NODE + 2] = rho * g * Lz;
            }

            // compute force vector due traction applied on one face of element
            feT_hex8_iso(feT, xeSt, nodalTraction, intData.Pts_n_Wts, NGT);

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

    // declare aMat object
    par::AMAT_TYPE matType;
    if (matFree) {
        matType = par::AMAT_TYPE::MAT_FREE;
    } else {
        matType = par::AMAT_TYPE::PETSC_SPARSE;
    }
    par::aMat<double, unsigned long int, unsigned int> stMat(matType);

    // set communicator
    stMat.set_comm(comm);

    // set global dof map
    stMat.set_map(nelem, localDofMap, ndofs_per_element, numLocalDofs, local2GlobalDofMap, start_global_dof,
                  end_global_dof, ndofs_total);

    // set boundary map
    stMat.set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, list_of_constraints.size());

    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact, error;
    stMat.petsc_create_vec(rhs);
    stMat.petsc_create_vec(out);
    stMat.petsc_create_vec(sol_exact);
    stMat.petsc_create_vec(error);

    // compute element stiffness matrix and force vector, then assemble
    // nodal value of body force
    double beN [24] = {0.0};

    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            gNodeId = globalMap[eid][nid];
            // get node coordinates
            x = (double)(gNodeId % (Nex + 1)) * hx;
            y = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(gNodeId / ((Nex + 1)*(Ney + 1))) * hz;

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
            ke_hex8_iso(kee[0], xe, E, nu, intData.Pts_n_Wts, NGT);
        } else {
            printf("Error: not yet implement element stiffness matrix which is not Eigen matrix format\n");
        }

        // assemble element stiffness matrix to global K
        if (useEigen){
            if (matFree) {
                // copy element matrix to store in m_epMat[eid]
                stMat.copy_element_matrix(eid, kee[0], 0, 0, 1);
            } else {
                stMat.petsc_set_element_matrix(eid, kee[0], 0, 0, ADD_VALUES);
            }
        } else {
            std::cout<<"Error: assembly matrix only works for Eigen matrix"<<std::endl;
        }
        
        // compute element force vector due to body force
        fe_hex8_iso(fee[0], xe, beN, intData.Pts_n_Wts, NGT);

        /* for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            printf("[e %d][n %d] fee= %f, %f, %f\n",eid,nid,fee[0](nid * NDOF_PER_NODE ),
            fee[0](nid * NDOF_PER_NODE +1),fee[0](nid * NDOF_PER_NODE +2));
        } */

        // assemble element load vector due to body force
        stMat.petsc_set_element_vec(rhs, eid, fee[0], 0, ADD_VALUES);

        // assemble element load vector due to traction
        if (elem_trac[eid].size() != 0){
            stMat.petsc_set_element_vec(rhs, eid, elem_trac[eid], 0, ADD_VALUES);
        }
    }

    delete [] xe;

    // Pestc begins and completes assembling the global stiffness matrix
    if (!matFree){
        stMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
        //stMat.dump_mat("matrix.dat");
    }

    // Pestc begins and completes assembling the global load vector
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);
    //stMat.dump_vec(rhs,"rhs_before.dat");

    //char fname[256];
    // apply dirichlet BCs
    if (!matFree){
        stMat.apply_bc_mat();
        stMat.petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat.petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
        //sprintf(fname,"matrix_%d.dat",size);    
        //stMat.dump_mat(fname);
    }
    stMat.apply_bc_rhs(rhs);
    stMat.petsc_init_vec(rhs);
    stMat.petsc_finalize_vec(rhs);
    //stMat.dump_vec(rhs, "rhs.dat");

    //sprintf(fname,"rhsVec_%d.dat",size);
    //stMat.dump_vec(rhs,fname);

    // solve
    stMat.petsc_solve((const Vec) rhs, out);

    stMat.petsc_init_vec(out);
    stMat.petsc_finalize_vec(out);
    stMat.dump_vec(out,"out_ex3.dat");

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
            gNodeId = globalMap[eid][nid];
            x = (double)(gNodeId % (Nex + 1)) * hx;
            y = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(gNodeId / ((Nex + 1)*(Ney + 1))) * hz;

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
        stMat.petsc_set_element_vec(sol_exact, eid, e_exact, 0, INSERT_VALUES);
    }

    stMat.petsc_init_vec(sol_exact);
    stMat.petsc_finalize_vec(sol_exact);


    // compute norm of exact solution
    VecNorm(sol_exact, NORM_2, &norm);
    if (rank == 0){
        printf("L2 norm of exact solution = %20.10f\n",norm);
    }

    // compute the error vector
    VecCopy(sol_exact, error);

    // subtract error = sol_exact - out
    VecAXPY(error, alpha, out);

    // compute norm of error
    //VecNorm(sol_exact, NORM_INFINITY, &norm);
    VecNorm(error, NORM_INFINITY, &norm);
    if (rank == 0){
        printf("Inf norm of error = %20.10f\n", norm);
    }
    
    std::ofstream myfile;
    if (!rank){
        myfile.open("ex3.vtk");
        myfile << "# vtk DataFile Version 2.0 " << std::endl;
        myfile << "Stress field" << std::endl;
        myfile << "ASCII" << std::endl;
        myfile << "DATASET UNSTRUCTURED_GRID" << std::endl;
        myfile << "POINTS " << nnode << " float" << std::endl;
        for (unsigned int nid = 0; nid < numLocalNodes; nid++){
            gNodeId = local2GlobalMap[nid];
            x = (double)(gNodeId % (Nex + 1)) * hx;
            y = (double)((gNodeId % ((Nex + 1)*(Ney + 1))) / (Nex + 1)) * hy;
            z = (double) (gNodeId / ((Nex + 1) * (Ney + 1))) * hz;

            // translate origin
            x = x - Lx/2;
            y = y - Ly/2;

            myfile << x << "  " << y << "  " << z << std::endl;
        }
        unsigned int size_cell_list = nelem * 9;
        myfile << "CELLS " << nelem << " " << size_cell_list << std::endl;
        for (unsigned int eid = 0; eid < nelem; eid++){
            myfile << "8 " << globalMap[eid][0] << " " << globalMap[eid][1] << " "
             << globalMap[eid][2] << " " << globalMap[eid][3] << " "
              << globalMap[eid][4] << " " << globalMap[eid][5] << " "
               << globalMap[eid][6] << " " << globalMap[eid][7] << std::endl;
        }
        myfile << "CELL_TYPES " << nelem << std::endl;
        for (unsigned int eid = 0; eid < nelem; eid++){
            myfile << "12" << std::endl;
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
    }

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