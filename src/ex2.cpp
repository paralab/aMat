//
// Created by Han Tran on 2/24/20.
//

/**
 * @file ex2.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Example: 2D square plate under uniform traction ty = 1 on the top surface
 * @brief Bottom surface is constrained by rollers
 * @brief Exact solution (origin at left-bottom corner)
 * @brief    uniform stress s_zz = tz = 1
 * @brief    displacement u = -(nu/E) * ty * x
 * @brief    displacement v = (1/E) * ty * y
 * @brief Partition of elements in y direction: owned elements in y direction ~ Ney/(number of ranks)
 * @brief Size of the domain: Lx = Ly = 1.0
 *
 * @version 0.1
 * @date 2020-02-24
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
#include "integration.hpp"

using Eigen::Matrix;

// number of cracks allowed in 1 element
#define AMAT_MAX_CRACK_LEVEL 0

// max number of block dimensions in one cracked element
#define AMAT_MAX_BLOCKSDIM_PER_ELEMENT (1u<<AMAT_MAX_CRACK_LEVEL)

//////////////////////////////////////////////////////////////////////////////////////////////////////

void usage()
{
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  ex2 <Nex> <Ney> <matrix based/free> <bc-method>\n";
    std::cout << "\n";
    std::cout << "     Nex: Number of elements in X\n";
    std::cout << "     Ney: Number of elements in y\n";
    std::cout << "     use matrix-free: 1 => yes.  0 => matrix-based method. \n";
    std::cout << "     use identity-matrix: 0    use penalty method: 1 \n";
    exit( 0) ;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]){
    // User provides: Nex = number of elements in x direction
    //                Ney = number of elements in y direction
    //                flag = 1 --> matrix-free method; 0 --> matrix-based method
    //                bcMethod = 0 --> identity matrix method; 1 --> penalty method
    if( argc < 5 ) {
        usage();
    }

    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]);
    const unsigned int matType = atoi(argv[3]);
    const unsigned int bcMethod = atoi(argv[4]); // method of applying BC

    const unsigned int NDOF_PER_NODE = 2;       // number of dofs per node
    const unsigned int NDIM = 2;                // number of dimension
    const unsigned int NNODE_PER_ELEM = 4;      // number of nodes per element

    // material properties
    const double E = 1.0;
    const double nu = 0.3;

    // element matrix and force vector
    Matrix<double,8,8> ke;
    Matrix<double,8,1> fe;

    // element nodal coordinates
    double* xe = new double [8];

    // domain size
    const double Lx = 1.0;
    const double Ly = 1.0;

    // element size
    const double hx = Lx/double(Nex);
    const double hy = Ly/double(Ney);

    const double zero_number = 1E-12;

    // MPI initialize
    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(!rank) {
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNex : "<< Nex << " Ney: " << Ney << "\n";
        std::cout << "\t\tLx : "<< Lx << " Ly: " << Ly << "\n";
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
        ymin -= tol * hy;
    }
    double ymax = ymin + d;
    if (rank == size){
        ymax += tol * hy;
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

    // map from elemental node to global node
    unsigned long gNodeId;
    unsigned long int* * globalMap;
    globalMap = new unsigned long* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        globalMap[eid] = new unsigned long [AMAT_MAX_BLOCKSDIM_PER_ELEMENT * NNODE_PER_ELEM];
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

    // map from elemental DOF to global DOF
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

    // build localMap from globalMap (to adapt the interface of bsamxx)
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalNodes;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;

    // counts of owned nodes: nnodeCount[0] = nnode0, nnodeCount[1] = nnode1, ...
    unsigned int* nnodeCount = new unsigned int [size];
    unsigned int* nnodeOffset = new unsigned int [size];

    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    nnodeOffset[0] = 0;
    for (unsigned int i = 1; i < size; i++){
        nnodeOffset[i] = nnodeOffset[i-1] + nnodeCount[i-1];
    }
    unsigned int nnode_total, ndof_total;
    nnode_total = nnodeOffset[size-1] + nnodeCount[size-1];
    ndof_total = nnode_total * NDOF_PER_NODE;

    // determine ghost nodes based on:
    // rank 0 owns [0,...,nnode0-1], rank 1 owns [nnode0,..., nnode0 + nnode1 - 1]...
    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            gNodeId = globalMap[eid][nid];
            if (gNodeId < nnodeOffset[rank]){
                preGhostGIds.push_back(gNodeId);
            } else if (gNodeId >= nnodeOffset[rank] + nnode){
                postGhostGIds.push_back(gNodeId);
            }
        }
    }

    // sort in ascending order to prepare for deleting repeated nodes in preGhostGIds and postGhostGIds
    std::sort(preGhostGIds.begin(), preGhostGIds.end());
    std::sort(postGhostGIds.begin(), postGhostGIds.end());

    // remove consecutive duplicates and erase all after .end()
    preGhostGIds.erase(std::unique(preGhostGIds.begin(), preGhostGIds.end()), preGhostGIds.end());
    postGhostGIds.erase(std::unique(postGhostGIds.begin(), postGhostGIds.end()), postGhostGIds.end());
    
    // number of pre and post ghost nodes of my rank
    numPreGhostNodes = preGhostGIds.size();
    numPostGhostNodes = postGhostGIds.size();

    // number of local dofs
    numLocalNodes = numPreGhostNodes + nnode + numPostGhostNodes;

    // number of local dofs
    unsigned int numLocalDofs = numLocalNodes * NDOF_PER_NODE;
    
    // map from elemental nodes to local nodes
    unsigned int* * localMap;
    localMap = new unsigned int* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        localMap[eid] = new unsigned int[AMAT_MAX_BLOCKSDIM_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int i = 0; i < NNODE_PER_ELEM; i++){
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
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
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
    /*for (unsigned int eid = 0; eid < nelem; eid++){
        printf("rank %d, eid %d, localMap= [%d,%d,%d,%d]\n",rank,eid,localMap[eid][0],localMap[eid][1],localMap[eid][2],localMap[eid][3]);
    }*/
    /*for (unsigned int nid = 0; nid < numLocalNodes; nid++){
        printf("rank %d, local node %d --> global node %d\n",rank,nid,local2GlobalMap[nid]);
    }*/

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
    for (unsigned int eid = 0; eid < nelem; eid++){
        bound_dofs[eid] = new unsigned int [ndofs_per_element[eid]];
        bound_values[eid] = new double [ndofs_per_element[eid]];
    }

    // construct elemental constrained DoFs and prescribed values
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){

            // global node id of elemental node n
            gNodeId = globalMap[eid][nid];

            // compute nodal coordinate
            double x = (double)(gNodeId % (Nex + 1)) * hx;
            double y = (double)(gNodeId / (Nex + 1)) * hy;

            // nodes on bottom surface have roller supports in y direction
            if (fabs(y) < zero_number){
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 1] = 1;
                bound_dofs[eid][(nid * NDOF_PER_NODE)] = 0; 
                bound_values[eid][(nid * NDOF_PER_NODE) + 1] = 0.0;
                bound_values[eid][(nid * NDOF_PER_NODE)] = -1000000;
            } else {
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                    bound_dofs[eid][(nid * NDOF_PER_NODE) + did] = 0; // free dof
                    bound_values[eid][(nid * NDOF_PER_NODE) + did] = -1000000;
                }
            }
            if ((fabs(x) < zero_number) && (fabs(y) < zero_number)){
                bound_dofs[eid][nid * NDOF_PER_NODE] = 1;
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 1] = 1;
                bound_values[eid][nid * NDOF_PER_NODE] = 0.0;
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
    /*printf("rank %d, number of constraints %d\n",rank,constrainedDofs.size());
    for (unsigned int i = 0; i < constrainedDofs.size(); i++){
        printf("rank %d constraint %d, global ID %d, prescribed value %f\n",rank,i,constrainedDofs_ptr[i],prescribedValues_ptr[i]);
    }*/

    // ad-hoc: elemental traction vector
    Matrix<double, Eigen::Dynamic, 1> * elem_trac;
    elem_trac = new Matrix<double, Eigen::Dynamic, 1> [nelem];

    double area, x2, x3;
    for (unsigned int eid = 0; eid < nelem; eid++){
        bool traction = false;
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            gNodeId = globalMap[eid][nid];
            double y = (double)(gNodeId / (Nex + 1)) * hy;
            if (fabs(y - Ly) < zero_number){
                // element eid has one face is the top surface with applied traction
                traction = true;
                break;
            }
        }
        if (traction) {
            // ad-hoc setting force vector due to uniform traction applied on face 2-3 of the element
            // compute area of the top face
            gNodeId = globalMap[eid][2];
            x2 = (double)(gNodeId % (Nex + 1)) * hx;
            
            gNodeId = globalMap[eid][3];
            x3 = (double)(gNodeId % (Nex + 1)) * hx;

            area = (x2 - x3);
            assert (area > zero_number);

            // put into load vector
            elem_trac[eid].resize(NNODE_PER_ELEM * NDOF_PER_NODE, 1);
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                    if (((nid == 2) || (nid == 3)) && (did == (NDOF_PER_NODE - 1))){
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = area/2;
                    }
                    else {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = 0.0;
                    }
                }
            }
        }
    }

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
                    end_global_dof, ndof_total);

    // set boundary map
    stMat->set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, list_of_constraints.size());

    Vec rhs, out, sol_exact;
    stMat->petsc_create_vec(rhs);
    stMat->petsc_create_vec(out);
    stMat->petsc_create_vec(sol_exact);

    // compute element stiffness matrix and assemble global stiffness matrix and load vector
    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            gNodeId = globalMap[eid][nid];
            double x = (double)(gNodeId % (Nex + 1)) * hx;
            double y = (double)(gNodeId / (Nex + 1)) * hy;
            xe[nid * 2] = x;
            xe[(nid * 2) + 1] = y;
        }

        // compute element stiffness matrix
        ke_quad4_iso(ke, xe, E, nu, intData.Pts_n_Wts, NGT);
        /* for (unsigned int row = 0; row < NNODE_PER_ELEM*NDOF_PER_NODE; row++){
            printf("[e %d, r %d]= %f,%f,%f,%f,%f,%f,%f,%f\n",eid,row,ke(row,0),ke(row,1)
            ,ke(row,2),ke(row,3),ke(row,4),ke(row,5),ke(row,6),ke(row,7));
        } */
        
        // assemble ke
        stMat->set_element_matrix(eid, ke, 0, 0, 1);

        // assemble fe
        if (elem_trac[eid].size() != 0){
            stMat->petsc_set_element_vec(rhs, eid, elem_trac[eid], 0, ADD_VALUES);
        }
    }

    delete [] xe;

    // Pestc begins and completes assembling the global stiffness matrix
    if (matType == 0){
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }

    // These are needed because we used ADD_VALUES for rhs when assembling
    // now we are going to use INSERT_VALUE for Fc in apply_bc_rhs
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // apply bc for rhs: this must be done before applying bc for the matrix
    // because we use the original matrix to compute KfcUc in matrix-based method
    stMat->apply_bc(rhs);
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    //char fname[256];

    // apply bc to the matrix
    if (matType == 0){
        //stMat.apply_bc_mat();
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
        //sprintf(fname,"matrix_%d.dat",size);    
        //stMat.dump_mat(fname);
    }

    //sprintf(fname,"rhsVec_%d.dat",size);
    //stMat.dump_vec(rhs,fname);

    // solve
    stMat->petsc_solve((const Vec) rhs, out);
    VecAssemblyBegin(out);
    VecAssemblyEnd(out);
    //stMat.dump_vec(out);

    PetscScalar norm, alpha = -1.0;

    // compute norm of solution
    VecNorm(out, NORM_2, &norm);
    if (rank == 0){
        printf("L2 norm of computed solution = %20.10f\n",norm);
    }

    // exact solution...
    Matrix< double, NDOF_PER_NODE * NNODE_PER_ELEM, 1 > e_exact;
    double disp [2];
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            gNodeId = globalMap[eid][nid];
            double x = (double)(gNodeId % (Nex + 1)) * hx;
            double y = (double)(gNodeId / (Nex + 1)) * hy;
            disp[0] = -nu * x/E;
            disp[1] = y/E;
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++){
                e_exact[(nid * NDOF_PER_NODE) + did] = disp[did];
            }
        }
        stMat->petsc_set_element_vec(sol_exact, eid, e_exact, 0, INSERT_VALUES);
    }
    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

    //stMat.dump_vec(sol_exact);

    // compute norm of exact solution
    VecNorm(sol_exact, NORM_2, &norm);
    if (rank == 0){
        printf("L2 norm of exact solution = %20.10f\n",norm);
    }

    // subtract sol_exact = sol_exact - out
    VecAXPY(sol_exact, alpha, out);

    // compute norm of error
    VecNorm(sol_exact, NORM_INFINITY, &norm);
    if (rank == 0){
        printf("Inf norm of error = %20.10f\n", norm);
    }

     for (unsigned int eid = 0; eid < nelem; eid++) {
        delete [] globalMap[eid];
        delete [] globalDofMap[eid];
    }
    delete [] globalMap;
    delete [] globalDofMap;

    for (unsigned int eid = 0; eid < nelem; eid++){
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

    for (unsigned int eid = 0; eid < nelem; eid++){
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