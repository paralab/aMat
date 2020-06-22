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

#include "maps.hpp"
#include "enums.hpp"
#include "aMat.hpp"
#include "aMatFree.hpp"
#include "aMatBased.hpp"
#include "constraintRecord.hpp"
#include "solve.hpp"

using Eigen::Matrix;

void usage()
{
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  fem1d <Nex> <use matrix/free> <bc method>\n";
    std::cout << "\n";
    std::cout << "     Nex: Number of elements in X\n";
    std::cout << "     use matrix-free: 1 => yes.  0 => matrix-based method.\n";
    std::cout << "     use identity-matrix: 0    use penalty method: 1 \n";
    std::cout << "\n";
    std::exit( 0 ) ;
}

int main(int argc, char *argv[]){
    // User provides: Ne = number of elements
    //                matType = 0 --> matrix-based method; 1 --> matrix-free method
    //                bcMethod = 0 --> identity matrix method; 1 --> penalty method
    if( argc < 4 ) {
        usage();
    }

    const unsigned int NDOF_PER_NODE = 1;       // number of dofs per node
    const unsigned int NDIM = 1;                // number of dimension
    const unsigned int NNODE_PER_ELEM = 2;      // number of nodes per element

    const unsigned int Ne = atoi(argv[1]);
    const unsigned int matType = atoi(argv[2]);
    const unsigned int bcMethod = atoi(argv[3]); // method of applying BC

    // element matrix and force vector
    Matrix<double,2,2> ke;
    Matrix<double,2,1> fe;

    // domain size
    const double L = 1.0;

    // element size
    const double h = L/double(Ne);

    const double zero_number = 1E-12;
    
    // MPI initialize
    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(!rank) {
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNe : "<< Ne << "\n";
        std::cout << "\t\tL : "<< L << "\n";
        std::cout << "\t\tMethod (0 = matrix based; 1 = matrix free) = " << matType << "\n";
        std::cout << "\t\tBC method (0 = 'identity-matrix'; 1 = penalty): " << bcMethod << "\n";
    }
    
    #ifdef VECTORIZED_AVX512
    if (!rank) {std::cout << "\t\tVectorization using AVX_512\n";}
    #elif VECTORIZED_AVX256
    if (!rank) {std::cout << "\t\tVectorization using AVX_256\n";}
    #elif VECTORIZED_OPENMP
    if (!rank) {std::cout << "\t\tVectorization using OpenMP\n";}
    #elif VECTORIZED_OPENMP_ALIGNED
    if (!rank) {std::cout << "\t\tVectorization using OpenMP with aligned memory\n";}
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

    // number of nodes owned by my rank (rank 0 owns 2 boundary nodes, other ranks own right boundary node)
    unsigned int nnode;
    if (rank == 0){
        nnode = nelem + 1;
    } else {
        nnode = nelem;
    }

    // determine globalMap
    unsigned int* ndofs_per_element = new unsigned int [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        ndofs_per_element[eid] = NNODE_PER_ELEM; //linear 2-node element
    }

    unsigned long int** globalMap;
    globalMap = new unsigned long* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        globalMap[eid] = new unsigned long [ndofs_per_element[eid]];
    }
    // todo hard-code 2 nodes per element:
    for (unsigned int eid = 0; eid < nelem; eid++){
        globalMap[eid][0] = (emin + eid);
        globalMap[eid][1] = globalMap[eid][0] + 1;
    }

    // build localDofMap from globalMap (to adapt the interface of bsamxx)
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalDofs;
    unsigned long gNodeId;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;
    unsigned int* nnodeCount = new unsigned int [size];
    unsigned int* nnodeOffset = new unsigned int [size];

    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    nnodeOffset[0] = 0;
    for (unsigned int i = 1; i < size; i++){
        nnodeOffset[i] = nnodeOffset[i-1] + nnodeCount[i-1];
    }
    unsigned int ndofs_total;
    ndofs_total = nnodeOffset[size-1] + nnodeCount[size-1];
    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++){
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
    numLocalDofs = numPreGhostNodes + nnode + numPostGhostNodes;
    unsigned int** localDofMap;
    localDofMap = new unsigned int* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        localDofMap[eid] = new unsigned int[ndofs_per_element[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int i = 0; i < ndofs_per_element[eid]; i++){
            gNodeId = globalMap[eid][i];
            if (gNodeId >= nnodeOffset[rank] &&
                gNodeId < (nnodeOffset[rank] + nnode)) {
                // nid is owned by me
                localDofMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
            } else if (gNodeId < nnodeOffset[rank]){
                // nid is owned by someone before me
                const unsigned int lookUp = std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) - preGhostGIds.begin();
                localDofMap[eid][i] = lookUp;
            } else if (gNodeId >= (nnodeOffset[rank] + nnode)){
                // nid is owned by someone after me
                const unsigned int lookUp = std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), gNodeId) - postGhostGIds.begin();
                localDofMap[eid][i] =  numPreGhostNodes + nnode + lookUp;
            }
        }
    }
    // build local2GlobalDofMap map (to adapt the interface of bsamxx)
    unsigned long * local2GlobalDofMap = new unsigned long[numLocalDofs];
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++){
            gNodeId = globalMap[eid][nid];
            local2GlobalDofMap[localDofMap[eid][nid]] = gNodeId;
        }
    }

    // compute constrained map
    unsigned int** bound_dofs = new unsigned int* [nelem];
    double** bound_values = new double* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        bound_dofs[eid] = new unsigned int [ndofs_per_element[eid]];
        bound_values[eid] = new double [ndofs_per_element[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++){
            unsigned long global_Id = globalMap[eid][nid];
            double x = (double)(global_Id * h);
            if ((fabs(x) < zero_number) || (fabs(x - L) < zero_number)){
                bound_dofs[eid][nid] = 1;
            } else {
                bound_dofs[eid][nid] = 0;
            }
            if (fabs(x) < zero_number){
                // left end
                bound_values[eid][nid] = 0.0;
            } else if (fabs(x - L) < zero_number){
                // right end
                bound_values[eid][nid] = 1.0;
            } else {
                // free dofs
                bound_values[eid][nid] = -1000000;
            }
        }
    }

    // create lists of constrained dofs
    std::vector< par::ConstraintRecord<double, unsigned long int> > list_of_constraints;
    par::ConstraintRecord<double, unsigned long int> cdof;
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                if (bound_dofs[eid][(nid * NDOF_PER_NODE) + did] == 1) {
                    // save the global id of constrained dof
                    cdof.set_dofId( globalMap[eid][(nid * NDOF_PER_NODE) + did] );
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

    unsigned long start_global_dof, end_global_dof;
    start_global_dof = nnodeOffset[rank];
    end_global_dof = start_global_dof + (nnode - 1);

    // declare Maps object  =================================
    par::Maps<double, unsigned long, unsigned int> meshMaps(comm);

    meshMaps.set_map(nelem, localDofMap, ndofs_per_element, numLocalDofs, local2GlobalDofMap, start_global_dof,
                  end_global_dof, ndofs_total);

    if (matType == 1){
        meshMaps.buildScatterMap();
    }

    meshMaps.set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, list_of_constraints.size());

    // declare aMat object =================================
    typedef par::aMatBased<double, unsigned long, unsigned int>* aMatBased_ptr;
    typedef par::aMatFree<double, unsigned long, unsigned int>* aMatFree_ptr;

    par::aMat<double, unsigned long, unsigned int> * stMat;
    if (matType == 0){
        stMat = new par::aMatBased<double, unsigned long, unsigned int>(meshMaps,(par::BC_METH)bcMethod);
    } else {
        stMat = new par::aMatFree<double, unsigned long, unsigned int>(meshMaps,(par::BC_METH)bcMethod);
    }
    

    Vec rhs, out;
    par::create_vec(meshMaps, rhs);
    par::create_vec(meshMaps, out);


    // element stiffness matrix and assembly
    for (unsigned int eid = 0; eid < nelem; eid++){
        ke(0,0) = 1.0/h;
        ke(0,1) = -1.0/h;
        ke(1,0) = -1.0/h;
        ke(1,1) = 1.0/h;
        stMat->set_element_matrix(eid, ke, 0, 0, 1);
        fe(0) = 0.0;
        fe(1) = 0.0;
        stMat->petsc_set_element_vec(rhs, eid, fe, 0, ADD_VALUES);
    }
    char fname[256];
    sprintf(fname,"matrix_%d.dat",size);
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
    stMat->apply_bc(rhs);  // this includes applying bc for matrix in matrix-based approach
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // communication for matrix-based approach
    if (matType == 0){
        //stMat->apply_bc_mat();
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
        stMat->dump_mat(fname);
    }
    
    // solve
    par::solve(*stMat, (const Vec)rhs, out);
    /* if (matType == 0){
        par::solve(*dynamic_cast<aMatBased_ptr>(stMat), (const Vec)rhs, out);
    } else {
        par::solve(*dynamic_cast<aMatFree_ptr>(stMat), (const Vec)rhs, out);
    } */
    // display solution on screen 
    if (!rank) std::cout << "Computed solution = \n";     
    
    //sprintf(fname,"solution_%d.dat",size); 
    par::dump_vec(meshMaps, out);
    
    // following exact solution is only for the problem of u(0) = 0 and u(L) = 1
    // exact solution: u(x) = (1/L)*x
    Vec sol_exact;
    PetscInt rowId;
    par::create_vec(meshMaps, sol_exact);
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++){
            rowId = globalMap[eid][nid];
            double x = globalMap[eid][nid] * h;
            PetscScalar u = 1/L*x;
            VecSetValue(sol_exact,rowId,u,INSERT_VALUES);
        }
    }
    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

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
        delete [] bound_dofs[eid];
        delete [] bound_values[eid];
    }
    delete [] bound_dofs;
    delete [] bound_values;

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] globalMap[eid];
    }
    delete [] globalMap;

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] localDofMap[eid];
    }
    delete [] localDofMap;
    delete [] nnodeCount;
    delete [] nnodeOffset;
    delete [] local2GlobalDofMap;
    delete [] ndofs_per_element;
    VecDestroy(&out);
    VecDestroy(&sol_exact);
    VecDestroy(&rhs);

    PetscFinalize();
    return 0;
}