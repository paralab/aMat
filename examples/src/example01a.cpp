/**
 * @file example01a.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Solving 1D FEM problem by FEM, in parallel using aMat, using linear 2-node bar elements
 * @brief (this example was fem1d in in aMat_for_paper/)
 * @brief Governing equation: d^u/dx^2 = 0 for x = [0,1] subjected to Dirichlet BCs u(0) = 0, u(1) = 1
 * @brief Exact solution u(x) = x
 *
 * @version 0.1
 * @date 2020-01-03
 *
 * @copyright Copyright (c) 2020 School of Computing, University of Utah
 *
 */

#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <functional>

#ifdef BUILD_WITH_PETSC
#include <petsc.h>
#endif

#include "Eigen/Dense"
#include "aMat.hpp"
#include "aMatBased.hpp"
#include "aMatFree.hpp"
#include "aVec.hpp"
#include "constraintRecord.hpp"
#include "enums.hpp"
#include "maps.hpp"
#include "solve.hpp"

using Eigen::Matrix;
void usage() {
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  fem1d <Nex> <use matrix/free> <bc method>\n";
    std::cout << "\n";
    std::cout << "     1) Nex: Number of elements in X\n";
    std::cout << "     2) Method (0, 1, 2, 3, 4, 5).\n";
    std::cout << "     3) BCs: use identity-matrix: 0    use penalty method: 1 \n";
    std::cout << "     4) number of streams used with GPU \n";
    std::cout << "     5) output filename \n";
    std::cout << "\n";
    std::exit(0);
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        usage();
    }

    const unsigned int NDOF_PER_NODE  = 1; // number of dofs per node
    const unsigned int NDIM           = 1; // number of dimension
    const unsigned int NNODE_PER_ELEM = 2; // number of nodes per element

    const unsigned int NDOF_PER_ELEM = NDOF_PER_NODE * NNODE_PER_ELEM;

    const unsigned int Ne       = atoi(argv[1]); // number of global elements
    const unsigned int matType  = atoi(argv[2]); // method of solving (0, 1, 2, 3, 4, 5)
    const unsigned int bcMethod = atoi(argv[3]); // method of applying BC
    const unsigned int nStreams = atoi(argv[4]); // number of streams used for GPU (method 3, 4, 5)
    const char* filename        = argv[5];       // output filename

    if (matType == 2){
        printf("Matrix-free method is not implemented for this example! Program stops... \n");
        exit(0);
    }
    
    // variables to hold elemental matrix and force vector
    Matrix<double, 2, 2> ke;
    Matrix<double, 2, 1> fe;

    const double L = 1.0; // domain size
    const double h = L / double(Ne); // element size
    const double zero_number = 1E-12;

    // MPI initialize
    PetscInitialize(&argc, &argv, NULL, NULL);

    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Display input parameters
    if (!rank) {
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNe : " << Ne << "\n";
        std::cout << "\t\tMethod (0 = 'matrix-assembled'; 1 = 'AFM'; 2 = 'matrix-free'; 3/4/5 = 'AFM with GPU'): " << matType << "\n";
        std::cout << "\t\tBC method (0 = 'identity-matrix'; 1 = penalty): " << bcMethod << "\n";
        if ((matType == 3) || (matType == 4) || (matType == 5)){
            std::cout << "\t\tNumber of streams: " << nStreams << "\n";
        }
    }

    #ifdef HYBRID_PARALLEL
    if (!rank) {
        std::cout << "\t\tHybrid parallel OpenMP + MPI\n";
        std::cout << "\t\tMax number of threads: " << omp_get_max_threads() << "\n";
        std::cout << "\t\tNumber of MPI processes: " << size << "\n";
    }
    #else
    if (!rank) {
        std::cout << "\t\tOnly MPI parallel\n";
        std::cout << "\t\tNumber of MPI processes: " << size << "\n";
    }
    #endif

    int rc;
    if (size > Ne) {
        if (!rank) {
            std::cout << "Number of ranks must be <= Ne, program stops..."
                      << "\n";
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // partition the domain [0,1]
    unsigned int emin = 0, emax = 0; //global index of element for my rank
    unsigned int nelem; // number of elements owned by my rank

    // minimum number of elements in x-dir for each rank
    unsigned int nxmin = Ne / size;
    // remaining
    unsigned int nRemain = Ne % size;
    // distribute nRemain uniformly from rank = 0 up to rank = nRemain - 1
    if (rank < nRemain) {
        nelem = nxmin + 1;
    } else {
        nelem = nxmin;
    }
    if (rank < nRemain) {
        emin = rank * nxmin + rank;
    } else {
        emin = rank * nxmin + nRemain;
    }
    emax = emin + nelem - 1;

    // assign number of nodes (dofs) owned by my rank: rank 0 owns 2 boundary dofs, other ranks own right boundary dof
    unsigned int nnode;
    if (rank == 0) {
        nnode = nelem + 1;
    } else {
        nnode = nelem;
    }

    // number of dofs per element (here we use linear 2-node element for this example)
    unsigned int* ndofs_per_element = new unsigned int[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        ndofs_per_element[eid] = NDOF_PER_ELEM;
    }

    // global map...
    unsigned long int** globalMap;
    globalMap = new unsigned long*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalMap[eid] = new unsigned long[ndofs_per_element[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalMap[eid][0] = (emin + eid);
        globalMap[eid][1] = globalMap[eid][0] + 1;
    }

    // build localDofMap from globalMap (to adapt the interface of bsamxx)
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalDofs;
    unsigned long gNodeId;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;
    unsigned int* nnodeCount  = new unsigned int[size];
    unsigned int* nnodeOffset = new unsigned int[size];

    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    nnodeOffset[0] = 0;
    for (unsigned int i = 1; i < size; i++) {
        nnodeOffset[i] = nnodeOffset[i - 1] + nnodeCount[i - 1];
    }
    unsigned int ndofs_total;
    ndofs_total = nnodeOffset[size - 1] + nnodeCount[size - 1];
    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            gNodeId = globalMap[eid][nid];
            if (gNodeId < nnodeOffset[rank]) {
                preGhostGIds.push_back(gNodeId);
            }
            else if (gNodeId >= nnodeOffset[rank] + nnode) {
                postGhostGIds.push_back(gNodeId);
            }
        }
    }

    // sort in ascending order
    std::sort(preGhostGIds.begin(), preGhostGIds.end());
    std::sort(postGhostGIds.begin(), postGhostGIds.end());
    // remove consecutive duplicates and erase all after .end()
    preGhostGIds.erase(std::unique(preGhostGIds.begin(), preGhostGIds.end()), preGhostGIds.end());
    postGhostGIds.erase(std::unique(postGhostGIds.begin(), postGhostGIds.end()),
                        postGhostGIds.end());
    numPreGhostNodes  = preGhostGIds.size();
    numPostGhostNodes = postGhostGIds.size();
    numLocalDofs      = numPreGhostNodes + nnode + numPostGhostNodes;
    unsigned int** localDofMap;
    localDofMap = new unsigned int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localDofMap[eid] = new unsigned int[ndofs_per_element[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int i = 0; i < ndofs_per_element[eid]; i++) {
            gNodeId = globalMap[eid][i];
            if (gNodeId >= nnodeOffset[rank] && gNodeId < (nnodeOffset[rank] + nnode)) {
                // nid is owned by me
                localDofMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
            }
            else if (gNodeId < nnodeOffset[rank]) {
                // nid is owned by someone before me
                const unsigned int lookUp =
                  std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) - preGhostGIds.begin();
                localDofMap[eid][i] = lookUp;
            }
            else if (gNodeId >= (nnodeOffset[rank] + nnode)) {
                // nid is owned by someone after me
                const unsigned int lookUp =
                  std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), gNodeId) - postGhostGIds.begin();
                localDofMap[eid][i] = numPreGhostNodes + nnode + lookUp;
            }
        }
    }
    // build local2GlobalDofMap map (to adapt the interface of bsamxx)
    unsigned long* local2GlobalDofMap = new unsigned long[numLocalDofs];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            gNodeId = globalMap[eid][nid];
            local2GlobalDofMap[localDofMap[eid][nid]] = gNodeId;
        }
    }

    // boundary conditions: u = 0 at node of x = 0, u = 1 at node of x = L
    unsigned int** bound_dofs = new unsigned int*[nelem];
    double** bound_values     = new double*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        bound_dofs[eid]   = new unsigned int[ndofs_per_element[eid]];
        bound_values[eid] = new double[ndofs_per_element[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            unsigned long global_Id = globalMap[eid][nid];
            double x = (double)(global_Id * h);
            if ((fabs(x) < zero_number) || (fabs(x - L) < zero_number)) {
                bound_dofs[eid][nid] = 1;
            }
            else {
                bound_dofs[eid][nid] = 0;
            }
            if (fabs(x) < zero_number) {
                // left end
                bound_values[eid][nid] = 0.0;
            }
            else if (fabs(x - L) < zero_number) {
                // right end
                bound_values[eid][nid] = 1.0;
            }
            else {
                // free dofs
                bound_values[eid][nid] = -1000000;
            }
        }
    }

    // create lists of constrained dofs
    std::vector<par::ConstraintRecord<double, unsigned long int>> list_of_constraints;
    par::ConstraintRecord<double, unsigned long int> cdof;
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                if (bound_dofs[eid][(nid * NDOF_PER_NODE) + did] == 1) {
                    // save the global id of constrained dof
                    cdof.set_dofId(globalMap[eid][(nid * NDOF_PER_NODE) + did]);
                    cdof.set_preVal(bound_values[eid][(nid * NDOF_PER_NODE) + did]);
                    list_of_constraints.push_back(cdof);
                }
            }
        }
    }

    // sort to prepare for deleting repeated constrained dofs in the list
    std::sort(list_of_constraints.begin(), list_of_constraints.end());
    list_of_constraints.erase(std::unique(list_of_constraints.begin(), list_of_constraints.end()),
                              list_of_constraints.end());

    // transform vector data to pointer (to be conformed with the aMat interface)
    unsigned long int* constrainedDofs_ptr;
    double* prescribedValues_ptr;
    const unsigned int n_constraints = list_of_constraints.size();
    constrainedDofs_ptr  = new unsigned long int[n_constraints];
    prescribedValues_ptr = new double[n_constraints];
    for (unsigned int i = 0; i < n_constraints; i++) {
        constrainedDofs_ptr[i]  = list_of_constraints[i].get_dofId();
        prescribedValues_ptr[i] = list_of_constraints[i].get_preVal();
    }

    unsigned long start_global_dof, end_global_dof;
    start_global_dof = nnodeOffset[rank];
    end_global_dof   = start_global_dof + (nnode - 1);

    // declare Maps object  =================================
    par::Maps<double, unsigned long, unsigned int> meshMaps(comm);
    meshMaps.set_map(nelem,
                     localDofMap,
                     ndofs_per_element,
                     numLocalDofs,
                     local2GlobalDofMap,
                     start_global_dof,
                     end_global_dof,
                     ndofs_total);

    meshMaps.set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, n_constraints);

    // declare aMat object =================================
    typedef par:: aMat<par::aMatBased<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> aMatBased;
    typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> aMatFree;

    aMatBased* stMatBased; // pointer of aMat taking aMatBased as derived
    aMatFree* stMatFree;   // pointer of aMat taking aMatFree as derived

    if (matType == 0) {
        // assign stMatBased to the derived class aMatBased
        stMatBased = new par::aMatBased<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
    } else {
        // assign stMatFree to the derived class aMatFree
        stMatFree = new par::aMatFree<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
        stMatFree->set_matfree_type((par::MATFREE_TYPE)matType);

        #ifdef USE_GPU
        if ((matType == 3) || (matType == 4) || (matType == 5)){
            #ifdef USE_GPU
                stMatFree->set_num_streams(nStreams);
            #endif
        }
        #endif
    }

    Vec rhs, out;
    par::create_vec(meshMaps, rhs);
    par::create_vec(meshMaps, out);

    // assemble element stiffness matrix and force vector
    for (unsigned int eid = 0; eid < nelem; eid++) {
        ke(0, 0) = 1.0 / h; ke(0, 1) = -1.0 / h; ke(1, 0) = -1.0 / h; ke(1, 1) = 1.0 / h;
        if (matType == 0)
            stMatBased->set_element_matrix(eid, ke, 0, 0, 1);
        else
            stMatFree->set_element_matrix(eid, ke, 0, 0, 1);

        fe(0) = 0.0; fe(1) = 0.0;
        par::set_element_vec(meshMaps, rhs, eid, fe, 0u, ADD_VALUES);
    }
    
    // Begins and completes assembling the global stiffness matrix
    if (matType == 0)
        stMatBased->finalize();
    else
        stMatFree->finalize();


    // These are needed because we used ADD_VALUES for rhs when assembling
    // now we are going to use INSERT_VALUE for Fc in apply_bc_rhs
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // apply bc for rhs: this must be done before applying bc for the matrix
    // because we use the original matrix to compute KfcUc in matrix-based method
    if (matType == 0)
        stMatBased->apply_bc(rhs); // this includes applying bc for matrix in matrix-based approach
    else
        stMatFree->apply_bc(rhs);
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // communication for matrix-based approach
    if (matType == 0) {
        stMatBased->finalize();
    }

    // solve
    if (matType == 0)
        par::solve(*stMatBased, (const Vec)rhs, out);
    else
        par::solve(*stMatFree, (const Vec)rhs, out);

    // display solution on screen
    if (!rank)
        std::cout << "Computed solution = \n";
    par::dump_vec(meshMaps, out);

    // exact solution: u(x) = (1/L)*x
    Vec sol_exact;
    PetscInt rowId;
    par::create_vec(meshMaps, sol_exact);
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            rowId         = globalMap[eid][nid];
            double x      = globalMap[eid][nid] * h;
            PetscScalar u = 1 / L * x;
            VecSetValue(sol_exact, rowId, u, INSERT_VALUES);
        }
    }
    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

    // compute the norm of error
    PetscScalar norm, alpha = -1.0;
    VecAXPY(sol_exact, alpha, out);
    VecNorm(sol_exact, NORM_INFINITY, &norm);

    if (!rank) {
        printf("Inf norm of error = %20.16f\n", norm);
    }

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] bound_dofs[eid];
        delete[] bound_values[eid];
    }
    delete[] bound_dofs;
    delete[] bound_values;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] globalMap[eid];
    }
    delete[] globalMap;
    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] localDofMap[eid];
    }
    delete[] localDofMap;
    delete[] nnodeCount;
    delete[] nnodeOffset;
    delete[] local2GlobalDofMap;
    delete[] ndofs_per_element;

    VecDestroy(&out);
    VecDestroy(&sol_exact);
    VecDestroy(&rhs);

    PetscFinalize();
    return 0;
}