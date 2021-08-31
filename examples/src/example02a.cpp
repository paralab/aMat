/**
 * @file example02a.cpp, example02a.hpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Solving 2D Laplace problem by FEM, in parallel, using aMat with linear 4-node quadrilateral elements
 * @brief (this example was fem2d in in aMat_for_paper/)
 * @brief    (d^2)u/d(x^2) + (d^2)u/d(y^2) = 0
 * @brief    BCs u(0,y) = u(1,y) = 0; u(x,0) = sin(pi*x); u(x,1) = sin(pi*x)exp(-pi)
 * @brief Exact solution: u(x,y) = sin(M_PI * x) * exp(-M_PI * y)
 *
 * @version 0.1
 * @date 2020-01-05
 *
 * @copyright Copyright (c) 2020 School of Computing, University of Utah
 *
 */

#include "example02a.hpp"
AppData example02aAppData;

void usage() {
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  example02a <Nex> <Ney> matrix method> <bc method> <nstreams> <outputfile>\n";
    std::cout << "\n";
    std::cout << "     1) Nex: Number of elements in X\n";
    std::cout << "     2) Ney: Number of elements in y\n";
    std::cout << "     3) method (0, 1, 2, 3, 4, 5).\n";
    std::cout << "     4) use identity-matrix: 0    use penalty method: 1 \n";
    std::cout << "     5) number of streams (used in method 3, 4, 5\n";
    std::cout << "     6) name of output file\n";
    std::cout << "\n";
    std::exit(0);
}

// function to compute element matrix used in method = 2
void computeElemMat(unsigned int eid, double *ke, double* xe) {
    const double hx = example02aAppData.hx;
    const double hy = example02aAppData.hy;
    const unsigned int Nex  = example02aAppData.Nex;
    const unsigned int NDOF_PER_ELEM = example02aAppData.NDOF_PER_ELEM;

    unsigned long ** globalMap = example02aAppData.globalMap;

    // get coordinates of all nodes
    unsigned long global_Id;
    for (unsigned int nid = 0; nid < NDOF_PER_ELEM; nid++) {
        global_Id = globalMap[eid][nid];
        xe[nid * 2] = (double)(global_Id % (Nex + 1)) * hx;
        xe[(nid * 2) + 1] = (double)(global_Id / (Nex + 1)) * hy;
    }
    ke_quad4(ke, xe, example02aAppData.intData->Pts_n_Wts, example02aAppData.NGT);
    
    return;
} // computeElemMat

int main(int argc, char* argv[]) {
    if (argc < 7) {
        usage();
    }

    const unsigned int NDOF_PER_NODE  = 1; // number of dofs per node
    const unsigned int NDIM           = 2; // number of dimension
    const unsigned int NNODE_PER_ELEM = 4; // number of nodes per element

    const unsigned int NDOF_PER_ELEM = NDOF_PER_NODE * NNODE_PER_ELEM;
    
    const unsigned int Nex      = atoi(argv[1]);
    const unsigned int Ney      = atoi(argv[2]);
    const unsigned int matType  = atoi(argv[3]);
    const unsigned int bcMethod = atoi(argv[4]);// method of applying BC
    const unsigned int nStreams = atoi(argv[5]);// number of streams used for method 3, 4, 5
    const char* filename        = argv[6];      // output filename

    // domain size
    const double Lx = 100.0;
    const double Ly = Lx; // the above-mentioned exact solution is only applied for Ly = Lx

    // element size
    const double hx = Lx / double(Nex);
    const double hy = Ly / double(Ney);

    // Gauss points and weights for computing element matrix and force vector
    const unsigned int NGT = 2;
    integration<double> intData(NGT);

    // give application data to global variable example02AppData so that they will be used in aMatFree if method 2 is chosen
    example02aAppData.Nex = Nex;
    example02aAppData.hx = hx;
    example02aAppData.hy = hy;
    example02aAppData.NDOF_PER_ELEM = NDOF_PER_ELEM;
    example02aAppData.NGT = NGT;
    example02aAppData.intData = &intData;

    // element matrix and force vector
    Matrix<double, NDOF_PER_ELEM, NDOF_PER_ELEM> ke;
    Matrix<double, NDOF_PER_ELEM, 1> fe;

    // element nodal coordinates
    double* xe = new double[NDIM * NNODE_PER_ELEM];

    // timing variables
    profiler_t elem_compute_time;
    profiler_t setup_time;
    profiler_t matvec_time;

    elem_compute_time.clear();
    setup_time.clear();
    matvec_time.clear();

    const double zero_number = 1E-12;

    // MPI initialize
    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // output file in csv format, open in append mode
    std::ofstream outFile;
    if (rank == 0)
        outFile.open(filename, std::fstream::app);

    if (!rank) {
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNex : " << Nex << " Ney: " << Ney << "\n";
        std::cout << "\t\tLx : " << Lx << " Ly: " << Ly << "\n";
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
    if (size > Ney) {
        if (!rank) {
            std::cout << "Number of ranks must be <= Ney, program stops..." << "\n";
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // partition in y direction...
    unsigned int emin = 0, emax = 0;
    unsigned int nelem_y;
    // minimum number of elements in y-dir for each rank
    unsigned int nymin = Ney / size;
    // remaining
    unsigned int nRemain = Ney % size;
    // distribute nRemain uniformly from rank = 0 up to rank = nRemain - 1
    if (rank < nRemain) {
        nelem_y = nymin + 1;
    } else {
        nelem_y = nymin;
    }
    if (rank < nRemain) {
        emin = rank * nymin + rank;
    } else {
        emin = rank * nymin + nRemain;
    }
    emax = emin + nelem_y - 1;

    // number of elements owned by my rank
    unsigned int nelem_x = Nex;
    unsigned int nelem   = nelem_x * nelem_y;

    // assign number of nodes owned by my rank: in y direction rank 0 owns 2 boundary nodes, other ranks own right boundary node)
    unsigned int nnode, nnode_y, nnode_x;
    if (rank == 0) {
        nnode_y = nelem_y + 1;
    } else {
        nnode_y = nelem_y;
    }
    nnode_x = Nex + 1;
    nnode   = nnode_x * nnode_y;

    // number of dofs per element (here we use linear 4-node element for this example)
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
    for (unsigned j = 0; j < nelem_y; j++) {
        for (unsigned i = 0; i < nelem_x; i++) {
            unsigned int eid  = nelem_x * j + i;
            globalMap[eid][0] = (emin * (Nex + 1) + i) + j * (Nex + 1);
            globalMap[eid][1] = globalMap[eid][0] + 1;
            globalMap[eid][3] = globalMap[eid][0] + (Nex + 1);
            globalMap[eid][2] = globalMap[eid][3] + 1;
        }
    }

    // set globalMap to AppData so that we can used in function compute element stiffness matrix
    example02aAppData.globalMap = globalMap;

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
    if (rank == 0)
        printf("Total dofs = %d\n", ndofs_total);

    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            gNodeId = globalMap[eid][nid];
            if (gNodeId < nnodeOffset[rank]) {
                preGhostGIds.push_back(gNodeId);
            } else if (gNodeId >= nnodeOffset[rank] + nnode) {
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
                  std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) -
                  preGhostGIds.begin();
                localDofMap[eid][i] = lookUp;
            }
            else if (gNodeId >= (nnodeOffset[rank] + nnode)) {
                // nid is owned by someone after me
                const unsigned int lookUp =
                  std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), gNodeId) -
                  postGhostGIds.begin();
                localDofMap[eid][i] = numPreGhostNodes + nnode + lookUp;
            }
        }
    }
    // build local2GlobalDofMap map (to adapt the interface of bsamxx)
    unsigned long* local2GlobalDofMap = new unsigned long[numLocalDofs];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            gNodeId                                   = globalMap[eid][nid];
            local2GlobalDofMap[localDofMap[eid][nid]] = gNodeId;
        }
    }

    // boundary conditions ...
    unsigned int** bound_nodes = new unsigned int* [nelem];
    double** bound_values      = new double* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        bound_nodes[eid]  = new unsigned int[ndofs_per_element[eid]];
        bound_values[eid] = new double[ndofs_per_element[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            unsigned long global_Id = globalMap[eid][nid];
            double x                = (double)(global_Id % (Nex + 1)) * hx;
            double y                = (double)(global_Id / (Nex + 1)) * hy;
            if ((fabs(x) < zero_number) || (fabs(x - Lx) < zero_number)) {
                // left or right boundary
                bound_nodes[eid][nid]  = 1;
                bound_values[eid][nid] = 0.0;
            }
            else if (fabs(y) < zero_number) {
                // bottom boundary
                bound_nodes[eid][nid]  = 1;
                bound_values[eid][nid] = sin(M_PI * x/Lx);
            }
            else if (fabs(y - Ly) < zero_number) {
                // top boundary
                bound_nodes[eid][nid]  = 1;
                bound_values[eid][nid] = sin(M_PI * x/Lx) * exp(-M_PI);
            }
            else {
                // iterior
                bound_nodes[eid][nid]  = 0;
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
                if (bound_nodes[eid][(nid * NDOF_PER_NODE) + did] == 1) {
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
    typedef par:: aMat<par::aMatBased<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> aMatBased; // aMat type taking aMatBased as derived class
    typedef par:: aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> aMatFree; // aMat type taking aMatBased as derived class

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

    // set function to compute element matrix if using matrix-free
    if (matType == 2){
        stMatFree->set_element_matrix_function(&computeElemMat);
    }

    Vec rhs, out, sol_exact;
    par::create_vec(meshMaps, rhs);
    par::create_vec(meshMaps, out);
    par::create_vec(meshMaps, sol_exact);

    // compute, then assemble element stiffness matrix and force vector
    for (unsigned int eid = 0; eid < nelem; eid++) {
        // compute nodal coordinates of element eid
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            unsigned long global_Id = globalMap[eid][nid];
            xe[nid * 2]             = (double)(global_Id % (Nex + 1)) * hx;
            xe[(nid * 2) + 1]       = (double)(global_Id / (Nex + 1)) * hy;
        }

        // compute element stiffness matrix
        setup_time.start();
        elem_compute_time.start();
        ke_quad4_eig(ke, xe, intData.Pts_n_Wts, NGT);
        elem_compute_time.stop();

        // assemble ke
        if (matType == 0)
            stMatBased->set_element_matrix(eid, ke, 0, 0, 1);
        else
            stMatFree->set_element_matrix(eid, ke, 0, 0, 1);
        setup_time.stop();

        // for this example, no force vector
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            fe(nid) = 0.0;
        }

        // assemble fe
        par::set_element_vec(meshMaps, rhs, eid, fe, 0u, ADD_VALUES);
    }
    delete[] xe;

    // Pestc begins and completes assembling the global stiffness matrix
    setup_time.start();
    if (matType == 0)
        stMatBased->finalize();
    else
        stMatFree->finalize(); // compute trace of matrix when using penalty method
    setup_time.stop();

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
        setup_time.start();
        stMatBased->finalize();
        setup_time.stop();
    }

    // solve
    matvec_time.start();
    if (matType == 0)
        par::solve(*stMatBased, (const Vec)rhs, out);
    else
        par::solve(*stMatFree, (const Vec)rhs, out);
    matvec_time.stop();

    PetscScalar norm, alpha = -1.0;

    VecNorm(out, NORM_2, &norm);
    if (!rank) {
        printf("L2 norm of computed solution = %f\n", norm);
    }

    // exact solution...
    PetscInt rowId;
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            rowId         = globalMap[eid][nid];
            double x      = (double)(rowId % (Nex + 1)) * hx;
            double y      = (double)(rowId / (Nex + 1)) * hy;
            PetscScalar u = sin(M_PI * x/Lx) * exp(-M_PI * y/Ly);
            VecSetValue(sol_exact, rowId, u, INSERT_VALUES);
        }
    }
    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

    VecNorm(sol_exact, NORM_2, &norm);
    if (!rank) {
        printf("L2 norm of exact solution = %f\n", norm);
    }

    // compute the norm of error
    VecAXPY(sol_exact, alpha, out);
    VecNorm(sol_exact, NORM_INFINITY, &norm);

    if (!rank) {
        printf("Inf norm of error = %20.10f\n", norm);
    }

    // computing time acrossing ranks and display
    long double elem_compute_maxTime;
    long double setup_maxTime;
    long double matvec_maxTime;
    
    MPI_Reduce(&elem_compute_time.seconds, &elem_compute_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&setup_time.seconds, &setup_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&matvec_time.seconds, &matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);

    if (matType == 0) {
        if (rank == 0) {
            std::cout << "(1) PETSc elem compute time = " << elem_compute_maxTime << "\n";
            std::cout << "(2) PETSc setup time = "        << setup_maxTime << "\n";
            std::cout << "(3) PETSc matvec time = "       << matvec_maxTime << "\n";
            outFile << "PETSc, " << elem_compute_maxTime << "," << setup_maxTime << "," << matvec_maxTime << "\n";
        }
    } else if (matType == 1) {
        if (rank == 0) {
            std::cout << "(1) aMat-hybrid elem compute time = " << elem_compute_maxTime << "\n";
            std::cout << "(2) aMat-hybrid setup time = "        << setup_maxTime << "\n";
            std::cout << "(3) aMat-hybrid matvec time = "       << matvec_maxTime << "\n";
            outFile << "aMat-hybrid, " << elem_compute_maxTime << ", " << setup_maxTime << ", " << matvec_maxTime << "\n";
        }
    } else if (matType == 2) {
        if (rank == 0) {
            std::cout << "(3) aMat-free matvec time = " << matvec_maxTime << "\n";
            outFile << "aMat-free, " << matvec_maxTime << "\n";
        }
    } else if ((matType == 3) || (matType == 4) || (matType == 5)) {
        if (rank == 0) {
            std::cout << "(1) aMatGpu elem compute time = " << elem_compute_maxTime << "\n";
            std::cout << "(2) aMatGpu setup time = " << setup_maxTime << "\n";
            std::cout << "(3) aMatGpu matvec time = " << matvec_maxTime << "\n";
            outFile << "aMatGpu, " << elem_compute_maxTime << ", " << setup_maxTime << ", " << matvec_maxTime << "\n";
        }
    }
    if (rank == 0) outFile.close();

    // free allocated memory...
    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] bound_nodes[eid];
        delete[] bound_values[eid];
    }
    delete[] bound_nodes;
    delete[] bound_values;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] globalMap[eid];
    }
    delete[] globalMap;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] localDofMap[eid];
    }
    delete [] localDofMap;
    delete [] nnodeCount;
    delete [] nnodeOffset;
    delete [] local2GlobalDofMap;
    delete[] ndofs_per_element;
    if (matType == 0) {
        delete stMatBased;
    } else {
        delete stMatFree;
    }
    
    // clean up Pestc vectors
    VecDestroy(&rhs);
    VecDestroy(&out);
    VecDestroy(&sol_exact);
    PetscFinalize();
    return 0;
}