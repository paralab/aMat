//
// Created by Han Tran on 1/5/20.
//

/**
 * @file fem2d.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Example of solving 2D Laplace problem using FEM :
 * @brief    (d^2)u/d(x^2) + (d^2)u/d(y^2) = 0
 * @brief    BCs u(0,y) = u(1,y) = 0; u(x,0) = sin(pi*x); u(x,1) = sin(pi*x)exp(-pi)
 * @brief Exact solution: u = sin(M_PI * x) * exp(-M_PI * y);
 *
 * @version 0.1
 * @date 2020-01-05
 *
 * @copyright Copyright (c) 2020 School of Computing, University of Utah
 *
 */

#include "fem2d.hpp"
AppData fem2dAppData;

void usage()
{
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  fem2d <Nex> <Ney> <matrix based/free> <bc method>\n";
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
    
    const double hx = fem2dAppData.hx;
    const double hy = fem2dAppData.hy;
    
    const unsigned int Nex  = fem2dAppData.Nex;

    const unsigned int NNODE_PER_ELEM = fem2dAppData.NNODE_PER_ELEM;

    // get coordinates of all nodes
    unsigned long global_Id;
    for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
        global_Id = fem2dAppData.globalMap[eid][nid];
        xe[nid * 2] = (double)(global_Id % (Nex + 1)) * hx;
        xe[(nid * 2) + 1] = (double)(global_Id / (Nex + 1)) * hy;
    }
    ke_quad4(ke, xe, fem2dAppData.intData->Pts_n_Wts, fem2dAppData.NGT);
    
    return;
} // computeElemMat

int main(int argc, char* argv[])
{
    // User provides: Nex = number of elements in x direction
    //                Ney = number of elements in y direction
    //                flag = 1 --> matrix-free method; 0 --> matrix-based method
    //                bcMethod = 0 --> identity matrix method; 1 --> penalty method

    const unsigned int NDOF_PER_NODE  = 1; // number of dofs per node
    const unsigned int NDIM           = 2; // number of dimension
    const unsigned int NNODE_PER_ELEM = 4; // number of nodes per element

    const unsigned int Nex      = atoi(argv[1]);
    const unsigned int Ney      = atoi(argv[2]);
    const unsigned int matType  = atoi(argv[3]);
    const unsigned int bcMethod = atoi(argv[4]); // method of applying BC
    const unsigned int nStreams = atoi(argv[5]); // number of streams used for method 3, 4, 5

    // domain size
    const double Lx = 100.0;
    const double Ly = Lx; // for this solution, Ly must be equal to Lx

    // element size
    const double hx = Lx / double(Nex);
    const double hy = Ly / double(Ney);

    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);

    fem2dAppData.Nex = Nex;
    fem2dAppData.Ney = Ney;
    fem2dAppData.Lx = Lx;
    fem2dAppData.Ly = Ly;
    fem2dAppData.hx = hx;
    fem2dAppData.hy = hy;
    fem2dAppData.NNODE_PER_ELEM = NNODE_PER_ELEM;
    fem2dAppData.NDOF_PER_NODE = NDOF_PER_NODE;
    fem2dAppData.NGT = NGT;
    fem2dAppData.intData = &intData;

    // element matrix and force vector
    Matrix<double, 4, 4> ke;
    Matrix<double, 4, 1> fe;

    // element nodal coordinates
    double* xe = new double[8];

    // timing variables
    profiler_t aMat_elem_compute_time;
    profiler_t aMat_setup_time;
    profiler_t aMat_matvec_time;

    profiler_t petsc_elem_compute_time;
    profiler_t petsc_setup_time;
    profiler_t petsc_matvec_time;

    if (matType != 0) {
        aMat_elem_compute_time.clear();
        aMat_setup_time.clear();
        aMat_matvec_time.clear();
    } else {
        petsc_elem_compute_time.clear();
        petsc_setup_time.clear();
        petsc_matvec_time.clear();
    }

    const double zero_number = 1E-12;

    // MPI initialize
    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // output file in csv format, open in append mode
    std::ofstream outFile;
    const char* filename = argv[6];
    if (rank == 0)
        outFile.open(filename, std::fstream::app);

    if (argc < 7){
        usage();
    }

    if (!rank) {
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNex : " << Nex << " Ney: " << Ney << "\n";
        std::cout << "\t\tLx : " << Lx << " Ly: " << Ly << "\n";
        std::cout << "\t\tMethod = " << matType << "\n";
        std::cout << "\t\tBC method (0 = 'identity-matrix'; 1 = penalty): " << bcMethod << "\n";
        if ((matType == 3) || (matType ==4) || (matType == 5)){
            std::cout << "\t\tNumber of streams: " << nStreams << "\n";
        }
    }

    #ifdef VECTORIZED_AVX512
    if (!rank) {
        std::cout << "\t\tVectorization using AVX_512\n";
    }
    #elif VECTORIZED_AVX256
    if (!rank) {
        std::cout << "\t\tVectorization using AVX_256\n";
    }
    #elif VECTORIZED_OPENMP
    if (!rank) {
        std::cout << "\t\tVectorization using OpenMP\n";
    }
    #elif VECTORIZED_OPENMP_ALIGNED
    if (!rank) {
        std::cout << "\t\tVectorization using OpenMP with aligned memory\n";
    }
    #else
    if (!rank) {
        std::cout << "\t\tNo vectorization\n";
    }
    #endif

    #ifdef HYBRID_PARALLEL
    if (!rank) {
        std::cout << "\t\tHybrid parallel OpenMP + MPI\n";
        std::cout << "\t\tMax number of threads: " << omp_get_max_threads() << "\n";
        std::cout << "\t\tNumber of MPI processes: " << size << "\n";
        if ((matType == 3) || (matType == 4) || (matType == 5)){
            std::cout << "\t\tNumber of streams: " << nStreams << "\n";
        }
    }
    #else
    if (!rank) {
        std::cout << "\t\tOnly MPI parallel\n";
        std::cout << "\t\tNumber of MPI processes: " << size << "\n";
        if ((matType == 3) || (matType == 4) || (matType == 5)){
            std::cout << "\t\tNumber of streams: " << nStreams << "\n";
        }
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

    // number of nodes owned by my rank (rank 0 owns 2 boundary nodes, other ranks own right
    // boundary node)
    unsigned int nnode, nnode_y, nnode_x;
    if (rank == 0) {
        nnode_y = nelem_y + 1;
    } else {
        nnode_y = nelem_y;
    }
    nnode_x = Nex + 1;
    nnode   = nnode_x * nnode_y;

    // determine globalMap
    unsigned int* ndofs_per_element = new unsigned int[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        ndofs_per_element[eid] = NNODE_PER_ELEM; // linear 4-node element
    }

    unsigned long int** globalMap;
    globalMap = new unsigned long*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        globalMap[eid] = new unsigned long[ndofs_per_element[eid]];
    }
    // todo hard-code 4 nodes per element:
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
    fem2dAppData.globalMap = globalMap;

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
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        localDofMap[eid] = new unsigned int[ndofs_per_element[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int i = 0; i < ndofs_per_element[eid]; i++)
        {
            gNodeId = globalMap[eid][i];
            if (gNodeId >= nnodeOffset[rank] && gNodeId < (nnodeOffset[rank] + nnode))
            {
                // nid is owned by me
                localDofMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
            }
            else if (gNodeId < nnodeOffset[rank])
            {
                // nid is owned by someone before me
                const unsigned int lookUp =
                  std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) -
                  preGhostGIds.begin();
                localDofMap[eid][i] = lookUp;
            }
            else if (gNodeId >= (nnodeOffset[rank] + nnode))
            {
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
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++)
        {
            gNodeId                                   = globalMap[eid][nid];
            local2GlobalDofMap[localDofMap[eid][nid]] = gNodeId;
        }
    }

    // compute constrained map
    unsigned int** bound_nodes = new unsigned int*[nelem];
    double** bound_values      = new double*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        bound_nodes[eid]  = new unsigned int[ndofs_per_element[eid]];
        bound_values[eid] = new double[ndofs_per_element[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++)
        {
            unsigned long global_Id = globalMap[eid][nid];
            double x                = (double)(global_Id % (Nex + 1)) * hx;
            double y                = (double)(global_Id / (Nex + 1)) * hy;
            if ((fabs(x) < zero_number) || (fabs(x - Lx) < zero_number))
            {
                // left or right boundary
                bound_nodes[eid][nid]  = 1;
                bound_values[eid][nid] = 0.0;
            }
            else if (fabs(y) < zero_number)
            {
                // bottom boundary
                bound_nodes[eid][nid]  = 1;
                bound_values[eid][nid] = sin(M_PI * x/Lx);
            }
            else if (fabs(y - Ly) < zero_number)
            {
                // top boundary
                bound_nodes[eid][nid]  = 1;
                bound_values[eid][nid] = sin(M_PI * x/Lx) * exp(-M_PI);
            }
            else
            {
                // iterior
                bound_nodes[eid][nid]  = 0;
                bound_values[eid][nid] = -1000000;
            }
        }
    }

    // create lists of constrained dofs
    std::vector<par::ConstraintRecord<double, unsigned long int>> list_of_constraints;
    par::ConstraintRecord<double, unsigned long int> cdof;
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++)
        {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
            {
                if (bound_nodes[eid][(nid * NDOF_PER_NODE) + did] == 1)
                {
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
    constrainedDofs_ptr  = new unsigned long int[list_of_constraints.size()];
    prescribedValues_ptr = new double[list_of_constraints.size()];
    for (unsigned int i = 0; i < list_of_constraints.size(); i++)
    {
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

    meshMaps.set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, list_of_constraints.size());

    // declare aMat object =================================
    typedef par::
      aMat<par::aMatBased<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>
        aMatBased; // aMat type taking aMatBased as derived class
    typedef par::
      aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>
        aMatFree; // aMat type taking aMatBased as derived class

    aMatBased* stMatBased; // pointer of aMat taking aMatBased as derived
    aMatFree* stMatFree;   // pointer of aMat taking aMatFree as derived

    if (matType == 0) {
        // assign stMatBased to the derived class aMatBased
        stMatBased = new par::aMatBased<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
    } else {
        // assign stMatFree to the derived class aMatFree
        stMatFree = new par::aMatFree<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
        stMatFree->set_matfree_type((par::MATFREE_TYPE)matType);
        if ((matType == 3) || (matType == 4) || (matType == 5)){
            #ifdef USE_GPU
                stMatFree->set_num_streams(nStreams);
            #endif
        }
    }

    // set function to compute element matrix if using matrix-free
    if (matType == 2){
        stMatFree->set_element_matrix_function(&computeElemMat);
    }

    Vec rhs, out;
    par::create_vec(meshMaps, rhs);
    par::create_vec(meshMaps, out);

    // element stiffness matrix, force vector, and assembly...
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            unsigned long global_Id = globalMap[eid][nid];
            xe[nid * 2]             = (double)(global_Id % (Nex + 1)) * hx;
            xe[(nid * 2) + 1]       = (double)(global_Id / (Nex + 1)) * hy;
        }

        if (matType == 0){
            petsc_setup_time.start();
            petsc_elem_compute_time.start();
        } else {
            aMat_setup_time.start();
            aMat_elem_compute_time.start();
        }
        ke_quad4_eig(ke, xe, intData.Pts_n_Wts, NGT);
        if (matType == 0){
            petsc_elem_compute_time.stop();
        } else {
            aMat_elem_compute_time.stop();
        }

        // assemble ke
        if (matType == 0)
            stMatBased->set_element_matrix(eid, ke, 0, 0, 1);
        else
            stMatFree->set_element_matrix(eid, ke, 0, 0, 1);
        if (matType == 0){
            petsc_setup_time.stop();
        } else {
            aMat_setup_time.stop();
        }

        // for this example, no force vector
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            fe(nid) = 0.0;
        }

        // assemble fe
        par::set_element_vec(meshMaps, rhs, eid, fe, 0u, ADD_VALUES);
    }
    delete[] xe;

    PetscScalar norm, alpha = -1.0;

    // Pestc begins and completes assembling the global stiffness matrix
    if (matType == 0){
        petsc_setup_time.start();
        stMatBased->finalize();
        petsc_setup_time.stop();
    } else {
        aMat_setup_time.start();
        stMatFree->finalize(); // compute trace of matrix when using penalty method
        aMat_setup_time.stop();
    }

    // char fname[256];
    // if (matType == 0) {
    //     sprintf(fname, "matrix_%d.dat", matType);
    //     stMatBased->dump_mat(fname);
    // }

    // These are needed because we used ADD_VALUES for rhs when assembling
    // now we are going to use INSERT_VALUE for Fc in apply_bc_rhs
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // sprintf(fname, "rhs_beforebc_%d.dat", matType);
    // par::dump_vec(meshMaps, rhs, fname);

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
        // stMat->apply_bc_mat();
        petsc_setup_time.start();
        stMatBased->finalize();
        petsc_setup_time.stop();
    }

    // sprintf(fname,"rhs_%d.dat",matType);
    // stMat->petsc_dump_vec(rhs,fname);

    // solve
    if (matType == 0) {
        petsc_matvec_time.start();
    } else {
        aMat_matvec_time.start();
    }

    if (matType == 0)
        par::solve(*stMatBased, (const Vec)rhs, out);
    else
        par::solve(*stMatFree, (const Vec)rhs, out);

    if (matType == 0) {
        petsc_matvec_time.stop();
    } else {
        aMat_matvec_time.stop();
    }

    // computing time acrossing ranks and display
    if (matType == 0) {
        if (size > 1) {
            long double petsc_elem_compute_maxTime;
            long double petsc_setup_maxTime;
            long double petsc_matvec_maxTime;
            MPI_Reduce(&petsc_elem_compute_time.seconds, &petsc_elem_compute_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&petsc_setup_time.seconds, &petsc_setup_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&petsc_matvec_time.seconds, &petsc_matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                std::cout << "(1) PETSc elem compute time = " << petsc_elem_compute_maxTime << "\n";
                std::cout << "(2) PETSc setup time = " << petsc_setup_maxTime << "\n";
                std::cout << "(3) PETSc matvec time = " << petsc_matvec_maxTime << "\n";
                outFile << "PETSc, " << petsc_elem_compute_maxTime << "," << petsc_setup_maxTime << "," << petsc_matvec_maxTime << "\n";
            }
        } else {
            std::cout << "(1) PETSc elem compute time = " << petsc_elem_compute_time.seconds << "\n";
            std::cout << "(2) PETSc setup time = " << petsc_setup_time.seconds << "\n";
            std::cout << "(3) PETSc matvec time = " << petsc_matvec_time.seconds << "\n";
            outFile << "PETSc, " << petsc_elem_compute_time.seconds << ", " << petsc_setup_time.seconds << ", " << petsc_matvec_time.seconds << "\n";
        }
    } else if (matType == 1) {
        if (size > 1) {
            long double aMat_hybrid_elem_compute_maxTime;
            long double aMat_hybrid_setup_maxTime;
            long double aMat_hybrid_matvec_maxTime;
            MPI_Reduce(&aMat_elem_compute_time.seconds, &aMat_hybrid_elem_compute_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&aMat_setup_time.seconds, &aMat_hybrid_setup_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&aMat_matvec_time.seconds, &aMat_hybrid_matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                std::cout << "(1) aMat-hybrid elem compute time = " << aMat_hybrid_elem_compute_maxTime << "\n";
                std::cout << "(2) aMat-hybrid setup time = " << aMat_hybrid_setup_maxTime << "\n";
                std::cout << "(3) aMat-hybrid matvec time = " << aMat_hybrid_matvec_maxTime << "\n";
                outFile << "aMat-hybrid, " << aMat_hybrid_elem_compute_maxTime << ", " << aMat_hybrid_setup_maxTime << ", " << aMat_hybrid_matvec_maxTime << "\n";
            }
        } else {
            std::cout << "(1) aMat-hybrid elem compute time = " << aMat_elem_compute_time.seconds << "\n";
            std::cout << "(2) aMat-hybrid setup time = " << aMat_setup_time.seconds << "\n";
            std::cout << "(3) aMat-hybrid matvec time = " << aMat_matvec_time.seconds << "\n";
            outFile << "aMat-hybrid, " << aMat_elem_compute_time.seconds << ", " << aMat_setup_time.seconds << ", " << aMat_matvec_time.seconds << "\n";
        }
    } else if (matType == 2) {
        if (size > 1) {
            long double aMat_free_matvec_maxTime;
            MPI_Reduce(&aMat_matvec_time.seconds, &aMat_free_matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                std::cout << "(1) aMat-free matvec time = " << aMat_free_matvec_maxTime << "\n";
                outFile << "aMat-free, " << aMat_free_matvec_maxTime << "\n";
            }
        } else {
            std::cout << "(1) aMat-free matvec time = " << aMat_matvec_time.seconds << "\n";
            outFile << "aMat-free, " << aMat_matvec_time.seconds << "\n";
        }
    } else if ((matType == 3) || (matType == 4) || (matType == 5)) {
        if (size > 1) {
            long double aMat_elem_compute_maxTime;
            long double aMat_setup_maxTime;
            long double aMat_matvec_maxTime;
            MPI_Reduce(&aMat_elem_compute_time.seconds, &aMat_elem_compute_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&aMat_setup_time.seconds, &aMat_setup_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&aMat_matvec_time.seconds, &aMat_matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                std::cout << "(1) aMatGpu elem compute time = " << aMat_elem_compute_maxTime << "\n";
                std::cout << "(2) aMatGpu setup time = " << aMat_setup_maxTime << "\n";
                std::cout << "(3) aMatGpu matvec time = " << aMat_matvec_maxTime << "\n";
                outFile << "aMatGpu, " << aMat_elem_compute_maxTime << ", " << aMat_setup_maxTime << ", " << aMat_matvec_maxTime << "\n";
            }
        } else {
            std::cout << "(1) aMatGpu elem compute time = " << aMat_elem_compute_time.seconds << "\n";
            std::cout << "(2) aMatGpu setup time = " << aMat_setup_time.seconds << "\n";
            std::cout << "(3) aMatGpu matvec time = " << aMat_matvec_time.seconds << "\n";
            outFile << "aMatGpu, " << aMat_elem_compute_time.seconds << ", " << aMat_setup_time.seconds << ", " << aMat_matvec_time.seconds << "\n";
        }
    }
    if (rank == 0) outFile.close();

    VecNorm(out, NORM_2, &norm);
    if (!rank) {
        printf("L2 norm of computed solution = %f\n", norm);
    }

    // exact solution...
    Vec sol_exact;
    PetscInt rowId;
    par::create_vec(meshMaps, sol_exact);

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
    delete[] localDofMap;
    delete[] nnodeCount;
    delete[] nnodeOffset;
    delete[] local2GlobalDofMap;
    delete[] ndofs_per_element;
    if (matType == 0) {
        delete stMatBased;
    } else {
        delete stMatFree;
    }
    
    VecDestroy(&rhs);
    VecDestroy(&out);
    VecDestroy(&sol_exact);
    PetscFinalize();
    return 0;
}