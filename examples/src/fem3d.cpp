/**
 * @file fem3d.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 * @author Milinda Fernando milinda@cs.utah.edu
 *
 * @brief Example of solving 3D Poisson equation by FEM, in parallel, using Petsc
 * @brief grad^2(u) + sin(2*M_PI*x) * sin(2*M_PI*y) * sin(2*M_PI*z) = 0 on domain L x L x L
 * @brief BCs u = 0 on all boundaries
 * @brief Exact solution: (L^2 / (12.0 * M_PI * M_PI)) * sin(2 * M_PI * x/L) * sin(2 * M_PI * y/L) * sin(2 * M_PI * z/L);
 *
 * @version 0.1
 * @date 2018-11-30
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#include "fem3d.hpp"
AppData fem3dAppData;

// number of cracks allowed in 1 element
#define MAX_CRACK_LEVEL 0

// max number of block dimensions in one cracked element
#define MAX_BLOCKS_PER_ELEMENT (1u << MAX_CRACK_LEVEL)

//////////////////////////////////////////////////////////////////////////////////////////////////////

void usage()
{
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  fem3d <Nex> <Ney> <Nez> <matrix based/free> <bc method>\n";
    std::cout << "\n";
    std::cout << "     1) Nex: Number of elements in x\n";
    std::cout << "     2) Ney: Number of elements in y\n";
    std::cout << "     3) Nez: Number of elements in z\n";
    std::cout << "     4) method (0, 1, 2, 3, 4, 5) \n";
    std::cout << "     5) use identity-matrix: 0    use penalty method: 1 \n";
    std::cout << "     6) number of streams (used in method 3, 4, 5)\n";
    std::cout << "     7) name of output file\n";
    std::cout << "\n";
    exit(0);
}

void compute_nodal_body_force(double* xe, unsigned int nnode, double L, double* be) {
    double x, y, z;
    for (unsigned int nid = 0; nid < nnode; nid++)
    {
        x       = xe[nid * 3];
        y       = xe[nid * 3 + 1];
        z       = xe[nid * 3 + 2];
        be[nid] = sin(2 * M_PI * (x/L)) * sin(2 * M_PI * (y/L)) * sin(2 * M_PI * (z/L));
    }
}

// function to compute element matrix, used in method 2
void computeElemMat(unsigned int eid, double *ke, double* xe) {
    
    const double hx = fem3dAppData.hx;
    const double hy = fem3dAppData.hy;
    const double hz = fem3dAppData.hz;
    
    const unsigned int Nex  = fem3dAppData.Nex;
    const unsigned int Ney  = fem3dAppData.Ney;
    const unsigned int Nez  = fem3dAppData.Nez;

    const unsigned int NNODE_PER_ELEM = fem3dAppData.NNODE_PER_ELEM;

    // get coordinates of all nodes
    unsigned long nid;
    for (unsigned int n = 0; n < NNODE_PER_ELEM; n++) {
        nid = fem3dAppData.globalMap[eid][n];
        xe[n * 3] = (double)(nid % (Nex + 1)) * hx;
        xe[(n * 3) + 1] = (double)((nid % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
        xe[(n * 3) + 2] = (double)(nid / ((Nex + 1) * (Ney + 1))) * hz;
    }
    ke_hex8(ke, xe, fem3dAppData.intData->Pts_n_Wts, fem3dAppData.NGT);

    return;
} // computeElemMat

//////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    // User provides: Nex - number of elements (global) in x direction
    //                Ney - number of elements (global) in y direction
    //                Nez - number of elements (global) in z direction
    //                flag = 1 --> matrix-free method; 0 --> matrix-based method
    //                bcMethod = 0 --> identity matrix method; 1 --> penalty method
    if (argc < 6)
    {
        usage();
    }

    int rc;
    double zmin, zmax;

    double x, y, z;
    double hx, hy, hz;

    unsigned int emin = 0, emax = 0;
    unsigned long nid, eid;
    const unsigned int NDOF_PER_NODE  = 1; // number of dofs per node
    const unsigned int NDIM           = 3; // number of dimension
    const unsigned int NNODE_PER_ELEM = 8; // number of nodes per element

    // number of (global) elements in x, y and z directions
    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]);
    const unsigned int Nez = atoi(argv[3]);

    // domain sizes: L x L x L - length of the (global) domain in x, y, z direction
    const double L = 100.0;

    // element sizes
    hx = L / double(Nex); // element size in x direction
    hy = L / double(Ney); // element size in y direction
    hz = L / double(Nez); // element size in z direction

    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);

    // set application data to AppData
    fem3dAppData.Nex = Nex;
    fem3dAppData.Ney = Ney;
    fem3dAppData.Nez = Nez;
    fem3dAppData.L = L;
    fem3dAppData.hx = hx;
    fem3dAppData.hy = hy;
    fem3dAppData.hz = hz;
    fem3dAppData.NGT = NGT;
    fem3dAppData.intData = &intData;

    // 05.19.20 only use Eigen matrix
    const bool useEigen = true;

    const unsigned int matType = atoi(argv[4]); // approach (matrix based/free)
    const unsigned int bcMethod = atoi(argv[5]); // method of applying BC
    const unsigned int nStreams = atoi(argv[6]); // number of streams used for method 3, 4, 5

    // element matrix (contains multiple matrix blocks)
    std::vector<Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, NDOF_PER_NODE * NNODE_PER_ELEM>> kee;
    kee.resize(MAX_BLOCKS_PER_ELEMENT * MAX_BLOCKS_PER_ELEMENT);

    // nodal coordinates of element
    double* xe = new double[NDIM * NNODE_PER_ELEM];

    // nodal body force
    double* be = new double[NNODE_PER_ELEM];

    // matrix block
    double* ke = new double[(NDOF_PER_NODE * NNODE_PER_ELEM) * (NDOF_PER_NODE * NNODE_PER_ELEM)];

    // element force vector (contains multiple vector blocks)
    std::vector<Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1>> fee;
    fee.resize(MAX_BLOCKS_PER_ELEMENT);
    
    const double zero_number = 1E-12;

    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Status Stat;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // output file in csv format, open in append mode
    std::ofstream outFile;
    const char* filename = argv[6];
    if (rank == 0)
        outFile.open(filename, std::fstream::app);

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

    if (!rank) {
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNex : " << Nex << " Ney: " << Ney << " Nez: " << Nez << "\n";
        std::cout << "\t\tL : " << L << " L: " << L << " L: " << L << "\n";
        std::cout << "\t\tMethod  = " << matType << "\n";
        std::cout << "\t\tBC method (0 = 'identity-matrix'; 1 = penalty): " << bcMethod << "\n";
        if ((matType == 3) || (matType == 4) || (matType == 5)){
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

    if (rank == 0) {
        if (size > Nez) {
            printf("The number of processes must be less than or equal Nez, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // partition number of elements in z direction...
    unsigned int nelem_z;
    // minimum number of elements in z-dir for each rank
    unsigned int nzmin = Nez / size;
    // remaining
    unsigned int nRemain = Nez % size;
    // distribute nRemain uniformly from rank = 0 up to rank = nRemain - 1
    if (rank < nRemain) {
        nelem_z = nzmin + 1;
    } else {
        nelem_z = nzmin;
    }
    if (rank < nRemain) {
        emin = rank * nzmin + rank;
    } else {
        emin = rank * nzmin + nRemain;
    }
    emax = emin + nelem_z - 1;

    // number of elements (partition of processes is only in z direction)
    unsigned int nelem_y = Ney;
    unsigned int nelem_x = Nex;
    unsigned int nelem   = (nelem_x) * (nelem_y) * (nelem_z);

    // number of nodes, specify what nodes I own based
    unsigned int nnode, nnode_x, nnode_y, nnode_z;
    if (rank == 0) {
        nnode_z = nelem_z + 1;
    } else {
        nnode_z = nelem_z;
    }
    nnode_y = nelem_y + 1;
    nnode_x = nelem_x + 1;
    nnode   = (nnode_x) * (nnode_y) * (nnode_z);

    // map from local nodes to global nodes
    unsigned int* ndofs_per_element = new unsigned int[nelem];
    for (unsigned e = 0; e < nelem; e++) {
        ndofs_per_element[e] = NNODE_PER_ELEM;
    }

    unsigned long int** globalMap;
    globalMap = new unsigned long int*[nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        globalMap[e] = new unsigned long int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned k = 0; k < nelem_z; k++) {
        for (unsigned j = 0; j < nelem_y; j++) {
            for (unsigned i = 0; i < nelem_x; i++) {
                eid = nelem_x * nelem_y * k + nelem_x * j + i;
                globalMap[eid][0] =
                  (emin * (Nex + 1) * (Ney + 1) + i) + j * (Nex + 1) + k * (Nex + 1) * (Ney + 1);
                globalMap[eid][1] = globalMap[eid][0] + 1;
                globalMap[eid][3] = globalMap[eid][0] + (Nex + 1);
                globalMap[eid][2] = globalMap[eid][3] + 1;
                globalMap[eid][4] = globalMap[eid][0] + (Nex + 1) * (Ney + 1);
                globalMap[eid][5] = globalMap[eid][4] + 1;
                globalMap[eid][7] = globalMap[eid][4] + (Nex + 1);
                globalMap[eid][6] = globalMap[eid][7] + 1;
            }
        }
    }

    // set globalMap to AppData so that we can used in function compute element stiffness matrix
    fem3dAppData.globalMap = globalMap;

    // build localDofMap from globalMap
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalDofs;
    unsigned long gNodeId;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;

    unsigned int* nnodeCount  = new unsigned int[size];
    unsigned int* nnodeOffset = new unsigned int[size];

    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    nnodeOffset[0] = 0;
    for (unsigned int i = 1; i < size; i++)
    {
        nnodeOffset[i] = nnodeOffset[i - 1] + nnodeCount[i - 1];
    }
    unsigned int ndofs_total;
    ndofs_total = nnodeOffset[size - 1] + nnodeCount[size - 1];
    if (rank == 0)
        printf("Total dofs = %d\n", ndofs_total);

    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < 8; nid++)
        {
            gNodeId = globalMap[eid][nid];
            if (gNodeId < nnodeOffset[rank])
            {
                preGhostGIds.push_back(gNodeId);
            }
            else if (gNodeId >= nnodeOffset[rank] + nnode)
            {
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
    for (unsigned int e = 0; e < nelem; e++)
    {
        localDofMap[e] = new unsigned int[8];
    }

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int i = 0; i < 8; i++)
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
        for (unsigned int nid = 0; nid < 8; nid++)
        {
            gNodeId                                   = globalMap[eid][nid];
            local2GlobalDofMap[localDofMap[eid][nid]] = gNodeId;
        }
    }

    // construct element maps of constrained DoFs and prescribed values
    unsigned int** bound_nodes = new unsigned int*[nelem];
    double** bound_values      = new double*[nelem];
    for (unsigned int e = 0; e < nelem; e++) {
        bound_nodes[e]  = new unsigned int[ndofs_per_element[e]];
        bound_values[e] = new double[ndofs_per_element[e]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int n = 0; n < ndofs_per_element[eid]; n++) {
            nid = globalMap[eid][n];

            // get node coordinates
            x = (double)(nid % (Nex + 1)) * hx;
            y = (double)((nid % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(nid / ((Nex + 1) * (Ney + 1))) * hz;

            // specify boundary nodes
            if ((fabs(x) < zero_number) || (fabs(x - L) < zero_number) ||
                (fabs(y) < zero_number) || (fabs(y - L) < zero_number) ||
                (fabs(z) < zero_number) || (fabs(z - L) < zero_number)) {
                bound_nodes[eid][n]  = 1; // boundary
                bound_values[eid][n] = 0; // prescribed value
            } else {
                bound_nodes[eid][n]  = 0;        // interior
                bound_values[eid][n] = -1000000; // for testing
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
    constrainedDofs_ptr  = new unsigned long int[list_of_constraints.size()];
    prescribedValues_ptr = new double[list_of_constraints.size()];
    for (unsigned int i = 0; i < list_of_constraints.size(); i++) {
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

    if (matType == 0){
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

    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact;
    par::create_vec(meshMaps, rhs);
    par::create_vec(meshMaps, out);
    par::create_vec(meshMaps, sol_exact);

    // compute element stiffness matrix and assemble global stiffness matrix and load vector
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int n = 0; n < NNODE_PER_ELEM; n++) {
            nid = globalMap[eid][n];

            // get node coordinates
            x               = (double)(nid % (Nex + 1)) * hx;
            y               = (double)((nid % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
            z               = (double)(nid / ((Nex + 1) * (Ney + 1))) * hz;
            xe[n * 3]       = x;
            xe[(n * 3) + 1] = y;
            xe[(n * 3) + 2] = z;
        }

        // compute element stiffness matrix
        if (matType == 0){
            petsc_setup_time.start();
            petsc_elem_compute_time.start();
        } else {
            aMat_setup_time.start();
            aMat_elem_compute_time.start();
        }
        if (useEigen) {
            ke_hex8_eig(kee[0], xe, intData.Pts_n_Wts, NGT);
        } else {
            printf("Error: no longer use non-Eigen matrix for element stiffness matrix \n");
            exit(0);
            //ke_hex8(ke, xe, intData.Pts_n_Wts, NGT);
        }
        if (matType == 0){
            petsc_elem_compute_time.stop();
        } else {
            aMat_elem_compute_time.stop();
        }

        // assemble element stiffness matrix to global K
        if (matType == 0)
            stMatBased->set_element_matrix(eid, kee[0], 0, 0, 1);
        else
            stMatFree->set_element_matrix(eid, kee[0], 0, 0, 1);

        if (matType == 0){
            petsc_setup_time.stop();
        } else {
            aMat_setup_time.stop();
        }

        // compute nodal values of body force
        compute_nodal_body_force(xe, 8, L, be);

        // compute element load vector due to body force
        fe_hex8_eig(fee[0], xe, be, intData.Pts_n_Wts, NGT);

        // assemble element load vector to global F
        par::set_element_vec(meshMaps, rhs, eid, fee[0], 0u, ADD_VALUES);
    }
    delete[] ke;
    delete[] xe;
    delete[] be;

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
    
    // apply bc to the matrix
    if (matType == 0) {
        petsc_setup_time.start();
        stMatBased->finalize();
        petsc_setup_time.stop();
    }

    // ====================== profiling matvec ====================================
    // generate random vector of length = number of owned dofs
    /* const unsigned int numDofsTotal = meshMaps.get_NumDofsTotal();
    const unsigned int numDofs = meshMaps.get_NumDofs();

    double* X = (double*) malloc(sizeof(double) * (numDofsTotal));
    for (unsigned int i = 0; i < (numDofsTotal); i++){
        //X[i] = (double)std::rand()/(double)(RAND_MAX/5.0);
        X[i] = 1.0;
    }
    // result vector Y = [K] * X
    double* Y = (double*) malloc(sizeof(double) * (numDofsTotal));

    // total number of matvec's we want to profile
    const unsigned int num_matvecs = 10;
    if (rank == 0) printf("Number of matvecs= %d\n", num_matvecs);

    if( matType==0) {
        Vec petsc_X, petsc_Y;
        par::create_vec(meshMaps, petsc_X, 1.0);
        par::create_vec(meshMaps, petsc_Y);

        for (unsigned int i = 0; i < num_matvecs; i++){
            petsc_matvec_time.start();
            stMatBased->matmult(petsc_Y, petsc_X);
            //VecAssemblyBegin(petsc_X);
            //VecAssemblyEnd(petsc_X);
            petsc_matvec_time.stop();
            VecSwap(petsc_Y, petsc_X);
        }
        VecDestroy(&petsc_X);
        VecDestroy(&petsc_Y);

    } else {
        for (unsigned int i = 0; i < num_matvecs; i++){
            
            aMat_matvec_time.start();     
                stMatFree->matvec(Y, X, true);
            aMat_matvec_time.stop();
            double * temp = X;
            X = Y;
            Y = temp;
        }
    }

    free (Y);
    free (X); */

    // ======================= solve =================================================
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
            long double aMat_elem_compute_maxTime;
            long double aMat_setup_maxTime;
            long double aMat_matvec_maxTime;
            MPI_Reduce(&aMat_elem_compute_time.seconds, &aMat_elem_compute_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&aMat_setup_time.seconds, &aMat_setup_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&aMat_matvec_time.seconds, &aMat_matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                std::cout << "(1) aMat-hybrid elem compute time = " << aMat_elem_compute_maxTime << "\n";
                std::cout << "(2) aMat-hybrid setup time = " << aMat_setup_maxTime << "\n";
                std::cout << "(3) aMat-hybrid matvec time = " << aMat_matvec_maxTime << "\n";
                outFile << "aMat-hybrid, " << aMat_elem_compute_maxTime << ", " << aMat_setup_maxTime << ", " << aMat_matvec_maxTime << "\n";
            }
        } else {
            std::cout << "(1) aMat-hybrid elem compute time = " << aMat_elem_compute_time.seconds << "\n";
            std::cout << "(2) aMat-hybrid setup time = " << aMat_setup_time.seconds << "\n";
            std::cout << "(3) aMat-hybrid matvec time = " << aMat_matvec_time.seconds << "\n";
            outFile << "aMat-hybrid, " << aMat_elem_compute_time.seconds << ", " << aMat_setup_time.seconds << ", " << aMat_matvec_time.seconds << "\n";
        }
    } else if (matType == 2) {
        if (size > 1) {
            long double aMat_matvec_maxTime;
            MPI_Reduce(&aMat_matvec_time.seconds, &aMat_matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                std::cout << "(3) aMat-free matvec time = " << aMat_matvec_maxTime << "\n";
                outFile << "aMat-free, " << aMat_matvec_maxTime << "\n";
            }
        } else {
            std::cout << "(3) aMat-free matvec time = " << aMat_matvec_time.seconds << "\n";
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

    PetscScalar norm, alpha = -1.0;

    VecNorm(out, NORM_2, &norm);
    if (!rank) {
        printf("L2 norm of computed solution = %f\n", norm);
    }

    // compute exact solution for comparison
    Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1> e_exact;

    for (unsigned int e = 0; e < nelem; e++) {
        for (unsigned int n = 0; n < NNODE_PER_ELEM; n++) {
            // global node ID
            nid = globalMap[e][n];
            // nodal coordinates
            x = (double)(nid % (Nex + 1)) * hx;
            y = (double)((nid % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(nid / ((Nex + 1) * (Ney + 1))) * hz;

            // exact solution at node n (and apply BCs)
            if ((std::abs(x) < zero_number) || (std::abs(x - L) < zero_number) ||
                (std::abs(y) < zero_number) || (std::abs(y - L) < zero_number) ||
                (std::abs(z) < zero_number) || (std::abs(z - L) < zero_number)) {
                e_exact(n) = 0.0; // boundary
            } else {
                e_exact(n) = ((L*L) / (12.0 * M_PI * M_PI)) * sin(2 * M_PI * (x/L)) * sin(2 * M_PI * (y/L)) *
                             sin(2 * M_PI * (z/L));
            }
        }
        // set exact solution to Pestc vector
        par::set_element_vec(meshMaps, sol_exact, e, e_exact, 0u, INSERT_VALUES);
    }

    // Pestc begins and completes assembling the exact solution
    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

    VecNorm(sol_exact, NORM_2, &norm);
    if (!rank) {
        printf("L2 norm of exact solution = %f\n", norm);
    }
    // stMat.dump_vec("exact_vec.dat", sol_exact);

    
    // compute the norm of error
    VecAXPY(sol_exact, alpha, out);
    VecNorm(sol_exact, NORM_INFINITY, &norm);

    if (rank == 0) {
        printf("L_inf norm of error = %20.10f\n", norm);
    }

    #ifdef AMAT_PROFILER
    stMat->profile_dump(std::cout);
    #endif

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] globalMap[eid];
        delete[] bound_nodes[eid];
        delete[] bound_values[eid];
    }
    delete[] globalMap;
    delete[] bound_nodes;
    delete[] bound_values;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] localDofMap[eid];
    }
    delete[] localDofMap;

    delete[] nnodeCount;
    delete[] nnodeOffset;

    // delete [] etype;
    // delete [] kee;
    delete[] local2GlobalDofMap;
    delete[] ndofs_per_element;

    delete[] constrainedDofs_ptr;
    delete[] prescribedValues_ptr;

    if (matType == 0) {
        delete stMatBased;
    } else {
        delete stMatFree;
    }

    // clean up Pestc vectors
    VecDestroy(&out);
    VecDestroy(&sol_exact);
    VecDestroy(&rhs);
    
    PetscFinalize();

    return 0;
}
