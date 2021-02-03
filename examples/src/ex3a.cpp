/**
 * @file ex3a.cpp: same as ex3 but all ranks own the same number of nodes
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Example: Stretching of a prismatic bar by its own weight (Timoshenko page 246)
 * @brief Exact solution (origin at centroid of bottom face)
 * @brief    uniform stress s_zz = rho * g * z
 * @brief    displacement u = -(nu * rho * g/E) * x * z
 * @brief    displacement v = -(nu * rho * g/E) * y * z
 * @brief    displacement w = (rho * g/2/E)(z^2 - Lz^2) + (nu * rho * g)/2/E(x^2 + y^2)
 * @brief Boundary condition: traction tz = rho * g * Lz applied on top surface + blocking rigid
 * motions
 * @brief Partition of elements in z direction: owned elements in z direction ~ Nez/(number of
 * ranks)
 * @brief Size of the domain: Lx = Ly = 1; Lz = 4.0
 *
 * @version 0.1
 * @date 2020-02-26
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#include "ex3.hpp"
AppData Ex3AppData;

// number of cracks allowed in 1 element
#define MAX_CRACK_LEVEL 3

// max number of block dimensions in one cracked element
#define MAX_BLOCKS_PER_ELEMENT (1u << MAX_CRACK_LEVEL)

//////////////////////////////////////////////////////////////////////////////////////////////////////

void usage() {
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  ex3 <Nex> <Ney> <Nez> <matrix based/free> <bc-method>\n";
    std::cout << "\n";
    std::cout << "     1) Nex: Number of elements in X\n";
    std::cout << "     2) Ney: Number of elements in y\n";
    std::cout << "     3) Nez: Number of elements in z\n";
    std::cout << "     4) method (0, 1, 2, 3, 4, 5)\n";
    std::cout << "     5) use identity-matrix: 0  ;  use penalty method: 1 \n";
    std::cout << "     6) number of streams (used in method 3, 4, 5)\n";
    std::cout << "     7) name of output file\n";
    exit(0);
}

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EigenMat;

// function to compute element matrix used for method = 2
void computeElemMat(unsigned int eid, double *ke, double* xe) {
    
    const double hx = Ex3AppData.hx;
    const double hy = Ex3AppData.hy;
    const double hz = Ex3AppData.hz;
    
    const unsigned int Nex  = Ex3AppData.Nex;
    const unsigned int Ney  = Ex3AppData.Ney;
    const unsigned int Nez  = Ex3AppData.Nez;
    const unsigned int NDOF_PER_NODE = Ex3AppData.NDOF_PER_NODE;
    const double Lx = Ex3AppData.Lx;
    const double Ly = Ex3AppData.Ly;

    // get coordinates of all nodes
    for (unsigned int nid = 0; nid < Ex3AppData.NNODE_PER_ELEM; nid++) {
        unsigned int gNodeId = Ex3AppData.ElementToGIDNode[eid][nid];
        xe[nid * NDOF_PER_NODE] = (double)(gNodeId % (Nex + 1)) * hx - Lx/2;
        xe[(nid * NDOF_PER_NODE) + 1] = (double)((gNodeId % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy - Ly/2;
        xe[(nid * NDOF_PER_NODE) + 2] = (double)(gNodeId / ((Nex + 1) * (Ney + 1))) * hz;
    }

    ke_hex8_iso(ke, xe, Ex3AppData.E, Ex3AppData.nu, Ex3AppData.intData->Pts_n_Wts, Ex3AppData.NGT);
    
    return;
} // computeElemMat

//////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    // User provides: 1) Nex - number of elements (global) in x direction
    //                2) Ney - number of elements (global) in y direction
    //                3) Nez - number of elements (global) in z direction
    //                4) method (0, 1, 2, 3, 4, 5)
    //                5)method for BCs (0 = identity matrix method; 1 = penalty method)
    //                6) number of streams (applied for method = 3, 4, 5)

    int rc;

    double x, y, z;
    double hx, hy, hz;

    unsigned int emin = 0;

    const unsigned int NDOF_PER_NODE  = 3; // number of dofs per node
    const unsigned int NDIM           = 3; // number of dimension
    const unsigned int NNODE_PER_ELEM = 8; // number of nodes per element

    // material properties of alumina
    const double E = 1.0E6;
    const double nu = 0.3;
    const double rho = 1.0;
    const double g = 1.0;
    // domain sizes: Lx, Ly, Lz - length of the (global) domain in x/y/z direction
    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);

    // number of (global) elements in x, y and z directions
    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]);
    const unsigned int Nez = atoi(argv[3]);

    const unsigned int matType = atoi(argv[4]); // matrix-free method (0, 1, 2, 3, 4, 5)
    const unsigned int bcMethod = atoi(argv[5]); // method of applying BC
    const unsigned int nStreams = atoi(argv[6]); // number of streams used for method 3, 4, 5

    // element sizes
    hx = Lx / double(Nex); // element size in x direction
    hy = Ly / double(Ney); // element size in y direction
    hz = Lz / double(Nez); // element size in z direction

    // 05.19.20 only use Eigen matrix
    const bool useEigen = true;
    
    Ex3AppData.E = E;
    Ex3AppData.nu = nu;
    Ex3AppData.rho = rho;
    Ex3AppData.g = g;

    Ex3AppData.Lx = Lx;
    Ex3AppData.Ly = Ly;
    Ex3AppData.Lz = Lz;
    
    Ex3AppData.Nex = Nex;
    Ex3AppData.Ney = Ney;
    Ex3AppData.Nez = Nez;

    Ex3AppData.hx = hx;
    Ex3AppData.hy = hy;
    Ex3AppData.hz = hz;

    Ex3AppData.NGT = NGT;
    Ex3AppData.intData = &intData;
    

    const double zero_number = 1E-12;

    PetscInitialize(&argc, &argv, NULL, NULL);

    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Status Stat;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // output file in csv format, open in append mode
    std::ofstream outFile;
    const char* filename = argv[7];
    if (rank == 0)
        outFile.open(filename, std::fstream::app);

    if (argc < 8) {
        usage();
    }

    // element matrix and vector
    EigenMat** kee;

    std::vector<Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1>> fee;
    fee.resize(MAX_BLOCKS_PER_ELEMENT);

    kee = new EigenMat*[MAX_BLOCKS_PER_ELEMENT];
    for (unsigned int i = 1; i < MAX_BLOCKS_PER_ELEMENT; i++) {
        kee[i] = nullptr;
    }
    kee[0] = new EigenMat;

    // nodal coordinates of element
    double* xe = new double[NDIM * NNODE_PER_ELEM];

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
        std::cout << "\t\tLx : " << Lx << " Ly: " << Ly << " Lz: " << Lz << "\n";
        std::cout << "\t\tMethod (0 = matrix based; 1 = hybrid; 2 = matrix free; 3 = gpu matvec) = " << matType << "\n";
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

    // for fixing rigid motions at centroid of the top/bottom face
    // number of elements in x and y directions must be even numbers
    if ((Nex % 2 != 0) || (Ney % 2 != 0)) {
        if (!rank) {
            printf("Number of elements in x and y must be even numbers, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // find min & max element index in z direction
    // partition number of elements in z direction
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

    // number of owned elements
    unsigned int nelem_y = Ney;
    unsigned int nelem_x = Nex;
    unsigned int nelem   = (nelem_x) * (nelem_y) * (nelem_z);

    // compute number of owned nodes
    unsigned int nnode;
    // number of nodes on each interface
    const unsigned int nInterface = (nelem_x + 1) * (nelem_y + 1);
    // total interface nodes
    const unsigned int nTotalInterface = (size - 1) * nInterface;
    // remainder after equally dividing to ranks
    const unsigned int nRemainInterface = nTotalInterface % size;

    if (rank < nRemainInterface) {
        nnode = (nelem_x + 1)*(nelem_y + 1)*(nelem_z + 1) - ((nTotalInterface / size) + 1);
    } else {
        nnode = (nelem_x + 1)*(nelem_y + 1)*(nelem_z + 1) - (nTotalInterface / size);
    }
    printf("=== rank %d, nnode= %d\n", rank, nnode);

    // map from elemental node to global nodes
    unsigned long gNodeId;
    unsigned long int** globalMap;
    globalMap = new unsigned long int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalMap[eid] = new unsigned long int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned k = 0; k < nelem_z; k++) {
        for (unsigned j = 0; j < nelem_y; j++) {
            for (unsigned i = 0; i < nelem_x; i++) {
                unsigned int elemID = nelem_x * nelem_y * k + nelem_x * j + i;
                globalMap[elemID][0] =
                  (emin * (Nex + 1) * (Ney + 1) + i) + j * (Nex + 1) + k * (Nex + 1) * (Ney + 1);
                globalMap[elemID][1] = globalMap[elemID][0] + 1;
                globalMap[elemID][3] = globalMap[elemID][0] + (Nex + 1);
                globalMap[elemID][2] = globalMap[elemID][3] + 1;
                globalMap[elemID][4] = globalMap[elemID][0] + (Nex + 1) * (Ney + 1);
                globalMap[elemID][5] = globalMap[elemID][4] + 1;
                globalMap[elemID][7] = globalMap[elemID][4] + (Nex + 1);
                globalMap[elemID][6] = globalMap[elemID][7] + 1;
            }
        }
    }

    // map from elemental dof to global dof
    unsigned long int** globalDofMap;
    globalDofMap = new unsigned long int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalDofMap[eid] =
          new unsigned long int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM * NDOF_PER_NODE];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                globalDofMap[eid][(nid * NDOF_PER_NODE) + did] =
                  (globalMap[eid][nid] * NDOF_PER_NODE) + did;
            }
        }
    }

    // set globalMap to AppData so that we can used in function compute element stiffness matrix
    Ex3AppData.ElementToGIDNode = globalMap;

    // build localMap from globalMap (this is just to conform with aMat interface used for bsamxx)
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalNodes;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;

    // counts of owned nodes: nnodeCount[0] = nnode0, nnodeCount[1] = nnode1, ...
    unsigned int* nnodeCount = new unsigned int[size];
    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    // offset of nnodeCount
    unsigned int* nnodeOffset = new unsigned int[size];
    nnodeOffset[0] = 0;
    for (unsigned int i = 1; i < size; i++) {
        nnodeOffset[i] = nnodeOffset[i - 1] + nnodeCount[i - 1];
    }
    // total number of nodes for all ranks
    unsigned long int nnode_total, ndofs_total;
    nnode_total = nnodeOffset[size - 1] + nnodeCount[size - 1];
    ndofs_total = nnode_total * NDOF_PER_NODE;
    if (rank == 0)
        printf("Total dofs = %d\n", ndofs_total);


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
    postGhostGIds.erase(std::unique(postGhostGIds.begin(), postGhostGIds.end()),
                        postGhostGIds.end());

    // number of pre and post ghost nodes of my rank
    numPreGhostNodes  = preGhostGIds.size();
    numPostGhostNodes = postGhostGIds.size();

    // number of local nodes
    numLocalNodes = numPreGhostNodes + nnode + numPostGhostNodes;

    // number of local dofs
    unsigned int numLocalDofs = numLocalNodes * NDOF_PER_NODE;

    // map from elemental nodes to local nodes
    unsigned int** localMap;
    localMap = new unsigned int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localMap[eid] = new unsigned int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int i = 0; i < NNODE_PER_ELEM; i++) {
            gNodeId = globalMap[eid][i];
            if ((gNodeId >= nnodeOffset[rank]) && (gNodeId < (nnodeOffset[rank] + nnode))) {
                // nid is owned by me
                localMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
            } else if (gNodeId < nnodeOffset[rank]) {
                // nid is owned by someone before me
                const unsigned int lookUp =
                  std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) -
                  preGhostGIds.begin();
                localMap[eid][i] = lookUp;
            } else if (gNodeId >= (nnodeOffset[rank] + nnode)) {
                // nid is owned by someone after me
                const unsigned int lookUp =
                  std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), gNodeId) -
                  postGhostGIds.begin();
                localMap[eid][i] = numPreGhostNodes + nnode + lookUp;
            }
        }
    }

    // map from local dof to global dof
    unsigned int** localDofMap;
    localDofMap = new unsigned int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localDofMap[eid] =
          new unsigned int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM * NDOF_PER_NODE];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                localDofMap[eid][(nid * NDOF_PER_NODE) + did] =
                  (localMap[eid][nid] * NDOF_PER_NODE) + did;
            }
        }
    }

    // local node to global node map
    unsigned long* local2GlobalMap = new unsigned long[numLocalNodes];
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
                local2GlobalDofMap[localDofMap[eid][(nid * NDOF_PER_NODE) + did]] =
                  globalDofMap[eid][(nid * NDOF_PER_NODE) + did];
            }
        }
    }

    // start and end (inclusive) global nodes owned by my rank
    unsigned long start_global_node, end_global_node;
    start_global_node = nnodeOffset[rank];
    end_global_node   = start_global_node + (nnode - 1);

    // start and end (inclusive) global dofs owned by my rank
    unsigned long start_global_dof, end_global_dof;
    start_global_dof = start_global_node * NDOF_PER_NODE;
    end_global_dof   = (end_global_node * NDOF_PER_NODE) + (NDOF_PER_NODE - 1);

    // number of dofs per element
    unsigned int* ndofs_per_element = new unsigned int[nelem];
    for (unsigned eid = 0; eid < nelem; eid++) {
        ndofs_per_element[eid] = NNODE_PER_ELEM * NDOF_PER_NODE;
    }

    // elemental boundary dofs and prescribed value
    unsigned int** bound_dofs = new unsigned int*[nelem];
    double** bound_values     = new double*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        bound_dofs[eid]   = new unsigned int[ndofs_per_element[eid]];
        bound_values[eid] = new double[ndofs_per_element[eid]];
    }

    // construct elemental constrained DoFs and prescribed values
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {

            // global node id of elemental node n
            gNodeId = globalMap[eid][nid];

            // compute nodal coordinate
            x = (double)(gNodeId % (Nex + 1)) * hx;
            y = (double)((gNodeId % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(gNodeId / ((Nex + 1) * (Ney + 1))) * hz;

            // translate origin to center of bottom face
            x = x - Lx / 2;
            y = y - Ly / 2;

            // node at centroid of top face: fix all directions
            if ((fabs(x) < zero_number) && (fabs(y) < zero_number) && (fabs(z - Lz) < zero_number)) {
                bound_dofs[eid][(nid * NDOF_PER_NODE)]     = 1;
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 1] = 1;
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 2] = 1;

                bound_values[eid][(nid * NDOF_PER_NODE)]     = 0.0;
                bound_values[eid][(nid * NDOF_PER_NODE) + 1] = 0.0;
                bound_values[eid][(nid * NDOF_PER_NODE) + 2] = 0.0;
            } else {
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                    bound_dofs[eid][(nid * NDOF_PER_NODE) + did]   = 0; // free dof
                    bound_values[eid][(nid * NDOF_PER_NODE) + did] = -1000000;
                }
            }

            // node at centroid of bottom surface: fix in x and y
            if ((fabs(x) < zero_number) && (fabs(y) < zero_number) && (fabs(z) < zero_number)) {
                bound_dofs[eid][nid * NDOF_PER_NODE]     = 1;
                bound_dofs[eid][nid * NDOF_PER_NODE + 1] = 1;

                bound_values[eid][nid * NDOF_PER_NODE]     = 0.0;
                bound_values[eid][nid * NDOF_PER_NODE + 1] = 0.0;
            }

            // node at center of right edge of bottom surface: fix in y
            if ((fabs(x - Lx / 2) < zero_number) && (fabs(y) < zero_number) &&
                (fabs(z) < zero_number)) {
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 1]   = 1;
                bound_values[eid][(nid * NDOF_PER_NODE) + 1] = 0.0;
            }
        }
    }

    // create lists of constrained dofs
    std::vector<par::ConstraintRecord<double, unsigned long int>> list_of_constraints;
    par::ConstraintRecord<double, unsigned long int> cdof;
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                if (bound_dofs[eid][(nid * NDOF_PER_NODE) + did] == 1) {
                    // save the global id of constrained dof
                    cdof.set_dofId(globalDofMap[eid][(nid * NDOF_PER_NODE) + did]);
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

    // elemental traction vector
    Matrix<double, Eigen::Dynamic, 1>* elem_trac;
    elem_trac = new Matrix<double, Eigen::Dynamic, 1>[nelem];

    // nodal traction of tractioned face
    double nodalTraction[12] = { 0.0 };
    // nodal coordinates of tractioned face
    double xeSt[12];
    // force vector due to traction
    Matrix<double, 12, 1> feT;

    
    for (unsigned int eid = 0; eid < nelem; eid++) {
        bool traction = false;
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            gNodeId = globalMap[eid][nid];
            z       = (double)(gNodeId / ((Nex + 1) * (Ney + 1))) * hz;
            if (fabs(z - Lz) < zero_number) {
                // element eid has one face is the top surface with applied traction
                traction = true;
                break;
            }
        }
        if (traction) {
            // get coordinates of all nodes
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
                gNodeId = globalMap[eid][nid];
                // get node coordinates
                x = (double)(gNodeId % (Nex + 1)) * hx;
                y = (double)((gNodeId % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
                z = (double)(gNodeId / ((Nex + 1) * (Ney + 1))) * hz;

                x = x - Lx / 2;
                y = y - Ly / 2;

                xe[nid * NDOF_PER_NODE]       = x;
                xe[(nid * NDOF_PER_NODE) + 1] = y;
                xe[(nid * NDOF_PER_NODE) + 2] = z;

                //printf("eid=%d nid=%d gid=%d ------> x = %f y = %f z = %f\n",eid,nid,gNodeId,xe[nid * NDOF_PER_NODE], xe[nid * NDOF_PER_NODE+1], xe[nid * NDOF_PER_NODE+2]);
            }

            // get coordinates of nodes belonging to the face where traction is applied
            // traction applied on face 4-5-6-7 ==> nodes [4,5,6,7] corresponds to nodes [0,1,2,3]
            // of 2D element
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                xeSt[did]                     = xe[4 * NDOF_PER_NODE + did];
                xeSt[NDOF_PER_NODE + did]     = xe[5 * NDOF_PER_NODE + did];
                xeSt[2 * NDOF_PER_NODE + did] = xe[6 * NDOF_PER_NODE + did];
                xeSt[3 * NDOF_PER_NODE + did] = xe[7 * NDOF_PER_NODE + did];
            }

            // get nodal traction of face where traction is applied (uniform traction t3 = rho*g*Lz
            // applied on top surface)
            for (unsigned int nid = 0; nid < 4; nid++) {
                nodalTraction[nid * NDOF_PER_NODE + 2] = rho * g * Lz;
            }

            // compute force vector due traction applied on one face of element
            feT_hex8_iso(feT, xeSt, nodalTraction, intData.Pts_n_Wts, NGT);

            // put traction force vector into element force vector
            elem_trac[eid].resize(NNODE_PER_ELEM * NDOF_PER_NODE, 1);
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                    // nodes [4,5,6,7] of 3D element are nodes [0,1,2,3] of 2D element where
                    // traction applied
                    if (nid == 4) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[did];
                    } else if (nid == 5) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[NDOF_PER_NODE + did];
                    } else if (nid == 6) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[2 * NDOF_PER_NODE + did];
                    } else if (nid == 7) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[3 * NDOF_PER_NODE + did];
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
                printf("[e %d][n %d] elem_trac = %f, %f, %f\n", eid, nid,
    elem_trac[eid](nid*NDOF_PER_NODE), elem_trac[eid](nid*NDOF_PER_NODE +
    1),elem_trac[eid](nid*NDOF_PER_NODE + 2));
            }
        }
    } */

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

    /// declare aMat object =================================
    typedef par::aMat<par::aMatBased<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>
        aMatBased; // aMat type taking aMatBased as derived class
    typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>
        aMatFree; // aMat type taking aMatBased as derived class

    aMatBased* stMatBased; // pointer of aMat taking aMatBased as derived
    aMatFree* stMatFree;   // pointer of aMat taking aMatFree as derived

    if (matType == 0) {
        // assign stMatBased to the derived class aMatBased
        stMatBased = new par::aMatBased<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
    } else {
        // assign stMatFree to the derived class aMatFree
        stMatFree = new par::aMatFree<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
        // set matrix-free type: 1 = hybrid-OpenMP, 2 = matrix-free, 3 = hybrid-GPU_OVER_CPU-matvec
        stMatFree->set_matfree_type((par::MATFREE_TYPE)matType);
        if ((matType == 3) || (matType == 4) || (matType == 5)){
            #ifdef USE_GPU
                stMatFree->set_num_streams(nStreams);
            #endif
        }
    }

    // compute element stiffness matrix and force vector, then assemble
    // nodal value of body force
    double beN[24] = { 0.0 };

    // set function to compute element matrix if using matrix-free
    if (matType == 2){
        stMatFree->set_element_matrix_function(&computeElemMat);
    }

    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact, error;
    par::create_vec(meshMaps, rhs);
    par::create_vec(meshMaps, out);
    par::create_vec(meshMaps, sol_exact);
    par::create_vec(meshMaps, error);

    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            gNodeId = globalMap[eid][nid];
            // get node coordinates
            x = (double)(gNodeId % (Nex + 1)) * hx;
            y = (double)((gNodeId % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(gNodeId / ((Nex + 1) * (Ney + 1))) * hz;

            // translate origin
            x = x - Lx / 2;
            y = y - Ly / 2;

            xe[nid * NDOF_PER_NODE]       = x;
            xe[(nid * NDOF_PER_NODE) + 1] = y;
            xe[(nid * NDOF_PER_NODE) + 2] = z;

            // const body force in z direction
            beN[(nid * NDOF_PER_NODE)]     = 0.0;
            beN[(nid * NDOF_PER_NODE) + 1] = 0.0;
            beN[(nid * NDOF_PER_NODE) + 2] = -rho * g;
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
            ke_hex8_iso(*kee[0], xe, E, nu, intData.Pts_n_Wts, NGT);
        } else {
            printf("Error: not yet implement element stiffness matrix which is not Eigen matrix "
                   "format\n");
            exit(0);
        }
        if (matType == 0){
            petsc_elem_compute_time.stop();
        } else {
            aMat_elem_compute_time.stop();
        }

        // assemble element stiffness matrix to global K
        if (matType == 0) {
            stMatBased->set_element_matrix(eid, *kee[0], 0, 0, 1);
        } else {
            stMatFree->set_element_matrix(eid, *kee[0], 0, 0, 1);
        }

        if (matType == 0){
            petsc_setup_time.stop();
        } else {
            aMat_setup_time.stop();
        }

        // compute element force vector due to body force
        fe_hex8_iso(fee[0], xe, beN, intData.Pts_n_Wts, NGT);

        par::set_element_vec(meshMaps, rhs, eid, fee[0], 0u, ADD_VALUES);

        // assemble element load vector due to traction
        if (elem_trac[eid].size() != 0) {
            par::set_element_vec(meshMaps, rhs, eid, elem_trac[eid], 0u, ADD_VALUES);
        }
    }

    delete [] xe;

    if (matType == 0) {
        petsc_setup_time.start();
        stMatBased->finalize(); // Pestc begins and completes assembling the global stiffness matrix
        petsc_setup_time.stop();
    } else {
        aMat_setup_time.start();
        stMatFree->finalize(); // compute trace of matrix when using penalty method
        aMat_setup_time.stop();
    }

    // Pestc begins and completes assembling the global load vector
    // These are needed because we used ADD_VALUES for rhs when assembling
    // now we are going to use INSERT_VALUE for Fc in apply_bc_rhs
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // apply bc for rhs: this must be done before applying bc for the matrix
    // because we use the original matrix to compute KfcUc in matrix-based method
    if (matType == 0)
        stMatBased->apply_bc(rhs);
    else
        stMatFree->apply_bc(rhs);

    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // char fname[256];
    // apply dirichlet BCs to the matrix
    if (matType == 0) {
        // stMat.apply_bc_mat();
        petsc_setup_time.start();
        stMatBased->finalize();
        petsc_setup_time.stop();
        // sprintf(fname,"matrix_%d.dat",size);
        // stMat.dump_mat(fname);
    }

    // sprintf(fname,"rhsVec_%d.dat",size);
    // stMat.dump_vec(rhs,fname);

    // ====================== profiling matvec ====================================
    const unsigned int numDofsTotal = meshMaps.get_NumDofsTotal(); // total dofs including ghost
    const unsigned int numDofs = meshMaps.get_NumDofs(); // total owned dofs

    double* X = (double*) malloc(sizeof(double) * (numDofsTotal));
    for (unsigned int i = 0; i < (numDofsTotal); i++){
        //X[i] = (double)std::rand()/(double)(RAND_MAX/5.0);
        X[i] = 1.5;
    }
    // result vector Y = [K] * X
    double* Y = (double*) malloc(sizeof(double) * (numDofsTotal));

    // total number of matvec's we want to profile
    const unsigned int num_matvecs = 10;
    if (rank == 0) printf("Number of matvecs= %d\n", num_matvecs);

    if( matType == 0) {
        Vec petsc_X, petsc_Y;
        par::create_vec(meshMaps, petsc_X, 1.5);
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
    free (X);

    // ===================== solve ================================================
    /* if (matType == 0) {
        petsc_matvec_time.start();
    } else {
        aMat_matvec_time.start();
    }

    if (matType == 0) {
        par::solve(*stMatBased, (const Vec)rhs, out);
    } else {
        par::solve(*stMatFree, (const Vec)rhs, out);
    }

    if (matType == 0) {
        petsc_matvec_time.stop();
    } else {
        aMat_matvec_time.stop();
    } */
    // ===================== finish solve =========================================

    // sprintf(fname,"outVec_%d.dat",size);
    // stMat.dump_vec(out,fname);

    // ============================ comparing with exact solution =================
    /* PetscScalar norm, alpha = -1.0;
    // compute norm of solution
    VecNorm(out, NORM_2, &norm);
    if (rank == 0) {
        printf("L2 norm of computed solution = %20.10f\n", norm);
    }

    // exact solution
    Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1> e_exact;
    double disp[3];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            gNodeId = globalMap[eid][nid];
            x       = (double)(gNodeId % (Nex + 1)) * hx;
            y       = (double)((gNodeId % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
            z       = (double)(gNodeId / ((Nex + 1) * (Ney + 1))) * hz;

            // transformed coordinates
            x = x - Lx / 2;
            y = y - Ly / 2;

            disp[0] = (-nu * rho * g / E) * x * z;
            disp[1] = (-nu * rho * g / E) * y * z;
            disp[2] =
              (rho * g / 2 / E) * (z * z - Lz * Lz) + (nu * rho * g / 2 / E) * (x * x + y * y);

            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                e_exact[(nid * NDOF_PER_NODE) + did] = disp[did];
            }
        }
        par::set_element_vec(meshMaps, sol_exact, eid, e_exact, 0u, INSERT_VALUES);
    }

    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

    // sprintf(fname,"exactVec_%d.dat",size);
    // stMat.dump_vec(sol_exact,fname);

    // compute norm of exact solution
    VecNorm(sol_exact, NORM_2, &norm);
    if (rank == 0) {
        printf("L2 norm of exact solution = %20.10f\n", norm);
    }

    // compute the error vector
    VecCopy(sol_exact, error);

    // subtract error = sol_exact - out
    VecAXPY(error, alpha, out);

    // compute norm of error
    // VecNorm(sol_exact, NORM_INFINITY, &norm);
    VecNorm(error, NORM_INFINITY, &norm);
    if (rank == 0) {
        printf("Inf norm of error = %20.10f\n", norm);
    } */
    // ============================ finish comparing with exact solution ============

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
                outFile << "aMatGpu_" << matType << ", " << aMat_elem_compute_maxTime << ", " << aMat_setup_maxTime << ", " << aMat_matvec_maxTime << "\n";
            }
        } else {
            std::cout << "(1) aMatGpu elem compute time = " << aMat_elem_compute_time.seconds << "\n";
            std::cout << "(2) aMatGpu setup time = " << aMat_setup_time.seconds << "\n";
            std::cout << "(3) aMatGpu matvec time = " << aMat_matvec_time.seconds << "\n";
            outFile << "aMatGpu_" << matType << "," << aMat_elem_compute_time.seconds << ", " << aMat_setup_time.seconds << ", " << aMat_matvec_time.seconds << "\n";
        }
    }
    if (rank == 0) outFile.close();


    // export ParaView
    /* std::ofstream myfile;
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
    } */

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] globalMap[eid];
        delete[] globalDofMap[eid];
    }
    delete[] globalMap;
    delete[] globalDofMap;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] localMap[eid];
    }
    delete[] localMap;
    delete[] nnodeCount;
    delete[] nnodeOffset;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] localDofMap[eid];
    }
    delete[] localDofMap;
    delete[] local2GlobalMap;
    delete[] local2GlobalDofMap;

    delete[] ndofs_per_element;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] bound_dofs[eid];
        delete[] bound_values[eid];
    }
    delete[] bound_dofs;
    delete[] bound_values;

    delete[] constrainedDofs_ptr;
    delete[] prescribedValues_ptr;

    delete[] elem_trac;

    for (unsigned int i = 0; i < MAX_BLOCKS_PER_ELEMENT; i++) {
        if (kee[i] != nullptr)
            delete kee[i];
    }
    delete[] kee;

    if (matType == 0) {
        delete stMatBased;
    } else {
        delete stMatFree;
    }

    VecDestroy(&out);
    VecDestroy(&sol_exact);
    VecDestroy(&rhs);
    VecDestroy(&error);
    PetscFinalize();
    return 0;
}