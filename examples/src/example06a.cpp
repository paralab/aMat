/**
 * @file example06a.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Solving elasticity problem by FEM using aMat, in parallel, with linear 8-node hex element
 * @brief (this example was ex3 or ex3a in in aMat_for_paper/)
 * @brief Stretching of a prismatic bar by its own weight (Timoshenko page 246)
 * @brief Exact solution (origin at centroid of bottom face)
 * @brief    uniform stress s_zz = rho * g * z
 * @brief    displacement u = -(nu * rho * g/E) * x * z
 * @brief    displacement v = -(nu * rho * g/E) * y * z
 * @brief    displacement w = (rho * g/2/E)(z^2 - Lz^2) + (nu * rho * g)/2/E(x^2 + y^2)
 * @brief Boundary condition: traction tz = rho * g * Lz applied on top surface + blocking rigid
 * motions
 * @brief Partition of elements in z direction: owned elements in z direction ~ Nez/(number of
 * ranks)
 * @brief Size of the domain: Lx, Ly, Lz
 *
 * @version 0.1
 * @date 2020-02-26
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#include "example06a.hpp"
AppData example06aAppData;

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
    std::cout << "\n";
    exit(0);
}

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EigenMat;

// function to compute element matrix used for method = 2
void computeElemMat(unsigned int eid, double *ke, double* xe) {
    
    const double hx = example06aAppData.hx;
    const double hy = example06aAppData.hy;
    const double hz = example06aAppData.hz;
    
    const unsigned int Nex  = example06aAppData.Nex;
    const unsigned int Ney  = example06aAppData.Ney;
    const unsigned int Nez  = example06aAppData.Nez;
    const unsigned int NDOF_PER_NODE = example06aAppData.NDOF_PER_NODE;
    const double Lx = example06aAppData.Lx;
    const double Ly = example06aAppData.Ly;

    // get coordinates of all nodes
    for (unsigned int nid = 0; nid < example06aAppData.NNODE_PER_ELEM; nid++) {
        unsigned int gNodeId = example06aAppData.ElementToGIDNode[eid][nid];
        xe[nid * NDOF_PER_NODE] = (double)(gNodeId % (Nex + 1)) * hx - Lx/2;
        xe[(nid * NDOF_PER_NODE) + 1] = (double)((gNodeId % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy - Ly/2;
        xe[(nid * NDOF_PER_NODE) + 2] = (double)(gNodeId / ((Nex + 1) * (Ney + 1))) * hz;
    }

    ke_hex8_iso(ke, xe, example06aAppData.E, example06aAppData.nu, example06aAppData.intData->Pts_n_Wts, example06aAppData.NGT);
    
    return;
} // computeElemMat

//////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {

    int rc;

    double x, y, z;
    double hx, hy, hz;

    unsigned int emin = 0;

    const unsigned int NDOF_PER_NODE  = 3; // number of dofs per node
    const unsigned int NDIM           = 3; // number of dimension
    const unsigned int NNODE_PER_ELEM = 8; // number of nodes per element

    const unsigned int NDOF_PER_ELEM = NDOF_PER_NODE * NNODE_PER_ELEM;

    // material properties of alumina
    const double E = 1.0E6;
    const double nu = 0.3;
    const double rho = 1.0;
    const double g = 1.0;
    // domain sizes: Lx, Ly, Lz - length of the (global) domain in x/y/z direction
    const double Lx = 100.0, Ly = 100.0, Lz = 100.0;
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
    
    example06aAppData.E = E;
    example06aAppData.nu = nu;
    example06aAppData.rho = rho;
    example06aAppData.g = g;

    example06aAppData.Lx = Lx;
    example06aAppData.Ly = Ly;
    example06aAppData.Lz = Lz;
    
    example06aAppData.Nex = Nex;
    example06aAppData.Ney = Ney;
    example06aAppData.Nez = Nez;

    example06aAppData.hx = hx;
    example06aAppData.hy = hy;
    example06aAppData.hz = hz;

    example06aAppData.NGT = NGT;
    example06aAppData.intData = &intData;

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

    std::vector<Matrix<double, NDOF_PER_ELEM, 1>> fee;
    fee.resize(MAX_BLOCKS_PER_ELEMENT);

    kee = new EigenMat*[MAX_BLOCKS_PER_ELEMENT];
    for (unsigned int i = 1; i < MAX_BLOCKS_PER_ELEMENT; i++) {
        kee[i] = nullptr;
    }
    kee[0] = new EigenMat;

    // nodal coordinates of element
    double* xe = new double[NDIM * NNODE_PER_ELEM];

    // timing variables
    profiler_t elem_compute_time;
    profiler_t setup_time;
    profiler_t matvec_time;
    
    elem_compute_time.clear();
    setup_time.clear();
    matvec_time.clear();

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

    // number of owned nodes
    unsigned int nnode_z;
    if (rank == 0) {
        nnode_z = nelem_z + 1;
    } else {
        nnode_z = nelem_z;
    }

    unsigned int nnode_y = nelem_y + 1;
    unsigned int nnode_x = nelem_x + 1;
    unsigned int nnode   = (nnode_x) * (nnode_y) * (nnode_z);

    // number of owned dofs
    const unsigned int ndof = nnode * NDOF_PER_NODE;

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
          new unsigned long int[MAX_BLOCKS_PER_ELEMENT * NDOF_PER_ELEM];
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
    example06aAppData.ElementToGIDNode = globalMap;

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
        ndofs_per_element[eid] = NDOF_PER_ELEM;
    }

    // boundary conditions...
    unsigned int** bound_dofs = new unsigned int*[nelem];
    double** bound_values     = new double*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        bound_dofs[eid]   = new unsigned int[ndofs_per_element[eid]];
        bound_values[eid] = new double[ndofs_per_element[eid]];
    }
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
    const unsigned int n_constraints = list_of_constraints.size();
    constrainedDofs_ptr  = new unsigned long int[n_constraints];
    prescribedValues_ptr = new double[n_constraints];
    for (unsigned int i = 0; i < n_constraints; i++) {
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
            // top face is subjected to traction t3 = sigma_33 = -rho * g
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

            // get nodal traction of face where traction is applied (uniform traction t3 = rho*g*Lz applied on top surface)
            for (unsigned int nid = 0; nid < 4; nid++) {
                nodalTraction[nid * NDOF_PER_NODE + 2] = rho * g * Lz;
            }

            // compute force vector due traction applied on one face of element
            feT_hex8_iso(feT, xeSt, nodalTraction, intData.Pts_n_Wts, NGT);

            // put traction force vector into element force vector
            elem_trac[eid].resize(NDOF_PER_ELEM, 1);
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                    // nodes [4,5,6,7] of 3D element are nodes [0,1,2,3] of 2D element where traction applied
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
    typedef par::aMat<par::aMatBased<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> aMatBased; // aMat type taking aMatBased as derived class
    typedef par::aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int> aMatFree; // aMat type taking aMatBased as derived class

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

        #ifdef USE_GPU
        if ((matType == 3) || (matType == 4) || (matType == 5)){
            stMatFree->set_num_streams(nStreams);
        }
        #endif
    }

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

    // nodal value of body force
    double beN[24] = { 0.0 };
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
        setup_time.start();
        elem_compute_time.start();
        ke_hex8_iso(*kee[0], xe, E, nu, intData.Pts_n_Wts, NGT);
        elem_compute_time.stop();

        // assemble element stiffness matrix to global K
        if (matType == 0) {
            stMatBased->set_element_matrix(eid, *kee[0], 0, 0, 1);
        } else {
            stMatFree->set_element_matrix(eid, *kee[0], 0, 0, 1);
        }
        setup_time.stop();

        // compute element force vector due to body force
        fe_hex8_iso(fee[0], xe, beN, intData.Pts_n_Wts, NGT);
        // assemble element load vector due to body force
        par::set_element_vec(meshMaps, rhs, eid, fee[0], 0u, ADD_VALUES);

        // assemble element load vector due to traction
        if (elem_trac[eid].size() != 0) {
            par::set_element_vec(meshMaps, rhs, eid, elem_trac[eid], 0u, ADD_VALUES);
        }
    }

    delete [] xe;

    setup_time.start();
    if (matType == 0) {
        stMatBased->finalize(); // Pestc begins and completes assembling the global stiffness matrix
    } else {
        stMatFree->finalize(); // compute trace of matrix when using penalty method
    }
    setup_time.stop();

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
        setup_time.start();
        stMatBased->finalize();
        setup_time.stop();
    }

    // ====================== profiling matvec ====================================
    /* const unsigned int numDofsTotal = meshMaps.get_NumDofsTotal(); // total dofs including ghost
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
    free (X); */

    // ===================== solve ================================================
    matvec_time.start();
    if (matType == 0) {
        par::solve(*stMatBased, (const Vec)rhs, out);
    } else {
        par::solve(*stMatFree, (const Vec)rhs, out);
    }
    matvec_time.stop();
    // ===================== finish solve =========================================

    // sprintf(fname,"outVec_%d.dat",size);
    // stMat.dump_vec(out,fname);

    // ============================ comparing with exact solution =================
    PetscScalar norm, alpha = -1.0;
    VecNorm(out, NORM_2, &norm);
    if (rank == 0) {
        printf("L2 norm of computed solution = %20.10f\n", norm);
    }

    // exact solution
    Matrix<double, NDOF_PER_ELEM, 1> e_exact;
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
            disp[2] = (rho * g / 2 / E) * (z * z - Lz * Lz) + (nu * rho * g / 2 / E) * (x * x + y * y);

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
    }
    // ============================ finish comparing with exact solution ============

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