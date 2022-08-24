/**
 * @file example07a.cpp, example07a.hpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief Solving elasticity problem by FEM using aMat, in parallel, with quadratic 20-node hex element (serendipity element)
 * @brief (this example was ex4 or ex4a in in aMat_for_paper/)
 * @brief Stretching of a prismatic bar by its own weight (Timoshenko page 246)
 * @brief          using 20-node quadratic element
 * @brief Exact solution (origin at centroid of bottom face)
 * @brief    uniform stress s_zz = rho * g * z
 * @brief    displacement u = -(nu * rho * g/E) * x * z
 * @brief    displacement v = -(nu * rho * g/E) * y * z
 * @brief    displacement w = (rho * g/2/E)(z^2 - Lz^2) + (nu * rho * g)/2/E(x^2 + y^2)
 * @brief Boundary condition: traction tz = rho * g * Lz applied on top surface + blocking rigid motions
 * @brief Partition of elements in z direction: owned elements in z direction ~ Nez/(number of ranks)
 * @brief Size of the domain: Lx, Ly, Lz
 *
 * @version 0.1
 * @date 2020-02-26
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#include "example07a.hpp"
AppData example07aAppData;

// number of cracks allowed in 1 element
#define MAX_CRACK_LEVEL 3

// max number of block dimensions in one cracked element
#define MAX_BLOCKS_PER_ELEMENT (1u << MAX_CRACK_LEVEL)

//////////////////////////////////////////////////////////////////////////////////////////////////////

void usage()
{
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  ./example07 <Nex> <Ney> <Nez> <matrix-method> <bc-method> <nStreams> <outputFile>\n";
    std::cout << "\n";
    std::cout << "     1) Nex: Number of elements in X\n";
    std::cout << "     2) Ney: Number of elements in y\n";
    std::cout << "     3) Nez: Number of elements in z\n";
    std::cout << "     4) method (0, 1, 2, 3, 4, 5) \n";
    std::cout << "     5) use identity-matrix: 0    use penalty method: 1 \n";
    std::cout << "     6) number of streams (used in method 3, 4, 5)\n";
    std::cout << "     7) name of output file\n";
    exit(0);
}

// function to compute element matrix used in method = 2
void computeElemMat(unsigned int eid, double *ke, double* xe) {
    
    const unsigned int NDOF_PER_NODE = example07aAppData.NDOF_PER_NODE;
    const unsigned int NNODE_PER_ELEM = example07aAppData.NNODE_PER_ELEM;
    
    // get coordinates of all nodes
    unsigned int lNodeId;
    for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
        unsigned int lNodeId = example07aAppData.localMap[eid][nid];
        xe[nid * NDOF_PER_NODE] = example07aAppData.localNodes[lNodeId].get_x() - example07aAppData.Lx / 2;
        xe[(nid * NDOF_PER_NODE) + 1] = example07aAppData.localNodes[lNodeId].get_y() - example07aAppData.Ly / 2;
        xe[(nid * NDOF_PER_NODE) + 2] = example07aAppData.localNodes[lNodeId].get_z();
    }

    ke_hex20_iso(ke, xe, example07aAppData.E, example07aAppData.nu, example07aAppData.intData->Pts_n_Wts, example07aAppData.NGT);
    
    return;
} // computeElemMat

unsigned long get_flop_per_matvec_petsc(const Mat & M)
{
    MatInfo info;
    MatGetInfo(M,MAT_GLOBAL_SUM,&info);
    unsigned long nnz = info.nz_used;
    
    PetscInt m, n;
    MatGetSize(M,&m,&n);
    assert(m==n);

    //printf("nnz %ul m %d\n",nnz,m);
    unsigned long flop = 2*nnz-n;
    return flop; 

}

unsigned long get_flop_per_matvec_afm(unsigned int ne, unsigned long gElems)
{   
    unsigned long flop = 0;
    // 
    // For each elemental matvec Ke ( nexne ) ve (nex1)
    unsigned long f1 = ne*(2*ne -1) + ne;
    flop = gElems*f1;
    //std::cout << "flop before mul with gE: " << f1 << "\n";
    return flop;

}

unsigned long get_flop_per_matvec_mf(unsigned int ne,unsigned long gElems)
{
    unsigned long flop = 0;
    
    const unsigned int NGT = 2*2*2;

    // for current code (not efficient) we need to add the flops for computing dN
    
    unsigned long b_flop = ne*6; // for dxds, dyds, dzds

    unsigned long jac_flop = 14; // for jaco

    unsigned long grad_phi_flop = (12 + 12 + 12) + 20 * 15; // for dNdx

    unsigned long cb_flop = (2*6-1)*6*ne; // for Bmat

    unsigned long btcb_flop = (2*6-1)*ne*ne; // for CB

    unsigned long ke_accum = ne*ne*5;

    unsigned long ke_flop = NGT * (b_flop + jac_flop + grad_phi_flop + cb_flop + btcb_flop + ke_accum);
    
    unsigned long f1 = 2*ne*ne;

    flop = gElems*(ke_flop + f1);
    return flop;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {

    int rc;

    double x, y, z;
    double hx, hy, hz;

    unsigned int emin = 0, emax = 0;

    const unsigned int NDOF_PER_NODE  = 3;  // number of dofs per node
    const unsigned int NDIM           = 3;  // number of dimension
    const unsigned int NNODE_PER_ELEM = 20; // number of nodes per element

    const unsigned int NDOF_PER_ELEM = NDOF_PER_NODE * NNODE_PER_ELEM;

    // material properties of alumina
    // const double E = 300.0; // GPa
    const double E = 1.0E6;
    // const double nu = 0.2;
    const double nu = 0.3;
    // const double rho = 3950;// kg.m^-3
    const double rho = 1.0;
    // const double g = 9.8;   // m.s^-2
    const double g = 1.0;

    // number of (global) elements in x, y and z directions
    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]);
    const unsigned int Nez = atoi(argv[3]);

    const unsigned int matType  = atoi(argv[4]); // approach (0, 1, 2, 3, 4, 5)
    const unsigned int bcMethod = atoi(argv[5]); // method of applying BC
    const unsigned int nStreams = atoi(argv[6]); // number of streams used for method 3, 4, 5
    const char* filename = argv[7];             // output file name

    // domain sizes: Lx, Ly, Lz - length of the (global) domain in x/y/z direction
    const double Lx = 1000, Ly = 1000, Lz = 1000;
    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);

    // element sizes
    hx = Lx / double(Nex); // element size in x direction
    hy = Ly / double(Ney); // element size in y direction
    hz = Lz / double(Nez); // element size in z direction

    example07aAppData.E = E;
    example07aAppData.nu = nu;
    example07aAppData.rho = rho;
    example07aAppData.g = g;

    example07aAppData.Lx = Lx;
    example07aAppData.Ly = Ly;
    example07aAppData.Lz = Lz;
    
    example07aAppData.Nex = Nex;
    example07aAppData.Ney = Ney;
    example07aAppData.Nez = Nez;

    example07aAppData.hx = hx;
    example07aAppData.hy = hy;
    example07aAppData.hz = hz;

    example07aAppData.NGT = NGT;
    example07aAppData.intData = &intData;

    example07aAppData.NDOF_PER_NODE = NDOF_PER_NODE;
    example07aAppData.NNODE_PER_ELEM = NNODE_PER_ELEM;

    const double zero_number = 1E-12;

    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Status Stat;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // output file in csv format, open in append mode
    std::ofstream outFile;
    if (rank == 0)
        outFile.open(filename, std::fstream::app);

    if (argc < 8) {
        usage();
    }

    // element matrix (contains multiple matrix blocks)
    typedef Eigen::Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, NDOF_PER_NODE * NNODE_PER_ELEM> EigenMat;
    typedef Eigen::Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1> EigenVec;

    EigenMat** kee;
    EigenVec** fee;

    kee = new EigenMat*[MAX_BLOCKS_PER_ELEMENT];
    fee = new EigenVec*[MAX_BLOCKS_PER_ELEMENT];
    for (unsigned int i = 1; i < MAX_BLOCKS_PER_ELEMENT; i++) {
        kee[i] = nullptr;
        fee[i] = nullptr;
    }
    kee[0] = new EigenMat;
    fee[0] = new EigenVec;

    // nodal coordinates of element
    double* xe = new double[NDIM * NNODE_PER_ELEM];

    // timing variables
    profiler_t elem_compute_time;
    profiler_t setup_time;
    profiler_t matvec_time;

    elem_compute_time.clear();
    setup_time.clear();
    matvec_time.clear();
    
    if (!rank)
    {
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

    // partition number of elements in z direction
    unsigned int nelem_z;
    // minimum number of elements in z-dir for each rank
    unsigned int nzmin = Nez / size;
    // remaining
    unsigned int nRemain = Nez % size;
    // distribute nRemain uniformly from rank = 0 up to rank = nRemain - 1
    if (rank < nRemain) {
        nelem_z = nzmin + 1;
    }
    else {
        nelem_z = nzmin;
    }
    if (rank < nRemain) {
        emin = rank * nzmin + rank;
    }
    else {
        emin = rank * nzmin + nRemain;
    }
    emax = emin + nelem_z - 1;

    // number of owned elements
    unsigned int nelem_y = Ney;
    unsigned int nelem_x = Nex;
    unsigned int nelem   = (nelem_x) * (nelem_y) * (nelem_z);

    double origin[3] = { 0.0 };
    origin[2]        = emin * hz;

    // generate nodes...
    std::vector<nodeData<double, unsigned int>> localNodes;
    nodeData<double, unsigned int> node;
    unsigned int nid = 0;
    for (unsigned int k = 0; k < (2 * nelem_z + 1); k++) {
        z = k * (hz / 2) + origin[2];
        for (unsigned int j = 0; j < (2 * nelem_y + 1); j++) {
            y = j * (hy / 2) + origin[1];
            for (unsigned int i = 0; i < (2 * nelem_x + 1); i++) {
                x = i * (hx / 2) + origin[0];
                if (!(((i % 2 == 1) && (j % 2 == 1)) || ((i % 2 == 1) && (k % 2 == 1)) ||
                      ((j % 2 == 1) && (k % 2 == 1)))) {
                    node.set_nodeId(nid);
                    node.set_x(x);
                    node.set_y(y);
                    node.set_z(z);
                    localNodes.push_back(node);
                    nid++;
                }
            }
        }
    }
    // total number of local nodes (inclulding ghost nodes)
    unsigned int numLocalNodes = localNodes.size();

    // number of local dofs (including ghost dofs)
    unsigned int numLocalDofs = numLocalNodes * NDOF_PER_NODE;

    // set localNodes to AppData
    example07aAppData.localNodes = localNodes.data();

    // local map...
    unsigned int** localMap;
    localMap = new unsigned int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localMap[eid] = new unsigned int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned int k = 0; k < nelem_z; k++) {
        for (unsigned int j = 0; j < nelem_y; j++) {
            for (unsigned int i = 0; i < nelem_x; i++) {
                unsigned int elemID = nelem_x * nelem_y * k + nelem_x * j + i;
                localMap[elemID][0] = (2 * i) + j * (3 * Nex + 2) +
                  k * ((2 * Nex + 1) * (2 * Ney + 1) - (Nex * Ney) + (Nex + 1) * (Ney + 1));
                localMap[elemID][1] = localMap[elemID][0] + 2;
                localMap[elemID][3] = localMap[elemID][0] + (3 * Nex + 2);
                localMap[elemID][2] = localMap[elemID][3] + 2;
                localMap[elemID][4] = localMap[elemID][0] + ((2 * Nex + 1) * (2 * Ney + 1) -
                                                             (Nex * Ney) + (Nex + 1) * (Ney + 1));
                localMap[elemID][5] = localMap[elemID][4] + 2;
                localMap[elemID][7] = localMap[elemID][4] + (3 * Nex + 2);
                localMap[elemID][6] = localMap[elemID][7] + 2;

                localMap[elemID][8]  = localMap[elemID][0] + 1;
                localMap[elemID][10] = localMap[elemID][3] + 1;
                localMap[elemID][11] = localMap[elemID][0] + (2 * Nex + 1) - i;
                localMap[elemID][9]  = localMap[elemID][11] + 1;

                localMap[elemID][12] = localMap[elemID][4] + 1;
                localMap[elemID][14] = localMap[elemID][7] + 1;
                localMap[elemID][15] = localMap[elemID][4] + (2 * Nex + 1) - i;
                localMap[elemID][13] = localMap[elemID][15] + 1;

                localMap[elemID][16] = localMap[elemID][0] +
                                       ((2 * Nex + 1) * (2 * Ney + 1) - (Nex * Ney) - i) -
                                       j * (2 * Nex + 1);
                localMap[elemID][17] = localMap[elemID][16] + 1;
                localMap[elemID][19] = localMap[elemID][16] + (Nex + 1);
                localMap[elemID][18] = localMap[elemID][19] + 1;
            }
        }
    }

    // set globalMap to AppData so that we can used in function compute element stiffness matrix
    example07aAppData.localMap = localMap;

    // local dof map ...
    unsigned int** localDofMap;
    localDofMap = new unsigned int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localDofMap[eid] =
          new unsigned int[MAX_BLOCKS_PER_ELEMENT * NDOF_PER_ELEM];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                localDofMap[eid][(nid * NDOF_PER_NODE) + did] =
                  (localMap[eid][nid] * NDOF_PER_NODE) + did;
            }
        }
    }

    // number of owned nodes
    unsigned int nnode;
    if (rank == 0) {
        nnode = numLocalNodes;
    }
    else {
        nnode = numLocalNodes - ((2 * Nex + 1) * (2 * Ney + 1) - (Nex * Ney));
    }

    // gather number of own nodes across ranks
    unsigned int* nnodeCount = new unsigned int[size];
    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    // offset of nnodeCount
    unsigned int* nnodeOffset = new unsigned int[size];
    nnodeOffset[0]            = 0;
    for (unsigned int i = 1; i < size; i++) {
        nnodeOffset[i] = nnodeOffset[i - 1] + nnodeCount[i - 1];
    }
    // total number of nodes and dofs for all ranks
    unsigned long int nnode_total, ndofs_total;
    nnode_total = nnodeOffset[size - 1] + nnodeCount[size - 1];
    ndofs_total = nnode_total * NDOF_PER_NODE;
    if (rank == 0)
        printf("Total dofs = %d\n", ndofs_total);

    // build global map from local map
    unsigned long gNodeId;
    unsigned long int** globalMap;
    globalMap = new unsigned long int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalMap[eid] = new unsigned long int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            if (rank == 0) {
                globalMap[eid][nid] = localMap[eid][nid];
            } else {
                globalMap[eid][nid] = localMap[eid][nid] + nnodeOffset[rank] -
                                      ((2 * Nex + 1) * (2 * Ney + 1) - (Nex * Ney));
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

    // local node to global node map
    unsigned long* local2GlobalMap = new unsigned long[numLocalNodes];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            gNodeId                             = globalMap[eid][nid];
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

    // map from eid to n dofs per element
    unsigned int* ndofs_per_element = new unsigned int[nelem];
    for (unsigned eid = 0; eid < nelem; eid++) {
        ndofs_per_element[eid] = NNODE_PER_ELEM * NDOF_PER_NODE;
    }

    // boundary conditions...
    unsigned int** bound_dofs = new unsigned int*[nelem];
    double** bound_values     = new double*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        bound_dofs[eid]   = new unsigned int[ndofs_per_element[eid]];
        bound_values[eid] = new double[ndofs_per_element[eid]];
    }

    // construct elemental constrained DoFs and prescribed values
    unsigned int lNodeId;
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            lNodeId = localMap[eid][nid];
            // get nodal coordinates
            x = localNodes[lNodeId].get_x();
            y = localNodes[lNodeId].get_y();
            z = localNodes[lNodeId].get_z();

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
    Eigen::Matrix<double, Eigen::Dynamic, 1>* elem_trac;
    elem_trac = new Eigen::Matrix<double, Eigen::Dynamic, 1>[nelem];

    // nodal traction of tractioned face
    double nodalTraction[24] = { 0.0 };

    // nodal coordinates of tractioned face
    double xeSt[24];

    // force vector due to traction
    Eigen::Matrix<double, 24, 1> feT;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        bool traction = false;
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            lNodeId = localMap[eid][nid];
            z       = localNodes[lNodeId].get_z();
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
                lNodeId = localMap[eid][nid];
                // get node coordinates
                x = localNodes[lNodeId].get_x();
                y = localNodes[lNodeId].get_y();
                z = localNodes[lNodeId].get_z();

                // tranlation origin
                x = x - Lx / 2;
                y = y - Ly / 2;

                xe[nid * NDOF_PER_NODE]       = x;
                xe[(nid * NDOF_PER_NODE) + 1] = y;
                xe[(nid * NDOF_PER_NODE) + 2] = z;
            }

            // get coordinates of nodes belonging to the face where traction is applied
            // traction applied on face 4-5-6-7-12-13-14-15 ==> nodes [4,5,6,7] corresponds to nodes
            // [0,1,2,3,4,5,6,7] of 2D element
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
            {
                // node 0 of 2D element <-- node 4 of 3D element
                xeSt[did] = xe[4 * NDOF_PER_NODE + did];

                // node 1 of 2D element <-- node 5 of 3D element
                xeSt[NDOF_PER_NODE + did] = xe[5 * NDOF_PER_NODE + did];

                // node 2 of 2D element <-- node 6 of 3D element
                xeSt[2 * NDOF_PER_NODE + did] = xe[6 * NDOF_PER_NODE + did];

                // node 3 of 2D element <-- node 7 of 3D element
                xeSt[3 * NDOF_PER_NODE + did] = xe[7 * NDOF_PER_NODE + did];

                // node 4 of 2D element <-- node 12 of 3D element
                xeSt[4 * NDOF_PER_NODE + did] = xe[12 * NDOF_PER_NODE + did];

                // node 5 of 2D element <-- node 13 of 3D element
                xeSt[5 * NDOF_PER_NODE + did] = xe[13 * NDOF_PER_NODE + did];

                // node 6 of 2D element <-- node 14 of 3D element
                xeSt[6 * NDOF_PER_NODE + did] = xe[14 * NDOF_PER_NODE + did];

                // node 7 of 2D element <-- node 15 of 3D element
                xeSt[7 * NDOF_PER_NODE + did] = xe[15 * NDOF_PER_NODE + did];
            }

            // get nodal traction of face where traction is applied (uniform traction t3 = rho*g*Lz applied on top surface)
            for (unsigned int nid = 0; nid < 8; nid++) {
                nodalTraction[nid * NDOF_PER_NODE + 2] = rho * g * Lz;
            }

            // compute force vector due traction applied on one face of element
            feT_hex20_iso(feT, xeSt, nodalTraction, intData.Pts_n_Wts, NGT);

            // put traction force vector into element force vector
            elem_trac[eid].resize(NDOF_PER_ELEM, 1);
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                    // nodes [4,5,6,7] of 3D element are nodes [0,1,2,3] of 2D element where traction applied
                    if (nid == 4) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[did];
                    }
                    else if (nid == 5) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[NDOF_PER_NODE + did];
                    }
                    else if (nid == 6) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[2 * NDOF_PER_NODE + did];
                    }
                    else if (nid == 7) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[3 * NDOF_PER_NODE + did];
                        // nodes [12,13,14,15] of 3D element are nodes [4,5,6,7] of 2D element where traction applied
                    }
                    else if (nid == 12) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[4 * NDOF_PER_NODE + did];
                    }
                    else if (nid == 13) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[5 * NDOF_PER_NODE + did];
                    }
                    else if (nid == 14) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[6 * NDOF_PER_NODE + did];
                    }
                    else if (nid == 15) {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[7 * NDOF_PER_NODE + did];
                    }
                    else {
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
    double beN[60] = { 0.0 };
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            lNodeId = localMap[eid][nid];
            // get node coordinates
            x = localNodes[lNodeId].get_x();
            y = localNodes[lNodeId].get_y();
            z = localNodes[lNodeId].get_z();

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
        ke_hex20_iso(*kee[0], xe, E, nu, intData.Pts_n_Wts, NGT);
        elem_compute_time.stop();

        // assemble element stiffness matrix to global K
        if (matType == 0)
            stMatBased->set_element_matrix(eid, *kee[0], 0, 0, 1);
        else
            stMatFree->set_element_matrix(eid, *kee[0], 0, 0, 1);
        setup_time.stop();

        // compute element force vector due to body force
        fe_hex20_iso(*fee[0], xe, beN, intData.Pts_n_Wts, NGT);
        // assemble element load vector due to body force
        par::set_element_vec(meshMaps, rhs, eid, *fee[0], 0u, ADD_VALUES);

        // assemble element load vector due to traction
        if (elem_trac[eid].size() != 0) {
            par::set_element_vec(meshMaps, rhs, eid, elem_trac[eid], 0u, ADD_VALUES);
        }
    }
    delete[] xe;

    setup_time.start();
    if (matType == 0) {
        stMatBased->finalize();
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

    // assemble matrix after matrix is modified by bc
    if (matType == 0) {
        setup_time.start();
        stMatBased->finalize();
        setup_time.stop();
    }

    // ====================== profiling matvec ====================================
    // generate random vector of length = number of owned dofs
    const unsigned int numDofsTotal = meshMaps.get_NumDofsTotal();
    const unsigned int numDofs = meshMaps.get_NumDofs();

    double* X = (double*) malloc(sizeof(double) * (numDofsTotal));
    for (unsigned int i = 0; i < (numDofsTotal); i++){
        //X[i] = (double)std::rand()/(double)(RAND_MAX/5.0);
        X[i] = 1.0;
    }
    // result vector Y = [K] * X
    double* Y = (double*) malloc(sizeof(double) * (numDofsTotal));

    // total number of matvec's we want to profile
    const unsigned int num_matvecs = 50;
    if (rank == 0) printf("Number of matvecs= %d\n", num_matvecs);

    if( matType == 0) {
        Vec petsc_X, petsc_Y;
        par::create_vec(meshMaps, petsc_X, 1.0);
        par::create_vec(meshMaps, petsc_Y);

        for (unsigned int i = 0; i < num_matvecs; i++){
            matvec_time.start();
            stMatBased->matmult(petsc_Y, petsc_X);
            //VecAssemblyBegin(petsc_X);
            //VecAssemblyEnd(petsc_X);
            matvec_time.stop();
            VecSwap(petsc_Y, petsc_X);

            // this is added on May 26, 2021, following ex6
            VecAXPY(petsc_X,1.020,petsc_Y);
        }
        VecDestroy(&petsc_X);
        VecDestroy(&petsc_Y);

    } else {
        for (unsigned int i = 0; i < num_matvecs; i++){
            
            matvec_time.start();     
            stMatFree->matvec(Y, X, true);
            matvec_time.stop();

            // this is the way ex4 done before:
            // double * temp = X;
            // X = Y;
            // Y = temp;

            // this is changed following ex6
            std::swap(X,Y);
            X[0]+=0.001;
            Y[0]+=0.001;
        }
    }
    free (Y);
    free (X);
    // ====================== finish profiling matvec ==============================


    // ====================== solve ================================================
    /* char fname[256];
    sprintf(fname, "matrix_%d.dat", size);
    if (matType == 0) {
        stMatBased->dump_mat(fname);
    } else {
        stMatFree->dump_mat(fname);
    } */

    /* matvec_time.start();
    if (matType == 0)
        par::solve(*stMatBased, (const Vec)rhs, out);
    else
        par::solve(*stMatFree, (const Vec)rhs, out);
    matvec_time.stop();
    // ======================finish solve ===========================================

    // sprintf(fname,"outVec_%d.dat",size);
    // stMat.dump_vec(out,fname);

    // ============================= comparing with exact solution ==========================
    PetscScalar norm, alpha = -1.0;
    // compute norm of solution
    VecNorm(out, NORM_2, &norm);
    if (rank == 0) {
        printf("L2 norm of computed solution = %20.10f\n", norm);
    }

    // exact solution
    Eigen::Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1> e_exact;
    double disp[3];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++) {
            lNodeId = localMap[eid][nid];
            // get node coordinates
            x = localNodes[lNodeId].get_x();
            y = localNodes[lNodeId].get_y();
            z = localNodes[lNodeId].get_z();

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

    // subtract error = error(=sol_exact) - out
    VecAXPY(error, alpha, out);

    // compute norm of error
    // VecNorm(sol_exact, NORM_INFINITY, &norm);
    VecNorm(error, NORM_INFINITY, &norm);
    if (rank == 0) {
        printf("Inf norm of error = %20.10f\n", norm);
    } */

    long double gather_time, scatter_time, mv_time, mvTotal_time;
    if ((matType == 3) || (matType == 4) || (matType == 5)) {
       stMatFree->get_timer(&scatter_time, &gather_time, &mv_time, &mvTotal_time);
    }
    // ============================= finish comparing with exact solution =================

    // computing time acrossing ranks and display
    long double elem_compute_maxTime;
    long double setup_maxTime;
    long double matvec_maxTime;
    long double gather_maxTime, scatter_maxTime, mv_maxTime, mvTotal_maxTime;

    MPI_Reduce(&elem_compute_time.seconds, &elem_compute_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&setup_time.seconds, &setup_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&matvec_time.seconds, &matvec_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&gather_time, &gather_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&scatter_time, &scatter_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&mv_time, &mv_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&mvTotal_time, &mvTotal_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);

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
            std::cout << "(4) aMatGpu (scatter, gather, mv, mvTotal) time = " << scatter_maxTime << ", " << gather_maxTime << ", " << mv_maxTime << ", " << mvTotal_maxTime << "\n";
            outFile << "aMatGpu, " << elem_compute_maxTime << ", " << setup_maxTime << ", " << matvec_maxTime << "\n";
        }
    }
    if (rank == 0) outFile.close();


    // export ParaView
    /* std::ofstream myfile;
    if (!rank){
        myfile.open("ex4.vtk");
        myfile << "# vtk DataFile Version 2.0 " << std::endl;
        myfile << "Stress field" << std::endl;
        myfile << "ASCII" << std::endl;
        myfile << "DATASET UNSTRUCTURED_GRID" << std::endl;
        myfile << "POINTS " << numLocalNodes << " float" << std::endl;
        for (unsigned int nid = 0; nid < numLocalNodes; nid++){
            x = localNodes[nid].get_x();
            y = localNodes[nid].get_y();
            z = localNodes[nid].get_z();
            // transformed coordinates
            x = x - Lx/2;
            y = y - Ly/2;
            myfile << x << "  " << y << "  " << z << std::endl;
        }
        unsigned int size_cell_list = nelem * 21;
        myfile << "CELLS " << nelem << " " << size_cell_list << std::endl;
        for (unsigned int eid = 0; eid < nelem; eid++){
            myfile << "20 " << localMap[eid][0] << " " << localMap[eid][1] << " "
             << localMap[eid][2] << " " << localMap[eid][3] << " "
              << localMap[eid][4] << " " << localMap[eid][5] << " "
               << localMap[eid][6] << " " << localMap[eid][7] << " "
               << localMap[eid][8] << " " << localMap[eid][9] << " "
               << localMap[eid][10] << " " << localMap[eid][11] << " "
               << localMap[eid][12] << " " << localMap[eid][13] << " "
               << localMap[eid][14] << " " << localMap[eid][15] << " "
               << localMap[eid][16] << " " << localMap[eid][17] << " "
               << localMap[eid][18] << " " << localMap[eid][19] << " " << std::endl;
        }
        myfile << "CELL_TYPES " << nelem << std::endl;
        for (unsigned int eid = 0; eid < nelem; eid++){
            myfile << "25" << std::endl;
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
    }*/
    

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
        if (fee[i] != nullptr)
            delete fee[i];
    }
    delete[] kee;
    delete[] fee;

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