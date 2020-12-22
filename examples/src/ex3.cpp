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

#include <fstream>
#include <iostream>
#include <mpi.h>

#ifdef BUILD_WITH_PETSC
#include <petsc.h>
#endif

#include "aMat.hpp"
#include "aMatBased.hpp"
#include "aMatFree.hpp"
#include "aVec.hpp"
#include "constraintRecord.hpp"
#include "enums.hpp"
#include "fe_vector.hpp"
#include "integration.hpp"
#include "ke_matrix.hpp"
#include "maps.hpp"
#include "profiler.hpp"
#include "solve.hpp"

#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// number of cracks allowed in 1 element
#define MAX_CRACK_LEVEL 3

// max number of block dimensions in one cracked element
#define MAX_BLOCKS_PER_ELEMENT (1u << MAX_CRACK_LEVEL)

//////////////////////////////////////////////////////////////////////////////////////////////////////

void usage()
{
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  ex3 <Nex> <Ney> <Nez> <matrix based/free> <bc-method>\n";
    std::cout << "\n";
    std::cout << "     Nex: Number of elements in X\n";
    std::cout << "     Ney: Number of elements in y\n";
    std::cout << "     Nez: Number of elements in z\n";
    std::cout << "     use matrix-free: 1 => yes.  0 => matrix-based method. \n";
    std::cout << "     use identity-matrix: 0    use penalty method: 1 \n";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    // User provides: Nex - number of elements (global) in x direction
    //                Ney - number of elements (global) in y direction
    //                Nez - number of elements (global) in z direction
    //                flag1 - 1 matrix-free method, 0 matrix-based method
    //                flag2 - 0 use identity-matrix method, 1 use penalty method for BC

    if (argc < 6)
    {
        usage();
        exit(0);
    }

    int rc;
    double zmin, zmax;

    double x, y, z;
    double hx, hy, hz;

    unsigned int emin = 0, emax = 0;

    const unsigned int NDOF_PER_NODE  = 3; // number of dofs per node
    const unsigned int NDIM           = 3; // number of dimension
    const unsigned int NNODE_PER_ELEM = 8; // number of nodes per element

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

    // 05.19.20 only use Eigen matrix
    const bool useEigen = true;

    const bool matType          = atoi(argv[4]); // use matrix-free method
    const unsigned int bcMethod = atoi(argv[5]); // method of applying BC

    // domain sizes: Lx, Ly, Lz - length of the (global) domain in x/y/z direction
    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

    // element sizes
    hx = Lx / double(Nex); // element size in x direction
    hy = Ly / double(Ney); // element size in y direction
    hz = Lz / double(Nez); // element size in z direction

    const double zero_number = 1E-12;

    PetscInitialize(&argc, &argv, NULL, NULL);

    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Status Stat;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // element matrix and vector
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EigenMat;
    EigenMat** kee;

    std::vector<Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1>> fee;
    fee.resize(MAX_BLOCKS_PER_ELEMENT);

    kee = new EigenMat*[MAX_BLOCKS_PER_ELEMENT];
    for (unsigned int i = 1; i < MAX_BLOCKS_PER_ELEMENT; i++)
    {
        kee[i] = nullptr;
    }
    kee[0] = new EigenMat;

    // nodal coordinates of element
    double* xe = new double[NDIM * NNODE_PER_ELEM];

    // timing variables
    profiler_t aMat_time;
    profiler_t petsc_time;
    profiler_t petsc_assemble_time;
    profiler_t total_time;
    if (matType != 0)
    {
        aMat_time.clear();
    }
    else
    {
        petsc_time.clear();
        petsc_assemble_time.clear();
    }
    total_time.clear();

    if (!rank)
    {
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNex : " << Nex << " Ney: " << Ney << " Nez: " << Nez << "\n";
        std::cout << "\t\tLx : " << Lx << " Ly: " << Ly << " Lz: " << Lz << "\n";
        std::cout << "\t\tMethod (0 = matrix based; 1 = matrix free) = " << matType << "\n";
        std::cout << "\t\tBC method (0 = 'identity-matrix'; 1 = penalty): " << bcMethod << "\n";
    }

#ifdef VECTORIZED_AVX512
    if (!rank)
    {
        std::cout << "\t\tVectorization using AVX_512\n";
    }
#elif VECTORIZED_AVX256
    if (!rank)
    {
        std::cout << "\t\tVectorization using AVX_256\n";
    }
#elif VECTORIZED_OPENMP
    if (!rank)
    {
        std::cout << "\t\tVectorization using OpenMP\n";
    }
#elif VECTORIZED_OPENMP_ALIGNED
    if (!rank)
    {
        std::cout << "\t\tVectorization using OpenMP with aligned memory\n";
    }
#else
    if (!rank)
    {
        std::cout << "\t\tNo vectorization\n";
    }
#endif

#ifdef HYBRID_PARALLEL
    if (!rank)
    {
        std::cout << "\t\tHybrid parallel OpenMP + MPI\n";
        std::cout << "\t\tMax number of threads: " << omp_get_max_threads() << "\n";
        std::cout << "\t\tNumber of MPI processes: " << size << "\n";
    }
#else
    if (!rank)
    {
        std::cout << "\t\tOnly MPI parallel\n";
        std::cout << "\t\tNumber of MPI processes: " << size << "\n";
    }
#endif


    if (rank == 0)
    {
        if (size > Nez)
        {
            printf("The number of processes must be less than or equal Nez, program stops.\n");
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // for fixing rigid motions at centroid of the top/bottom face
    // number of elements in x and y directions must be even numbers
    if ((Nex % 2 != 0) || (Ney % 2 != 0))
    {
        if (!rank)
        {
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
    if (rank < nRemain)
    {
        nelem_z = nzmin + 1;
    }
    else
    {
        nelem_z = nzmin;
    }
    if (rank < nRemain)
    {
        emin = rank * nzmin + rank;
    }
    else
    {
        emin = rank * nzmin + nRemain;
    }
    emax = emin + nelem_z - 1;

    // number of owned elements
    unsigned int nelem_y = Ney;
    unsigned int nelem_x = Nex;
    unsigned int nelem   = (nelem_x) * (nelem_y) * (nelem_z);

    // number of owned nodes
    unsigned int nnode_z;
    if (rank == 0)
    {
        nnode_z = nelem_z + 1;
    }
    else
    {
        nnode_z = nelem_z;
    }

    unsigned int nnode_y = nelem_y + 1;
    unsigned int nnode_x = nelem_x + 1;
    unsigned int nnode   = (nnode_x) * (nnode_y) * (nnode_z);

    // map from elemental node to global nodes
    unsigned long gNodeId;
    unsigned long int** globalMap;
    globalMap = new unsigned long int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        globalMap[eid] = new unsigned long int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned k = 0; k < nelem_z; k++)
    {
        for (unsigned j = 0; j < nelem_y; j++)
        {
            for (unsigned i = 0; i < nelem_x; i++)
            {
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
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        globalDofMap[eid] =
          new unsigned long int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM * NDOF_PER_NODE];
    }
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
        {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
            {
                globalDofMap[eid][(nid * NDOF_PER_NODE) + did] =
                  (globalMap[eid][nid] * NDOF_PER_NODE) + did;
            }
        }
    }

    // build localMap from globalMap (this is just to conform with aMat interface used for bsamxx)
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalNodes;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;

    // counts of owned nodes: nnodeCount[0] = nnode0, nnodeCount[1] = nnode1, ...
    unsigned int* nnodeCount = new unsigned int[size];
    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    // offset of nnodeCount
    unsigned int* nnodeOffset = new unsigned int[size];
    nnodeOffset[0]            = 0;
    for (unsigned int i = 1; i < size; i++)
    {
        nnodeOffset[i] = nnodeOffset[i - 1] + nnodeCount[i - 1];
    }
    // total number of nodes for all ranks
    unsigned long int nnode_total, ndofs_total;
    nnode_total = nnodeOffset[size - 1] + nnodeCount[size - 1];
    ndofs_total = nnode_total * NDOF_PER_NODE;
    if (rank == 0)
        printf("Total dofs = %d\n", ndofs_total);

    // PetscFinalize();
    // return 0;


    // determine ghost nodes based on:
    // rank 0 owns [0,...,nnode0-1], rank 1 owns [nnode0,..., nnode0 + nnode1 - 1]...
    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
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
    // sort (in ascending order) to prepare for deleting repeated nodes in preGhostGIds and
    // postGhostGIds
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
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        localMap[eid] = new unsigned int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int i = 0; i < NNODE_PER_ELEM; i++)
        {
            gNodeId = globalMap[eid][i];
            if ((gNodeId >= nnodeOffset[rank]) && (gNodeId < (nnodeOffset[rank] + nnode)))
            {
                // nid is owned by me
                localMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
            }
            else if (gNodeId < nnodeOffset[rank])
            {
                // nid is owned by someone before me
                const unsigned int lookUp =
                  std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) -
                  preGhostGIds.begin();
                localMap[eid][i] = lookUp;
            }
            else if (gNodeId >= (nnodeOffset[rank] + nnode))
            {
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
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        localDofMap[eid] =
          new unsigned int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM * NDOF_PER_NODE];
    }
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
        {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
            {
                localDofMap[eid][(nid * NDOF_PER_NODE) + did] =
                  (localMap[eid][nid] * NDOF_PER_NODE) + did;
            }
        }
    }

    // local node to global node map
    unsigned long* local2GlobalMap = new unsigned long[numLocalNodes];
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
        {
            gNodeId                             = globalMap[eid][nid];
            local2GlobalMap[localMap[eid][nid]] = gNodeId;
        }
    }

    // local dof to global dof map
    unsigned long* local2GlobalDofMap = new unsigned long[numLocalNodes * NDOF_PER_NODE];
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
        {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
            {
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
    for (unsigned eid = 0; eid < nelem; eid++)
    {
        ndofs_per_element[eid] = NNODE_PER_ELEM * NDOF_PER_NODE;
    }

    // elemental boundary dofs and prescribed value
    unsigned int** bound_dofs = new unsigned int*[nelem];
    double** bound_values     = new double*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        bound_dofs[eid]   = new unsigned int[ndofs_per_element[eid]];
        bound_values[eid] = new double[ndofs_per_element[eid]];
    }

    // construct elemental constrained DoFs and prescribed values
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
        {

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
            if ((fabs(x) < zero_number) && (fabs(y) < zero_number) && (fabs(z - Lz) < zero_number))
            {
                bound_dofs[eid][(nid * NDOF_PER_NODE)]     = 1;
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 1] = 1;
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 2] = 1;

                bound_values[eid][(nid * NDOF_PER_NODE)]     = 0.0;
                bound_values[eid][(nid * NDOF_PER_NODE) + 1] = 0.0;
                bound_values[eid][(nid * NDOF_PER_NODE) + 2] = 0.0;
            }
            else
            {
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
                {
                    bound_dofs[eid][(nid * NDOF_PER_NODE) + did]   = 0; // free dof
                    bound_values[eid][(nid * NDOF_PER_NODE) + did] = -1000000;
                }
            }

            // node at centroid of bottom surface: fix in x and y
            if ((fabs(x) < zero_number) && (fabs(y) < zero_number) && (fabs(z) < zero_number))
            {
                bound_dofs[eid][nid * NDOF_PER_NODE]     = 1;
                bound_dofs[eid][nid * NDOF_PER_NODE + 1] = 1;

                bound_values[eid][nid * NDOF_PER_NODE]     = 0.0;
                bound_values[eid][nid * NDOF_PER_NODE + 1] = 0.0;
            }

            // node at center of right edge of bottom surface: fix in y
            if ((fabs(x - Lx / 2) < zero_number) && (fabs(y) < zero_number) &&
                (fabs(z) < zero_number))
            {
                bound_dofs[eid][(nid * NDOF_PER_NODE) + 1]   = 1;
                bound_values[eid][(nid * NDOF_PER_NODE) + 1] = 0.0;
            }
        }
    }

    // create lists of constrained dofs
    std::vector<par::ConstraintRecord<double, unsigned long int>> list_of_constraints;
    par::ConstraintRecord<double, unsigned long int> cdof;
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
        {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
            {
                if (bound_dofs[eid][(nid * NDOF_PER_NODE) + did] == 1)
                {
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
    for (unsigned int i = 0; i < list_of_constraints.size(); i++)
    {
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

    // Gauss points and weights
    const unsigned int NGT = 4;
    integration<double> intData(NGT);

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        bool traction = false;
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
        {
            gNodeId = globalMap[eid][nid];
            z       = (double)(gNodeId / ((Nex + 1) * (Ney + 1))) * hz;
            if (fabs(z - Lz) < zero_number)
            {
                // element eid has one face is the top surface with applied traction
                traction = true;
                break;
            }
        }
        if (traction)
        {
            // get coordinates of all nodes
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
            {
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
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
            {
                xeSt[did]                     = xe[4 * NDOF_PER_NODE + did];
                xeSt[NDOF_PER_NODE + did]     = xe[5 * NDOF_PER_NODE + did];
                xeSt[2 * NDOF_PER_NODE + did] = xe[6 * NDOF_PER_NODE + did];
                xeSt[3 * NDOF_PER_NODE + did] = xe[7 * NDOF_PER_NODE + did];
            }

            // get nodal traction of face where traction is applied (uniform traction t3 = rho*g*Lz
            // applied on top surface)
            for (unsigned int nid = 0; nid < 4; nid++)
            {
                nodalTraction[nid * NDOF_PER_NODE + 2] = rho * g * Lz;
            }

            // compute force vector due traction applied on one face of element
            feT_hex8_iso(feT, xeSt, nodalTraction, intData.Pts_n_Wts, NGT);

            // put traction force vector into element force vector
            elem_trac[eid].resize(NNODE_PER_ELEM * NDOF_PER_NODE, 1);
            for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
            {
                for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
                {
                    // nodes [4,5,6,7] of 3D element are nodes [0,1,2,3] of 2D element where
                    // traction applied
                    if (nid == 4)
                    {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[did];
                    }
                    else if (nid == 5)
                    {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[NDOF_PER_NODE + did];
                    }
                    else if (nid == 6)
                    {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[2 * NDOF_PER_NODE + did];
                    }
                    else if (nid == 7)
                    {
                        elem_trac[eid]((nid * NDOF_PER_NODE) + did) = feT[3 * NDOF_PER_NODE + did];
                    }
                    else
                    {
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

    total_time.start();

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
    if (matType == 1)
    {
        meshMaps.buildScatterMap();
    }
    meshMaps.set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, list_of_constraints.size());

    /// declare aMat object =================================
    typedef par::
      aMat<par::aMatBased<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>
        aMatBased; // aMat type taking aMatBased as derived class
    typedef par::
      aMat<par::aMatFree<double, unsigned long, unsigned int>, double, unsigned long, unsigned int>
        aMatFree; // aMat type taking aMatBased as derived class

    aMatBased* stMatBased; // pointer of aMat taking aMatBased as derived
    aMatFree* stMatFree;   // pointer of aMat taking aMatFree as derived

    if (matType == 0)
    {
        // assign stMatBased to the derived class aMatBased
        stMatBased =
          new par::aMatBased<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
    }
    else
    {
        // assign stMatFree to the derived class aMatFree
        stMatFree =
          new par::aMatFree<double, unsigned long, unsigned int>(meshMaps, (par::BC_METH)bcMethod);
    }

    // create rhs, solution and exact solution vectors
    Vec rhs, out, sol_exact, error;
    par::create_vec(meshMaps, rhs);
    par::create_vec(meshMaps, out);
    par::create_vec(meshMaps, sol_exact);
    par::create_vec(meshMaps, error);

    // compute element stiffness matrix and force vector, then assemble
    // nodal value of body force
    double beN[24] = { 0.0 };

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
        {
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
        if (useEigen)
        {
            ke_hex8_iso(*kee[0], xe, E, nu, intData.Pts_n_Wts, NGT);
        }
        else
        {
            printf("Error: not yet implement element stiffness matrix which is not Eigen matrix "
                   "format\n");
        }

        // assemble element stiffness matrix to global K
        if (matType == 0){
            petsc_time.start();
            petsc_assemble_time.start();
        } else
            aMat_time.start();
        if (matType == 0)
            stMatBased->set_element_matrix(eid, *kee[0], 0, 0, 1);
        else
            stMatFree->set_element_matrix(eid, *kee[0], 0, 0, 1);
        if (matType == 0){
            petsc_assemble_time.stop();
            petsc_time.stop();
        } else
            aMat_time.stop();

        // compute element force vector due to body force
        fe_hex8_iso(fee[0], xe, beN, intData.Pts_n_Wts, NGT);

        /* for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++){
            printf("[e %d][n %d] fee= %f, %f, %f\n",eid,nid,fee[0](nid * NDOF_PER_NODE ),
            fee[0](nid * NDOF_PER_NODE +1),fee[0](nid * NDOF_PER_NODE +2));
        } */

        // assemble element load vector due to body force
        if (matType == 0)
            petsc_time.start();
        else
            aMat_time.start();

        par::set_element_vec(meshMaps, rhs, eid, fee[0], 0u, ADD_VALUES);

        if (matType == 0)
            petsc_time.stop();
        else
            aMat_time.stop();

        // assemble element load vector due to traction
        if (elem_trac[eid].size() != 0)
        {
            if (matType == 0)
                petsc_time.start();
            else
                aMat_time.start();
            par::set_element_vec(meshMaps, rhs, eid, elem_trac[eid], 0u, ADD_VALUES);
            if (matType == 0)
                petsc_time.stop();
            else
                aMat_time.stop();
        }
    }

    delete[] xe;

    // Pestc begins and completes assembling the global stiffness matrix
    if (matType == 0)
    {
        petsc_time.start();
        stMatBased->finalize();
        petsc_time.stop();
    }
    else
    {
        aMat_time.start();
        stMatFree->finalize(); // compute trace of matrix when using penalty method
        aMat_time.stop();
    }

    // Pestc begins and completes assembling the global load vector
    if (matType != 0)
    {
        aMat_time.start();
    }
    else
    {
        petsc_time.start();
    }
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
    if (matType != 0)
    {
        aMat_time.stop();
    }
    else
    {
        petsc_time.stop();
    }

    // char fname[256];
    // apply dirichlet BCs to the matrix
    if (matType == 0)
    {
        petsc_time.start();
        // stMat.apply_bc_mat();
        stMatBased->finalize();
        petsc_time.stop();
        // sprintf(fname,"matrix_%d.dat",size);
        // stMat.dump_mat(fname);
    }

    // sprintf(fname,"rhsVec_%d.dat",size);
    // stMat.dump_vec(rhs,fname);

    // solve
    if (matType != 0)
    {
        aMat_time.start();
    }
    else
    {
        petsc_time.start();
    }
    if (matType == 0)
    {
        par::solve(*stMatBased, (const Vec)rhs, out);
    }
    else
    {
        par::solve(*stMatFree, (const Vec)rhs, out);
    }

    /* if (matType == 0){
        par::solve(*dynamic_cast<aMatBased_ptr>(stMat), (const Vec)rhs, out);
    } else {
        par::solve(*dynamic_cast<aMatFree_ptr>(stMat), (const Vec)rhs, out);
    } */

    // VecAssemblyBegin(out);
    // VecAssemblyEnd(out);
    if (matType != 0)
    {
        aMat_time.stop();
    }
    else
    {
        petsc_time.stop();
    }

    total_time.stop();

    // display timing
    if (matType != 0)
    {
        if (size > 1)
        {
            long double aMat_maxTime;
            MPI_Reduce(&aMat_time.seconds, &aMat_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0)
            {
                std::cout << "aMat time = " << aMat_maxTime << "\n";
            }
        }
        else
        {
            if (rank == 0)
            {
                std::cout << "aMat time = " << aMat_time.seconds << "\n";
            }
        }
    }
    else
    {
        if (size > 1)
        {
            long double petsc_maxTime;
            long double petsc_assemble_maxTime;
            MPI_Reduce(&petsc_time.seconds, &petsc_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&petsc_assemble_time.seconds, &petsc_assemble_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0)
            {
                std::cout << "PETSC time = " << petsc_maxTime << "\n";
                std::cout << "PETSC assemble time = " << petsc_assemble_maxTime << "\n";
            }
        }
        else
        {
            if (rank == 0)
            {
                std::cout << "PETSC time = " << petsc_time.seconds << "\n";
                std::cout << "PETSC assemble time = " << petsc_assemble_time.seconds << "\n";
            }
        }
    }
    /* if (size > 1)
    {
        long double total_time_max;
        MPI_Reduce(&total_time.seconds, &total_time_max, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0)
        {
            std::cout << "total time = " << total_time_max << "\n";
        }
    }
    else
    {
        if (rank == 0)
        {
            std::cout << "total time = " << total_time.seconds << "\n";
        }
    } */

    // sprintf(fname,"outVec_%d.dat",size);
    // stMat.dump_vec(out,fname);

    PetscScalar norm, alpha = -1.0;

    // compute norm of solution
    VecNorm(out, NORM_2, &norm);
    if (rank == 0)
    {
        printf("L2 norm of computed solution = %20.10f\n", norm);
    }

    // exact solution
    Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1> e_exact;
    double disp[3];
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int nid = 0; nid < NNODE_PER_ELEM; nid++)
        {
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

            for (unsigned int did = 0; did < NDOF_PER_NODE; did++)
            {
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
    if (rank == 0)
    {
        printf("L2 norm of exact solution = %20.10f\n", norm);
    }

    // compute the error vector
    VecCopy(sol_exact, error);

    // subtract error = sol_exact - out
    VecAXPY(error, alpha, out);

    // compute norm of error
    // VecNorm(sol_exact, NORM_INFINITY, &norm);
    //VecNorm(error, NORM_INFINITY, &norm);
    VecNorm(error, NORM_2, &norm);
    if (rank == 0)
    {
        //printf("Inf norm of error = %20.10f\n", norm);
        printf("L2 norm of error = %20.12f\n", norm);
    }

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

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        delete[] globalMap[eid];
        delete[] globalDofMap[eid];
    }
    delete[] globalMap;
    delete[] globalDofMap;

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        delete[] localMap[eid];
    }
    delete[] localMap;
    delete[] nnodeCount;
    delete[] nnodeOffset;

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        delete[] localDofMap[eid];
    }
    delete[] localDofMap;
    delete[] local2GlobalMap;
    delete[] local2GlobalDofMap;

    delete[] ndofs_per_element;

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        delete[] bound_dofs[eid];
        delete[] bound_values[eid];
    }
    delete[] bound_dofs;
    delete[] bound_values;

    delete[] constrainedDofs_ptr;
    delete[] prescribedValues_ptr;

    delete[] elem_trac;
    for (unsigned int i = 0; i < MAX_BLOCKS_PER_ELEMENT; i++)
    {
        if (kee[i] != nullptr)
            delete kee[i];
    }
    delete[] kee;

    VecDestroy(&out);
    VecDestroy(&sol_exact);
    VecDestroy(&rhs);

    PetscFinalize();
    return 0;
}