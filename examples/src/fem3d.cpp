/**
 * @file fem3d.cpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 * @author Milinda Fernando milinda@cs.utah.edu
 *
 * @brief Example of solving 3D Poisson equation by FEM, in parallel, using Petsc
 * @brief grad^2(u) + sin(2*M_PI*x) * sin(2*M_PI*y) * sin(2*M_PI*z) = 0
 * @brief BCs u = 0 on all boundaries
 * @brief Exact solution: (1.0 / (12.0 * M_PI * M_PI)) * sin(2 * M_PI * x) * sin(2 * M_PI * y) *
 * sin(2 * M_PI * z);
 *
 * @version 0.1
 * @date 2018-11-30
 *
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 *
 */

#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

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
#include "solve.hpp"

#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

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
    std::cout << "     Nex: Number of elements in x\n";
    std::cout << "     Ney: Number of elements in y\n";
    std::cout << "     Nez: Number of elements in z\n";
    std::cout << "     use matrix-free: 1 => yes.  0 => matrix-based method.\n";
    std::cout << "     use identity-matrix: 0    use penalty method: 1 \n";
    std::cout << "\n";
    exit(0);
}

void compute_nodal_body_force(double* xe, unsigned int nnode, double* be)
{
    double x, y, z;
    for (unsigned int nid = 0; nid < nnode; nid++)
    {
        x       = xe[nid * 3];
        y       = xe[nid * 3 + 1];
        z       = xe[nid * 3 + 2];
        be[nid] = sin(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * z);
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
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

    // 05.19.20 only use Eigen matrix
    const bool useEigen = true;

    const bool matType          = atoi(argv[4]); // approach (matrix based/free)
    const unsigned int bcMethod = atoi(argv[5]); // method of applying BC

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

    // domain sizes: Lx, Ly, Lz - length of the (global) domain in x, y, z direction
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

    // timing variables
    profiler_t aMat_time;
    profiler_t petsc_time;
    profiler_t petsc_assemble_time;
    if (matType != 0)
    {
        aMat_time.clear();
    }
    else
    {
        petsc_time.clear();
        petsc_assemble_time.clear();
    }

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

    // partition number of elements in z direction...
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

    // number of elements (partition of processes is only in z direction)
    unsigned int nelem_y = Ney;
    unsigned int nelem_x = Nex;
    unsigned int nelem   = (nelem_x) * (nelem_y) * (nelem_z);

    // number of nodes, specify what nodes I own based
    unsigned int nnode, nnode_x, nnode_y, nnode_z;
    if (rank == 0)
    {
        nnode_z = nelem_z + 1;
    }
    else
    {
        nnode_z = nelem_z;
    }
    nnode_y = nelem_y + 1;
    nnode_x = nelem_x + 1;
    nnode   = (nnode_x) * (nnode_y) * (nnode_z);

    // map from local nodes to global nodes
    unsigned int* ndofs_per_element = new unsigned int[nelem];
    for (unsigned e = 0; e < nelem; e++)
    {
        ndofs_per_element[e] = NNODE_PER_ELEM;
    }

    unsigned long int** globalMap;
    globalMap = new unsigned long int*[nelem];
    for (unsigned int e = 0; e < nelem; e++)
    {
        globalMap[e] = new unsigned long int[MAX_BLOCKS_PER_ELEMENT * NNODE_PER_ELEM];
    }
    for (unsigned k = 0; k < nelem_z; k++)
    {
        for (unsigned j = 0; j < nelem_y; j++)
        {
            for (unsigned i = 0; i < nelem_x; i++)
            {
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
    for (unsigned int e = 0; e < nelem; e++)
    {
        bound_nodes[e]  = new unsigned int[ndofs_per_element[e]];
        bound_values[e] = new double[ndofs_per_element[e]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int n = 0; n < ndofs_per_element[eid]; n++)
        {
            nid = globalMap[eid][n];

            // get node coordinates
            x = (double)(nid % (Nex + 1)) * hx;
            y = (double)((nid % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(nid / ((Nex + 1) * (Ney + 1))) * hz;

            // specify boundary nodes
            if ((fabs(x) < zero_number) || (fabs(x - Lx) < zero_number) ||
                (fabs(y) < zero_number) || (fabs(y - Ly) < zero_number) ||
                (fabs(z) < zero_number) || (fabs(z - Lz) < zero_number))
            {
                bound_nodes[eid][n]  = 1; // boundary
                bound_values[eid][n] = 0; // prescribed value
            }
            else
            {
                bound_nodes[eid][n]  = 0;        // interior
                bound_values[eid][n] = -1000000; // for testing
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

    if (matType == 1)
    {
        meshMaps.buildScatterMap();
    }

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
    Vec rhs, out, sol_exact;
    par::create_vec(meshMaps, rhs);
    par::create_vec(meshMaps, out);
    par::create_vec(meshMaps, sol_exact);


    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);

    // compute element stiffness matrix and assemble global stiffness matrix and load vector
    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        for (unsigned int n = 0; n < NNODE_PER_ELEM; n++)
        {
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
        if (useEigen)
        {
            ke_hex8_eig(kee[0], xe, intData.Pts_n_Wts, NGT);
        }
        else
        {
            ke_hex8(ke, xe, intData.Pts_n_Wts, NGT);
        }

        // compute nodal values of body force
        compute_nodal_body_force(xe, 8, be);

        // compute element load vector due to body force
        fe_hex8_eig(fee[0], xe, be, intData.Pts_n_Wts, NGT);

        // assemble element stiffness matrix to global K
        if (matType == 0){
            petsc_assemble_time.start();
        }
        if (matType == 0)
            stMatBased->set_element_matrix(eid, kee[0], 0, 0, 1);
        else
            stMatFree->set_element_matrix(eid, kee[0], 0, 0, 1);
        if (matType == 0){
            petsc_assemble_time.stop();
        }

        // assemble element load vector to global F
        par::set_element_vec(meshMaps, rhs, eid, fee[0], 0u, ADD_VALUES);
    }
    delete[] ke;
    delete[] xe;
    delete[] be;

    // Pestc begins and completes assembling the global stiffness matrix
    if (matType == 0){
        petsc_assemble_time.start();
        stMatBased->finalize();
        petsc_assemble_time.stop();
    } else {
        stMatFree->finalize(); // compute trace of matrix when using penalty method
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
    if (matType == 0)
    {
        //petsc_assemble_time.start();
        stMatBased->finalize();
        //petsc_assemble_time.stop();
    }

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
        par::solve(*stMatBased, (const Vec)rhs, out);
    else
        par::solve(*stMatFree, (const Vec)rhs, out);
    /* if (matType == 0){
        par::solve(*dynamic_cast<aMatBased_ptr>(stMat), (const Vec)rhs, out);
    } else {
        par::solve(*dynamic_cast<aMatFree_ptr>(stMat), (const Vec)rhs, out);
    } */

    //VecAssemblyBegin(out);
    //VecAssemblyEnd(out);
    if (matType != 0)
    {
        aMat_time.stop();
    }
    else
    {
        petsc_time.stop();
    }

    // display timing
    if (matType != 0){
        if (size > 1){
            long double aMat_maxTime;
            MPI_Reduce(&aMat_time.seconds, &aMat_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0){
                std::cout << "aMat time = " << aMat_maxTime << "\n";
            }
        } else {
            if (rank == 0){
                std::cout << "aMat time = " << aMat_time.seconds << "\n";
            }
        }
    } else {
        if (size > 1) {
            long double petsc_maxTime;
            long double petsc_assemble_maxTime;
            MPI_Reduce(&petsc_time.seconds, &petsc_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&petsc_assemble_time.seconds, &petsc_assemble_maxTime, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                std::cout << "PETSC time = " << petsc_maxTime << "\n";
                std::cout << "PETSC assembly time = " << petsc_assemble_maxTime << "\n";
            }
        } else {
            if (rank == 0) {
                std::cout << "PETSC time = " << petsc_time.seconds << "\n";
                std::cout << "PETSC assemble time = " << petsc_assemble_time.seconds << "\n";
            }
        }
    }

    // compute exact solution for comparison
    Matrix<double, NDOF_PER_NODE * NNODE_PER_ELEM, 1> e_exact;

    for (unsigned int e = 0; e < nelem; e++)
    {
        for (unsigned int n = 0; n < NNODE_PER_ELEM; n++)
        {
            // global node ID
            nid = globalMap[e][n];
            // nodal coordinates
            x = (double)(nid % (Nex + 1)) * hx;
            y = (double)((nid % ((Nex + 1) * (Ney + 1))) / (Nex + 1)) * hy;
            z = (double)(nid / ((Nex + 1) * (Ney + 1))) * hz;

            // exact solution at node n (and apply BCs)
            if ((std::abs(x) < zero_number) || (std::abs(x - Lx) < zero_number) ||
                (std::abs(y) < zero_number) || (std::abs(y - Ly) < zero_number) ||
                (std::abs(z) < zero_number) || (std::abs(z - Lz) < zero_number))
            {
                e_exact(n) = 0.0; // boundary
            }
            else
            {
                e_exact(n) = (1.0 / (12.0 * M_PI * M_PI)) * sin(2 * M_PI * x) * sin(2 * M_PI * y) *
                             sin(2 * M_PI * z);
            }
        }
        // set exact solution to Pestc vector
        par::set_element_vec(meshMaps, sol_exact, e, e_exact, 0u, INSERT_VALUES);
    }

    // Pestc begins and completes assembling the exact solution
    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

    // stMat.dump_vec("exact_vec.dat", sol_exact);

    // subtract out from sol_exact
    PetscScalar norm, alpha = -1.0;
    VecAXPY(sol_exact, alpha, out);

    // compute the norm of sol_exact
    //VecNorm(sol_exact, NORM_INFINITY, &norm);
    VecNorm(sol_exact, NORM_2, &norm);

    if (rank == 0)
    {
        //printf("L_inf norm= %20.10f\n", norm);
        printf("L2 norm= %20.10f\n", norm);
    }

#ifdef AMAT_PROFILER
    stMat->profile_dump(std::cout);
#endif

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
        delete[] globalMap[eid];
        delete[] bound_nodes[eid];
        delete[] bound_values[eid];
    }
    delete[] globalMap;
    delete[] bound_nodes;
    delete[] bound_values;

    for (unsigned int eid = 0; eid < nelem; eid++)
    {
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

    // clean up Pestc vectors
    VecDestroy(&out);
    VecDestroy(&sol_exact);
    VecDestroy(&rhs);
    PetscFinalize();

    return 0;
}
