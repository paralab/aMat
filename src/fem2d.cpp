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
#include "aMat.hpp"
#include "integration.hpp"

using Eigen::Matrix;

void usage()
{
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  fem2d <Nex> <Ney> <matrix based/free> <bc method>\n";
    std::cout << "\n";
    std::cout << "     Nex: Number of elements in X\n";
    std::cout << "     Ney: Number of elements in y\n";
    std::cout << "     use matrix-free: 1 => yes.  0 => matrix-based method.\n";
    std::cout << "     use identity-matrix: 0    use penalty method: 1 \n";
    std::cout << "\n";
    std::exit( 0 ) ;
}

int main(int argc, char *argv[]){
    // User provides: Nex = number of elements in x direction
    //                Ney = number of elements in y direction
    //                flag = 1 --> matrix-free method; 0 --> matrix-based method
    //                bcMethod = 0 --> identity matrix method; 1 --> penalty method
    if( argc < 5 ) {
        usage();
    }

    const unsigned int NDOF_PER_NODE = 1;       // number of dofs per node
    const unsigned int NDIM = 2;                // number of dimension
    const unsigned int NNODE_PER_ELEM = 4;      // number of nodes per element

    const unsigned int Nex = atoi(argv[1]);
    const unsigned int Ney = atoi(argv[2]);
    const unsigned int matType = atoi(argv[3]);
    const unsigned int bcMethod = atoi(argv[4]); // method of applying BC

    // element matrix and force vector
    Matrix<double,4,4> ke;
    Matrix<double,4,1> fe;

    // element nodal coordinates
    double* xe = new double [8];

    // domain size
    const double Lx = 1.0;
    const double Ly = 1.0;

    // element size
    const double hx = Lx/double(Nex);
    const double hy = Ly/double(Ney);

    const double zero_number = 1E-12;

    // MPI initialize
    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(!rank) {
        std::cout << "============ parameters read  =======================\n";
        std::cout << "\t\tNex : "<< Nex << " Ney: " << Ney << "\n";
        std::cout << "\t\tLx : "<< Lx << " Ly: " << Ly << "\n";
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
    if (size > Ney){
        if (!rank){
            std::cout << "Number of ranks must be <= Ney, program stops..." << "\n";
            MPI_Abort(comm, rc);
            exit(0);
        }
    }

    // partition in y direction...
    unsigned int emin = 0, emax = 0;
    unsigned int nelem_y;
    // minimum number of elements in y-dir for each rank
    unsigned int nymin = Ney/size;
    // remaining
    unsigned int nRemain = Ney % size;
    // distribute nRemain uniformly from rank = 0 up to rank = nRemain - 1
    if (rank < nRemain){
        nelem_y = nymin + 1;
    } else {
        nelem_y = nymin;
    }
    if (rank < nRemain){
        emin = rank * nymin + rank;
    } else {
        emin = rank * nymin + nRemain;
    }
    emax = emin + nelem_y - 1;

    // number of elements owned by my rank
    unsigned int nelem_x = Nex;
    unsigned int nelem = nelem_x * nelem_y;

    // number of nodes owned by my rank (rank 0 owns 2 boundary nodes, other ranks own right boundary node)
    unsigned int nnode, nnode_y, nnode_x;
    if (rank == 0){
        nnode_y = nelem_y + 1;
    } else {
        nnode_y = nelem_y;
    }
    nnode_x = Nex + 1;
    nnode = nnode_x * nnode_y;

    // determine globalMap
    unsigned int* nnode_per_elem = new unsigned int [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        nnode_per_elem[eid] = NNODE_PER_ELEM; //linear 4-node element
    }

    unsigned long int** globalMap;
    globalMap = new unsigned long* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        globalMap[eid] = new unsigned long [nnode_per_elem[eid]];
    }
    // todo hard-code 4 nodes per element:
    for (unsigned j = 0; j < nelem_y; j++){
        for (unsigned i = 0; i < nelem_x; i++){
            unsigned int eid = nelem_x * j + i;
            globalMap[eid][0] = (emin*(Nex + 1) + i) + j*(Nex + 1);
            globalMap[eid][1] = globalMap[eid][0] + 1;
            globalMap[eid][3] = globalMap[eid][0] + (Nex + 1);
            globalMap[eid][2] = globalMap[eid][3] + 1;
        }
    }

    // build localMap from globalMap (to adapt the interface of bsamxx)
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalNodes;
    unsigned long gNodeId;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;
    unsigned int* nnodeCount = new unsigned int [size];
    unsigned int* nnodeOffset = new unsigned int [size];

    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    nnodeOffset[0] = 0;
    for (unsigned int i = 1; i < size; i++){
        nnodeOffset[i] = nnodeOffset[i-1] + nnodeCount[i-1];
    }
    unsigned int nnode_total;
    nnode_total = nnodeOffset[size-1] + nnodeCount[size-1];
    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
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
    numLocalNodes = numPreGhostNodes + nnode + numPostGhostNodes;
    unsigned int** localMap;
    localMap = new unsigned int* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        localMap[eid] = new unsigned int[nnode_per_elem[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int i = 0; i < nnode_per_elem[eid]; i++){
            gNodeId = globalMap[eid][i];
            if (gNodeId >= nnodeOffset[rank] &&
                gNodeId < (nnodeOffset[rank] + nnode)) {
                // nid is owned by me
                localMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
            } else if (gNodeId < nnodeOffset[rank]){
                // nid is owned by someone before me
                const unsigned int lookUp = std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) - preGhostGIds.begin();
                localMap[eid][i] = lookUp;
            } else if (gNodeId >= (nnodeOffset[rank] + nnode)){
                // nid is owned by someone after me
                const unsigned int lookUp = std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), gNodeId) - postGhostGIds.begin();
                localMap[eid][i] =  numPreGhostNodes + nnode + lookUp;
            }
        }
    }
    // build local2GlobalMap map (to adapt the interface of bsamxx)
    unsigned long * local2GlobalMap = new unsigned long[numLocalNodes];
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            gNodeId = globalMap[eid][nid];
            local2GlobalMap[localMap[eid][nid]] = gNodeId;
        }
    }

    // compute constrained map
    unsigned int** bound_nodes = new unsigned int* [nelem];
    double** bound_values = new double* [nelem];
    for (unsigned int eid = 0; eid < nelem; eid++){
        bound_nodes[eid] = new unsigned int [nnode_per_elem[eid]];
        bound_values[eid] = new double [nnode_per_elem[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            unsigned long global_Id = globalMap[eid][nid];
            double x = (double)(global_Id % (Nex + 1)) * hx;
            double y = (double)(global_Id / (Nex + 1)) * hy;
            if ((fabs(x) < zero_number) || (fabs(x - Lx) < zero_number)){
                // left or right boundary
                bound_nodes[eid][nid] = 1;
                bound_values[eid][nid] = 0.0;
            } else if (fabs(y) < zero_number){
                // bottom boundary
                bound_nodes[eid][nid] = 1;
                bound_values[eid][nid] = sin(M_PI * x);
            } else if (fabs(y - Ly) < zero_number){
                // top boundary
                bound_nodes[eid][nid] = 1;
                bound_values[eid][nid] = sin(M_PI * x) * exp(-M_PI);
            } else {
                // iterior
                bound_nodes[eid][nid] = 0;
                bound_values[eid][nid] = -1000000;
            }
        }
    }

    // create lists of constrained dofs
    std::vector< par::ConstrainedRecord<double, unsigned long int> > list_of_constraints;
    par::ConstrainedRecord<double, unsigned long int> cdof;
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++) {
            for (unsigned int did = 0; did < NDOF_PER_NODE; did++) {
                if (bound_nodes[eid][(nid * NDOF_PER_NODE) + did] == 1) {
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

    unsigned long start_global_node, end_global_node;
    start_global_node = nnodeOffset[rank];
    end_global_node = start_global_node + (nnode - 1);


    // declare aMat object =================================
    par::aMat<double, unsigned long, unsigned int> * stMat;
    if (matType == 0){
        stMat = new par::aMatBased<double, unsigned long, unsigned int>((par::BC_METH)bcMethod);
    } else {
        stMat = new par::aMatFree<double, unsigned long, unsigned int>((par::BC_METH)bcMethod);
    }

    stMat->set_comm(comm);
    stMat->set_map(nelem, localMap, nnode_per_elem, numLocalNodes, local2GlobalMap, start_global_node, end_global_node, nnode_total);
    stMat->set_bdr_map(constrainedDofs_ptr, prescribedValues_ptr, list_of_constraints.size());

    Vec rhs, out;
    stMat->petsc_create_vec(rhs);
    stMat->petsc_create_vec(out);

    // Gauss points and weights
    const unsigned int NGT = 2;
    integration<double> intData(NGT);

    // element stiffness matrix, force vector, and assembly...
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            unsigned long global_Id = globalMap[eid][nid];
            xe[nid * 2] = (double)(global_Id % (Nex + 1)) * hx;
            xe[(nid * 2) + 1] = (double)(global_Id / (Nex + 1)) * hy;
        }
        ke_quad4_eig(ke, xe, intData.Pts_n_Wts, NGT);
        // for this example, no force vector
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            fe(nid) = 0.0;
        }
        // assemble ke
        stMat->set_element_matrix(eid, ke, 0, 0, 1);
        // assemble fe
        stMat->petsc_set_element_vec(rhs, eid, fe, 0, ADD_VALUES);
    }
    delete [] xe;

    PetscScalar norm, alpha = -1.0;

    // Pestc begins and completes assembling the global stiffness matrix
    if (matType == 0){
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }

    char fname[256];
    sprintf(fname,"matrix_%d.dat",matType);
    stMat->petsc_dump_mat(fname);

    // These are needed because we used ADD_VALUES for rhs when assembling
    // now we are going to use INSERT_VALUE for Fc in apply_bc_rhs
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    sprintf(fname,"rhs_beforebc_%d.dat",matType);
    stMat->petsc_dump_vec(rhs,fname);
    
    // apply bc for rhs: this must be done before applying bc for the matrix
    // because we use the original matrix to compute KfcUc in matrix-based method
    stMat->apply_bc(rhs); // this includes applying bc for matrix in matrix-based approach
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    // communication for matrix-based approach
    if (matType == 0){
        //stMat->apply_bc_mat();
        stMat->petsc_init_mat(MAT_FINAL_ASSEMBLY);
        stMat->petsc_finalize_mat(MAT_FINAL_ASSEMBLY);
    }

    sprintf(fname,"rhs_%d.dat",matType);
    stMat->petsc_dump_vec(rhs,fname);

    // solve
    stMat->petsc_solve((const Vec) rhs, out);

    VecNorm(out, NORM_2, &norm);
    if (!rank){
        printf("L2 norm of computed solution = %f\n",norm);
    }

    // exact solution...
    Vec sol_exact;
    PetscInt rowId;
    stMat->petsc_create_vec(sol_exact);
    for (unsigned int eid = 0; eid < nelem; eid++){
        for (unsigned int nid = 0; nid < nnode_per_elem[eid]; nid++){
            rowId = globalMap[eid][nid];
            double x = (double)(rowId % (Nex + 1)) * hx;
            double y = (double)(rowId / (Nex + 1)) * hy;
            PetscScalar u = sin(M_PI * x) * exp(-M_PI * y);
            VecSetValue(sol_exact, rowId, u, INSERT_VALUES);
        }
    }
    VecAssemblyBegin(sol_exact);
    VecAssemblyEnd(sol_exact);

    VecNorm(sol_exact, NORM_2, &norm);
    if (!rank){
        printf("L2 norm of exact solution = %f\n",norm);
    }

    // compute the norm of error
    VecAXPY(sol_exact, alpha, out);
    VecNorm(sol_exact, NORM_INFINITY, &norm);

    if (!rank){
        printf("Inf norm of error = %20.10f\n", norm);
    }

    // free allocated memory...
    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] bound_nodes[eid];
        delete [] bound_values[eid];
    }
    delete [] bound_nodes;
    delete [] bound_values;

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] globalMap[eid];
    }
    delete [] globalMap;

    for (unsigned int eid = 0; eid < nelem; eid++){
        delete [] localMap[eid];
    }
    delete [] localMap;
    delete [] nnodeCount;
    delete [] nnodeOffset;
    delete [] local2GlobalMap;
    delete [] nnode_per_elem;

    PetscFinalize();
    return 0;
}