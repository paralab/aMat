/**
 * @brief this code to read an unstructured mesh of all hex from text file
 * @brief then generate the matrix resulted from Laplace equation grad^2(u)
 * @brief to run the code: ./matgen mesh_filename
 * @brief the matrix is stored in file "matrix.out"
 */
#include <iostream>
#include <fstream>
#include <string>

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

// number of cracks allowed in 1 element
#define MAX_CRACK_LEVEL 0

// max number of block dimensions in one cracked element
#define MAX_BLOCKS_PER_ELEMENT (1u << MAX_CRACK_LEVEL)

void compute_nodal_body_force(double* xe, unsigned int nnode, double L, double* be) {
    double x, y, z;
    for (unsigned int nid = 0; nid < nnode; nid++){
        x       = xe[nid * 3];
        y       = xe[nid * 3 + 1];
        z       = xe[nid * 3 + 2];
        be[nid] = sin(2 * M_PI * (x/L)) * sin(2 * M_PI * (y/L)) * sin(2 * M_PI * (z/L));
    }
}

int main(int argc, char* argv[]){

   typedef unsigned int IT; // tyype for index
   typedef double DT; // type for data

   const unsigned int NNODE_PER_ELEM = 8; // number of nodes per element
   const unsigned int NDOF_PER_NODE  = 1; // number of dofs per node
   const unsigned int NDIM           = 3; // number of dimension
   const DT zero_number = 1E-12;
   unsigned long nid, eid;
   DT x, y, z;
   const DT L = 100.0;
   unsigned int matType = 0; // use matrix-based method
   unsigned int bcMethod = 0; // bc method
   const bool useEigen = true;

   // element matrix (contains multiple matrix blocks)
   std::vector<Matrix<DT, NDOF_PER_NODE * NNODE_PER_ELEM, NDOF_PER_NODE * NNODE_PER_ELEM>> kee;
   kee.resize(MAX_BLOCKS_PER_ELEMENT * MAX_BLOCKS_PER_ELEMENT);
   // nodal coordinates of element
   DT* xe = new DT[NDIM * NNODE_PER_ELEM];
   // nodal body force
   DT* be = new DT[NNODE_PER_ELEM];
   // matrix block
   DT* ke = new DT[(NDOF_PER_NODE * NNODE_PER_ELEM) * (NDOF_PER_NODE * NNODE_PER_ELEM)];
   // element force vector (contains multiple vector blocks)
   std::vector<Matrix<DT, NDOF_PER_NODE * NNODE_PER_ELEM, 1>> fee;
   fee.resize(MAX_BLOCKS_PER_ELEMENT);

   // Gauss points and weights
   const unsigned int NGT = 2;
   integration<DT> intData(NGT);

   PetscInitialize(&argc, &argv, NULL, NULL);

   int rank, size;
   MPI_Comm comm = PETSC_COMM_WORLD;
   MPI_Status Stat;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);
   
   // mesh file name
   const std::string str = argv[1];

   std::ifstream input;
   
   // open mesh file
   input.open(str);
   
   // ignore 4 lines
   std::string str_temp;
   for (unsigned int i = 0; i < 4; i++){
      std::getline(input, str_temp);
   }

   // read number of nodes
   IT nnode;
   input >> nnode;
   
   // read nodal coordinates
   IT * nodeId = new IT [nnode];
   DT * nodeCoord = new DT [nnode * 3];
   for (IT i = 0; i < nnode; i++){
      input >> nodeId[i] >> nodeCoord[3*i] >> nodeCoord[3*i + 1] >> nodeCoord[3*i + 2];
   }
   
   // ignore next 2 lines (after reading the last coordinate, the cursor is still in the same line
   // that is why we need to go 3 times to read the next 2 lines
   for (unsigned int i = 0; i < 3; i++){
      std::getline(input, str_temp);
   }

   // read number of elements
   IT nelem, nBdrElem, nIntElem;
   input >> nelem;
   
   // read element connectivity
   IT * elemId = new IT [nelem];
   IT * elemType = new IT [nelem];
   IT * * elemFlag = new IT* [nelem];
   IT * * globalMap_temp = new IT* [nelem];
   for (IT e = 0; e < nelem; e++){
      elemFlag[e] = new IT [3];
      globalMap_temp[e] = new IT [8];
   }
   
   nBdrElem = 0;
   nIntElem = 0;
   for (IT e = 0; e < nelem; e++){
      input >> elemId[e] >> elemType[e] >> elemFlag[e][0] >> elemFlag[e][1] >> elemFlag[e][2];
      if (elemType[e] == 3){
         nBdrElem += 1;
         input >> globalMap_temp[e][0] >> globalMap_temp[e][1] >> globalMap_temp[e][2] >> globalMap_temp[e][3];
      } else if (elemType[e] == 5){
         nIntElem += 1;
         input >> globalMap_temp[e][0] >> globalMap_temp[e][1] >> globalMap_temp[e][2] >> globalMap_temp[e][3]
            >> globalMap_temp[e][4] >> globalMap_temp[e][5] >> globalMap_temp[e][6] >> globalMap_temp[e][7];
      } else {
         printf("error: element %d has un-recognized id = %d\n", e, elemId[e]);
         exit(1);
      }
   }

   // close the input file
   input.close();

   // check
   if ((nBdrElem + nIntElem) != nelem){
      printf("N boundary elems = %d, N interior elems = %d, not equal total elems = %d\n", 
            nBdrElem, nIntElem, nelem);
      exit(1);
   }

   // convert to 0 index of node id in globalMap
   for (IT e = 0; e < nBdrElem; e++){
      for (IT i = 0; i < 4; i++){
         globalMap_temp[e][i] -= 1;
      }
   }
   for (IT e = nBdrElem; e < nelem; e++){
      for (IT i = 0; i < NNODE_PER_ELEM; i++){
         globalMap_temp[e][i] -= 1;
      }
   }

   // extract boundary nodes from boundary elements
   // assume that boundary elements are always at the beginning of the list
   // create lists of constrained dofs
   std::vector<par::ConstraintRecord<double, unsigned long int>> list_of_constraints;
   par::ConstraintRecord<double, unsigned long int> cdof;
   for (IT eid = 0; eid < nBdrElem; eid++) {
      for (unsigned int nid = 0; nid < 4; nid++) {
         cdof.set_dofId(globalMap_temp[eid][nid]);
         cdof.set_preVal(0.0);
         list_of_constraints.push_back(cdof);
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

   // connectivity of interior elements
   IT * * globalMap = new IT* [nIntElem];
   for (IT e = 0; e < nIntElem; e++){
      globalMap[e] = globalMap_temp[e + nBdrElem];
   }
   // get rid of boundary elements, update nelem
   nelem = nIntElem;

   // switch the order of nodes to 4-5-6-8-0-1-2-3
   IT nodes1234[4];
   for (IT e = 0; e < nelem; e++){
      memcpy(nodes1234, globalMap[e], 4 * sizeof(IT));
      memcpy(globalMap[e], &globalMap[e][4], 4 * sizeof(IT));
      memcpy(&globalMap[e][4], nodes1234, 4 * sizeof(IT));
   }

 
   // output read parameters
   /* std::cout << "nnode= " << nnode << "\n";
   for (IT i = 0; i < nnode; i++){
      std::cout << nodeId[i] << "; " << nodeCoord[3*i] <<
      "; " << nodeCoord[3*i + 1] << "; " << nodeCoord[3*i + 2] << "\n";
   }
   std::cout << "nelem= " << nelem << "\n";
   for (IT e = 0; e < nelem; e++){
      for (unsigned int i = 0; i < NNODE_PER_ELEM; i++){
         std::cout << globalMap[e][i] << "; ";
      }
      std::cout << "\n";
   }
   std::cout << "total boundary nodes= " << list_of_constraints.size() << "\n";
   for (IT n = 0; n < list_of_constraints.size(); n++){
      std::cout << "node: " << constrainedDofs_ptr[n] << "; value= " << prescribedValues_ptr[n] << "\n";
   } */

   
   IT* ndofs_per_element = new IT [nelem];
   for (IT e = 0; e < nelem; e++) {
      ndofs_per_element[e] = NNODE_PER_ELEM;
   }

   // build localDofMap from globalMap
   IT numPreGhostNodes, numPostGhostNodes, numLocalDofs;
   unsigned long gNodeId;
   std::vector<IT> preGhostGIds, postGhostGIds;

   IT* nnodeCount  = new IT[size];
   IT* nnodeOffset = new IT[size];

   MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

   nnodeOffset[0] = 0;
   for (unsigned int i = 1; i < size; i++) {
      nnodeOffset[i] = nnodeOffset[i - 1] + nnodeCount[i - 1];
   }
   IT ndofs_total;
   ndofs_total = nnodeOffset[size - 1] + nnodeCount[size - 1];
   if (rank == 0)
      printf("Total dofs = %d\n", ndofs_total);

   preGhostGIds.clear();
   postGhostGIds.clear();
   for (IT eid = 0; eid < nelem; eid++) {
      for (unsigned int nid = 0; nid < 8; nid++) {
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

   IT** localDofMap;
   localDofMap = new IT*[nelem];
   for (IT e = 0; e < nelem; e++) {
      localDofMap[e] = new IT[8];
   }

   for (IT eid = 0; eid < nelem; eid++) {
      for (unsigned int i = 0; i < 8; i++) {
         gNodeId = globalMap[eid][i];
         if (gNodeId >= nnodeOffset[rank] && gNodeId < (nnodeOffset[rank] + nnode)) {
            // nid is owned by me
            localDofMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
         } else if (gNodeId < nnodeOffset[rank]) {
            // nid is owned by someone before me
            const IT lookUp =
            std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) -
            preGhostGIds.begin();
            localDofMap[eid][i] = lookUp;
         } else if (gNodeId >= (nnodeOffset[rank] + nnode)) {
            // nid is owned by someone after me
            const IT lookUp =
            std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), gNodeId) -
            postGhostGIds.begin();
            localDofMap[eid][i] = numPreGhostNodes + nnode + lookUp;
         }
      }
   }

   // build local2GlobalDofMap map (to adapt the interface of bsamxx)
   unsigned long* local2GlobalDofMap = new unsigned long[numLocalDofs];
   for (IT eid = 0; eid < nelem; eid++) {
      for (unsigned int nid = 0; nid < 8; nid++) {
         gNodeId                                   = globalMap[eid][nid];
         local2GlobalDofMap[localDofMap[eid][nid]] = gNodeId;
      }
   }

   unsigned long start_global_dof, end_global_dof;
   start_global_dof = nnodeOffset[rank];
   end_global_dof   = start_global_dof + (nnode - 1);

   printf("start global dof = %d, end global dof = %d\n", start_global_dof, end_global_dof);

   // declare Maps object  =================================
   par::Maps<DT, unsigned long, IT> meshMaps(comm);

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
   typedef par::aMat<par::aMatBased<DT, unsigned long, IT>, DT, unsigned long, IT>
      aMatBased; // aMat type taking aMatBased as derived class
   typedef par::aMat<par::aMatFree<DT, unsigned long, IT>, DT, unsigned long, IT>
      aMatFree; // aMat type taking aMatBased as derived class

   aMatBased* stMatBased; // pointer of aMat taking aMatBased as derived
   aMatFree* stMatFree;   // pointer of aMat taking aMatFree as derived

   if (matType == 0){
      // assign stMatBased to the derived class aMatBased
      stMatBased = new par::aMatBased<DT, unsigned long, IT>(meshMaps, (par::BC_METH)bcMethod);
   } else {
      // assign stMatFree to the derived class aMatFree
      stMatFree = new par::aMatFree<DT, unsigned long, IT>(meshMaps, (par::BC_METH)bcMethod);
      stMatFree->set_matfree_type((par::MATFREE_TYPE)matType);
   }

   // create rhs, solution and exact solution vectors
   Vec rhs, out, sol_exact;
   par::create_vec(meshMaps, rhs);
   par::create_vec(meshMaps, out);
   par::create_vec(meshMaps, sol_exact);

   // compute element stiffness matrix and assemble global stiffness matrix and load vector
   for (IT eid = 0; eid < nelem; eid++) {
      for (unsigned int n = 0; n < NNODE_PER_ELEM; n++) {
         // node id
         nid = globalMap[eid][n];
         // get node coordinates
         x = nodeCoord[nid*3];
         y = nodeCoord[nid*3 + 1];
         z = nodeCoord[nid*3 + 2];
         xe[n * 3]       = x;
         xe[(n * 3) + 1] = y;
         xe[(n * 3) + 2] = z;
      }

      // compute element stiffness matrix
      if (useEigen) {
         ke_hex8_eig(kee[0], xe, intData.Pts_n_Wts, NGT);
      } else {
         printf("Error: no longer use non-Eigen matrix for element stiffness matrix \n");
         exit(0);
      }

      // assemble element stiffness matrix to global K
      if (matType == 0)
         stMatBased->set_element_matrix(eid, kee[0], 0, 0, 1);
      else
         stMatFree->set_element_matrix(eid, kee[0], 0, 0, 1);

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
      stMatBased->finalize();
   } else {
      stMatFree->finalize();
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

   if (matType == 0) {
      stMatBased->finalize();
   }

   /* if (matType == 0)
      par::solve(*stMatBased, (const Vec)rhs, out);
   else
      par::solve(*stMatFree, (const Vec)rhs, out); */

   // print out matrix to file
   if (matType == 0){
      stMatBased->dump_mat("matrix.out");
   } else {
      stMatFree->dump_mat("matrix.out");
   }

   // print out rhs to file
   par::dump_vec(meshMaps, rhs, "rhs.out");


   delete [] nodeCoord;
   delete [] nodeId;
   delete [] elemId;
   delete [] elemType;
   delete [] elemFlag;
   for (IT i = 0; i < nelem; i++){
      delete [] globalMap_temp[i];
   }
   delete [] globalMap_temp;
   delete [] globalMap;

   // clean up Pestc vectors
   VecDestroy(&out);
   VecDestroy(&sol_exact);
   VecDestroy(&rhs);
   PetscFinalize();

   return 0;
}