#include "GmshMesh.hpp"
// ================================================================================
GmshMesh::GmshMesh(MPI_Comm comm, unsigned int nDofPerNode) {
   m_comm = comm;
   MPI_Comm_rank(comm, (int*)&m_rank);
   MPI_Comm_size(comm, (int*)&m_size);
   m_NDOF_PER_NODE = nDofPerNode;

   // initialize...
   m_nNodes_global = 0;

   m_nodeLabel_global = nullptr;
   m_nodeCoord_global = nullptr;

   m_nElems_global = 0;
   m_nBdrElems_global = 0;
   m_nItrElems_global = 0;

   m_elemLabel_global = nullptr;
   m_elemType_global = nullptr;
   m_elemFlag_global = nullptr;
   m_elemPart_global = nullptr;

   m_globalMap_global = nullptr;

   m_itrElemLabel_global = nullptr;
   m_itrElemType_global = nullptr;
   m_itrElemFlag_global = nullptr;
   m_itrElemPart_global = nullptr;

   m_itrGlobalMap_global = nullptr;

   m_nElemCount = nullptr;
   m_owned_nodes_rank = nullptr;
   m_nNodeCount = nullptr;
   m_nNodeOffset = nullptr;
   
   m_nOwnedElems = 0;
   m_nOwnedNodes = 0;
   m_nLocalNodes = 0;
   m_nPreGhostNodes = 0;
   m_nPostGhostNodes = 0;

   m_preGhostGIds.clear();
   m_postGhostGIds.clear();

   m_Gmsh2aMat_globalNid = nullptr;
   m_aMat2Gmsh_globalNid = nullptr;

   m_globalMap = nullptr;
   m_localMap = nullptr;

   m_local2GlobalMap = nullptr;

   m_nDofsPerElem = nullptr;
   m_local2GlobalDofMap = nullptr;

   m_nConstraints = 0;
   m_constrainedDofs = nullptr;
   m_prescribedValues = nullptr;

   m_bdrNodes.clear();

   m_globalDofMap = nullptr;
   m_localDofMap = nullptr;
   m_globalDofMap_allocated = false;
   m_localDofMap_allocated = false;

}// GmshMesh constructor


// ================================================================================
GmshMesh::~GmshMesh() {
   delete [] m_nodeLabel_global;
   delete [] m_nodeCoord_global;

   delete [] m_elemLabel_global;
   delete [] m_elemType_global;
   delete [] m_elemFlag_global;
   delete [] m_elemPart_global;

   for (unsigned int e = 0; e < m_nElems_global; e++){
      delete [] m_globalMap_global[e];
   }
   delete [] m_globalMap_global;
   
   delete [] m_nElemCount;
   delete [] m_owned_nodes_rank;
   delete [] m_nNodeCount;
   delete [] m_nNodeOffset;

   delete [] m_Gmsh2aMat_globalNid;
   delete [] m_aMat2Gmsh_globalNid;

   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      delete [] m_globalMap[e];
   }
   delete [] m_globalMap;

   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      delete [] m_localMap[e];
   }
   delete [] m_localMap;
   delete [] m_local2GlobalMap;
   delete [] m_nNodesPerElem;

   if (m_globalDofMap_allocated){
      for (unsigned int e = 0; e < m_nOwnedElems; e++) {
         delete [] m_globalDofMap[e];
      }
      delete [] m_globalDofMap;
   }
   
   if (m_localDofMap_allocated) {
      for (unsigned int e = 0; e < m_nOwnedElems; e++) {
         delete [] m_localDofMap[e];
      }
      delete [] m_localDofMap;
   }
   
   delete [] m_nDofsPerElem;
   delete [] m_local2GlobalDofMap;

   delete [] m_constrainedDofs;
   delete [] m_prescribedValues;
   
}// GmshMesh desstructor


// ================================================================================
// NOTE: to get correct number of iterior elements (which is what aMat want), we assume that
// Gmsh lists all boundary elements before listing iterior elements in the $Elements of Gmsh file
void GmshMesh::setGmshMesh(std::string meshFile){
   
   std::ifstream input;
   input.open(meshFile);

   std::string line_of_text;
   bool start_reading;

   // skip lines until hitting keyword "$Nodes"
   start_reading = false;
   while (!start_reading) 
   {
      std::getline(input, line_of_text);
      if (line_of_text == "$Nodes")
         start_reading = true;
   }

   // read total number of nodes, then allocate variables holding nodal coordinates...
   input >> m_nNodes_global;
   m_nodeLabel_global = new unsigned long [m_nNodes_global];
   m_nodeCoord_global = new double [3 * m_nNodes_global];

   // read global node Id and its coordinates
   for (unsigned int n = 0; n < m_nNodes_global; n++) {
      input >> m_nodeLabel_global[n] >> m_nodeCoord_global[3 * n] >> m_nodeCoord_global[(3 * n) + 1] >> m_nodeCoord_global[(3 * n) + 2];
   }

   // skip lines until hitting keyword "$Elements"
   start_reading = false;
   while (!start_reading) {
      std::getline(input, line_of_text);
      if (line_of_text == "$Elements")
         start_reading = true;
   }

   // read total number of elems including bdr and interior elems
   input >> m_nElems_global;

   // element global Id
   m_elemLabel_global = new unsigned long [m_nElems_global];

   // map from element to global node Id
   m_globalMap_global = new unsigned long * [m_nElems_global];
   for (unsigned int e = 0; e < m_nElems_global; e++){
      m_globalMap_global[e] = new unsigned long [MAX_NNODE_PER_ELEM];
   }

   // element type
   m_elemType_global = new unsigned int [m_nElems_global];

   // partition of element
   m_elemPart_global = new unsigned int [m_nElems_global];

   // element flags (todo: this is internally to Gmsh, still do not understand...)
   m_elemFlag_global = new unsigned int [MAX_ELEM_FLAG * m_nElems_global];
   
   // initialize number of bdr and interior elements
   m_nBdrElems_global = 0;
   m_nItrElems_global = 0;
   m_nPtrElems_global = 0;
   m_nLneElems_global = 0;

   // read global elem id, elem type, elem flags, elem connectivity
   for (unsigned int e = 0; e < m_nElems_global; e++) {
      if (m_size > 1) {
         input >> m_elemLabel_global[e] >> m_elemType_global[e] 
               >> m_elemFlag_global[e * MAX_ELEM_FLAG] >> m_elemFlag_global[e * MAX_ELEM_FLAG + 1] 
               >> m_elemFlag_global[e * MAX_ELEM_FLAG + 2] >> m_elemFlag_global[e * MAX_ELEM_FLAG + 3] 
               >> m_elemPart_global[e];
      } else {
         input >> m_elemLabel_global[e] >> m_elemType_global[e] 
               >> m_elemFlag_global[e * MAX_ELEM_FLAG] >> m_elemFlag_global[e * MAX_ELEM_FLAG + 1] 
               >> m_elemFlag_global[e * MAX_ELEM_FLAG + 2];
         // NOTE: Gmsh use 1-based index, i.e. partitions 1, 2, 3, ...
         m_elemPart_global[e] = 1;
      }
      switch (m_elemType_global[e]) {
         case (unsigned int)ELTYPE::ONE_POINT:
            // 1-node point --> not used by aMat
            m_nPtrElems_global += 1;
            input >> m_globalMap_global[e][0];
            break;

         case (unsigned int)ELTYPE::TWO_LINE:
            // 2-node line --> not used by aMat
            m_nLneElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1];
            break;

         case (unsigned int)ELTYPE::THREE_2ND_LINE:
            // 3-node line --> not used by aMat
            m_nLneElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1] >> m_globalMap_global[e][2];
            break;

         case (unsigned int)ELTYPE::THREE_TRI:
            // 3-node triangle: boundary
            m_nBdrElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1] >> m_globalMap_global[e][2];
            break;

         case (unsigned int)ELTYPE::FOUR_QUAD:
            // 4-node quadrilateral: boundary
            m_nBdrElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1] >> m_globalMap_global[e][2] >> m_globalMap_global[e][3];
            break;

         case (unsigned int)ELTYPE::SIX_2ND_TRI:
            // 6-node triangle (boundary of 10-node tet): boundary
            m_nBdrElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1] >> m_globalMap_global[e][2] >> m_globalMap_global[e][3]
                  >> m_globalMap_global[e][4] >> m_globalMap_global[e][5];
            break;

         case (unsigned int)ELTYPE::NINE_2ND_QUAD:
            // 9-node quadrilateral: boundary
            m_nBdrElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1] >> m_globalMap_global[e][2] >> m_globalMap_global[e][3]
                  >> m_globalMap_global[e][4] >> m_globalMap_global[e][5] >> m_globalMap_global[e][6] >> m_globalMap_global[e][7]
                  >> m_globalMap_global[e][8];
            break;

         case (unsigned int)ELTYPE::FOUR_TET:
            // 4-node tetrahedra: interior
            m_nItrElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1] >> m_globalMap_global[e][2] >> m_globalMap_global[e][3];
            break;

         case (unsigned int)ELTYPE::TEN_2ND_TET:
            // 10-node tetrahedra: interior
            m_nItrElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1] >> m_globalMap_global[e][2] >> m_globalMap_global[e][3]
                  >> m_globalMap_global[e][4] >> m_globalMap_global[e][5] >> m_globalMap_global[e][6] >> m_globalMap_global[e][7]
                  >> m_globalMap_global[e][8] >> m_globalMap_global[e][9];
            break;

         case (unsigned int)ELTYPE::EIGHT_HEX:
            // 8-node hex: interior
            m_nItrElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1] >> m_globalMap_global[e][2] >> m_globalMap_global[e][3]
                  >> m_globalMap_global[e][4] >> m_globalMap_global[e][5] >> m_globalMap_global[e][6] >> m_globalMap_global[e][7];
            break;

         case (unsigned int)ELTYPE::TWENTYSEVEN_2ND_HEX:
            // 27-node hex: interior
            m_nItrElems_global += 1;
            input >> m_globalMap_global[e][0] >> m_globalMap_global[e][1] >> m_globalMap_global[e][2] >> m_globalMap_global[e][3]
                  >> m_globalMap_global[e][4] >> m_globalMap_global[e][5] >> m_globalMap_global[e][6] >> m_globalMap_global[e][7]
                  >> m_globalMap_global[e][8] >> m_globalMap_global[e][9] >> m_globalMap_global[e][10] >> m_globalMap_global[e][11]
                  >> m_globalMap_global[e][12] >> m_globalMap_global[e][13] >> m_globalMap_global[e][14] >> m_globalMap_global[e][15]
                  >> m_globalMap_global[e][16] >> m_globalMap_global[e][17] >> m_globalMap_global[e][18] >> m_globalMap_global[e][19]
                  >> m_globalMap_global[e][20] >> m_globalMap_global[e][21] >> m_globalMap_global[e][22] >> m_globalMap_global[e][23]
                  >> m_globalMap_global[e][24] >> m_globalMap_global[e][25] >> m_globalMap_global[e][26];
            break;
         default:
            std::cout << "element " << e << " has unsupported element type: " << m_elemType_global[e] << "\n";
            exit(1);
            break;
      }
   }
   // close the mesh file
   input.close();

   // a small check
   if (m_nElems_global != (m_nItrElems_global + m_nBdrElems_global + m_nPtrElems_global + m_nLneElems_global)) {
      std::cout << "boundary elements= " << m_nBdrElems_global << ", and interior elements= " << m_nItrElems_global 
               << ", points= " << m_nPtrElems_global << ", lines= " << m_nLneElems_global
               << ", total elements= " << m_nElems_global << "\n";
      exit(1);
   }

   // having number of point, line, surface elements, pointer variables holding data of interior element
   // NOTE: we assume that Gmsh lists elements in order: Points, Lines, Surface, Volume; thus the volume elements starting from (Points + Lines + Surfaces)
   const unsigned int total_nonItr_elements = m_nBdrElems_global + m_nPtrElems_global + m_nLneElems_global;
   m_itrElemLabel_global = &m_elemLabel_global[total_nonItr_elements];
   m_itrElemType_global = &m_elemType_global[total_nonItr_elements];
   m_itrElemFlag_global = &m_elemFlag_global[total_nonItr_elements];
   m_itrElemPart_global = &m_elemPart_global[total_nonItr_elements];
   m_itrGlobalMap_global = &m_globalMap_global[total_nonItr_elements];

   // convert to 0-based index
   convertZeroBased();

   // swap node order of Gmsh elements to make it compatible with aMat element
   swap_node_order();

   // compute data for my rank
   compute_owned_data();
   
   // compute n nodes per element
   compute_nNodesPerElem();
   
   // build global maps
   build_globalMap();

   // build local maps...
   build_localMap();

} // setGmshMesh


// ================================================================================
void GmshMesh::convertZeroBased() {
   // convert node label in nodeLabel array
   for (unsigned int nid = 0; nid < m_nNodes_global; nid++){
      if (m_nodeLabel_global[nid] < 1) {
         std::cout << "Global node id= " << nid << " is out of range\n";
         exit(1);
      }
      m_nodeLabel_global[nid] -= 1;
   }
   // convert node label in globalMap for ALL elements including 1D, 2D elements
   for (unsigned int eid = 0; eid < m_nElems_global; eid++) {
      const unsigned int nnode = NNODES_PER_ELEM[m_elemType_global[eid]];
      for (unsigned int nid = 0; nid < nnode; nid++) {
         m_globalMap_global[eid][nid] -= 1;
      }
   }
   // debug
   /* for (unsigned int eid = 0; eid < m_nElems_global; eid++) {
      const unsigned int nnode = NNODES_PER_ELEM[m_elemType_global[eid]];
      printf("Elem %d: ",eid);
      for (unsigned int nid = 0; nid < nnode; nid++) {
         printf("%d, ", m_globalMap_global[eid][nid]);
      }
      printf("\n");
   } */
} // convertZeroBased


// ================================================================================
void GmshMesh::swap_node_order() {
   // so far, I can see that only quadratic tet element needs to be re-ordered
   for (unsigned int i = 0; i < m_nItrElems_global; i++) {
      if (m_itrElemType_global[i] == (unsigned int)ELTYPE::TEN_2ND_TET) {
         // TET10 element Gmsh uses this order: 0-1-2-3-4-5-6-7-9-8, we need to swap 10 and 9
         const unsigned int node_9 = m_itrGlobalMap_global[i][9];
         m_itrGlobalMap_global[i][9] = m_itrGlobalMap_global[i][8];
         m_itrGlobalMap_global[i][8] = node_9;
      }
   }
} // switch_node_order


// ================================================================================
// m_nElemCount[], m_nNodeCount[], m_nNodeOffset[], m_nOwnedElems, m_nOwnedNodes, m_owned_nodes_rank[p][] 
void GmshMesh::compute_owned_data() {
   // local variables
   bool * nodeOwnerFlag;

   // allocate class-member variables
   m_nElemCount = new unsigned int [m_size];
   m_owned_nodes_rank = new std::vector<unsigned long> [m_size];
   m_nNodeCount = new unsigned int [m_size];
   m_nNodeOffset = new unsigned long [m_size];

   for (unsigned int p = 0; p < m_size; p++) {
      m_nNodeCount[p] = 0;
      m_nElemCount[p] = 0;
   }

   // flag (false/true) indicating if a node is claimed by a rank
   nodeOwnerFlag = new bool [m_nNodes_global];
   for (unsigned int n = 0; n < m_nNodes_global; n++) {
      nodeOwnerFlag[n] = false;
   }

   // loop over all global elements:
   // (1) count number of elements owned by each rank (which is stored the same in all ranks)
   // (2) pick element that owned by my rank and add to the list m_owned_elems
   // (3) pick node composing the element owned by my rank (and not claimed by any other rank yet) and add to the list of owned node (which is stored the same in all ranks)
   for (unsigned int e = 0; e < m_nItrElems_global; e++) {
      const unsigned int p = m_itrElemPart_global[e] - 1; // partition of global element e (since Gmsh uses 1-based index -> convert to 0-based index)
      
      // count 1 to number of owned elements to rank p
      m_nElemCount[p] += 1;

      // add e to the list of owned elements of my rank; NOTE e is the Gmsh global id (only for interior, not include bdr elements)
      if (p == m_rank) m_owned_elems.push_back(e);

      for (unsigned int n = 0; n < NNODES_PER_ELEM[m_itrElemType_global[e]]; n++) {
         const unsigned int gNodeId = m_itrGlobalMap_global[e][n]; // gmsh global node id
         // if gNodeId is not claimed by any rank, add it to the list of owned nodes of rank p
         if (nodeOwnerFlag[gNodeId] == false) {
            m_owned_nodes_rank[p].push_back(gNodeId);
            nodeOwnerFlag[gNodeId] = true;
         }
      }
   }
   
   // sort the list owned nodes for all ranks, then delete repeated nodes
   for (unsigned int p = 0; p < m_size; p++) {
      std::sort(m_owned_nodes_rank[p].begin(), m_owned_nodes_rank[p].end());
      m_owned_nodes_rank[p].erase(std::unique(m_owned_nodes_rank[p].begin(), m_owned_nodes_rank[p].end()),
                                 m_owned_nodes_rank[p].end());
   }
   
   // get n owned nodes & n owned elems by my rank
   m_nOwnedNodes = m_owned_nodes_rank[m_rank].size();
   m_nOwnedElems = m_owned_elems.size();
   
   // a small check ...
   if (m_nOwnedElems != m_nElemCount[m_rank]) {
      std::cout << "GmshMesh::compute_owned_data: rank " << m_rank << "has wrong n owned elements= " << m_nOwnedElems << "\n";
      exit(1);
   }
   
   // one more check...
   unsigned int temp = 0;
   if (m_rank == 0){
      for (unsigned int p = 0; p < m_size; p++) temp += m_nElemCount[p];
      if (temp != m_nItrElems_global) {
         std::cout << "GmshMesh::compute_owned_data, wrong total elements= " << temp << "\n";
         exit(1);
      }
   }
   
   // communicate to get n owned nodes of all ranks (not needed, we already have it)
   //MPI_Allgather(&m_nOwnedNodes, 1, MPI_UNSIGNED, m_nNodeCount, 1, MPI_UNSIGNED, m_comm);
   for (unsigned int p = 0; p < m_size; p++) {
      m_nNodeCount[p] = m_owned_nodes_rank[p].size();
   }
   
   // compute nNodes offset
   m_nNodeOffset[0] = 0;
   for (unsigned int p = 1; p < m_size; p++) {
      m_nNodeOffset[p] = m_nNodeOffset[p - 1] + m_nNodeCount[p - 1];
   }

   delete [] nodeOwnerFlag;

} // compute_owned_data


// ================================================================================
// based on element type provided by Gmsh, compute n nodes per element
void GmshMesh::compute_nNodesPerElem() {
   m_nNodesPerElem = new unsigned int [m_nOwnedElems];
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      const unsigned int nnodes = NNODES_PER_ELEM[m_itrElemType_global[m_owned_elems[e]]];
      m_nNodesPerElem[e] = nnodes;
   }
} // compute_nNodesPerElem


// ================================================================================
// globalMap[e][]
void GmshMesh::build_globalMap() {   

   // build map from Gmsh global node -> aMat node (that satisfy the "contiguous condition")
   m_Gmsh2aMat_globalNid = new unsigned long [m_nNodes_global];
   for (unsigned int p = 0; p < m_size; p++) {
      for (unsigned int n = 0; n < m_nNodeCount[p]; n++) {
         const unsigned int gmshNid = m_owned_nodes_rank[p][n];
         m_Gmsh2aMat_globalNid[gmshNid] = (n + m_nNodeOffset[p]);
      }
   }

   // build map from aMat global node -> Gmsh global node
   m_aMat2Gmsh_globalNid = new unsigned long [m_nNodes_global];
   for (unsigned int i = 0; i < m_nNodes_global; i++) {
      m_aMat2Gmsh_globalNid[m_Gmsh2aMat_globalNid[i]] = i;
   }

   // convert Gmsh global map to aMat global map
   m_globalMap = new unsigned long * [m_nOwnedElems];
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      m_globalMap[e] = new unsigned long [MAX_NNODE_PER_ELEM];
   }

   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      const unsigned int gmshEid = m_owned_elems[e]; // gmsh global element id
      for (unsigned int n = 0; n < NNODES_PER_ELEM[m_itrElemType_global[gmshEid]]; n++) {
         const unsigned int gmshNid = m_itrGlobalMap_global[gmshEid][n]; // gmsh global node id
         m_globalMap[e][n] = m_Gmsh2aMat_globalNid[gmshNid];
      }
      //debug
      /* if (e == 0) {
         printf("gmsh global id of %d is %d\n",e, gmshEid);
      } */
   }

   //debug
   /* printf("Rank %d, n owned elems= %d, n bdr elems= %d, n Ptr elems= %d, n Lne elems= %d \n", m_rank, m_nOwnedElems, m_nBdrElems_global, m_nPtrElems_global, m_nLneElems_global);
   for (unsigned int e = 0; e < m_nOwnedElems; e++){
      printf("Element %d: ",e);
      for (unsigned int n = 0; n < m_nNodesPerElem[e]; n++) {
         printf(" %d",m_globalMap[e][n]);
      }
      printf("\n");
   } */
   /* for (unsigned int n = 0; n < m_nNodes_global; n++) {
      printf("Gmsh node %d, aMat node %d\n",n,m_Gmsh2aMat_globalNid[n]);
      printf("aMat node %d, Gmsh node %d\n",n,m_aMat2Gmsh_globalNid[n]);
   } */

} // build_globalMap


// ================================================================================
void GmshMesh::build_localMap() {
   m_localMap = new unsigned int * [m_nOwnedElems];
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      m_localMap[e] = new unsigned int [MAX_NNODE_PER_ELEM];
   }
   
   // determine pre and post ghost nodes based on global map
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      const unsigned int nnodes = m_nNodesPerElem[e]; // number of nodes of element e
      for (unsigned int n = 0; n < nnodes; n++) {
         const unsigned int aMatGNid = m_globalMap[e][n]; // global node id
         if (aMatGNid < m_nNodeOffset[m_rank]) {
            m_preGhostGIds.push_back(aMatGNid);
         }
         else if (aMatGNid >= m_nNodeOffset[m_rank]) {
            m_postGhostGIds.push_back(aMatGNid);
         }
      }
   }
   
   // sort and delete repeated ghost nodes...
   std::sort(m_preGhostGIds.begin(), m_preGhostGIds.end());
   std::sort(m_postGhostGIds.begin(), m_postGhostGIds.end());
   m_preGhostGIds.erase(std::unique(m_preGhostGIds.begin(), m_preGhostGIds.end()), m_preGhostGIds.end());
   m_postGhostGIds.erase(std::unique(m_postGhostGIds.begin(), m_postGhostGIds.end()), m_postGhostGIds.end());
   m_nPreGhostNodes = m_preGhostGIds.size();
   m_nPostGhostNodes = m_postGhostGIds.size();
   
   // update number of local nodes
   m_nLocalNodes = m_nPreGhostNodes + m_nPostGhostNodes + m_nOwnedNodes;

   // build localMap
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      const unsigned int nnodes = m_nNodesPerElem[e];
      for (unsigned int n = 0; n < nnodes; n++) {
         const unsigned int aMatGNid = m_globalMap[e][n];
         if ((aMatGNid >= m_nNodeOffset[m_rank]) && (aMatGNid < (m_nNodeOffset[m_rank] + m_nOwnedNodes))) {
            m_localMap[e][n] = aMatGNid - m_nNodeOffset[m_rank] + m_nPreGhostNodes;
         } 
         else if (aMatGNid < m_nNodeOffset[m_rank]) {
            const unsigned int lookUp = std::lower_bound(m_preGhostGIds.begin(), m_preGhostGIds.end(), aMatGNid) - m_preGhostGIds.begin();
            m_localMap[e][n] = lookUp;
         } 
         else if (aMatGNid >= (m_nNodeOffset[m_rank] + m_nOwnedNodes)) {
            const unsigned int lookUp = std::lower_bound(m_postGhostGIds.begin(), m_postGhostGIds.end(), aMatGNid) - m_postGhostGIds.begin();
            m_localMap[e][n] = m_nPreGhostNodes + m_nOwnedNodes + lookUp;
         }
      }
   }

   // build map from aMat local node --> aMat global node
   m_local2GlobalMap = new unsigned long [m_nLocalNodes];

   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      const unsigned int nnodes = m_nNodesPerElem[e];
      for (unsigned int n = 0; n < nnodes; n++) {
         m_local2GlobalMap[m_localMap[e][n]] = m_globalMap[e][n];
      }
   }

}// build localMap


// ================================================================================
unsigned long * * GmshMesh::get_globalDofMap() {
   m_globalDofMap = new unsigned long * [m_nOwnedElems];
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      m_globalDofMap[e] = new unsigned long [MAX_NNODE_PER_ELEM * m_NDOF_PER_NODE];
   }
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      const unsigned int nnodes = m_nNodesPerElem[e];
      for (unsigned int n = 0; n < nnodes; n++) {
         for (unsigned int d = 0; d < m_NDOF_PER_NODE; d++) {
            m_globalDofMap[e][n * m_NDOF_PER_NODE + d] = m_globalMap[e][n] * m_NDOF_PER_NODE + d;
         }
      }
   }
   // debug
   /* for (unsigned int e = 0; e < m_nOwnedElems; e++){
      printf("Element %d: ",e);
      for (unsigned int n = 0; n < m_nNodesPerElem[e]; n++) {
         for (unsigned int d = 0; d < m_NDOF_PER_NODE; d++) {
            printf("%d: %d, ",(n*m_NDOF_PER_NODE + d), m_globalDofMap[e][n*m_NDOF_PER_NODE + d]);
         }
         
      }
      printf("\n");
   } */
   return m_globalDofMap;
} // get_globalDofMap


// ================================================================================
unsigned int * * GmshMesh::get_localDofMap() {
   m_localDofMap = new unsigned int * [m_nOwnedElems];
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      m_localDofMap[e] = new unsigned int [MAX_NNODE_PER_ELEM * m_NDOF_PER_NODE];
   }
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      const unsigned int nnodes = m_nNodesPerElem[e];
      for (unsigned int n = 0; n < nnodes; n++) {
         for (unsigned int d = 0; d < m_NDOF_PER_NODE; d++) {
            m_localDofMap[e][n * m_NDOF_PER_NODE + d] = m_localMap[e][n] * m_NDOF_PER_NODE + d;
         }
      }
   }
   return m_localDofMap;
} // get_localDofMap


// ================================================================================
unsigned int * GmshMesh::get_nDofsPerElem() {
   m_nDofsPerElem = new unsigned int [m_nOwnedElems];
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      const unsigned int nnodes = m_nNodesPerElem[e];
      m_nDofsPerElem[e] = m_NDOF_PER_NODE * nnodes;
   }
   return m_nDofsPerElem;
} // compute_nDofsPerElem


// ================================================================================
unsigned long * GmshMesh::get_local2GlobalDofMap() {
   m_local2GlobalDofMap = new unsigned long [m_nLocalNodes * m_NDOF_PER_NODE];
   for (unsigned int e = 0; e < m_nOwnedElems; e++) {
      const unsigned int nnodes = m_nNodesPerElem[e];
      for (unsigned int n = 0; n < nnodes; n++) {
         for (unsigned int d = 0; d < m_NDOF_PER_NODE; d++) {
            m_local2GlobalDofMap[(m_localMap[e][n] * m_NDOF_PER_NODE) + d] = (m_globalMap[e][n] * m_NDOF_PER_NODE) + d;
         }
      }
   }
   return m_local2GlobalDofMap;
} // get_local2GlobalDofMap


// ================================================================================
unsigned int GmshMesh::get_startGlobalDof() {
   const unsigned int start_global_node = m_nNodeOffset[m_rank];
   return start_global_node * m_NDOF_PER_NODE;
} // get_startGlobalDof


// ================================================================================
unsigned int GmshMesh::get_endGlobalDof() {
   const unsigned int end_global_node = m_nNodeOffset[m_rank] + m_nOwnedNodes - 1;
   return ((end_global_node * m_NDOF_PER_NODE) + (m_NDOF_PER_NODE - 1));
} // get_endGlobalDof


// ================================================================================
// this function to extract nodes that compose bdr elements owned by my rank
// these bdr nodes could be NOT owned by me, but it is ok for aMat to have that
// output is the list m_bdrNodes containing aMat global id of bdr nodes composing bdr elements owned by my rank
void GmshMesh::extract_bdr_nodes() {
   
   //debug
   //printf("n points elems= %d, n lines elems= %d, n bdr elems= %d, n itr elems= %d\n", m_nPtrElems_global, m_nLneElems_global, m_nBdrElems_global, m_nItrElems_global);
   
   // loop on bdr elements of all ranks
   // NOTE: we assume that Gmsh lists elements in the order: point elements, line elements, surface elements, volume elements
   for (unsigned int e = (m_nPtrElems_global + m_nLneElems_global); e < (m_nPtrElems_global + m_nLneElems_global) + m_nBdrElems_global; e++) {
      // only continue if eid belongs to my rank (NOTE: m_elemPart_global is not converted to 0-based index yet)
      if (m_elemPart_global[e] == (m_rank + 1)) {
         const unsigned int nnodes = NNODES_PER_ELEM[m_elemType_global[e]];
         
         for (unsigned int n = 0; n < nnodes; n++) {
            //printf("elem %d, nnodes= %d, n= %d, gmsh id= %d, aMat id= %d\n",e,nnodes,n,m_globalMap_global[e][n],m_Gmsh2aMat_globalNid[m_globalMap_global[e][n]]);
            m_bdrNodes.push_back(m_Gmsh2aMat_globalNid[m_globalMap_global[e][n]]); // aMat global node id of bdr node
         }
      }
   }
   
   // sort in order to delete repeated nodes (sort is in #include<algorithm>)
   std::sort(m_bdrNodes.begin(), m_bdrNodes.end());
   // delete repeated nodes
   m_bdrNodes.erase(std::unique(m_bdrNodes.begin(), m_bdrNodes.end()), m_bdrNodes.end());

} // extract_bdr_nodes


// ================================================================================
unsigned int GmshMesh::get_n_constraints() { 
   // extract boundary nodes
   extract_bdr_nodes();
   return (m_bdrNodes.size() * m_NDOF_PER_NODE); 
}


// ================================================================================
unsigned long * GmshMesh::get_constrained_dofs(){

   const unsigned int nBdrNodes = m_bdrNodes.size(); // number of boundary nodes composing elements belong to my rank

   // total number of constraints
   m_constrainedDofs = new unsigned long [m_NDOF_PER_NODE * nBdrNodes];

   unsigned int count = 0;
   for (unsigned int i = 0; i < nBdrNodes; i++) {
      // get global id of boundary node i
      const unsigned int bdrNodeId = m_bdrNodes[i];
      // each node has n dofs
      for (unsigned int j = 0; j < m_NDOF_PER_NODE; j++){
         m_constrainedDofs[count] = (bdrNodeId * m_NDOF_PER_NODE) + j;
         count += 1;
      }
   }

   // debug
   /* printf("rank %d, number of constrained nodes= %d\n",m_rank, nBdrNodes);
   for (unsigned int i = 0; i < nBdrNodes; i++){
      printf("rank %d, constrained node# %d, aMat global node id= %d, gmsh global node id= %d\n", m_rank, i, m_bdrNodes[i], m_aMat2Gmsh_globalNid[m_bdrNodes[i]]);
   } */

   return m_constrainedDofs;
} // compute_constrained_dofs


// ================================================================================
double* GmshMesh::get_prescribed_values(void (*coord2Displ)(const double *, double *)){

   const unsigned int nBdrNodes = m_bdrNodes.size();
   m_prescribedValues = new double [m_NDOF_PER_NODE * nBdrNodes];
   double * uvw = new double [m_NDOF_PER_NODE];

   unsigned int count = 0;
   for (unsigned int i = 0; i < nBdrNodes; i++) {
      const unsigned int aMat_gBdrNid = m_bdrNodes[i]; // aMat global id of boundary node i

      //const double * xyz = &m_nodeCoord_global[bdrNodeId]; // bug found 2021_12_23: bdrNodeId is aMat (global) node id, not gmsh (global) node id
      const unsigned int gmsh_gBdrNid = m_aMat2Gmsh_globalNid[aMat_gBdrNid]; // gmsh global node id of bdr node i
      const double * xyz = &m_nodeCoord_global[gmsh_gBdrNid * m_NDOF_PER_NODE]; //pointer to position storing x,y,z of this global node id
      
      /* printf("bdr node %d, amat id= %d, gmsh id= %d\n", i, aMat_gBdrNid, gmsh_gBdrNid);
      printf("coordinates= %f,%f,%f\n", xyz[0], xyz[1], xyz[2]); */

      // compute displacement using provided function
      coord2Displ(xyz, uvw);

      for (unsigned int j = 0; j < m_NDOF_PER_NODE; j++){
         m_prescribedValues[count] = uvw[j];
         // debug
         //printf("get_prescribed_values: constrained dof %d, prescribed value= %f\n", count, m_prescribedValues[count]);
         count += 1;
      }
   }
   delete [] uvw;
   return m_prescribedValues;
} // compute_prescribed_values


// ================================================================================
double GmshMesh::get_x(unsigned int localNid) {
   const unsigned int aMat_gNid = m_local2GlobalMap[localNid];
   const unsigned int Gmsh_gNid = m_aMat2Gmsh_globalNid[aMat_gNid];
   return m_nodeCoord_global[Gmsh_gNid * 3];
}


// ================================================================================
double GmshMesh::get_y(unsigned int localNid) {
   const unsigned long aMat_gNid = m_local2GlobalMap[localNid];
   const unsigned long Gmsh_gNid = m_aMat2Gmsh_globalNid[aMat_gNid];
   return m_nodeCoord_global[Gmsh_gNid * 3 + 1];
}


// ================================================================================
double GmshMesh::get_z(unsigned int localNid) {
   const unsigned long aMat_gNid = m_local2GlobalMap[localNid];
   const unsigned long Gmsh_gNid = m_aMat2Gmsh_globalNid[aMat_gNid];
   return m_nodeCoord_global[Gmsh_gNid * 3 + 2];
}