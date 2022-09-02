#pragma once
#include <fstream>
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <vector>

//====================================================================================================
/*@brief class to get mesh created by Gmsh, then convert to mesh compatible with aMat 
Notes:
1) in Gmsh, mesh must be exported in ASCII version 2 format
2) every rank reads the same mesh file which is the global mesh created by Gmsh, i.e. every rank stores global mesh
3) future version: each rank should read its owned mesh file, not stores global mesh
4) the mesh could be used for problems having 1/2/3... dofs per node (ndofs per node is provided in the constructor)
5) use get_ functions to have the pointers to data that are fed to aMat
*/
class GmshMesh {
private:
   // properties:
   MPI_Comm m_comm;
   unsigned int m_size;
   unsigned int m_rank;

   // Gmsh element types: http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_63.php
   enum class ELTYPE {UNDEFINED=0, TWO_LINE=1, THREE_TRI=2, FOUR_QUAD=3, FOUR_TET=4,
                     EIGHT_HEX=5, SIX_PRIS=6, FIVE_PYR=7, THREE_2ND_LINE=8, SIX_2ND_TRI=9,
                     NINE_2ND_QUAD=10, TEN_2ND_TET=11, TWENTYSEVEN_2ND_HEX=12,
                     EIGHTEEN_2ND_PRIS=13, FOURTEEN_2ND_PYR=14, ONE_POINT=15,
                     EIGHT_2ND_QUAD=16, TWENTY_2ND_HEX=17, FIFTEEN_2ND_PRIS=18,
                     THIRTEEN_2ND_PYR=19, NINE_3RD_TRI=20, TEN_3RD_TRI=21,
                     TWELVE_4TH_TRI=22, FIFTEEN_4TH_TRI=23, THIRTEEN_5TH_TRI=24,
                     TWENTYONE_5TH_TRI=25, FOUR_3RD_EDGE=26, FIVE_4TH_EDGE=27,
                     SIX_5TH_EDGE=28, TWENTY_3RD_TET=29, THIRTYFIVE_4TH_TET=30, FIFTYSIX_5TH_TET=31};
   
   // number of nodes per element based on ELTYPE
   const unsigned int NNODES_PER_ELEM [32] = {0, 2, 3, 4, 4, 8, 6, 5, 3, 6, 9, 10, 27, 18, 14, 1, 8,
                                             20, 15, 13, 9, 10, 12, 15, 13, 21, 4, 5, 6, 20, 35, 56};
   
   const unsigned int MAX_NNODE_PER_ELEM = 27;  // todo: update this when more elements available
   const unsigned int MAX_ELEM_FLAG = 4;        // todo: currently, Gmsh uses 3 flags for sequential, 4 flags for parallel
   unsigned int m_NDOF_PER_NODE;                // n dofs per node, e.g. potential problem is 1, 2D elasticity is 2, 3D elasticity is 3


   // variables holding data of Gmsh mesh, global ========================================================
   unsigned long m_nNodes_global;               // n nodes for all ranks

   unsigned long * m_nodeLabel_global;          // node label (not used by aMat, since we use the order that a node appears in the list as its id)
   double * m_nodeCoord_global;                 // node coordinates

   unsigned long m_nElems_global;               // n elems for all ranks, bdr and interiror elems
   unsigned long m_nBdrElems_global;            // n bdr elems for all ranks
   unsigned long m_nItrElems_global;            // n interior elems for all ranks
   unsigned long m_nPtrElems_global;             // 2021.12.23, number of point elements
   unsigned long m_nLneElems_global;             // 2021.12.23, number of lines elements

   unsigned long * m_elemLabel_global;          // elem label, bdr & interior elems
   
   unsigned int * m_elemType_global;            // elem type, bdr & interior elems
   unsigned int * m_elemFlag_global;            // elem flags, bdr & interior elems
   unsigned int * m_elemPart_global;            // elem partition, bdr & interior elems

   unsigned long ** m_globalMap_global;         // global map, bdr & interior elems

   unsigned long * m_itrElemLabel_global;       // elem label, only interior elems

   unsigned int * m_itrElemType_global;         // elem type, only interior elems
   unsigned int * m_itrElemFlag_global;         // elem flags, only interior elems
   unsigned int * m_itrElemPart_global;         // elem partition, only interior elems

   unsigned long ** m_itrGlobalMap_global;      // global map, only interior elems

   // variables across ranks =============================================================================
   unsigned int * m_nElemCount;                       // number of owned elems
   std::vector<unsigned long> * m_owned_nodes_rank;   // list of owned global id nodes
   unsigned int * m_nNodeCount;                       // number of owned nodes
   unsigned long * m_nNodeOffset;                     // node offset


   // variables holding data for my rank =================================================================
   std::vector<unsigned long> m_owned_elems;    // list of elements (global) owned by my rank
   unsigned int m_nOwnedElems;                  // n elems owned by my rank
   unsigned int m_nOwnedNodes;                  // n nodes owned by my rank
   unsigned int m_nLocalNodes;                  // n local nodes = (preghost + postghost + owned) nodes
   unsigned int m_nPreGhostNodes;               // n pre ghost nodes
   unsigned int m_nPostGhostNodes;              // n post ghost nodes

   std::vector<unsigned long> m_preGhostGIds;   // list of global-id (aMat global id) of pre ghost nodes
   std::vector<unsigned long> m_postGhostGIds;  // list of global-id (aMat global id) of post ghost nodes
   
   unsigned long * m_Gmsh2aMat_globalNid;       // map Gmsh global node -> aMat global node
   unsigned long * m_aMat2Gmsh_globalNid;       // map aMat global node -> Gmsh global node

   // globalMap and localMap are automatically built in set_GmshMesh()
   unsigned long ** m_globalMap;                // global map (aMat global node) of elements owned by my rank
   unsigned int ** m_localMap;                  // local map of elements owned by my rak

   unsigned long * m_local2GlobalMap;           // map local node id -> global node id
   unsigned int * m_nNodesPerElem;

   // globalDofMap and localDofMap are only built when function get_globalDofMap and get_localDofMap are called
   unsigned long ** m_globalDofMap;             // global dof map of elements owned by my rank
   unsigned int ** m_localDofMap;               // local dof map of elements owned by my rank
   bool m_globalDofMap_allocated;                 // flag to indicate globalDofMap is allocated inside GmshMesh
   bool m_localDofMap_allocated;                  // flag to indicate localDofMap is allocated inside GmshMesh

   unsigned int * m_nDofsPerElem;               // n dofs per element owned by my rank
   unsigned long * m_local2GlobalDofMap;        // local dof --> global dof map

   unsigned int m_nConstraints;                 // n constraints (n dofs that are constrained)
   unsigned long * m_constrainedDofs;           // list of dofs (global) that are constrained
   double * m_prescribedValues;                 // list of prescribed values of constrained dofs

   std::vector<unsigned long> m_bdrNodes;       // bdr nodes (aMat global id) composing elements owned by my rank


   /*===========private methods ============================== */
   /*@brief convert global ID of node, element and GlobalMap to zero-based index */
   void convertZeroBased();

   /*@brief swap order of nodes, making Gmsh element to be equivalent with aMat element matrix */
   void swap_node_order();

   /*@brief determine elements owned by my rank, nElemCount, owned nodes for all rank, nNodeCount, nNodeOffset */
   void compute_owned_data();

   /*@brief build map of element node to global node (aMat global node) */
   void build_globalMap();

   /*@brief build map of element node to local node (i.e. local nodes include ghost nodes & owned nodes) */
   void build_localMap();

   /*@brief extract boundary nodes from mesh */
   void extract_bdr_nodes();

   void compute_nNodesPerElem();

public:
   GmshMesh() {

   }
   
   GmshMesh(MPI_Comm comm, unsigned int nDofPerNode);

   ~GmshMesh();

   /*@brief read ASCII file of mesh generated by Gmsh, after this we have */
   void setGmshMesh(std::string meshFile);

   /*@brief return n owned elements */
   unsigned int get_nOwnedElems() const { return m_nOwnedElems; }

   /*@brief return n owned dofs */
   unsigned int get_nOwnedDofs() const { return m_NDOF_PER_NODE * m_nOwnedNodes; }

   /*@brief return n nodes per element */
   unsigned int * get_nNodesPerElem() { return m_nNodesPerElem; }

   /*@brief return local node map */
   unsigned int * * get_localMap() { return m_localMap; }

   /*@brief return local node map */
   unsigned long * * get_globalMap() { return m_globalMap; }

   /*@brief build and return global dof map */
   unsigned long * * get_globalDofMap();

   /*@brief build and return local dof map */
   unsigned int * * get_localDofMap();

   /*@brief build and return array of ndofs of each element*/
   unsigned int * get_nDofsPerElem();

   /*@brief return number of local dofs */
   unsigned int get_nLocalDofs() const { return m_NDOF_PER_NODE * m_nLocalNodes; };

   /*@brief return number of local dofs */
   unsigned int get_nLocalNodes() const { return m_nLocalNodes; };

   unsigned long * get_local2GlobalMap() const { return m_local2GlobalMap; }

   /*@brief build and return local to global Dof map */
   unsigned long * get_local2GlobalDofMap();

   /*@brief return start dof id owned by my rank */
   // this is not used by aMat, have it here because of Keith's (AFRL) request
   unsigned int get_startGlobalDof();

   /*@brief return end dof id owned by my rank */
   // this is not used by aMat, have it here because of Keith's (AFRL) request
   unsigned int get_endGlobalDof();

   /*@brief return number of dofs of all ranks */
   // this is not used by aMat, have it here because of Keith's (AFRL) request
   unsigned int get_nDofsTotal() const { return m_nNodes_global * m_NDOF_PER_NODE; };

   /*@brief return number of constraints (n dofs that are constrained) */
   // NOTE: THIS FUNCTION ASSUMES ALL BOUNDARY NODES ARE PRESCRIBED
   unsigned int get_n_constraints();

   /*@brief compute list of constrained dofs */
   unsigned long* get_constrained_dofs();

   /*@brief compute prescribed value of displacement for boundary nodes */
   double* get_prescribed_values(void (*coord2Displ)(const double *, double*));

   /*@brief return coordinates of nodes of local node id */
   double get_x(unsigned int localNid);
   double get_y(unsigned int localNid);
   double get_z(unsigned int localNid);
}; // class GmshMesh


class eMatrix {
   private:
   double * m_eMat;
   unsigned int m_nRows;
   unsigned int m_nCols;
   bool m_rowMajor;

   public:
   eMatrix();
   ~eMatrix();
   unsigned int get_nRows() { return m_nRows; }
   unsigned int get_nCols() { return m_nCols; }
   double get_component(unsigned int row, unsigned int col);
   
};