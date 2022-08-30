#ifndef MESHGRAPH_H
#define MESHGRAPH_H

// A C++ program to implement greedy algorithm for graph coloring
// Source: https://www.geeksforgeeks.org/graph-coloring-set-2-greedy-algorithm/
#include <iostream>
#include <list>
#include <vector> 

// A class that take given mesh to convert to a graph, coloring it, then return color of vertices
class meshGraph
{
protected:
	unsigned int V; // No. of vertices (i.e. elements of the mesh)
	std::list<unsigned int> *adj; // A dynamic array of adjacency lists
	unsigned int * color;
	unsigned int rank;
public:
	unsigned int nColors;
public:
	// Constructor and destructor
	meshGraph(unsigned int V, unsigned int rank = 0) { 
		this->V = V;
		this->rank = rank;
		adj = new std::list<unsigned int>[V];
		color = new unsigned int [V];
		nColors = 0;
	}
	~meshGraph(){
		delete [] adj;
		delete [] color;
	}

	// Create graph from a mesh
	void mesh2Graph(unsigned int nDofs, unsigned int nElements, unsigned int * nDofsPerElem, unsigned int ** mesh);
	void mesh2Graph_xfem(unsigned int nDofs, unsigned int nMats, unsigned int matSize, 
		unsigned int * mat2eid, unsigned int * mat2i, unsigned int ** localMesh);

	// greedy coloring of the vertices
	void greedyColoring();
	
	void computeVerticesPerColor(std::vector<unsigned int> * verticesPerColor);

	void printColors();

private:
	// function to add an edge to graph
	void addEdge(unsigned int v, unsigned int w);
};

void meshGraph::addEdge(unsigned int v, unsigned int w)
{
	adj[v].push_back(w);
	adj[w].push_back(v); // Note: the graph is undirected
}

// Assigns colors (starting from 0) to all vertices and prints
// the assignment of colors
void meshGraph::greedyColoring()
{
	// Assign the first color to first vertex
	color[0] = 0;
	nColors = 1; // Han added, for the case of only 1 vertex in the graph

	// Initialize remaining V-1 vertices as unassigned
	for (int u = 1; u < V; u++)
		color[u] = -1; // no color is assigned to u

	// A temporary array to store the available colors. True
	// value of available[cr] would mean that the color cr is
	// assigned to one of its adjacent vertices
	bool available[V];
	for (int cr = 0; cr < V; cr++)
		available[cr] = false;

	// Assign colors to remaining V-1 vertices
	for (int u = 1; u < V; u++)
	{
		// Process all adjacent vertices and flag their colors
		// as unavailable
		std::list<unsigned int>::iterator i;
		for (i = adj[u].begin(); i != adj[u].end(); ++i)
			if (color[*i] != -1)
				available[color[*i]] = true;

		// Find the first available color
		int cr;
		for (cr = 0; cr < V; cr++)
			if (available[cr] == false)
				break;

		color[u] = cr; // Assign the found color

		// Han added, update number of colors
		if (cr + 1 > nColors) {
			nColors = cr + 1;
		}

		// Reset the values back to false for the next iteration
		for (i = adj[u].begin(); i != adj[u].end(); ++i)
			if (color[*i] != -1)
				available[color[*i]] = false;
	}
	
	// print the color
	// for (int u = 0; u < V; u++)
	// 	std::cout << "Vertex " << u << " ---> Color "
	// 		<< color[u] << std::endl;
} // greedyColoring

void meshGraph::computeVerticesPerColor(std::vector<unsigned int> * verticesPerColor) {
	for (unsigned int v = 0; v < V; v++) {
		verticesPerColor[color[v]].push_back(v);
	}
	
	/* for (unsigned int c = 0; c < nColors; c++) {
		printf("Rank %d, color %d, number of vertices: ", rank, c, verticesPerColor[c].size());
		for (unsigned int v = 0; v < verticesPerColor[c].size(); v++) {
			printf("%d, ",verticesPerColor[c][v]);
		}
		printf("\n");
	} */
}

void meshGraph::printColors() {
	std::cout << "Number of colors " << nColors << "\n";
	for (unsigned int u = 0; u < V; u++) {
		std::cout << "Vertex " << u << " ---> Color " << color[u] << std::endl;
	}
}
// create a graph from given mesh
void meshGraph::mesh2Graph(unsigned int nDofs, unsigned int nElements,  unsigned int * nDofsPerElem, unsigned int ** mesh) {
	std::vector<unsigned int> * dof2elem;

	// build map from dof id to element ids contributing to the dof
	dof2elem = new std::vector<unsigned int>[nDofs];
	for (unsigned int e = 0; e < nElements; e++) {
		for (unsigned int d = 0; d < nDofsPerElem[e]; d++) {
			const unsigned int dId = mesh[e][d];
			dof2elem[dId].push_back(e);
		}
	}

	// create graph: loop over dofs, create edges between all elements contributing to the dof
	for (unsigned int gd = 0; gd < nDofs; gd++) {
		for (unsigned int e1 = 0; e1 < dof2elem[gd].size(); e1++) {
			for (unsigned int e2 = (e1 + 1); e2 < dof2elem[gd].size(); e2++) {
				addEdge(dof2elem[gd][e1], dof2elem[gd][e2]);
			}
		}
	}

	// free memory used for dof2elem
	delete [] dof2elem;
} // mesh2Graph


// xfem version: given nMats of non-zero blocks (which are blocks inside an element matrix)
// for each block b = [0,nMat):
//	mat2eid[b] = element id of block b
//	mat2i[b] = i_index of block b
//	mat2j[b] = j_index of block b
// localMesh[e][d] = local index of DoF d of element e
void meshGraph::mesh2Graph_xfem(unsigned int nDofs, unsigned int nMats, unsigned int matSize, 
	unsigned int * mat2eid, unsigned int * mat2i, unsigned int ** localMesh) {
	std::vector<unsigned int> * dof2elem;

	// build map from dof id to block ids contributing to the dof
	dof2elem = new std::vector<unsigned int>[nDofs];
	for (unsigned int b = 0; b < nMats; b++) {
		const unsigned int eid = mat2eid[b];
		// off-diagonal blocks have different local dofs of the diagonal terms of the blocks
		// as this is used for gather_vHost2v, we need only block_i, not block_j
		const unsigned int block_i = mat2i[b];
		const unsigned int block_row_offset = block_i * matSize;
		for (unsigned int r = 0; r < matSize; r++) {
			const unsigned int rowId = localMesh[eid][block_row_offset + r];
			dof2elem[rowId].push_back(b);
		}
	}

	// create graph: loop over dofs, create edges between all elements contributing to the dof
	for (unsigned int gd = 0; gd < nDofs; gd++) {
		for (unsigned int e1 = 0; e1 < dof2elem[gd].size(); e1++) {
			for (unsigned int e2 = (e1 + 1); e2 < dof2elem[gd].size(); e2++) {
				addEdge(dof2elem[gd][e1], dof2elem[gd][e2]);
			}
		}
	}

	// free memory used for dof2elem
	delete [] dof2elem;
} // mesh2Graph_xfem

#endif //MESHGRAPH