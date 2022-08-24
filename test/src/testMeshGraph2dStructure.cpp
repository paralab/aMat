#include "../../include/meshGraph.hpp"
#include <mpi.h>
#include <functional>
#include <math.h>
#include <algorithm>

int main(int argc, char* argv[]) {
    
    const unsigned int NDOF_PER_NODE  = 1; // number of dofs per node
    const unsigned int NDIM           = 2; // number of dimension
    const unsigned int NNODE_PER_ELEM = 4; // number of nodes per element

    const unsigned int NDOF_PER_ELEM = NDOF_PER_NODE * NNODE_PER_ELEM;
    
    const unsigned int Nex      = atoi(argv[1]);
    const unsigned int Ney      = atoi(argv[2]);

    // domain size
    const double Lx = 100.0;
    const double Ly = Lx; // the above-mentioned exact solution is only applied for Ly = Lx

    // element size
    const double hx = Lx / double(Nex);
    const double hy = Ly / double(Ney);

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // partition in y direction...
    unsigned int emin = 0, emax = 0;
    unsigned int nelem_y;
    // minimum number of elements in y-dir for each rank
    unsigned int nymin = Ney / size;
    // remaining
    unsigned int nRemain = Ney % size;
    // distribute nRemain uniformly from rank = 0 up to rank = nRemain - 1
    if (rank < nRemain) {
        nelem_y = nymin + 1;
    } else {
        nelem_y = nymin;
    }
    if (rank < nRemain) {
        emin = rank * nymin + rank;
    } else {
        emin = rank * nymin + nRemain;
    }
    emax = emin + nelem_y - 1;

    // number of elements owned by my rank
    unsigned int nelem_x = Nex;
    unsigned int nelem   = nelem_x * nelem_y;

    // assign number of nodes owned by my rank: in y direction rank 0 owns 2 boundary nodes, other ranks own right boundary node)
    unsigned int nnode, nnode_y, nnode_x;
    if (rank == 0) {
        nnode_y = nelem_y + 1;
    } else {
        nnode_y = nelem_y;
    }
    nnode_x = Nex + 1;
    nnode   = nnode_x * nnode_y;

    // number of dofs per element (here we use linear 4-node element for this example)
    unsigned int* ndofs_per_element = new unsigned int[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        ndofs_per_element[eid] = NDOF_PER_ELEM;
    }

    // global map...
    unsigned long int** globalMap;
    globalMap = new unsigned long*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        globalMap[eid] = new unsigned long[ndofs_per_element[eid]];
    }
    for (unsigned j = 0; j < nelem_y; j++) {
        for (unsigned i = 0; i < nelem_x; i++) {
            unsigned int eid  = nelem_x * j + i;
            globalMap[eid][0] = (emin * (Nex + 1) + i) + j * (Nex + 1);
            globalMap[eid][1] = globalMap[eid][0] + 1;
            globalMap[eid][3] = globalMap[eid][0] + (Nex + 1);
            globalMap[eid][2] = globalMap[eid][3] + 1;
        }
    }

    // build localDofMap from globalMap (to adapt the interface of bsamxx)
    unsigned int numPreGhostNodes, numPostGhostNodes, numLocalDofs;
    unsigned long gNodeId;
    std::vector<unsigned int> preGhostGIds, postGhostGIds;
    unsigned int* nnodeCount  = new unsigned int[size];
    unsigned int* nnodeOffset = new unsigned int[size];

    MPI_Allgather(&nnode, 1, MPI_UNSIGNED, nnodeCount, 1, MPI_UNSIGNED, comm);

    nnodeOffset[0] = 0;
    for (unsigned int i = 1; i < size; i++) {
        nnodeOffset[i] = nnodeOffset[i - 1] + nnodeCount[i - 1];
    }
    unsigned int ndofs_total;
    ndofs_total = nnodeOffset[size - 1] + nnodeCount[size - 1];
    if (rank == 0)
        printf("Total dofs = %d\n", ndofs_total);

    preGhostGIds.clear();
    postGhostGIds.clear();
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
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
    unsigned int** localDofMap;
    localDofMap = new unsigned int*[nelem];
    for (unsigned int eid = 0; eid < nelem; eid++) {
        localDofMap[eid] = new unsigned int[ndofs_per_element[eid]];
    }
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int i = 0; i < ndofs_per_element[eid]; i++) {
            gNodeId = globalMap[eid][i];
            if (gNodeId >= nnodeOffset[rank] && gNodeId < (nnodeOffset[rank] + nnode)) {
                // nid is owned by me
                localDofMap[eid][i] = gNodeId - nnodeOffset[rank] + numPreGhostNodes;
            }
            else if (gNodeId < nnodeOffset[rank]) {
                // nid is owned by someone before me
                const unsigned int lookUp =
                  std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), gNodeId) -
                  preGhostGIds.begin();
                localDofMap[eid][i] = lookUp;
            }
            else if (gNodeId >= (nnodeOffset[rank] + nnode)) {
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
    for (unsigned int eid = 0; eid < nelem; eid++) {
        for (unsigned int nid = 0; nid < ndofs_per_element[eid]; nid++) {
            gNodeId                                   = globalMap[eid][nid];
            local2GlobalDofMap[localDofMap[eid][nid]] = gNodeId;
        }
    }

    meshGraph m1(nelem, rank);
    m1.mesh2Graph(numLocalDofs, nelem, ndofs_per_element, localDofMap);
    m1.greedyColoring();
    m1.printColors();

    // free allocated memory...
    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] globalMap[eid];
    }
    delete[] globalMap;

    for (unsigned int eid = 0; eid < nelem; eid++) {
        delete[] localDofMap[eid];
    }
    delete [] localDofMap;
    delete [] nnodeCount;
    delete [] nnodeOffset;
    delete [] local2GlobalDofMap;
    delete[] ndofs_per_element;
    
    MPI_Finalize();
    return 0;
}