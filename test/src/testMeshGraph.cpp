#include "../../include/meshGraph.hpp"
int main()
{
    const unsigned int nElems = 12;
    const unsigned int nDofs = 17;
    unsigned int * nDofsPerElem;
    unsigned int ** mesh;

    nDofsPerElem = new unsigned int[nElems];
    for (unsigned int e = 0; e < nElems; e++) {
        nDofsPerElem[e] = 4;
    }
    mesh = new unsigned*[nElems];
    for (unsigned int e = 0; e < nElems; e++) {
        mesh[e] = new unsigned int[nDofsPerElem[e]];
    }

    // define mesh
    mesh[0][0] = 0; mesh[0][1] = 1; mesh[0][2] = 4; mesh[0][3] = 3;
    mesh[1][0] = 1; mesh[1][1] = 2; mesh[1][2] = 5; mesh[1][3] = 4;
    mesh[2][0] = 3; mesh[2][1] = 4; mesh[2][2] = 5; mesh[2][3] = 6;
    mesh[3][0] = 6; mesh[3][1] = 5; mesh[3][2] = 10; mesh[3][3] = 9;
    mesh[4][0] = 5; mesh[4][1] = 2; mesh[4][2] = 7; mesh[4][3] = 10;
    mesh[5][0] = 9; mesh[5][1] = 10; mesh[5][2] = 7; mesh[5][3] = 8;
    mesh[6][0] = 0; mesh[6][1] = 3; mesh[6][2] = 12; mesh[6][3] = 11;
    mesh[7][0] = 3; mesh[7][1] = 6; mesh[7][2] = 13; mesh[7][3] = 12;
    mesh[8][0] = 11; mesh[8][1] = 12; mesh[8][2] = 13; mesh[8][3] = 14;
    mesh[9][0] = 13; mesh[9][1] = 16; mesh[9][2] = 15; mesh[9][3] = 14;
    mesh[10][0] = 6; mesh[10][1] = 9; mesh[10][2] = 16; mesh[10][3] = 13;
    mesh[11][0] = 9; mesh[11][1] = 8; mesh[11][2] = 15; mesh[11][3] = 16;

    // graph object
    meshGraph m1(nElems);
    
    // convert mesh to graph
    m1.mesh2Graph(nDofs, nElems, nDofsPerElem, mesh);

    // coloring graph and print out
    std::cout << "Coloring of graph 1 \n";
    m1.greedyColoring();
    m1.printColors();

    delete [] nDofsPerElem;
    for (unsigned int e = 0; e < nElems; e++) {
        delete [] mesh[e];
    }
    delete [] mesh;
    return 0;
}