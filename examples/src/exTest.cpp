#include "exUtils.hpp"
#include <fstream>
#include <iostream>

#ifdef BUILD_WITH_PETSC
#include <petsc.h>
#endif
void computeDispl(const double * xyz, double * uvw) {
   const double E = 1.0E6, nu = 0.3, rho = 1.0, g = 1.0, L = 1.0;
   uvw[0] = (-nu * rho * g)/E * xyz[0] * xyz[2];
   uvw[1] = (-nu * rho * g)/E * xyz[1] * xyz[2];
   uvw[2] =0.5*(rho * g)/E * (xyz[2]*xyz[2] - L*L) + 0.5*(nu * rho * g)/E*(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
}
int main(int argc, char* argv[]) {

   PetscInitialize(&argc, &argv, NULL, NULL);

   int rank, size;
   MPI_Comm comm = PETSC_COMM_WORLD;
   MPI_Status Stat;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   const std::string gmshFile = "cube_24e.msh";
   unsigned long nNodeTotal, nElemTotal, nBdrElemTotal, nItrElemTotal;
   unsigned int nNodeOwned, nElemOwned, nPartition;
   double *nodeCoord;
   unsigned long *globalNodeId, *globalElemId, *globalBdrElemMap, *globalItrElemMap;
   unsigned int *globalElemType;

   GmshMesh mesh(comm, 3);

   mesh.setGmshMesh(gmshFile);

   unsigned int nelem = mesh.get_nOwnedElems();
   printf("number of owned elements= %d\n", nelem);

   unsigned int * * localDofMap = mesh.get_localDofMap();
   unsigned int * nDofsPerElem = mesh.get_nDofsPerElem();
   unsigned int nLocalDofs = mesh.get_nLocalDofs();
   unsigned long * local2GlobalDofMap = mesh.get_local2GlobalDofMap();
   unsigned long startGlobalDof = mesh.get_startGlobalDof();
   unsigned long endGlobalDof = mesh.get_endGlobalDof();

   unsigned long nDofsTotal = mesh.get_nDofsTotal();
   printf("number of dofs total= %d\n", nDofsTotal);

   unsigned int nConstraints = mesh.get_n_constraints();
   printf("number of constraints= %d\n", nConstraints);
   unsigned long * constrainedDofs = mesh.get_constrained_dofs();
   double * prescribedValues = mesh.get_prescribed_values(&computeDispl);

   PetscFinalize();

   return 0;
}