/**
 * @file read_gmsh.cpp
 * @author Han Tran
 * @author Milinda Fernando
 * @brief 
 * @version 0.1
 * @date 2021-04-08
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "aMatUtils.hpp"
#include <mpi.h>

int main(int argc, char* argv[])
{
    const std::string fname = argv[1];
    const unsigned int npe_bdy = atoi(argv[2]);
    const unsigned int npe_interior = atoi(argv[3]);

    MPI_Init(&argc,&argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;

    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&npes);

    read_tet_msh_file_partitioned<unsigned int, unsigned long, double>(fname,npes,npe_interior,npe_bdy,comm);

    
    MPI_Finalize();
    return 0;


}