#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h>

#include <petsc.h>

#include <Eigen/Dense>
#include "sparse_linear_algebra/parlab/aMat.h"

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;

using GlobalIndex = int;
using LocalIndex = int;

//////////////////////////////////////////////////////////////////////////////////////////////////////

int main( int argc, char *argv[] ) {

    PetscInitialize(&argc, &argv, NULL, NULL);

    int rank, size;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Status Stat;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    par::aMat<double,GlobalIndex,LocalIndex> mat(par::AMAT_TYPE::MAT_FREE);
    mat.set_comm(comm);
    
    std::string line;
    std::ifstream debug_dump_file("parlab_dump_r" + std::to_string(rank) + ".txt");

    ////////////////////////////////////////////////////
    // Read dof information from dump file
    
    // Skip line:   # of elements:
    std::getline(debug_dump_file, line);
    LocalIndex n_elements_on_rank;
    debug_dump_file >> n_elements_on_rank;

    // Skip line:   # dofs per element and element dof maps (# of dofs   dof1   dof2   etc.):
    std::getline(debug_dump_file, line);
    std::vector<LocalIndex> element_n_dofs(n_elements_on_rank, 0);
    std::vector<LocalIndex*> element_to_rank_maps(n_elements_on_rank, nullptr);
    for (LocalIndex e = 0; e < n_elements_on_rank; e++)
    {
        debug_dump_file >> element_n_dofs[e];
        element_to_rank_maps[e] = new LocalIndex[element_n_dofs[e]];
        for (LocalIndex i = 0; i < element_n_dofs[e]; i++)
        {
            debug_dump_file >> element_to_rank_maps[e][i];
        }
    }

    // Skip line:   # rank dofs:
    std::getline(debug_dump_file, line);
    LocalIndex n_rank_dofs;
    debug_dump_file >> n_rank_dofs;
    
    // Skip line:   rank-to-global map:
    std::getline(debug_dump_file, line);
    std::vector<GlobalIndex> rank_to_global_map(n_rank_dofs, 0);
    for (LocalIndex i = 0; i < n_rank_dofs; i++)
    {
        LocalIndex local_dof;
        debug_dump_file >> local_dof;
        GlobalIndex global_dof;
        debug_dump_file >> global_dof;
        rank_to_global_map[local_dof] = global_dof;
    }

    // Skip line:   owned global begin:
    std::getline(debug_dump_file, line);
    GlobalIndex owned_global_dof_begin;
    debug_dump_file >> owned_global_dof_begin;

    // Skip line:   owned global end:
    std::getline(debug_dump_file, line);
    GlobalIndex owned_global_dof_end;
    debug_dump_file >> owned_global_dof_end;

    // Skip line:   # global dofs:
    std::getline(debug_dump_file, line);
    GlobalIndex n_global_dofs;
    debug_dump_file >> n_global_dofs;
    
    mat.set_map(
        n_elements_on_rank,
        element_to_rank_maps.data(),
        element_n_dofs.data(),
        n_rank_dofs,
        rank_to_global_map.data(),
        owned_global_dof_begin,
        owned_global_dof_end,
        n_global_dofs);


    ////////////////////////////////////////////////////
    // Read element matrices

    std::vector<int> element_n_blocks(n_elements_on_rank);
    std::vector<Eigen::MatrixXd*> element_matrices(n_elements_on_rank);

    for (LocalIndex e = 0; e < n_elements_on_rank; e++)
    {
        // Skip:  Element
        debug_dump_file >> line;
        LocalIndex element_number;
        debug_dump_file >> element_number;
        // Skip:  blocks:
        debug_dump_file >> line;
        LocalIndex n_block_rows;
        debug_dump_file >> n_block_rows;

        element_n_blocks[e] = n_block_rows;
        element_matrices[e] = new Eigen::MatrixXd[n_block_rows*n_block_rows];

        for (auto block_i = 0; block_i < n_block_rows; block_i++)
        {
            for (auto block_j = 0; block_j < n_block_rows; block_j++)
            {
                LocalIndex block_i_assert;
                debug_dump_file >> block_i_assert;
                LocalIndex block_j_assert;
                debug_dump_file >> block_j_assert;
                assert(block_i == block_i_assert && block_j == block_j_assert);

                int n_dofs_in_block;
                debug_dump_file >> n_dofs_in_block;

                auto index = block_i * n_block_rows + block_j;
                auto& curr_block = element_matrices[e][index];
                curr_block.resize(n_dofs_in_block, n_dofs_in_block);
                for (auto i = 0; i < n_dofs_in_block; i++)
                {
                    for (auto j = 0; j < n_dofs_in_block; j++)
                    {
                        debug_dump_file >> curr_block(i, j);
                    }
                }
            }
        }
    }

    ////////////////////////////////////////////////////
    // Read and set boundary conditions

    // Skip line:   #
    debug_dump_file >> line;
    // Skip line:   of constrained dofs:
    std::getline(debug_dump_file, line);
    LocalIndex n_constraints;
    debug_dump_file >> n_constraints;

    std::vector<GlobalIndex> constrained_global_dofs(n_constraints);
    std::vector<double> constrained_values(n_constraints);
    for (LocalIndex i = 0; i < n_constraints; i++)
    {
        debug_dump_file >> constrained_global_dofs[i];
        debug_dump_file >> constrained_values[i];
    }

    mat.set_bdr_map(
            constrained_global_dofs.data(),
            constrained_values.data(),
            n_constraints);

    debug_dump_file.close();

    ////////////////////////////////////////////////////
    // Set element matrices

    for (LocalIndex e = 0; e < n_elements_on_rank; e++)
    {
        const auto n_block_rows = element_n_blocks[e];
        for (auto block_i = 0; block_i < n_block_rows; block_i++)
        {
            for (auto block_j = 0; block_j < n_block_rows; block_j++)
            {
                auto index = block_i * n_block_rows + block_j;
                auto& curr_block = element_matrices[e][index];
                if (curr_block.rows() > 0)
                {
                    mat.copy_element_matrix(
                        e, curr_block, block_i, block_j, n_block_rows);
                }
            }
        }
    }

    ////////////////////////////////////////////////////
    // Solve boundary value problem

    Vec rhs, solution;
    mat.petsc_create_vec(rhs);
    mat.petsc_create_vec(solution);

    mat.petsc_init_vec(rhs);
    mat.petsc_finalize_vec(rhs);
    
    mat.apply_bc_rhs(rhs);
    mat.petsc_init_vec(rhs);
    mat.petsc_finalize_vec(rhs);

    mat.petsc_solve(rhs, solution);
    mat.petsc_init_vec(solution);
    mat.petsc_finalize_vec(solution);

    
    ////////////////////////////////////////////////////
    // Convert solution to Eigen vector for comparison

    Vec local_vec;

    // Need to copy the whole vector to rank 0 for this...
    //
    VecScatter to_zero;
    VecScatterCreateToZero(solution, &to_zero, &local_vec);
    VecScatterBegin(to_zero, solution, local_vec, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(to_zero, solution, local_vec, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterDestroy(&to_zero);

    PetscViewer viewer;
    PetscViewerCreate(MPI_COMM_WORLD, &viewer);
    PetscViewerSetType(viewer, PETSCVIEWERASCII);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);

    VecView(solution, viewer);
    VecView(local_vec, viewer);

    PetscScalar* raw_data;
    VecGetArray(local_vec, &raw_data);

    PetscInt lsize, gsize;
    VecGetSize(local_vec, &gsize);
    VecGetLocalSize(local_vec, &lsize);

    Eigen::Map<Eigen::VectorXd> solution_eigen(&raw_data[0], lsize, 1);

    // TODO compare to reference

    PetscFinalize();
    return 0;
}