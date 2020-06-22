/**
 * @file aVec.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Tran         hantran@cs.utah.edu
 *
 * @brief Hold vectors created based on maps
 * 
 * @version
 * @date   2020.06.19
 * 
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 * 
 */

#ifndef ADAPTIVEMATRIX_VECTOR_H
#define ADAPTIVEMATRIX_VECTOR_H

#include "enums.hpp"
#include "maps.hpp"

#include <petsc.h>
#include <petscvec.h>

namespace par {
    
    /**@brief allocate memory for a PETSc vector "vec", initialized by alpha */
    template<typename MapType>
    static Error create_vec(const MapType& maps, Vec &vec, const PetscScalar alpha = 0.0){
        MPI_Comm m_comm = maps.get_comm();
        unsigned int m_uiRank;
        unsigned int m_uiSize;
        MPI_Comm_rank(m_comm, (int*)&m_uiRank);
        MPI_Comm_size(m_comm, (int*)&m_uiSize);

        VecCreate(m_comm, &vec);
        if (m_uiSize>1) {
            VecSetType(vec,VECMPI);
            VecSetSizes(vec, maps.get_NumDofs(), PETSC_DECIDE);
            VecSet(vec, alpha);
        } else {
            VecSetType(vec,VECSEQ);
            VecSetSizes(vec, maps.get_NumDofs(), PETSC_DECIDE);
            VecSet(vec, alpha);
        }

        return Error::SUCCESS;
    } // create_vec


    /**@brief write PETSc vector "vec" to filename "fvec" */
    template <typename MapType>
    static Error dump_vec( const MapType& maps, Vec vec, const char* filename = nullptr ) {
        MPI_Comm m_comm = maps.get_comm();

        if( filename == nullptr ) {
            VecView( vec, PETSC_VIEWER_STDOUT_WORLD );

        }else {
            PetscViewer viewer;
            // write to ASCII file
            PetscViewerASCIIOpen( m_comm, filename, &viewer );
            // write to file readable by Matlab (filename must be name.m)
            //PetscViewerPushFormat( viewer, PETSC_VIEWER_ASCII_MATLAB );
            VecView( vec, viewer );
            PetscViewerDestroy( &viewer );
        }
        
        return Error::SUCCESS;
    } // dump_vec


    template <typename MapType, typename LI, typename EigenVec>
    static Error set_element_vec( const MapType& maps, Vec vec, LI eid, const EigenVec& e_vec, LI block_i, InsertMode mode = ADD_VALUES ){

        using GI = typename MapType::GIType;
        
        GI** const m_ulpMap = maps.get_Map();

        LI num_dofs_per_block = e_vec.size();
        assert(e_vec.size() == e_vec.rows()); // since EigenVec is defined as matrix with 1 column

        PetscScalar value;
        PetscInt rowId;

        for (LI r = 0; r < num_dofs_per_block; ++r) {
            // this ONLY WORKS with assumption that all blocks have the same number of dofs (that is true for R-XFEM ?)
            rowId = m_ulpMap[eid][block_i * num_dofs_per_block + r];
            value = e_vec(r);
            VecSetValue(vec, rowId, value, mode);
        }
        return Error::SUCCESS;
    }


    /**@brief free memory allocated for PETSc vector*/
    Error delete_vec( Vec &vec ) {
        VecDestroy( &vec );
        return Error::SUCCESS;
    } // delete_vec

} // namespace par
#endif// APTIVEMATRIX_VECTOR_H