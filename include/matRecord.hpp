/**
 * @file matRecord.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Tran         hantran@cs.utah.edu
 *
 * @brief Classes to hold record of matrix terms (row, column, value, ...)
 * 
 * @version
 * @date   2020.06.18
 * 
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 * 
 */

#ifndef ADAPTIVEMATRIX_MATRECORD_H
#define ADAPTIVEMATRIX_MATRECORD_H

#include <mpi.h>
#include <vector>

namespace par {

    // Class MatRecord
    //      DT => type of data stored in matrix (eg: double). LI => size of local index.
    template <typename Dt, typename Li>
    class MatRecord {
    private:
        unsigned int m_uiRank;
        Li m_uiRowId;
        Li m_uiColId;
        Dt m_dtVal;
    public:
        MatRecord(){
            m_uiRank = 0;
            m_uiRowId = 0;
            m_uiColId = 0;
            m_dtVal = 0;
        }
        MatRecord(unsigned int rank, Li rowId, Li colId, Dt val){
            m_uiRank = rank;
            m_uiRowId = rowId;
            m_uiColId = colId;
            m_dtVal = val;
        }

        unsigned int getRank() const { return m_uiRank; }
        Li getRowId() const { return m_uiRowId; }
        Li getColId() const { return m_uiColId; }
        Dt getVal()   const { return m_dtVal; }

        void setRank(  unsigned int rank ) { m_uiRank = rank; }
        void setRowId( Li rowId ) { m_uiRowId = rowId; }
        void setColId( Li colId ) { m_uiColId = colId; }
        void setVal(   Dt val ) {   m_dtVal = val; }

        bool operator == (MatRecord const &other) const {
            return ((m_uiRank == other.getRank())&&(m_uiRowId == other.getRowId())&&(m_uiColId == other.getColId()));
        }

        bool operator < (MatRecord const &other) const {
            if (m_uiRank < other.getRank()) return true;
            else if (m_uiRank == other.getRank()) {
                if (m_uiRowId < other.getRowId()) return true;
                else if (m_uiRowId == other.getRowId()) {
                    if (m_uiColId < other.getColId()) return true;
                    else return false;
                }
                else return false;
            }
            else {
                return false;
            }
        }

        bool operator <= (MatRecord const &other) const { return (((*this) < other) || (*this) == other); }

        ~MatRecord() {}
    }; // class MatRecord
    
    //==============================================================================================================
    // DT => type of data stored in matrix (eg: double). LI => size of local index.
    // class is used to define MPI_Datatype for MatRecord
    template <typename dt, typename li>
    class MPI_datatype_matrecord{
    public:
        static MPI_Datatype value(){
            static bool first = true;
            static MPI_Datatype mpiDatatype;
            if (first){
                first=false;
                MPI_Type_contiguous(sizeof(MatRecord<dt,li>),MPI_BYTE,&mpiDatatype);
                MPI_Type_commit(&mpiDatatype);
            }
            return mpiDatatype;
        }
    }; // class MPI_datatype_matrecord

} // namespace par
#endif// APTIVEMATRIX_MATRECORD_H