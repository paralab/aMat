/**
 * @file constraintRecord
 *.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Tran         hantran@cs.utah.edu
 *
 * @brief Class to hold record of constraints (constrained dof, prescribed value)
 * 
 * @version
 * @date   2020.06.18
 * 
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 * 
 */

#ifndef ADAPTIVEMATRIX_CONSTRAINT_H
#define ADAPTIVEMATRIX_CONSTRAINT_H

#include <mpi.h>
#include <vector>

namespace par {
    // Class ConstraintRecord
    //      DT => type of data stored in matrix (eg: double)
    template <typename DT, typename GI>
    class ConstraintRecord {
        private:
        GI dofId; // global dof Id that is constrained (i.e. Dirichlet BC)
        DT preVal;// prescribed valua for the constrained dof

        public:
        ConstraintRecord() {
            dofId = 0;
            preVal = 0.0;
        }

        GI get_dofId() const { return dofId; }
        DT get_preVal() const {return preVal; }

        void set_dofId(GI id) { dofId = id; }
        void set_preVal(DT value) { preVal = value; }

        bool operator == (ConstraintRecord const &other) const {
            return (dofId == other.get_dofId());
        }
        bool operator < (ConstraintRecord const &other) const {
            if (dofId < other.get_dofId()) return true;
            else return false;
        }
        bool operator <= (ConstraintRecord const &other) const {
            return (((*this) < other) || ((*this) == other));
        }

        ~ConstraintRecord() {}
    };// class ConstrainedDof
    
} // namespace par
#endif// APTIVEMATRIX_CONSTRAINT_H