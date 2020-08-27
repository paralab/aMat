/**
 * @file Enums.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Duc Tran     hantran@cs.utah.edu
 *
 * @brief all enum classes used in par::aMat
 * 
 * @version 0.1
 * @date 2020-06-09
 * 
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 * 
 */

#ifndef ADAPTIVEMATRIX_ENUMS_H
#define ADAPTIVEMATRIX_ENUMS_H

namespace par {
/**@brief method of applying boundary conditions */
enum class BC_METH { BC_IMATRIX,
    BC_PENALTY };

/**@brief types of error used in functions of aMat class */
enum class Error { SUCCESS,
    INDEX_OUT_OF_BOUNDS,
    UNKNOWN_ELEMENT_TYPE,
    UNKNOWN_ELEMENT_STATUS,
    NULL_L2G_MAP,
    GHOST_NODE_NOT_FOUND,
    UNKNOWN_MAT_TYPE,
    WRONG_COMMUNICATION,
    UNKNOWN_DOF_TYPE,
    UNKNOWN_CONSTRAINT,
    UNKNOWN_BC_METH,
    NOT_IMPLEMENTED,
    GLOBAL_DOF_ID_NOT_FOUND };

/**@brief  */
enum class DOF_TYPE { FREE,
    PRESCRIBED,
    UNDEFINED };

/**@brief timers for profiling */
enum class PROFILER { MATVEC = 0,
    MATVEC_MUL,
    MATVEC_ACC,
    PETSC_ASS,
    PETSC_MATVEC,
    PETSC_KfcUc,
    LAST };

enum class MATRIX_TYPE { MATRIX_BASED,
    MATRIX_FREE };

} // namespace par
#endif // APTIVEMATRIX_ENUMS_H