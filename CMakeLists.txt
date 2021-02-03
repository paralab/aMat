cmake_minimum_required(VERSION 3.7)

# set name of project, which can be used by ${PROJECT_NAME}
project(fem3d)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)

# find and load settings from external project, if not found then stop with error message
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)


set(LAPACK_LINKER_FLAGS -llapacke -llapack -lblas -lgfortran -lquadmath)
set(LAPACKE_DIR $ENV{LAPACK}/LAPACKE)
set(LINK_FLAGS "${LINK_FLAGS} ${LAPACK_LINKER_FLAGS}")
set(LAPACK_LIBRARIES ${LAPACK_LIBRARIES} ${LAPACKE_LIB})
message(STATUS ${LAPACK_LIBRARIES})
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    message("${CMAKE_CXX_COMPILER_ID} compiler detected adding -mkl flag for BLAS LAPACK")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mkl")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")
endif()

# when OpenMP is found
if(OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(MPI_COMPILE_FLAGS)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set(LINK_FLAGS "${LINK_FLAGS} ${MPI_LINK_FLAGS}")
endif()

set(INCLUDE_FILES
        include/aMat.hpp
        include/aMatBased.hpp
        include/aMatFree.hpp
        include/asyncExchangeCtx.hpp
        include/aVec.hpp
        include/constraintRecord.hpp
        include/enums.hpp
        include/maps.hpp
        include/matRecord.hpp
        include/profiler.hpp
        include/lapack_extern.hpp
        examples/include/fe_vector.hpp
        examples/include/integration.hpp
        examples/include/ke_matrix.hpp
        examples/include/me_matrix.hpp
        examples/include/shapeFunc.hpp
        examples/include/solve.hpp
        include/aMatGpu.hpp)

set(SOURCE_FILES
        examples/src/ke_matrix.cpp
        examples/src/fe_vector.cpp)

#set (MAGMA_DIR /uufs/chpc.utah.edu/sys/installdir/magma/2.3.0-i18.1-cuda9.1/)
#set(CUDA_DIR /uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85)
# Note: the environmental variables MAGMA_DIR and CUDA_DIR has to be correct path to where MAGMA and CUDA are installed
set (MAGMA_DIR $ENV{MAGMA_DIR})
set (CUDA_DIR $ENV{CUDA_DIR})

add_definitions(-DNOCHANGE)
add_definitions(-DMAGMA_WITH_MKL)

# cmake options, which will be visible at ccmake ../
option(BUILD_WITH_PETSC "Build code with the petsc" ON)
option(AMAT_PROFILER "turn on the amat profiler counters" OFF)
option(VECTORIZED_AVX512 "vectorization using AVX-512" OFF)
option(VECTORIZED_AVX256 "vectorization using AVX-256" OFF)
option(VECTORIZED_OPENMP "vectorization using OpenMP SIMD" OFF)
option(VECTORIZED_OPENMP_ALIGNED "vectorization using OpenMP SIMD with aligned memory" OFF)

option(HYBRID_PARALLEL "hybrid parallelism OpenMP and MPI" ON)
option(USE_GPU "use GPU for matvec" OFF)

# if BUILD_WITH_PETSC ON , #define BUILD_WITH_PETSC
if(BUILD_WITH_PETSC)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
    find_package(PETSc REQUIRED)
    add_definitions(-DBUILD_WITH_PETSC)
endif(BUILD_WITH_PETSC)

if(AMAT_PROFILER)
    add_definitions(-DAMAT_PROFILER)
endif(AMAT_PROFILER)

if(VECTORIZED_AVX512)
    add_definitions(-DVECTORIZED_AVX512)
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=corei7-avx")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=corei7-avx")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif(VECTORIZED_AVX512)

if(VECTORIZED_AVX256)
    add_definitions(-DVECTORIZED_AVX256)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif(VECTORIZED_AVX256)

if(VECTORIZED_OPENMP)
    add_definitions(-DVECTORIZED_OPENMP)
endif(VECTORIZED_OPENMP)

if(VECTORIZED_OPENMP_ALIGNED)
    add_definitions(-DVECTORIZED_OPENMP_ALIGNED)
endif(VECTORIZED_OPENMP_ALIGNED)

if(HYBRID_PARALLEL)
    add_definitions(-DHYBRID_PARALLEL)
endif(HYBRID_PARALLEL)

if(USE_GPU)
    add_definitions(-DUSE_GPU)
endif(USE_GPU)

#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout")

set(EIGEN_HEADER_DIR .)

add_executable(fem3d examples/include/fem3d.hpp examples/src/fem3d.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(fem3d PUBLIC include)
target_include_directories(fem3d PUBLIC examples/include)
target_include_directories(fem3d PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(fem3d PRIVATE ${EIGEN_HEADER_DIR})
#target_include_directories(fem3d PRIVATE ${MAGMA_DIR}/include)
#target_include_directories(fem3d PRIVATE ${CUDA_DIR}/include)
#target_link_directories(fem3d PRIVATE ${MAGMA_DIR}/lib)
#target_link_directories(fem3d PRIVATE ${CUDA_DIR}/lib64)
#target_link_libraries(fem3d ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
target_link_libraries(fem3d ${MPI_LIBRARIES} m)

add_executable(fem1d examples/src/fem1d.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(fem1d PUBLIC include)
target_include_directories(fem1d PUBLIC examples/include)
target_include_directories(fem1d PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(fem1d PRIVATE ${EIGEN_HEADER_DIR})
#target_include_directories(fem1d PRIVATE ${MAGMA_DIR}/include)
#target_include_directories(fem1d PRIVATE ${CUDA_DIR}/include)
#target_link_directories(fem1d PRIVATE ${MAGMA_DIR}/lib)
#target_link_directories(fem1d PRIVATE ${CUDA_DIR}/lib64)
#target_link_libraries(fem1d ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
target_link_libraries(fem1d ${MPI_LIBRARIES} m)

add_executable(fem2d examples/include/fem2d.hpp examples/src/fem2d.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(fem2d PUBLIC include)
target_include_directories(fem2d PUBLIC examples/include)
target_include_directories(fem2d PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(fem2d PRIVATE ${EIGEN_HEADER_DIR})
#target_include_directories(fem2d PRIVATE ${MAGMA_DIR}/include)
#target_include_directories(fem2d PRIVATE ${CUDA_DIR}/include)
#target_link_directories(fem2d PRIVATE ${MAGMA_DIR}/lib)
#target_link_directories(fem2d PRIVATE ${CUDA_DIR}/lib64)
#target_link_libraries(fem2d ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
target_link_libraries(fem2d ${MPI_LIBRARIES} m)

add_executable(ex1 examples/src/ex1.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex1 PUBLIC include)
target_include_directories(ex1 PUBLIC examples/include)
target_include_directories(ex1 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex1 PRIVATE ${EIGEN_HEADER_DIR})
#target_include_directories(ex1 PRIVATE ${MAGMA_DIR}/include)
#target_include_directories(ex1 PRIVATE ${CUDA_DIR}/include)
#target_link_directories(ex1 PRIVATE ${MAGMA_DIR}/lib)
#target_link_directories(ex1 PRIVATE ${CUDA_DIR}/lib64)
#target_link_libraries(ex1 ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
target_link_libraries(ex1 ${MPI_LIBRARIES} m)

add_executable(ex2 examples/src/ex2.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex2 PUBLIC include)
target_include_directories(ex2 PUBLIC examples/include)
target_include_directories(ex2 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex2 PRIVATE ${EIGEN_HEADER_DIR})
#target_include_directories(ex2 PRIVATE ${MAGMA_DIR}/include)
#target_include_directories(ex2 PRIVATE ${CUDA_DIR}/include)
#target_link_directories(ex2 PRIVATE ${MAGMA_DIR}/lib)
#target_link_directories(ex2 PRIVATE ${CUDA_DIR}/lib64)
#target_link_libraries(ex2 ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
target_link_libraries(ex2 ${MPI_LIBRARIES} m)

add_executable(ex3 examples/include/ex3.hpp examples/src/ex3.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex3 PUBLIC include)
target_include_directories(ex3 PUBLIC examples/include)
target_include_directories(ex3 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex3 PRIVATE ${EIGEN_HEADER_DIR})
#target_include_directories(ex3 PRIVATE ${MAGMA_DIR}/include)
#target_include_directories(ex3 PRIVATE ${CUDA_DIR}/include)
#target_link_directories(ex3 PRIVATE ${MAGMA_DIR}/lib)
#target_link_directories(ex3 PRIVATE ${CUDA_DIR}/lib64)
#target_link_libraries(ex3 ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
target_link_libraries(ex3 ${MPI_LIBRARIES} m)

add_executable(ex4 examples/include/ex4.hpp examples/src/ex4.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex4 PUBLIC include)
target_include_directories(ex4 PUBLIC examples/include)
target_include_directories(ex4 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex4 PRIVATE ${EIGEN_HEADER_DIR})
#target_include_directories(ex4 PRIVATE ${MAGMA_DIR}/include)
#target_include_directories(ex4 PRIVATE ${CUDA_DIR}/include)
#target_link_directories(ex4 PRIVATE ${MAGMA_DIR}/lib)
#target_link_directories(ex4 PRIVATE ${CUDA_DIR}/lib64)
#target_link_libraries(ex4 ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
target_link_libraries(ex4 ${MPI_LIBRARIES} m)

add_executable(ex4a examples/src/ex4a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex4a PUBLIC include)
target_include_directories(ex4a PUBLIC examples/include)
target_include_directories(ex4a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex4a PRIVATE ${EIGEN_HEADER_DIR})
#target_include_directories(ex4a PRIVATE ${MAGMA_DIR}/include)
#target_include_directories(ex4a PRIVATE ${CUDA_DIR}/include)
#target_link_directories(ex4a PRIVATE ${MAGMA_DIR}/lib)
#target_link_directories(ex4a PRIVATE ${CUDA_DIR}/lib64)
#target_link_libraries(ex4a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
target_link_libraries(ex4a ${MPI_LIBRARIES} m)

add_executable(ex3a examples/src/ex3a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex3a PUBLIC include)
target_include_directories(ex3a PUBLIC examples/include)
target_include_directories(ex3a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex3a PRIVATE ${EIGEN_HEADER_DIR})
#target_include_directories(ex3a PRIVATE ${MAGMA_DIR}/include)
#target_include_directories(ex3a PRIVATE ${CUDA_DIR}/include)
#target_link_directories(ex3a PRIVATE ${MAGMA_DIR}/lib)
#target_link_directories(ex3a PRIVATE ${CUDA_DIR}/lib64)
#target_link_libraries(ex3a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
target_link_libraries(ex3a ${MPI_LIBRARIES} m)

if(BUILD_WITH_PETSC)
    target_include_directories(fem3d PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(fem3d ${PETSC_LIBRARIES})

    target_include_directories(fem1d PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(fem1d ${PETSC_LIBRARIES})

    target_include_directories(fem2d PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(fem2d ${PETSC_LIBRARIES})

    target_include_directories(ex1 PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex1 ${PETSC_LIBRARIES})

    target_include_directories(ex2 PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex2 ${PETSC_LIBRARIES})

    target_include_directories(ex3 PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex3 ${PETSC_LIBRARIES})

    target_include_directories(ex4 PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex4 ${PETSC_LIBRARIES})

    target_include_directories(ex4a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex4a ${PETSC_LIBRARIES})

    target_include_directories(ex3a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex3a ${PETSC_LIBRARIES})
endif()
