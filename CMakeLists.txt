cmake_minimum_required(VERSION 3.7)
project(fem3d)

set(CMAKE_CXX_STANDARD 14)

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
        include/aMatGpu.hpp
        include/aMatGpu_1.hpp
        examples/include/GmshMesh.hpp)

set(SOURCE_FILES
        examples/src/ke_matrix.cpp
        examples/src/fe_vector.cpp
        examples/src/GmshMesh.cpp)

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
option(USE_BLAS_MATVEC "use MKL BLAS for matvec" OFF)

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

if(USE_BLAS_MATVEC)
    add_definitions(-DUSE_BLAS_MATVEC)
endif(USE_BLAS_MATVEC)

#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout")

set(EIGEN_HEADER_DIR .)

add_executable(example01a examples/src/example01a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(example01a PUBLIC include)
target_include_directories(example01a PUBLIC examples/include)
target_include_directories(example01a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(example01a PRIVATE ${EIGEN_HEADER_DIR})

add_executable(example02a examples/include/example02a.hpp examples/src/example02a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(example02a PUBLIC include)
target_include_directories(example02a PUBLIC examples/include)
target_include_directories(example02a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(example02a PRIVATE ${EIGEN_HEADER_DIR})

add_executable(example03a examples/include/example03a.hpp examples/src/example03a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(example03a PUBLIC include)
target_include_directories(example03a PUBLIC examples/include)
target_include_directories(example03a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(example03a PRIVATE ${EIGEN_HEADER_DIR})

add_executable(example04a examples/include/example04a.hpp examples/src/example04a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(example04a PUBLIC include)
target_include_directories(example04a PUBLIC examples/include)
target_include_directories(example04a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(example04a PRIVATE ${EIGEN_HEADER_DIR})

add_executable(example05a examples/include/example05a.hpp examples/src/example05a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(example05a PUBLIC include)
target_include_directories(example05a PUBLIC examples/include)
target_include_directories(example05a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(example05a PRIVATE ${EIGEN_HEADER_DIR})

add_executable(example06a examples/include/example06a.hpp examples/src/example06a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(example06a PUBLIC include)
target_include_directories(example06a PUBLIC examples/include)
target_include_directories(example06a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(example06a PRIVATE ${EIGEN_HEADER_DIR})

add_executable(example07a examples/include/example07a.hpp examples/src/example07a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(example07a PUBLIC include)
target_include_directories(example07a PUBLIC examples/include)
target_include_directories(example07a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(example07a PRIVATE ${EIGEN_HEADER_DIR})

add_executable(example08a examples/include/example08a.hpp examples/src/example08a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(example08a PUBLIC include)
target_include_directories(example08a PUBLIC examples/include)
target_include_directories(example08a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(example08a PRIVATE ${EIGEN_HEADER_DIR})

add_executable(example11 examples/include/example11.hpp examples/src/example11.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(example11 PUBLIC include)
target_include_directories(example11 PUBLIC examples/include)
target_include_directories(example11 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(example11 PRIVATE ${EIGEN_HEADER_DIR})

if(USE_GPU)
    target_include_directories(example01a PRIVATE ${MAGMA_DIR}/include)
    target_include_directories(example01a PRIVATE ${CUDA_DIR}/include)
    target_link_directories(example01a PRIVATE ${MAGMA_DIR}/lib)
    target_link_directories(example01a PRIVATE ${CUDA_DIR}/lib64)
    target_link_libraries(example01a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)

    target_include_directories(example02a PRIVATE ${MAGMA_DIR}/include)
    target_include_directories(example02a PRIVATE ${CUDA_DIR}/include)
    target_link_directories(example02a PRIVATE ${MAGMA_DIR}/lib)
    target_link_directories(example02a PRIVATE ${CUDA_DIR}/lib64)
    target_link_libraries(example02a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)

    target_include_directories(example03a PRIVATE ${MAGMA_DIR}/include)
    target_include_directories(example03a PRIVATE ${CUDA_DIR}/include)
    target_link_directories(example03a PRIVATE ${MAGMA_DIR}/lib)
    target_link_directories(example03a PRIVATE ${CUDA_DIR}/lib64)
    target_link_libraries(example03a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)

    target_include_directories(example04a PRIVATE ${MAGMA_DIR}/include)
    target_include_directories(example04a PRIVATE ${CUDA_DIR}/include)
    target_link_directories(example04a PRIVATE ${MAGMA_DIR}/lib)
    target_link_directories(example04a PRIVATE ${CUDA_DIR}/lib64)
    target_link_libraries(example04a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)

    target_include_directories(example05a PRIVATE ${MAGMA_DIR}/include)
    target_include_directories(example05a PRIVATE ${CUDA_DIR}/include)
    target_link_directories(example05a PRIVATE ${MAGMA_DIR}/lib)
    target_link_directories(example05a PRIVATE ${CUDA_DIR}/lib64)
    target_link_libraries(example05a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)

    target_include_directories(example06a PRIVATE ${MAGMA_DIR}/include)
    target_include_directories(example06a PRIVATE ${CUDA_DIR}/include)
    target_link_directories(example06a PRIVATE ${MAGMA_DIR}/lib)
    target_link_directories(example06a PRIVATE ${CUDA_DIR}/lib64)
    target_link_libraries(example06a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)

    target_include_directories(example07a PRIVATE ${MAGMA_DIR}/include)
    target_include_directories(example07a PRIVATE ${CUDA_DIR}/include)
    target_link_directories(example07a PRIVATE ${MAGMA_DIR}/lib)
    target_link_directories(example07a PRIVATE ${CUDA_DIR}/lib64)
    target_link_libraries(example07a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)

    target_include_directories(example08a PRIVATE ${MAGMA_DIR}/include)
    target_include_directories(example08a PRIVATE ${CUDA_DIR}/include)
    target_link_directories(example08a PRIVATE ${MAGMA_DIR}/lib)
    target_link_directories(example08a PRIVATE ${CUDA_DIR}/lib64)
    target_link_libraries(example08a ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)

    target_include_directories(example11 PRIVATE ${MAGMA_DIR}/include)
    target_include_directories(example11 PRIVATE ${CUDA_DIR}/include)
    target_link_directories(example11 PRIVATE ${MAGMA_DIR}/lib)
    target_link_directories(example11 PRIVATE ${CUDA_DIR}/lib64)
    target_link_libraries(example11 ${MPI_LIBRARIES} m magma magma_sparse cublas cudart cusparse)
endif(USE_GPU)

target_link_libraries(example01a ${MPI_LIBRARIES} m)
target_link_libraries(example02a ${MPI_LIBRARIES} m)
target_link_libraries(example03a ${MPI_LIBRARIES} m)
target_link_libraries(example04a ${MPI_LIBRARIES} m)
target_link_libraries(example05a ${MPI_LIBRARIES} m)
target_link_libraries(example06a ${MPI_LIBRARIES} m)
target_link_libraries(example07a ${MPI_LIBRARIES} m)
target_link_libraries(example08a ${MPI_LIBRARIES} m)
target_link_libraries(example11 ${MPI_LIBRARIES} m)

if(BUILD_WITH_PETSC)
    target_include_directories(example01a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(example01a ${PETSC_LIBRARIES})

    target_include_directories(example02a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(example02a ${PETSC_LIBRARIES})

    target_include_directories(example03a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(example03a ${PETSC_LIBRARIES})

    target_include_directories(example04a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(example04a ${PETSC_LIBRARIES})

    target_include_directories(example05a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(example05a ${PETSC_LIBRARIES})

    target_include_directories(example06a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(example06a ${PETSC_LIBRARIES})

    target_include_directories(example07a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(example07a ${PETSC_LIBRARIES})

    target_include_directories(example08a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(example08a ${PETSC_LIBRARIES})

    target_include_directories(example11 PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(example11 ${PETSC_LIBRARIES})
endif(BUILD_WITH_PETSC)
