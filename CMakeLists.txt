cmake_minimum_required(VERSION 3.7)
project(fem3d)

set(CMAKE_CXX_STANDARD 14)
FIND_PACKAGE(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
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
        include/ke_matrix.hpp
        include/fe_vector.hpp
        include/aMat.hpp
        include/profiler.hpp
        include/integration.hpp
        include/shapeFunc.hpp)

set(SOURCE_FILES
        src/ke_matrix.cpp
        src/fe_vector.cpp
        src/profiler.cpp)

# cmake options, which will be visible at ccmake ../
option(BUILD_WITH_PETSC "Build code with the petsc" ON)
option(AMAT_PROFILER "turn on the amat profiler counters" OFF)
option(AVX_512 "turn on use AVX-512" OFF)
option(AVX_256 "turn on use AVX-256" OFF)
option(OMP_SIMD "turn on use OMP-SIMD" OFF)
option(USE_OMP "turn on use OMP" OFF)

# if BUILD_WITH_PETSC ON , #define BUILD_WITH_PETSC
if(BUILD_WITH_PETSC)
    list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
    find_package(PETSc REQUIRED)
    add_definitions(-DBUILD_WITH_PETSC)
endif(BUILD_WITH_PETSC)


if(AMAT_PROFILER)
    add_definitions(-DAMAT_PROFILER)
endif(AMAT_PROFILER)

if(AVX_512)
    add_definitions(-DAVX_512)
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=corei7-avx")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=corei7-avx")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif(AVX_512)

if(AVX_256)
    add_definitions(-DAVX_256)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif(AVX_256)

if(OMP_SIMD)
    add_definitions(-DOMP_SIMD)
endif(OMP_SIMD)

if(USE_OMP)
    add_definitions(-DUSE_OMP)
endif(USE_OMP)

#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout")

set(EIGEN_HEADER_DIR .)

add_executable(fem3d src/fem3d.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(fem3d PUBLIC include)
target_include_directories(fem3d PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(fem3d PRIVATE ${EIGEN_HEADER_DIR})
target_link_libraries(fem3d ${MPI_LIBRARIES} m)

add_executable(fem1d src/fem1d.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(fem1d PUBLIC include)
target_include_directories(fem1d PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(fem1d PRIVATE ${EIGEN_HEADER_DIR})
target_link_libraries(fem1d ${MPI_LIBRARIES} m)

add_executable(fem2d src/fem2d.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(fem2d PUBLIC include)
target_include_directories(fem2d PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(fem2d PRIVATE ${EIGEN_HEADER_DIR})
target_link_libraries(fem2d ${MPI_LIBRARIES} m)

add_executable(ex1 src/ex1.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex1 PUBLIC include)
target_include_directories(ex1 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex1 PRIVATE ${EIGEN_HEADER_DIR})
target_link_libraries(ex1 ${MPI_LIBRARIES} m)

add_executable(ex1a src/ex1a.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex1a PUBLIC include)
target_include_directories(ex1a PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex1a PRIVATE ${EIGEN_HEADER_DIR})
target_link_libraries(ex1a ${MPI_LIBRARIES} m)

add_executable(ex2 src/ex2.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex2 PUBLIC include)
target_include_directories(ex2 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex2 PRIVATE ${EIGEN_HEADER_DIR})
target_link_libraries(ex2 ${MPI_LIBRARIES} m)

add_executable(ex3 src/ex3.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex3 PUBLIC include)
target_include_directories(ex3 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex3 PRIVATE ${EIGEN_HEADER_DIR})
target_link_libraries(ex3 ${MPI_LIBRARIES} m)

add_executable(ex4 src/ex4.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(ex4 PUBLIC include)
target_include_directories(ex4 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(ex4 PRIVATE ${EIGEN_HEADER_DIR})
target_link_libraries(ex4 ${MPI_LIBRARIES} m)

if(BUILD_WITH_PETSC)
    target_include_directories(fem3d PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(fem3d ${PETSC_LIBRARIES})

    target_include_directories(fem1d PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(fem1d ${PETSC_LIBRARIES})

    target_include_directories(fem2d PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(fem2d ${PETSC_LIBRARIES})

    target_include_directories(ex1 PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex1 ${PETSC_LIBRARIES})

    target_include_directories(ex1a PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex1a ${PETSC_LIBRARIES})

    target_include_directories(ex2 PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex2 ${PETSC_LIBRARIES})

    target_include_directories(ex3 PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex3 ${PETSC_LIBRARIES})

    target_include_directories(ex4 PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(ex4 ${PETSC_LIBRARIES})
endif()
