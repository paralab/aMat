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

set(INCLUDE_FILES include/shfunction.hpp
        include/ke_matrix.hpp
        include/me_matrix.hpp
        include/fe_vector.hpp
        include/aMat.hpp)

set(SOURCE_FILES
        src/shfunction.cpp
        src/ke_matrix.cpp
        src/me_matrix.cpp
        src/fe_vector.cpp)

# cmake options, which will be visible at ccmake ../
option(BUILD_WITH_PETSC "Build code with the petsc" ON)

# if BUILD_WITH_PETSC ON , #define BUILD_WITH_PETSC
if(BUILD_WITH_PETSC)
    list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
    find_package(PETSc REQUIRED)
    add_definitions(-DBUILD_WITH_PETSC)
endif(BUILD_WITH_PETSC)


set( EIGEN_HEADER_DIR . )

add_executable(fem3d src/fem3d.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
target_include_directories(fem3d PUBLIC include)
target_include_directories(fem3d PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(fem3d PRIVATE ${EIGEN_HEADER_DIR})
target_link_libraries(fem3d ${MPI_LIBRARIES} m)

#add_executable(matfree src/matfree.cpp ${INCLUDE_FILES} ${SOURCE_FILES})
#target_include_directories(matfree PUBLIC include)
#target_include_directories(matfree PRIVATE ${MPI_INCLUDE_PATH})
#target_include_directories(matfree PRIVATE ${EIGEN_HEADER_DIR})
#target_link_libraries(matfree ${MPI_LIBRARIES} m)


#add_executable(testEigen ${INCLUDE_FILES} ${SOURCE_FILES} test/src/testEigen.cpp test/include/aMat.h)
#target_include_directories(testEigen PUBLIC include)
#target_include_directories(testEigen PUBLIC test/include)
#target_include_directories(testEigen PRIVATE ${MPI_INCLUDE_PATH})
#target_include_directories(testEigen PRIVATE ${EIGEN_HEADER_DIR})
#target_link_libraries(testEigen ${MPI_LIBRARIES} m)

if(BUILD_WITH_PETSC)
    target_include_directories(fem3d PUBLIC ${PETSC_INCLUDES})
    target_link_libraries(fem3d ${PETSC_LIBRARIES})

#    target_include_directories(matfree PUBLIC ${PETSC_INCLUDES})
#    target_link_libraries(matfree ${PETSC_LIBRARIES})

#    target_include_directories(testEigen PUBLIC ${PETSC_INCLUDES})
#    target_link_libraries(testEigen ${PETSC_LIBRARIES})
endif()
