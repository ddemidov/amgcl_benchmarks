cmake_minimum_required(VERSION 3.0)
project(amgcl-deflation-code)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# --- OpenMP ---------------------------------------------------------------
find_package(OpenMP)
add_library(openmp_target INTERFACE)
target_include_directories(openmp_target INTERFACE ${OpenMP_CXX_FLAGS})
target_link_libraries(openmp_target INTERFACE ${OpenMP_CXX_FLAGS})

# --- Command line handler -------------------------------------------------
add_library(argh_target INTERFACE)
target_include_directories(argh_target INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# --- MPI ------------------------------------------------------------------
find_package(MPI)
add_library(mpi_target INTERFACE)
target_compile_options(mpi_target INTERFACE ${MPI_CXX_COMPILE_FLAGS})
target_include_directories(mpi_target INTERFACE ${MPI_CXX_INCLUDE_PATH})
string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)
string(STRIP "${MPI_CXX_LIBRARIES}" MPI_CXX_LIBRARIES)
target_link_libraries(mpi_target INTERFACE "${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES}")

# --- AMGCL ----------------------------------------------------------------
find_package(amgcl)
get_property(AMGCL_INCLUDE_DIRS TARGET amgcl::amgcl PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

# --- Boost ----------------------------------------------------------------
find_package(Boost)

#----------------------------------------------------------------------------
# Eigen backend
#----------------------------------------------------------------------------
find_path(EIGEN_INCLUDE Eigen/SparseCore PATH_SUFFIXES eigen3)
if (EIGEN_INCLUDE)
    add_library(eigen_target INTERFACE)
    target_include_directories(eigen_target INTERFACE ${EIGEN_INCLUDE})
    target_compile_options(eigen_target INTERFACE
        $<$<CXX_COMPILER_ID:GNU>:-Wno-int-in-bool-context>
        $<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-long-long>
        )
    target_compile_definitions(eigen_target INTERFACE AMGCL_HAVE_EIGEN)
endif()

#----------------------------------------------------------------------------
# Find Pastix
#----------------------------------------------------------------------------
find_package(Metis  QUIET)
find_package(Scotch QUIET)
find_package(Pastix QUIET)
find_package(BLAS   QUIET)

if (PASTIX_INCLUDES AND SCOTCH_INCLUDES AND BLAS_FOUND)
    add_library(pastix_target INTERFACE)
    target_include_directories(pastix_target INTERFACE
        ${PASTIX_INCLUDES}
        ${SCOTCH_INCLUDES}
        )
    target_link_libraries(pastix_target INTERFACE
        ${PASTIX_LIBRARIES}
        ${SCOTCH_LIBRARIES}
        ${BLAS_LIBRARIES}
        )
    target_compile_definitions(pastix_target INTERFACE AMGCL_HAVE_PASTIX)
endif()

# --- CUDA -----------------------------------------------------------------
find_package(CUDA QUIET)
if (CUDA_FOUND)
    set(CUDA_TARGET_ARCH "Kepler Maxwell Pascal" CACHE STRING "Target architecture(s) for CUDA")
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_TARGET_ARCH})

    list(APPEND CUDA_NVCC_FLAGS
        ${CUDA_ARCH_FLAGS} -Wno-deprecated-gpu-targets)

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        list(APPEND CUDA_NVCC_FLAGS
            -std=c++11
            -Xcompiler -Wno-vla
            -Xcompiler -fopenmp
            -Xcompiler -fPIC
            )
    endif()

    add_library(cusp_target INTERFACE)
    target_include_directories(cusp_target INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/cusplibrary)
endif()

# --- VexCL ----------------------------------------------------------------
find_package(VexCL QUIET)

# --- Trilinos -------------------------------------------------------------
add_library(trilinos_target INTERFACE)
target_include_directories(trilinos_target INTERFACE $ENV{TRILINOS_DIR}/include)
target_link_libraries(trilinos_target INTERFACE ${BLAS_LIBRARIES})

foreach(lib 
        ml
        amesos
        aztecoo
        ifpack
        epetra
        epetraext
        zoltan
        zoltan2
        teuchosparameterlist
        teuchosremainder
        teuchosnumerics
        teuchoscomm
        teuchoscore
        kokkoscore
        triutils
        )
    find_library(TRILINOS_${lib}_LIBRARY ${lib} PATHS ENV TRILINOS_DIR PATH_SUFFIXES lib)
    target_link_libraries(trilinos_target INTERFACE ${TRILINOS_${lib}_LIBRARY})
endforeach()

# --- PETSC ----------------------------------------------------------------
add_library(petsc_target INTERFACE)
target_include_directories(petsc_target INTERFACE $ENV{PETSC_DIR}/include)
find_library(PETSC_LIBRARY petsc PATHS ENV PETSC_DIR PATH_SUFFIXES lib)
target_link_libraries(petsc_target INTERFACE ${PETSC_LIBRARY})

# --------------------------------------------------------------------------
add_subdirectory(shared_mem)
add_subdirectory(distr_mem)
