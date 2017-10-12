cmake_minimum_required(VERSION 3.0)
project(amgcl-deflation-code)

# --- OpenMP ---------------------------------------------------------------
find_package(OpenMP)
add_library(openmp_target INTERFACE)
target_include_directories(openmp_target INTERFACE ${OpenMP_CXX_FLAGS})
target_link_libraries(openmp_target INTERFACE ${OpenMP_CXX_FLAGS})

# --- Boost ----------------------------------------------------------------
find_package(Boost COMPONENTS program_options)
add_library(program_options INTERFACE)
target_include_directories(program_options INTERFACE ${Boost_INCLUDE_DIRS})
target_link_libraries(program_options INTERFACE ${Boost_PROGRAM_OPTIONS_LIBRARY})

# --- MPI ------------------------------------------------------------------
find_package(MPI)
add_library(mpi_target INTERFACE)
target_compile_options(mpi_target INTERFACE ${MPI_CXX_COMPILE_FLAGS})
target_include_directories(mpi_target INTERFACE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(mpi_target INTERFACE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})

# --- AMGCL ----------------------------------------------------------------
find_package(amgcl)
get_property(AMGCL_INCLUDE_DIRS TARGET amgcl::amgcl PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

# --- CUDA -----------------------------------------------------------------
find_package(CUDA)
if (CUDA_FOUND)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)

    list(APPEND CUDA_NVCC_FLAGS
        ${CUDA_ARCH_FLAGS} -Wno-deprecated-gpu-targets)

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        list(APPEND CUDA_NVCC_FLAGS
            -Xcompiler -std=c++03
            -Xcompiler -Wno-vla
            -Xcompiler -fopenmp
            -Xcompiler -fPIC
            )
    endif()

    add_library(cusp_target INTERFACE)
    target_include_directories(cusp_target INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/cusplibrary)
endif()

# --- Trilinos -------------------------------------------------------------
add_library(trilinos_target INTERFACE)
target_include_directories(trilinos_target INTERFACE $ENV{TRILINOS_DIR}/include)

foreach(lib 
        aztecoo
        epetra
        epetraext
        ml
        teuchoscore
        teuchosparameterlist
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
