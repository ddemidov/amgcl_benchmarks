add_executable(ns_amgcl ns_amgcl.cpp)
target_link_libraries(ns_amgcl
    amgcl::amgcl
    mpi_target
    program_options
    )

add_executable(ns_trilinos ns_trilinos.cpp)
target_link_libraries(ns_trilinos mpi_target trilinos_target amgcl::amgcl)

if (CUDA_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/ns_amgcl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ns_amgcl.cu
        COPYONLY
        )

    cuda_add_executable(ns_amgcl_cuda
        ${CMAKE_CURRENT_BINARY_DIR}/ns_amgcl.cu
        OPTIONS -DSOLVER_BACKEND_CUDA)

    target_link_libraries(ns_amgcl_cuda
        amgcl::amgcl
        mpi_target
        program_options
        ${CUDA_cusparse_LIBRARY}
        )
endif()

find_path(METIS_INCLUDES NAMES metis.h PATHS $ENV{METISDIR} PATH_SUFFIXES metis)
find_library(METIS_LIBRARY metis PATHS $ENV{METISDIR})
add_library(metis_target INTERFACE)
target_include_directories(metis_target INTERFACE ${METIS_INCLUDES})
target_link_libraries(metis_target INTERFACE ${METIS_LIBRARY})

add_executable(partition partition.cpp)
target_link_libraries(partition amgcl::amgcl metis_target program_options)
