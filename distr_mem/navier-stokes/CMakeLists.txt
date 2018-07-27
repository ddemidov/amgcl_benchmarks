foreach(script run_amgcl.py run_amgcl.sh spc.json)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${script}
        ${CMAKE_CURRENT_BINARY_DIR}/${script}
        COPYONLY
        )
endforeach()

add_executable(dmem_ns_amgcl amgcl.cpp)
target_link_libraries(dmem_ns_amgcl
    amgcl::amgcl
    mpi_target
    argh_target
    )

if (TARGET eigen_target)
    target_link_libraries(dmem_ns_amgcl eigen_target)
endif()

add_executable(dmem_ns_amgcl_sdd amgcl_sdd.cpp)
target_link_libraries(dmem_ns_amgcl_sdd
    amgcl::amgcl
    mpi_target
    argh_target
    )

if (TARGET eigen_target)
    target_link_libraries(dmem_ns_amgcl_sdd eigen_target)
endif()

add_executable(dmem_ns_trilinos trilinos.cpp)
target_link_libraries(dmem_ns_trilinos mpi_target trilinos_target amgcl::amgcl)

if (CUDA_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/amgcl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/amgcl.cu
        COPYONLY
        )

    cuda_add_executable(dmem_ns_amgcl_cuda
        ${CMAKE_CURRENT_BINARY_DIR}/amgcl.cu
        OPTIONS -DSOLVER_BACKEND_CUDA)

    target_link_libraries(dmem_ns_amgcl_cuda
        amgcl::amgcl
        mpi_target
        ${CUDA_cusparse_LIBRARY}
        argh_target
        )

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/amgcl_sdd.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/amgcl_sdd.cu
        COPYONLY
        )

    cuda_add_executable(dmem_ns_amgcl_sdd_cuda
        ${CMAKE_CURRENT_BINARY_DIR}/amgcl_sdd.cu
        OPTIONS -DSOLVER_BACKEND_CUDA)

    target_link_libraries(dmem_ns_amgcl_sdd_cuda
        amgcl::amgcl
        mpi_target
        ${CUDA_cusparse_LIBRARY}
        argh_target
        )
endif()

find_path(METIS_INCLUDES NAMES metis.h PATHS $ENV{METISDIR} PATH_SUFFIXES metis)
find_library(METIS_LIBRARY metis PATHS $ENV{METISDIR})
add_library(metis_target INTERFACE)
target_include_directories(metis_target INTERFACE ${METIS_INCLUDES})
target_link_libraries(metis_target INTERFACE ${METIS_LIBRARY})

add_executable(partition partition.cpp)
target_link_libraries(partition amgcl::amgcl metis_target argh_target)
