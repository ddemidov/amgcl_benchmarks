add_executable(poisson_amgcl poisson_amgcl.cpp)
target_link_libraries(poisson_amgcl
    amgcl::amgcl
    mpi_target
    program_options
    )

add_executable(poisson_trilinos poisson_trilinos.cpp)
target_link_libraries(poisson_trilinos mpi_target trilinos_target)

if (CUDA_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/poisson_amgcl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/poisson_amgcl.cu
        COPYONLY
        )

    cuda_add_executable(poisson_amgcl_cuda
        ${CMAKE_CURRENT_BINARY_DIR}/poisson_amgcl.cu
        OPTIONS -DSOLVER_BACKEND_CUDA)

    target_include_directories(poisson_amgcl_cuda
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(poisson_amgcl_cuda
        amgcl::amgcl
        mpi_target
        program_options
        ${CUDA_cusparse_LIBRARY}
        )
endif()
