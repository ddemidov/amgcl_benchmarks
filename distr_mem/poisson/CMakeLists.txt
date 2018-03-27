foreach(script run_trilinos.py run_petsc.py run_amgcl.py)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${script}
        ${CMAKE_CURRENT_BINARY_DIR}/${script}
        COPYONLY
        )
endforeach()

add_executable(dmem_poisson_amgcl amgcl.cpp)
target_link_libraries(dmem_poisson_amgcl
    amgcl::amgcl
    mpi_target
    program_options
    )

add_executable(dmem_poisson_petsc petsc.cpp)
target_link_libraries(dmem_poisson_petsc
    mpi_target
    petsc_target
    )

add_executable(dmem_poisson_trilinos trilinos.cpp)
target_link_libraries(dmem_poisson_trilinos mpi_target trilinos_target)

if (CUDA_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/amgcl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/amgcl.cu
        COPYONLY
        )

    cuda_add_executable(dmem_poisson_amgcl_cuda
        ${CMAKE_CURRENT_BINARY_DIR}/amgcl.cu
        OPTIONS -DSOLVER_BACKEND_CUDA)

    target_include_directories(dmem_poisson_amgcl_cuda
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(dmem_poisson_amgcl_cuda
        amgcl::amgcl
        mpi_target
        program_options
        ${CUDA_cusparse_LIBRARY}
        )
endif()
