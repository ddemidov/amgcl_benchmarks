foreach(script
        run_trilinos.py run_petsc.py run_amgcl.py
        run_amgcl_strong.sh run_amgcl_weak.sh
        run_trilinos_strong.sh run_trilinos_weak.sh
        )
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
    argh_target
    )

add_executable(dmem_poisson_amgcl_sdd amgcl_sdd.cpp)
target_link_libraries(dmem_poisson_amgcl_sdd
    amgcl::amgcl
    mpi_target
    argh_target
    )

if (TARGET pastix_target)
    target_link_libraries(dmem_poisson_amgcl pastix_target)
    target_link_libraries(dmem_poisson_amgcl_sdd pastix_target)
endif()

if (TARGET eigen_target)
    target_link_libraries(dmem_poisson_amgcl eigen_target)
    target_link_libraries(dmem_poisson_amgcl_sdd eigen_target)
endif()

if (TARGET Metis::metis)
    target_link_libraries(dmem_poisson_amgcl Metis::metis)
endif()

add_executable(dmem_poisson_petsc petsc.cpp)
target_link_libraries(dmem_poisson_petsc
    mpi_target
    petsc_target
    )
target_include_directories(dmem_poisson_petsc PRIVATE ${Boost_INCLUDE_DIRS})

add_executable(dmem_poisson_trilinos trilinos.cpp)
target_link_libraries(dmem_poisson_trilinos mpi_target trilinos_target)
target_include_directories(dmem_poisson_trilinos PRIVATE ${Boost_INCLUDE_DIRS})

if (TARGET VexCL::CUDA)
    add_executable(dmem_poisson_amgcl_vexcl amgcl.cpp)
    target_compile_definitions(dmem_poisson_amgcl_vexcl PRIVATE SOLVER_BACKEND_VEXCL)
    target_link_libraries(dmem_poisson_amgcl_vexcl
        amgcl::amgcl
        mpi_target
        argh_target
        VexCL::CUDA
        )

    add_executable(dmem_poisson_amgcl_sdd_vexcl amgcl_sdd.cpp)
    target_compile_definitions(dmem_poisson_amgcl_sdd_vexcl PRIVATE SOLVER_BACKEND_VEXCL)
    target_link_libraries(dmem_poisson_amgcl_sdd_vexcl
        amgcl::amgcl
        mpi_target
        argh_target
        VexCL::CUDA
        )

    if (TARGET pastix_target)
        target_link_libraries(dmem_poisson_amgcl_vexcl pastix_target)
        target_link_libraries(dmem_poisson_amgcl_sdd_vexcl pastix_target)
    endif()

    if (TARGET eigen_target)
        target_link_libraries(dmem_poisson_amgcl_vexcl eigen_target)
        target_link_libraries(dmem_poisson_amgcl_sdd_vexcl eigen_target)
    endif()
endif()

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
        ${CUDA_cusparse_LIBRARY}
        argh_target
        )

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/amgcl_sdd.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/amgcl_sdd.cu
        COPYONLY
        )

    cuda_add_executable(dmem_poisson_amgcl_sdd_cuda
        ${CMAKE_CURRENT_BINARY_DIR}/amgcl_sdd.cu
        OPTIONS -DSOLVER_BACKEND_CUDA)

    target_include_directories(dmem_poisson_amgcl_sdd_cuda
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(dmem_poisson_amgcl_sdd_cuda
        amgcl::amgcl
        mpi_target
        ${CUDA_cusparse_LIBRARY}
        argh_target
        )
endif()
