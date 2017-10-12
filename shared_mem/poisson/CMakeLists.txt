add_executable(smem_poisson_amgcl amgcl.cpp)
target_link_libraries(smem_poisson_amgcl amgcl::amgcl program_options)

if (TARGET VexCL::CUDA)
    add_executable(smem_poisson_amgcl_vexcl_cuda amgcl-vexcl.cpp)
    target_link_libraries(smem_poisson_amgcl_vexcl_cuda VexCL::CUDA amgcl::amgcl program_options)
endif()

if (TARGET VexCL::OpenCL)
    add_executable(smem_poisson_amgcl_vexcl_opencl amgcl-vexcl.cpp)
    target_link_libraries(smem_poisson_amgcl_vexcl_opencl VexCL::OpenCL amgcl::amgcl program_options)
endif()

if (CUDA_FOUND)
    cuda_add_executable(smem_poisson_amgcl_cuda amgcl.cu)
    target_link_libraries(smem_poisson_amgcl_cuda amgcl::amgcl program_options ${CUDA_cusparse_LIBRARY})

    cuda_add_executable(smem_poisson_cusp cusp.cu)
    target_link_libraries(smem_poisson_cusp cusp_target program_options)
endif()

if (TARGET trilinos_target)
    add_executable(smem_poisson_trilinos trilinos.cpp)
    target_link_libraries(smem_poisson_trilinos trilinos_target)
endif()

if (TARGET petsc_target)
    add_executable(smem_poisson_petsc petsc.cpp)
    target_link_libraries(smem_poisson_petsc petsc_target)
endif()
