add_executable(smem_ns_amgcl amgcl.cpp)
target_link_libraries(smem_ns_amgcl amgcl::amgcl argh_target)

add_executable(smem_ns_amgcl_scalar amgcl-scalar.cpp)
target_link_libraries(smem_ns_amgcl_scalar amgcl::amgcl argh_target)

if (TARGET VexCL::OpenCL)
    add_executable(smem_ns_amgcl_vexcl_opencl amgcl-vexcl.cpp)
    target_link_libraries(smem_ns_amgcl_vexcl_opencl
        amgcl::amgcl VexCL::OpenCL argh_target)

    add_executable(smem_ns_schur_vexcl_opencl schur-vexcl.cpp)
    target_link_libraries(smem_ns_schur_vexcl_opencl
        amgcl::amgcl VexCL::OpenCL argh_target)
    target_compile_definitions(smem_ns_schur_vexcl_opencl PRIVATE AMGCL_DEBUG)
endif()

add_executable(smem_ns_schur schur.cpp)
target_link_libraries(smem_ns_schur amgcl::amgcl argh_target)
target_compile_definitions(smem_ns_schur PRIVATE AMGCL_DEBUG)

if (TARGET VexCL::CUDA)
    add_executable(smem_ns_amgcl_vexcl_cuda amgcl-vexcl.cpp)
    target_link_libraries(smem_ns_amgcl_vexcl_cuda
        amgcl::amgcl VexCL::CUDA argh_target)

    add_executable(smem_ns_schur_vexcl_cuda schur-vexcl.cpp)
    target_link_libraries(smem_ns_schur_vexcl_cuda
        amgcl::amgcl VexCL::CUDA argh_target)
    target_compile_definitions(smem_ns_schur_vexcl_cuda PRIVATE AMGCL_DEBUG)
endif()

if (CUDA_FOUND)
    cuda_add_executable(smem_ns_amgcl_cuda amgcl.cu)
    target_link_libraries(smem_ns_amgcl_cuda amgcl::amgcl argh_target ${CUDA_cusparse_LIBRARY})

    cuda_add_executable(smem_ns_schur_cuda schur.cu)
    target_compile_definitions(smem_ns_schur_cuda PRIVATE AMGCL_DEBUG)
    target_link_libraries(smem_ns_schur_cuda amgcl::amgcl argh_target ${CUDA_cusparse_LIBRARY})

    cuda_add_executable(smem_ns_cusp cusp.cu)
    target_link_libraries(smem_ns_cusp cusp_target argh_target)

endif()

if (TARGET trilinos_target)
    add_executable(smem_ns_trilinos trilinos.cpp)
    target_include_directories(smem_ns_trilinos PRIVATE ${AMGCL_INCLUDE_DIRS})
    target_link_libraries(smem_ns_trilinos trilinos_target mpi_target)
endif()

if (TARGET petsc_target)
    add_executable(smem_ns_petsc petsc.cpp)
    target_include_directories(smem_ns_petsc PRIVATE ${AMGCL_INCLUDE_DIRS})
    target_link_libraries(smem_ns_petsc petsc_target mpi_target)
endif()
