add_library(mac_core STATIC
  la/sparse_matrix.cpp
  la/blas.cpp
  solvers/iterative/iterative.cpp
  solvers/direct/dense_cholesky.cpp
  constraints/constraint.cpp
  constraints/assembler.cpp
  physics/integrator.cpp
)

if(USE_MPI)
  target_sources(mac_core PRIVATE
    solvers/mpi/distributed_csr.cpp
    solvers/mpi/distributed_blas.cpp
    solvers/mpi/dist_solvers.cpp
  )
endif()
target_include_directories(mac_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mac_core PUBLIC mac_flags)
