add_library(mac_core STATIC
  la/sparse_matrix.cpp
  la/blas.cpp
  solvers/iterative/iterative.cpp
  solvers/direct/dense_cholesky.cpp
)
target_include_directories(mac_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mac_core PUBLIC mac_flags)
