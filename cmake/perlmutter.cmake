# Perlmutter (NERSC) toolchain hints.
#
# Usage:
#   module load PrgEnv-gnu cray-mpich cmake
#   module load cray-libsci  # BLAS/LAPACK if needed by CHOLMOD
#   module load cpu          # CPU partition compute env
#   export MAC_TARGET_ARCH=znver3
#   export CC=cc CXX=CC
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/perlmutter.cmake -DUSE_CHOLMOD=ON
#
# Cray's `cc` / `CC` wrappers already inject MPI; we still call find_package(MPI)
# in the main CMakeLists so the imported target exists for IDE tooling, but no
# extra mpi link flags are needed here.

set(CMAKE_C_COMPILER   cc)
set(CMAKE_CXX_COMPILER CC)

# Milan = znver3, AVX2 + FMA, no AVX-512.
if(NOT DEFINED ENV{MAC_TARGET_ARCH})
  set(ENV{MAC_TARGET_ARCH} "znver3")
endif()

# Cray MPICH is auto-linked by `CC`; tell find_package not to add anything extra.
set(MPIEXEC_EXECUTABLE "srun" CACHE FILEPATH "")
