cmake_minimum_required(VERSION 3.20)
project(mac_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(USE_MPI         "Build MPI solvers"          ON)
option(USE_OPENMP      "Build OpenMP-parallel kernels" ON)

# AVX2 only makes sense on x86_64. Default OFF on aarch64/arm64 (Apple Silicon).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_default_avx2 ON)
else()
  set(_default_avx2 OFF)
endif()
option(USE_AVX2        "Enable AVX2 intrinsics"     ${_default_avx2})
option(USE_CHOLMOD     "Link SuiteSparse CHOLMOD"   OFF)
option(USE_EIGEN_TESTS "Use Eigen as test oracle"   ON)
option(BUILD_TESTS     "Build Catch2 tests"         ON)

add_library(mac_flags INTERFACE)
target_compile_options(mac_flags INTERFACE
  -O3 -Wall -Wextra -Wpedantic -Wno-unused-parameter
  $<$<BOOL:${USE_AVX2}>:-mavx2>
  $<$<BOOL:${USE_AVX2}>:-mfma>
)
if(DEFINED ENV{MAC_TARGET_ARCH})
  target_compile_options(mac_flags INTERFACE -march=$ENV{MAC_TARGET_ARCH})
endif()
if(USE_AVX2)
  target_compile_definitions(mac_flags INTERFACE MAC_USE_AVX2=1)
endif()

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(mac_flags INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(mac_flags INTERFACE MAC_USE_OPENMP=1)
endif()

if(USE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(mac_flags INTERFACE MPI::MPI_CXX)
  target_compile_definitions(mac_flags INTERFACE MAC_USE_MPI=1)
endif()

if(USE_CHOLMOD)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(CHOLMOD REQUIRED IMPORTED_TARGET cholmod)
  target_link_libraries(mac_flags INTERFACE PkgConfig::CHOLMOD)
  target_compile_definitions(mac_flags INTERFACE MAC_USE_CHOLMOD=1)
endif()

include(FetchContent)

if(BUILD_TESTS)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)
  enable_testing()
endif()

if(USE_EIGEN_TESTS)
  FetchContent_Declare(Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW ON
  )
  set(EIGEN_BUILD_DOC      OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING  OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING        OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen)
endif()

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(bench)
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
