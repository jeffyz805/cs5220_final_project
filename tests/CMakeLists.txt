add_executable(mac_tests
  test_smoke.cpp
  test_solvers.cpp
)
target_link_libraries(mac_tests PRIVATE mac_core Catch2::Catch2WithMain)
if(USE_EIGEN_TESTS)
  target_link_libraries(mac_tests PRIVATE Eigen3::Eigen)
  target_compile_definitions(mac_tests PRIVATE MAC_HAS_EIGEN=1)
endif()

include(CTest)
include(Catch)
catch_discover_tests(mac_tests)
