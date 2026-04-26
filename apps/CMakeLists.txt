add_executable(solver_bench solver_bench.cpp)
target_link_libraries(solver_bench PRIVATE mac_core mac_bench)

add_executable(mac_sim mac_sim.cpp)
target_link_libraries(mac_sim PRIVATE mac_core)
