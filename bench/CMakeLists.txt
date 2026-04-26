add_library(mac_bench STATIC
  spd_generator.cpp
)
target_include_directories(mac_bench PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mac_bench PUBLIC mac_core)
