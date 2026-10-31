cmake_minimum_required(VERSION 3.16)
project(podmonitor CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# no fused multiply-add: the monitor must match the numpy reference bit for bit
add_compile_options(-O2 -ffp-contract=off -Wall -Wextra -Werror)

add_library(podmonitor SHARED monitor.cpp)
add_executable(test_monitor test_monitor.cpp monitor.cpp)
target_compile_options(test_monitor PRIVATE -UNDEBUG)

enable_testing()
add_test(NAME monitor COMMAND test_monitor)
