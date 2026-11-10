cmake_minimum_required(VERSION 3.24)
project(mlxvs VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# MLX's C++ library (e.g. `brew install mlx`): mlxvs pins the MLX version it was built against
find_package(MLX CONFIG REQUIRED)

add_library(mlxvs SHARED src/knn.cpp src/kmeans.cpp src/nndescent.cpp)
target_include_directories(mlxvs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mlxvs PUBLIC mlx)

add_executable(test_knn tests/test_knn.cpp)
target_link_libraries(test_knn PRIVATE mlxvs)

add_executable(test_nnd tests/test_nnd.cpp)
target_link_libraries(test_nnd PRIVATE mlxvs)

add_executable(bench_knn bench/bench_knn.cpp)
target_link_libraries(bench_knn PRIVATE mlxvs)

enable_testing()
add_test(NAME test_knn COMMAND test_knn)
add_test(NAME test_nnd COMMAND test_nnd)
