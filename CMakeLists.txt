cmake_minimum_required(VERSION 3.25)
project(mlxvs VERSION 0.15.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MLXVS_BUILD_PYTHON "Build the Python bindings (nanobind)" OFF)
option(MLXVS_BUILD_TESTS "Build the C++ tests and benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(MLXVS_BUILD_C_API "Build libmlxvs_c, the C API" ${PROJECT_IS_TOP_LEVEL})

# MLX's C++ library. For the Python bindings, the one the `mlx` wheel ships, so
# that one process holds one libmlx; otherwise an installed MLX (e.g. `brew
# install mlx`), or -DMLX_ROOT. mlxvs pins the MLX version it was built against.
if(MLXVS_BUILD_PYTHON)
  find_package(
    Python 3.10
    COMPONENTS Interpreter Development.Module
    REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_ROOT)
endif()
find_package(MLX CONFIG REQUIRED)

add_library(mlxvs STATIC src/knn.cpp src/kmeans.cpp src/nndescent.cpp
                         src/all_neighbors.cpp src/search.cpp src/cagra.cpp
                         src/vamana.cpp src/export.cpp src/ivf.cpp
                         src/vpq.cpp)
target_include_directories(mlxvs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mlxvs PUBLIC mlx)
# IOKit: the GPU core count (graph_search picks single- or multi-CTA by it)
target_link_libraries(mlxvs PUBLIC "-framework IOKit" "-framework CoreFoundation")

# The C API: a shared library exporting only the mlxvs_* C functions
if(MLXVS_BUILD_C_API AND NOT MLXVS_BUILD_PYTHON)
  add_library(mlxvs_c SHARED src/c_api.cpp)
  target_link_libraries(mlxvs_c PRIVATE mlxvs)
  target_compile_definitions(mlxvs_c PRIVATE MLXVS_VERSION="${PROJECT_VERSION}")
  # only the C functions: not mlxvs's (static) C++ symbols linked in
  target_link_options(mlxvs_c PRIVATE "LINKER:-exported_symbol,_mlxvs_*")
  set_target_properties(
    mlxvs_c
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
  include(GNUInstallDirs)
  install(TARGETS mlxvs_c LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES include/mlxvs/mlxvs_c.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlxvs)
endif()

if(MLXVS_BUILD_TESTS)
  add_executable(test_knn tests/test_knn.cpp)
  target_link_libraries(test_knn PRIVATE mlxvs)
  add_executable(test_nnd tests/test_nnd.cpp)
  target_link_libraries(test_nnd PRIVATE mlxvs)
  add_executable(test_all_neighbors tests/test_all_neighbors.cpp)
  target_link_libraries(test_all_neighbors PRIVATE mlxvs)
  add_executable(bench_knn bench/bench_knn.cpp)
  target_link_libraries(bench_knn PRIVATE mlxvs)
  enable_testing()
  add_test(NAME test_knn COMMAND test_knn)
  add_test(NAME test_nnd COMMAND test_nnd)
  add_executable(test_search tests/test_search.cpp)
  target_link_libraries(test_search PRIVATE mlxvs)
  add_test(NAME test_all_neighbors COMMAND test_all_neighbors)
  add_test(NAME test_search COMMAND test_search)
  add_executable(test_cagra tests/test_cagra.cpp)
  target_link_libraries(test_cagra PRIVATE mlxvs)
  add_test(NAME test_cagra COMMAND test_cagra)
  add_executable(test_vamana tests/test_vamana.cpp)
  target_link_libraries(test_vamana PRIVATE mlxvs)
  add_test(NAME test_vamana COMMAND test_vamana)
  add_executable(test_ivf tests/test_ivf.cpp)
  target_link_libraries(test_ivf PRIVATE mlxvs)
  add_test(NAME test_ivf COMMAND test_ivf)
  if(TARGET mlxvs_c)
    add_executable(test_c_api tests/test_c_api.c)
    set_target_properties(test_c_api PROPERTIES C_STANDARD 11)
    target_include_directories(test_c_api
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(test_c_api PRIVATE mlxvs_c)
    add_test(NAME test_c_api COMMAND test_c_api)
  endif()
endif()

if(MLXVS_BUILD_PYTHON)
  # nanobind from the `nanobind` wheel (python -m nanobind --cmake_dir)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  # NB_DOMAIN mlx: mlx.core's array, Stream and Device types are shared with
  # the extension, as MLX's own extension examples do
  nanobind_add_module(
    _core
    NB_STATIC
    LTO
    NOMINSIZE
    NB_DOMAIN
    mlx
    python/bindings.cpp)
  target_link_libraries(_core PRIVATE mlxvs)
  install(TARGETS _core LIBRARY DESTINATION mlxvs)
endif()
