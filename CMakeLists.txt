cmake_minimum_required(VERSION 3.18)
project(torch_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python (needed for torch/extension.h -> Python.h)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find PyTorch
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED NO_DEFAULT_PATH)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find Vulkan SDK
find_package(Vulkan REQUIRED)

set(KOMPUTE_OPT_LOG_LEVEL Warn CACHE STRING "Minimize per-dispatch Kompute logging" FORCE)

# Find Kompute
find_package(kompute QUIET)
if(NOT kompute_FOUND)
  # Try pkg-config or manual path
  find_path(KOMPUTE_INCLUDE_DIR kompute/Kompute.hpp
    HINTS /usr/local/include /usr/include
    PATH_SUFFIXES kompute)
  find_library(KOMPUTE_LIBRARY kompute
    HINTS /usr/local/lib /usr/lib)

  if(KOMPUTE_INCLUDE_DIR AND KOMPUTE_LIBRARY)
    message(STATUS "Found Kompute: ${KOMPUTE_LIBRARY}")
  else()
    message(STATUS "Kompute not found, building from source...")
    include(FetchContent)
    FetchContent_Declare(
      kompute
      GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
      GIT_TAG v0.9.0)
    set(KOMPUTE_OPT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KOMPUTE_OPT_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(KOMPUTE_OPT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    # Pin and patch the bundled v0.9.0 runtime. Kompute originally compiles
    # a fresh VkPipeline for each tensor binding; the patch shares immutable
    # pipelines across per-binding Algorithm descriptors. Reproducible, checked
    # patch source is under scripts/patch_kompute.py.
    FetchContent_GetProperties(kompute)
    if(NOT kompute_POPULATED)
      FetchContent_Populate(kompute)
    endif()
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_kompute.py" "${kompute_SOURCE_DIR}"
      RESULT_VARIABLE KOMPUTE_PATCH_RC
      OUTPUT_VARIABLE KOMPUTE_PATCH_OUT
      ERROR_VARIABLE KOMPUTE_PATCH_ERR)
    if(NOT KOMPUTE_PATCH_RC EQUAL 0)
      message(FATAL_ERROR "Kompute patch failed: ${KOMPUTE_PATCH_ERR}")
    endif()
    message(STATUS "${KOMPUTE_PATCH_OUT}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_descriptor_arenas.py" "${kompute_SOURCE_DIR}"
      RESULT_VARIABLE KOMPUTE_DESCRIPTORS_RC
      OUTPUT_VARIABLE KOMPUTE_DESCRIPTORS_OUT
      ERROR_VARIABLE KOMPUTE_DESCRIPTORS_ERR)
    if(NOT KOMPUTE_DESCRIPTORS_RC EQUAL 0)
      message(FATAL_ERROR "Kompute descriptor patch failed: ${KOMPUTE_DESCRIPTORS_ERR}")
    endif()
    message(STATUS "${KOMPUTE_DESCRIPTORS_OUT}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_fence_pool.py" "${kompute_SOURCE_DIR}"
      RESULT_VARIABLE KOMPUTE_FENCES_RC
      OUTPUT_VARIABLE KOMPUTE_FENCES_OUT
      ERROR_VARIABLE KOMPUTE_FENCES_ERR)
    if(NOT KOMPUTE_FENCES_RC EQUAL 0)
      message(FATAL_ERROR "Kompute fence patch failed: ${KOMPUTE_FENCES_ERR}")
    endif()
    message(STATUS "${KOMPUTE_FENCES_OUT}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_barrier_batch.py" "${kompute_SOURCE_DIR}"
      RESULT_VARIABLE KOMPUTE_BARRIERS_RC
      OUTPUT_VARIABLE KOMPUTE_BARRIERS_OUT
      ERROR_VARIABLE KOMPUTE_BARRIERS_ERR)
    if(NOT KOMPUTE_BARRIERS_RC EQUAL 0)
      message(FATAL_ERROR "Kompute barrier batch patch failed: ${KOMPUTE_BARRIERS_ERR}")
    endif()
    message(STATUS "${KOMPUTE_BARRIERS_OUT}")
    add_subdirectory("${kompute_SOURCE_DIR}" "${kompute_BINARY_DIR}")
    set(KOMPUTE_LIBRARY kompute)
    set(KOMPUTE_INCLUDE_DIR "")
  endif()
endif()

# The extension library
add_library(torch_vulkan_C SHARED
  csrc/torch_vulkan.cpp
  csrc/vulkan_generator.cpp
  csrc/ops/fft.cpp
  csrc/ops/linalg.cpp
  csrc/ops/attention.cpp
  csrc/ops/optimizer_primitives.cpp
  csrc/ops/embedding_bag.cpp
  csrc/ops/pool3d.cpp
  csrc/ops/resample.cpp
  csrc/ops/padding.cpp
  csrc/ops/grid_sample.cpp
  csrc/ops/sparse_coo.cpp
  csrc/ops/out_adapters.cpp
  csrc/ops/factories.cpp
  csrc/ops/reduction_general.cpp
  csrc/ops/svd.cpp
  csrc/ops/bitwise.cpp
  csrc/ops/advanced_indexing.cpp
  csrc/ops/tensor_selection.cpp
  csrc/ops/pointwise_math.cpp
  csrc/ops/views.cpp
  csrc/ops/comparison.cpp
  csrc/ops/device_copy.cpp
  csrc/ops/integer_ops.cpp
  csrc/ops/collection_primitives.cpp
  csrc/ops/additional_losses.cpp
  csrc/ops/order_statistics.cpp
  csrc/ops/indexing.cpp
  csrc/ops/random.cpp
  csrc/vulkan_context.cpp
  csrc/vulkan_allocator.cpp
  csrc/ops/mm.cpp
  csrc/ops/scalar_add.cpp
  csrc/ops/activation.cpp
  csrc/ops/loss_primitives.cpp
  csrc/ops/nll_loss.cpp
  csrc/ops/cat.cpp
  csrc/ops/pooling.cpp
  csrc/ops/predicates_math.cpp
  csrc/ops/average_pool.cpp
  csrc/ops/resize_linear_cubic.cpp
  csrc/ops/upsample.cpp
  csrc/ops/batch_norm.cpp
  csrc/ops/group_norm.cpp
  csrc/ops/diffusion_unary.cpp
  csrc/ops/softmax_axis_backward.cpp
  csrc/ops/bmm.cpp
  csrc/ops/convolution_nd.cpp
  csrc/ops/recurrent.cpp
  csrc/ops/convolution.cpp
  csrc/ops/adamw.cpp
  csrc/ops/layer_norm_backward.cpp
  csrc/ops/reduce.cpp
  csrc/ops/log_softmax.cpp
  csrc/vulkan_engine.cpp
)

# Public algorithm-cache counters exist only with our verified bundled patch.
if(TARGET kompute)
  target_compile_definitions(torch_vulkan_C PRIVATE TORCH_VULKAN_PATCHED_KOMPUTE=1)
endif()

target_include_directories(torch_vulkan_C PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${Vulkan_INCLUDE_DIRS}
  ${KOMPUTE_INCLUDE_DIR}
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
  csrc/
)

target_link_libraries(torch_vulkan_C PRIVATE Python3::Python)

target_link_libraries(torch_vulkan_C PRIVATE
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  ${Vulkan_LIBRARIES}
  ${KOMPUTE_LIBRARY}
)

# Output as _C.so for Python import
set_target_properties(torch_vulkan_C PROPERTIES
  OUTPUT_NAME "_C"
  PREFIX ""
  SUFFIX ".${Python3_SOABI}.so"
)

option(TORCH_VULKAN_BUILD_RUNTIME_TESTS "Build bundled Kompute lifecycle regression tests" OFF)
if(TORCH_VULKAN_BUILD_RUNTIME_TESTS)
  if(NOT TARGET kompute OR NOT EXISTS "${kompute_SOURCE_DIR}/src/include/kompute/FencePool.hpp")
    message(FATAL_ERROR "Runtime tests require the bundled patched Kompute")
  endif()
  find_package(Threads REQUIRED)
  add_executable(test_fence_pool tests/cpp/test_fence_pool.cpp)
  target_link_libraries(test_fence_pool PRIVATE kompute Vulkan::Vulkan Threads::Threads)
endif()

# Capture the actual completed build, not a later edited working tree.
add_custom_command(TARGET torch_vulkan_C POST_BUILD
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/record_build_sources.py
    --extension $<TARGET_FILE:torch_vulkan_C>
    --output ${CMAKE_BINARY_DIR}/build-source-manifest.json
  VERBATIM)
