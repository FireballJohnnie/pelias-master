# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# Pelias: one tree, two shared layers and one program (masterwork 14.2). Host-only by default so the whole tree builds
# and tests on a machine without a GPU; -DPELIAS_CUDA=ON adds the CUDA backend (the runtime linked statically, 3.2 B2).
cmake_minimum_required(VERSION 3.24)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PELIAS_VERSION LIMIT_COUNT 1)
project(pelias VERSION ${PELIAS_VERSION} LANGUAGES CXX)

include(cmake/PeliasOptions.cmake)
include(cmake/PeliasModule.cmake)

if(PELIAS_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(PELIAS_HAS_CUDA_BOOL true)
else()
  set(PELIAS_HAS_CUDA_BOOL false)
endif()

# version.h from VERSION and the git commit (empty when git is absent)
find_package(Git QUIET)
set(PELIAS_GIT_COMMIT "unknown")
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTPUT_VARIABLE PELIAS_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
configure_file(include/pelias/version.h.in ${CMAKE_BINARY_DIR}/generated/pelias/version.h @ONLY)

# ---- Layer 0: os, hal, plan (libpelias_substrate later; one library now)
pelias_add_module(pelias_os
  SOURCES src/os/status.cpp src/os/log.cpp src/os/file_map.cpp src/os/clock.cpp src/os/dynlib.cpp)
pelias_add_module(pelias_quant
  SOURCES src/quant/block_types.cpp
  DEPENDS pelias_os)
set(PELIAS_HAL_SOURCES src/hal/device.cpp src/hal/nvml.cpp src/hal/vram_governor.cpp)
if(PELIAS_CUDA)
  list(APPEND PELIAS_HAL_SOURCES src/hal/backend_cuda/cuda_device.cpp)
else()
  list(APPEND PELIAS_HAL_SOURCES src/hal/backend_cpu/no_cuda_device.cpp)
endif()
pelias_add_module(pelias_hal SOURCES ${PELIAS_HAL_SOURCES} DEPENDS pelias_os)
if(PELIAS_CUDA)
  target_link_libraries(pelias_hal PRIVATE CUDA::cudart_static)
  if(NOT WIN32)
    target_link_libraries(pelias_hal PRIVATE ${CMAKE_DL_LIBS} pthread rt)
  endif()
endif()
if(NOT WIN32)
  target_link_libraries(pelias_os PRIVATE ${CMAKE_DL_LIBS})
endif()
pelias_add_module(pelias_io
  SOURCES src/io/sha256.cpp src/io/json.cpp src/io/gguf_reader.cpp src/io/safetensors.cpp src/io/number_text.cpp
  DEPENDS pelias_os pelias_hal pelias_quant)
pelias_add_module(pelias_plan
  SOURCES src/plan/memory_plan.cpp
  DEPENDS pelias_hal pelias_io)

# ---- the program
add_executable(pelias src/cli/main.cc)
pelias_apply_flags(pelias)
target_link_libraries(pelias PRIVATE pelias_plan pelias_io pelias_hal pelias_quant pelias_os)

# ---- tests (own harness, no framework)
if(PELIAS_BUILD_TESTS)
  enable_testing()
  add_executable(pelias_unit tests/unit/main.cpp tests/unit/test_sha256.cpp tests/unit/test_json.cpp tests/unit/test_gguf.cpp
                 tests/unit/test_safetensors.cpp tests/unit/test_plan.cpp tests/unit/test_block_types.cpp tests/unit/test_number_text.cpp tests/unit/test_nvml_pick.cpp tests/unit/gguf_builder.cpp)
  pelias_apply_flags(pelias_unit)
  target_include_directories(pelias_unit PRIVATE tests)
  target_link_libraries(pelias_unit PRIVATE pelias_plan pelias_io pelias_hal pelias_quant pelias_os)
  add_test(NAME unit COMMAND pelias_unit --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
  add_executable(pelias_mutate tests/unit/mutate.cpp tests/unit/gguf_builder.cpp)
  pelias_apply_flags(pelias_mutate)
  target_include_directories(pelias_mutate PRIVATE tests)
  target_link_libraries(pelias_mutate PRIVATE pelias_io pelias_quant pelias_os)
  add_test(NAME mutate COMMAND pelias_mutate --count 1500 --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
endif()
