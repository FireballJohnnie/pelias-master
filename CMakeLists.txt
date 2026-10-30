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
  SOURCES src/os/status.cpp src/os/log.cpp src/os/file_map.cpp src/os/clock.cpp src/os/dynlib.cpp src/os/membw.cpp)
pelias_add_module(pelias_quant
  SOURCES src/quant/block_types.cpp src/quant/quant_cpu.cpp src/quant/tdecode.cpp src/quant/planes.cpp src/quant/dequant_more.cpp src/quant/quant_weighted.cpp src/quant/iq_tables.cpp src/quant/type_rules.cpp
  DEPENDS pelias_os)
# The reference quantizers must round exactly as the reference does: plain float arithmetic, no fused multiply-add
# (contraction) and no reassociation, whatever architecture flags the rest of the build gets later (P-E6).
if(MSVC)
  target_compile_options(pelias_quant PRIVATE /fp:precise)
else()
  target_compile_options(pelias_quant PRIVATE -ffp-contract=off -fno-fast-math)
endif()
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
find_package(Threads REQUIRED)
target_link_libraries(pelias_os PUBLIC Threads::Threads)  # os/membw: the probe's reader threads
if(NOT WIN32)
  target_link_libraries(pelias_os PRIVATE ${CMAKE_DL_LIBS})
endif()
pelias_add_module(pelias_io
  SOURCES src/io/sha256.cpp src/io/json.cpp src/io/gguf_reader.cpp src/io/safetensors.cpp src/io/number_text.cpp src/io/gguf_writer.cpp src/io/requantize.cpp src/io/imatrix_weights.cpp src/io/gguf_compare.cpp src/io/text_file.cpp src/io/planes_build.cpp src/io/tier_build.cpp
  DEPENDS pelias_os pelias_hal pelias_quant)
pelias_add_module(pelias_ingest
  SOURCES src/ingest/hf_metadata.cpp src/ingest/hf_vocab.cpp src/ingest/qwen35moe.cpp src/ingest/stage_a.cpp
  DEPENDS pelias_io pelias_quant pelias_hal)
pelias_add_module(pelias_tok
  SOURCES src/tok/unicode.cpp src/tok/unicode_tables.cpp src/tok/tokenizer.cpp
  DEPENDS pelias_io)
pelias_add_module(pelias_plan
  SOURCES src/plan/memory_plan.cpp
  DEPENDS pelias_hal pelias_io)
pelias_add_module(pelias_model
  SOURCES src/model/spec.cpp src/model/arch_qwen35moe.cpp src/model/imatrix.cpp src/model/sampler.cpp
  DEPENDS pelias_io pelias_quant pelias_hal)
if(NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(pelias_model PUBLIC Threads::Threads)
endif()
# The reference forward pass is plain C++. On an x86-64 CPU with AVX2 and FMA (any Ryzen, Intel since Haswell) it may
# use the wider vector unit. Off by default so a build runs on any x86-64 / ARM64 machine. The arithmetic does not
# change: multiply-adds stay unfused (-ffp-contract=off) and each sum keeps its order, so the logits are the same bit for
# bit as the portable build (MEASURED on the tiny oracle model). -march=native measured 3.6x slower here; not offered.
if(PELIAS_CPU_AVX2 AND NOT MSVC)
  target_compile_options(pelias_model PRIVATE -mavx2 -mfma -ffp-contract=off)
endif()
pelias_add_module(pelias_verify
  SOURCES src/verify/kl.cpp
  DEPENDS pelias_io pelias_quant pelias_hal)

# ---- the program
add_executable(pelias src/cli/main.cc)
pelias_apply_flags(pelias)
target_link_libraries(pelias PRIVATE pelias_model pelias_verify pelias_tok pelias_plan pelias_ingest pelias_io pelias_hal pelias_quant pelias_os)

# ---- tests (own harness, no framework)
if(PELIAS_BUILD_TESTS)
  enable_testing()
  add_executable(pelias_unit tests/unit/main.cpp tests/unit/test_sha256.cpp tests/unit/test_json.cpp tests/unit/test_gguf.cpp
                 tests/unit/test_safetensors.cpp tests/unit/test_plan.cpp tests/unit/test_block_types.cpp tests/unit/test_number_text.cpp tests/unit/test_nvml_pick.cpp tests/unit/test_quant.cpp tests/unit/test_tdecode.cpp tests/unit/test_planes.cpp tests/unit/test_type_rules.cpp tests/unit/test_requantize.cpp tests/unit/test_ingest.cpp tests/unit/test_tok.cpp tests/unit/test_model.cpp tests/unit/test_sampler.cpp tests/unit/test_membw.cpp tests/unit/synth_hf.cpp tests/unit/synth_model.cpp tests/unit/gguf_builder.cpp)
  pelias_apply_flags(pelias_unit)
  target_include_directories(pelias_unit PRIVATE tests)
  target_link_libraries(pelias_unit PRIVATE pelias_model pelias_verify pelias_tok pelias_plan pelias_ingest pelias_io pelias_hal pelias_quant pelias_os)
  add_test(NAME unit COMMAND pelias_unit --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
  add_executable(pelias_mutate tests/unit/mutate.cpp tests/unit/gguf_builder.cpp)
  pelias_apply_flags(pelias_mutate)
  target_include_directories(pelias_mutate PRIVATE tests)
  target_link_libraries(pelias_mutate PRIVATE pelias_io pelias_quant pelias_os)
  add_test(NAME mutate COMMAND pelias_mutate --count 1500 --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME recipe_tool COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_recipe.py)
    add_test(NAME route_stats_tool COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_route_stats.py)
    add_test(NAME imatrix_merge_tool COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_imatrix_merge.py)
  endif()

  # Test-only oracle (masterwork 3.3): the reference library at llama.cpp commit 2ca15f5, built unchanged from a
  # checkout outside this repository, compared byte for byte with Pelias's quantizers. Off unless a path is given;
  # nothing in the product links it (boundary check BC2).
  set(PELIAS_GGML_ORACLE_DIR "" CACHE PATH "llama.cpp source at commit 2ca15f5 (test-only oracle)")
  if(PELIAS_GGML_ORACLE_DIR)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${PELIAS_GGML_ORACLE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml_oracle EXCLUDE_FROM_ALL)
    add_executable(pelias_quant_oracle tests/oracle/quant_oracle.cpp)
    pelias_apply_flags(pelias_quant_oracle)
    target_include_directories(pelias_quant_oracle SYSTEM PRIVATE ${PELIAS_GGML_ORACLE_DIR}/ggml/include ${PELIAS_GGML_ORACLE_DIR}/ggml/src)
    target_include_directories(pelias_quant_oracle PRIVATE tests)
    target_link_libraries(pelias_quant_oracle PRIVATE pelias_quant pelias_io ggml-base)
  endif()
endif()
