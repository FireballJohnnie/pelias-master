# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# pelias_add_module(name SOURCES ... DEPENDS ...): a static library with the project's flags and include paths.
# Every module sees include/ (shared headers) and src/ (module headers by path, e.g. "io/gguf_reader.h"); the module
# dependency graph of 14.3 is enforced by tools/check_module_deps.py, not by CMake.
function(pelias_apply_flags target)
  target_include_directories(${target} PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/generated)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /EHsc /Zc:__cplusplus)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    if(PELIAS_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
    if(PELIAS_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

function(pelias_add_module name)
  cmake_parse_arguments(M "" "" "SOURCES;DEPENDS" ${ARGN})
  add_library(${name} STATIC ${M_SOURCES})
  pelias_apply_flags(${name})
  if(M_DEPENDS)
    target_link_libraries(${name} PUBLIC ${M_DEPENDS})
  endif()
endfunction()
