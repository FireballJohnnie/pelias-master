# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
option(PELIAS_CUDA "Build the CUDA backend (needs the CUDA toolkit)" OFF)
option(PELIAS_BUILD_TESTS "Build the unit and mutation tests" ON)
option(PELIAS_WARNINGS_AS_ERRORS "Treat warnings as errors (release gate PG9)" ON)
set(PELIAS_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures (sm_75 to sm_120 plus PTX at release, N6)")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(PELIAS_CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_ARCHITECTURES ${PELIAS_CUDA_ARCHITECTURES})
endif()
