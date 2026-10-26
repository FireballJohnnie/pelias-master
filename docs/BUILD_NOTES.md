# Build notes

**Status 2026-10-26: no product code yet.** This file grows step by step as the build proceeds (Masterwork Section 15). Each step adds: what was built, the exact commands used on each platform, the tool versions, what the author ran, and where the results are.

## Toolchains on record (the reference machine)

| Platform | Compiler | CUDA | CMake | Ninja | Git | Label |
|---|---|---|---|---|---|---|
| WSL2 (Ubuntu, kernel 6.18.54.1) | GCC 15.2.0 (also Clang 21) | nvcc 13.0.88 | 4.2.3 | 1.13.2 | 2.53.0 | MEASURED (census 2026-10-23) |
| Windows 11 Pro (build 26300) | MSVC 19.44.35229.0, Visual Studio 18 Professional, toolset 14.44.35207 | nvcc 13.4.92 | 4.4.3 | 1.13.2 | 2.55.0.windows.5 | MEASURED (2026-10-23 01:09; configure log) |
| Driver | 617.42 (CUDA 13.4 user mode) | | | | | MEASURED |
| Avoid | CUDA toolkit 13.2 | | | | | AUTHOR-STATED |

Missing or unconfirmed for a build (from `PELIAS_SYSTEM_MASTER.md` Section 6): Python 3 on Windows (only for two repository checks); the Windows SDK version; `compute-sanitizer` and Nsight on Windows; `ncu` on WSL2; free disk space on both sides.

## Rules for every build step

- Host-only builds and CPU tests run in Claude's sandbox; every GPU build and run is done by the author on WSL2 and on Windows.
- Windows builds run from the x64 Native Tools prompt (Visual Studio 18).
- Warnings are errors in release (`-Wall -Wextra -Werror` / `/W4 /WX`), gate PG9.
- The boundary checks BC1 to BC7 run on every build.

## Steps

*(none yet: the build starts after the Fable review, step R0)*
