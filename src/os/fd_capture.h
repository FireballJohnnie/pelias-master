// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// fd_capture.h - run a function with stdout and stderr redirected into a temporary file and hand back what it wrote.
// v58c (author) moved from cli/main.cc into os/ (BC6: the dup / dup2 / _dup2 calls and <unistd.h> / <io.h> live here).
#pragma once

#include <functional>
#include <string>

namespace pelias::os {
// Runs `operation` with stdout and stderr captured; `exit_code` gets its return value, `output` what it printed.
// Returns false when the descriptors could not be redirected (then nothing ran).
bool capture_stdio(const std::function<int()>& operation, int& exit_code, std::string& output);
}  // namespace pelias::os
