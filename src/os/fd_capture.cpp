// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "os/fd_capture.h"

#include <cstdio>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace pelias::os {
bool capture_stdio(const std::function<int()>& operation, int& exit_code, std::string& output) {
  std::FILE* capture = std::tmpfile();
  if (!capture) return false;
  std::fflush(nullptr);
#if defined(_WIN32)
  const int saved_out = _dup(_fileno(stdout));
  const int saved_err = _dup(_fileno(stderr));
  if (saved_out < 0 || saved_err < 0) {
    if (saved_out >= 0) _close(saved_out);
    if (saved_err >= 0) _close(saved_err);
    std::fclose(capture);
    return false;
  }
  if (_dup2(_fileno(capture), _fileno(stdout)) != 0 || _dup2(_fileno(capture), _fileno(stderr)) != 0) {
    _dup2(saved_out, _fileno(stdout));
    _dup2(saved_err, _fileno(stderr));
    _close(saved_out);
    _close(saved_err);
    std::fclose(capture);
    return false;
  }
#else
  const int saved_out = dup(fileno(stdout));
  const int saved_err = dup(fileno(stderr));
  if (saved_out < 0 || saved_err < 0) {
    if (saved_out >= 0) close(saved_out);
    if (saved_err >= 0) close(saved_err);
    std::fclose(capture);
    return false;
  }
  if (dup2(fileno(capture), fileno(stdout)) < 0 || dup2(fileno(capture), fileno(stderr)) < 0) {
    dup2(saved_out, fileno(stdout));
    dup2(saved_err, fileno(stderr));
    close(saved_out);
    close(saved_err);
    std::fclose(capture);
    return false;
  }
#endif
  try {
    exit_code = operation();
  } catch (...) {
    std::fflush(nullptr);
#if defined(_WIN32)
    _dup2(saved_out, _fileno(stdout));
    _dup2(saved_err, _fileno(stderr));
    _close(saved_out);
    _close(saved_err);
#else
    dup2(saved_out, fileno(stdout));
    dup2(saved_err, fileno(stderr));
    close(saved_out);
    close(saved_err);
#endif
    std::fclose(capture);
    throw;
  }
  std::fflush(nullptr);
#if defined(_WIN32)
  _dup2(saved_out, _fileno(stdout));
  _dup2(saved_err, _fileno(stderr));
  _close(saved_out);
  _close(saved_err);
#else
  dup2(saved_out, fileno(stdout));
  dup2(saved_err, fileno(stderr));
  close(saved_out);
  close(saved_err);
#endif
  if (std::fseek(capture, 0, SEEK_SET) != 0) {
    std::fclose(capture);
    return false;
  }
  char buffer[4096];
  for (;;) {
    const std::size_t n = std::fread(buffer, 1, sizeof(buffer), capture);
    output.append(buffer, n);
    if (n < sizeof(buffer)) break;
  }
  std::fclose(capture);
  return true;
}
}  // namespace pelias::os
