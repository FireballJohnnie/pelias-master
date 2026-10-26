// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// status.h - the error type every Pelias module returns. No exceptions cross a module boundary: a function that can fail
// returns a Status (or a Result<T>), and the caller decides. A Status carries a code, a short message meant for the
// user, and, where the failure is inside a file, the byte offset and the tensor or key it concerns (rule 7: logs carry
// names, offsets and counts, never file paths or dataset text).
#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace pelias {

enum class Code : int {
  ok = 0,
  invalid_argument,  // a caller passed something the function cannot act on
  not_found,         // file, key, tensor or device missing
  corrupt,           // the input violates its own format; offset and name set when known
  unsupported,       // valid input the product does not handle yet (format version, tensor type)
  io_error,          // the operating system refused a read, map or open
  no_device,         // no usable GPU (CUDA runtime missing or no device)
  over_ceiling,      // a plan would exceed the measured VRAM ceiling
  internal,          // a bug: an invariant broke
};

const char* code_name(Code c);

struct Status {
  Code code = Code::ok;
  std::string message;       // one line, user-readable, no secrets
  std::uint64_t offset = 0;  // byte offset in the input where the problem was found (0 when not applicable)
  std::string name;          // tensor or key name concerned, empty when not applicable

  [[nodiscard]] bool ok() const { return code == Code::ok; }
  explicit operator bool() const { return ok(); }
  std::string text() const;  // "code: message (name 'x' at byte N)"

  static Status good() { return {}; }
  static Status fail(Code c, std::string msg, std::uint64_t off = 0, std::string nm = {}) {
    Status s;
    s.code = c;
    s.message = std::move(msg);
    s.offset = off;
    s.name = std::move(nm);
    return s;
  }
};

template <typename T>
struct Result {
  Status status;
  T value{};
  [[nodiscard]] bool ok() const { return status.ok(); }
  static Result good(T v) {
    Result r;
    r.value = std::move(v);
    return r;
  }
  static Result fail(Status s) {
    Result r;
    r.status = std::move(s);
    return r;
  }
};

}  // namespace pelias
