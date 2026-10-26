// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "pelias/status.h"

namespace pelias {

const char* code_name(Code c) {
  switch (c) {
    case Code::ok: return "ok";
    case Code::invalid_argument: return "invalid argument";
    case Code::not_found: return "not found";
    case Code::corrupt: return "corrupt input";
    case Code::unsupported: return "unsupported";
    case Code::io_error: return "I/O error";
    case Code::no_device: return "no device";
    case Code::over_ceiling: return "over the VRAM ceiling";
    case Code::internal: return "internal error";
  }
  return "unknown";
}

std::string Status::text() const {
  std::string s = code_name(code);
  if (!message.empty()) s += ": " + message;
  if (!name.empty()) s += " (name '" + name + "'";
  if (offset != 0) s += (name.empty() ? " (" : ", ") + std::string("at byte ") + std::to_string(offset);
  if (!name.empty() || offset != 0) s += ")";
  return s;
}

}  // namespace pelias
