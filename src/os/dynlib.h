// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// dynlib.h - loads a shared library at run time (NVML, the Vulkan loader, the system's TLS library; 3.2 B2 and B4), so
// the program starts without them. The one OS-specific loader; nothing else includes dlfcn.h or windows.h for this.
#pragma once

#include <string>
#include <vector>

namespace pelias::os {

class DynLib {
 public:
  DynLib() = default;
  DynLib(const DynLib&) = delete;
  DynLib& operator=(const DynLib&) = delete;
  ~DynLib();
  // Tries each name in order; true when one loaded.
  bool open(const std::vector<std::string>& names);
  void close();
  [[nodiscard]] bool is_open() const { return h_ != nullptr; }
  [[nodiscard]] void* symbol(const char* name) const;
  [[nodiscard]] const std::string& loaded_name() const { return name_; }

 private:
  void* h_ = nullptr;
  std::string name_;
};

}  // namespace pelias::os
