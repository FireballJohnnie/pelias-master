// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "os/dynlib.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000  // documented value (LoadLibraryEx); older headers may lack it
#endif
#else
#include <dlfcn.h>
#endif

namespace pelias::os {

DynLib::~DynLib() { close(); }

bool DynLib::open(const std::vector<std::string>& names) {
  close();
  for (const std::string& n : names) {
#ifdef _WIN32
    // System folders and the program's own folder only: never the current directory (DLL planting).
    HMODULE h = LoadLibraryExA(n.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (h) {
      h_ = h;
      name_ = n;
      return true;
    }
#else
    void* h = dlopen(n.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h) {
      h_ = h;
      name_ = n;
      return true;
    }
#endif
  }
  return false;
}

void DynLib::close() {
  if (!h_) return;
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(h_));
#else
  dlclose(h_);
#endif
  h_ = nullptr;
  name_.clear();
}

void* DynLib::symbol(const char* name) const {
  if (!h_) return nullptr;
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(h_), name));
#else
  return dlsym(h_, name);
#endif
}

}  // namespace pelias::os
