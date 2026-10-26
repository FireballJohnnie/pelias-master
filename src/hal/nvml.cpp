// provenance: original (public NVML API by name)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "hal/nvml.h"

#include "os/dynlib.h"

namespace pelias::hal {
namespace {

using Device = void*;
struct Memory {
  unsigned long long total, free, used;
};
using InitFn = int (*)();
using ShutdownFn = int (*)();
using GetHandleFn = int (*)(unsigned, Device*);
using GetHandleByPciFn = int (*)(const char*, Device*);
using MemInfoFn = int (*)(Device, Memory*);
using DriverFn = int (*)(char*, unsigned);

}  // namespace

NvmlMemory nvml_memory(const std::string& pci_bus_id) {
  NvmlMemory m;
  os::DynLib lib;
#ifdef _WIN32
  if (!lib.open({"nvml.dll"})) {
#else
  if (!lib.open({"libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1", "libnvidia-ml.so"})) {
#endif
    m.note = "NVML library not found";
    return m;
  }
  auto init = reinterpret_cast<InitFn>(lib.symbol("nvmlInit_v2"));
  auto shutdown = reinterpret_cast<ShutdownFn>(lib.symbol("nvmlShutdown"));
  auto by_index = reinterpret_cast<GetHandleFn>(lib.symbol("nvmlDeviceGetHandleByIndex_v2"));
  auto by_pci = reinterpret_cast<GetHandleByPciFn>(lib.symbol("nvmlDeviceGetHandleByPciBusId_v2"));
  auto meminfo = reinterpret_cast<MemInfoFn>(lib.symbol("nvmlDeviceGetMemoryInfo"));
  auto driver = reinterpret_cast<DriverFn>(lib.symbol("nvmlSystemGetDriverVersion"));
  if (!init || !shutdown || !by_index || !meminfo) {
    m.note = "NVML functions missing";
    return m;
  }
  if (init() != 0) {
    m.note = "NVML init failed";
    return m;
  }
  Device dev = nullptr;
  int rc = -1;
  if (!pci_bus_id.empty() && by_pci) rc = by_pci(pci_bus_id.c_str(), &dev);
  if (rc != 0) rc = by_index(0, &dev);
  if (rc == 0) {
    Memory mem{};
    if (meminfo(dev, &mem) == 0) {
      m.ok = true;
      m.total = mem.total;
      m.free = mem.free;
      m.used = mem.used;
    } else {
      m.note = "NVML memory query failed";
    }
  } else {
    m.note = "NVML device handle failed";
  }
  if (driver) {
    char buf[96] = {};
    if (driver(buf, sizeof buf) == 0) m.driver_version = buf;
  }
  shutdown();
  return m;
}

}  // namespace pelias::hal
