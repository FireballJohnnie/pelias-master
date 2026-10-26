// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// vk_extmem.h - P-E16: a small block of CPU-visible VRAM, allocated by Vulkan (DEVICE_LOCAL | HOST_VISIBLE, reached by
// the CPU through the PCIe BAR) and exported to CUDA by external-memory interop (opaque Win32 handle on Windows, opaque
// file descriptor on Linux). Vulkan is loaded at run time from the driver (vulkan-1.dll / libvulkan.so.1); only the
// Vulkan-Headers are needed to build. Under WSL2 there is no NVIDIA Vulkan driver, so this reports "unavailable".
#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
// From vulkan_win32.h (VK_KHR_external_memory_win32), which the vendored headers do not include.
typedef struct VkMemoryGetWin32HandleInfoKHR_ {
  VkStructureType sType;
  const void* pNext;
  VkDeviceMemory memory;
  VkExternalMemoryHandleTypeFlagBits handleType;
} VkMemoryGetWin32HandleInfoKHR_;
typedef VkResult(VKAPI_PTR* PFN_vkGetMemoryWin32HandleKHR_)(VkDevice, const VkMemoryGetWin32HandleInfoKHR_*, HANDLE*);
#endif

namespace e16 {

class VkVram {
 public:
  // Results of open(): where the block lives and what the device offers.
  void* host = nullptr;    // CPU pointer (through the BAR)
  void* dev = nullptr;     // CUDA device pointer to the same bytes
  size_t bytes = 0;
  uint64_t host_visible_vram_heap_bytes = 0;  // size of the heap behind the chosen memory type
  bool coherent = false;
  std::string vk_device_name;

  VkVram() = default;
  VkVram(const VkVram&) = delete;
  VkVram& operator=(const VkVram&) = delete;
  ~VkVram() { close(); }

  bool open(const unsigned char cuda_uuid[16], size_t want_bytes, std::string& why) {
    if (!load(why)) return false;
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "pelias_e16";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance_(&ici, nullptr, &inst_) != VK_SUCCESS) return fail(why, "vkCreateInstance failed");
    auto gip = [&](const char* n) { return vkGetInstanceProcAddr_(inst_, n); };
    auto enumPD = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gip("vkEnumeratePhysicalDevices"));
    auto props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gip("vkGetPhysicalDeviceProperties2"));
    auto memprops = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(gip("vkGetPhysicalDeviceMemoryProperties"));
    auto qfp = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(gip("vkGetPhysicalDeviceQueueFamilyProperties"));
    auto devext = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(gip("vkEnumerateDeviceExtensionProperties"));
    auto createDev = reinterpret_cast<PFN_vkCreateDevice>(gip("vkCreateDevice"));
    vkGetDeviceProcAddr_ = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gip("vkGetDeviceProcAddr"));
    vkDestroyInstance_ = reinterpret_cast<PFN_vkDestroyInstance>(gip("vkDestroyInstance"));
    if (!enumPD || !props2 || !memprops || !qfp || !devext || !createDev || !vkGetDeviceProcAddr_) return fail(why, "Vulkan 1.1 functions missing");
    uint32_t n = 0;
    enumPD(inst_, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    if (n) enumPD(inst_, &n, pds.data());
    for (VkPhysicalDevice p : pds) {  // the Vulkan device that is the CUDA device (same UUID)
      VkPhysicalDeviceIDProperties idp{};
      idp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
      VkPhysicalDeviceProperties2 pp{};
      pp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      pp.pNext = &idp;
      props2(p, &pp);
      if (std::memcmp(idp.deviceUUID, cuda_uuid, 16) == 0) {
        pd_ = p;
        vk_device_name = pp.properties.deviceName;
        break;
      }
    }
    if (!pd_) return fail(why, "no Vulkan device with the CUDA GPU's UUID");
#ifdef _WIN32
    const char* ext = "VK_KHR_external_memory_win32";
#else
    const char* ext = "VK_KHR_external_memory_fd";
#endif
    uint32_t ne = 0;
    devext(pd_, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne);
    if (ne) devext(pd_, nullptr, &ne, exts.data());
    bool has = false;
    for (const auto& e : exts) has = has || std::strcmp(e.extensionName, ext) == 0;
    if (!has) return fail(why, std::string("device lacks ") + ext);
    uint32_t nq = 0;
    qfp(pd_, &nq, nullptr);
    if (nq == 0) return fail(why, "no queue family");
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char* exts_on[] = {"VK_KHR_external_memory", ext};
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = exts_on;
    if (createDev(pd_, &dci, nullptr, &dev_) != VK_SUCCESS) return fail(why, "vkCreateDevice failed");
    auto gdp = [&](const char* nm) { return vkGetDeviceProcAddr_(dev_, nm); };
    vkDestroyDevice_ = reinterpret_cast<PFN_vkDestroyDevice>(gdp("vkDestroyDevice"));
    auto createBuf = reinterpret_cast<PFN_vkCreateBuffer>(gdp("vkCreateBuffer"));
    vkDestroyBuffer_ = reinterpret_cast<PFN_vkDestroyBuffer>(gdp("vkDestroyBuffer"));
    auto memreq = reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(gdp("vkGetBufferMemoryRequirements"));
    auto alloc = reinterpret_cast<PFN_vkAllocateMemory>(gdp("vkAllocateMemory"));
    vkFreeMemory_ = reinterpret_cast<PFN_vkFreeMemory>(gdp("vkFreeMemory"));
    auto bind = reinterpret_cast<PFN_vkBindBufferMemory>(gdp("vkBindBufferMemory"));
    auto mapm = reinterpret_cast<PFN_vkMapMemory>(gdp("vkMapMemory"));
    vkUnmapMemory_ = reinterpret_cast<PFN_vkUnmapMemory>(gdp("vkUnmapMemory"));
    if (!vkDestroyDevice_ || !createBuf || !vkDestroyBuffer_ || !memreq || !alloc || !vkFreeMemory_ || !bind || !mapm || !vkUnmapMemory_)
      return fail(why, "Vulkan device functions missing");
#ifdef _WIN32
    const VkExternalMemoryHandleTypeFlagBits ht = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    auto gethandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR_>(gdp("vkGetMemoryWin32HandleKHR"));
#else
    const VkExternalMemoryHandleTypeFlagBits ht = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto getfd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(gdp("vkGetMemoryFdKHR"));
#endif
    VkExternalMemoryBufferCreateInfo ebi{};
    ebi.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ebi.handleTypes = ht;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ebi;
    bci.size = want_bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (createBuf(dev_, &bci, nullptr, &buf_) != VK_SUCCESS) return fail(why, "vkCreateBuffer failed");
    VkMemoryRequirements mr{};
    memreq(dev_, buf_, &mr);
    VkPhysicalDeviceMemoryProperties mp{};
    memprops(pd_, &mp);
    const VkMemoryPropertyFlags need = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    int type = -1;
    for (int pass = 0; pass < 2 && type < 0; ++pass) {  // prefer HOST_COHERENT
      for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if (!(mr.memoryTypeBits & (1u << i)) || (f & need) != need) continue;
        if (pass == 0 && !(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) continue;
        type = static_cast<int>(i);
        break;
      }
    }
    if (type < 0) return fail(why, "no DEVICE_LOCAL | HOST_VISIBLE memory type for an exportable buffer");
    coherent = (mp.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    host_visible_vram_heap_bytes = mp.memoryHeaps[mp.memoryTypes[type].heapIndex].size;
    VkExportMemoryAllocateInfo emi{};
    emi.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    emi.handleTypes = ht;
    VkMemoryDedicatedAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dai.pNext = &emi;
    dai.buffer = buf_;
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = &dai;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(type);
    if (alloc(dev_, &mai, nullptr, &mem_) != VK_SUCCESS) return fail(why, "vkAllocateMemory (exportable, CPU-visible VRAM) failed");
    if (bind(dev_, buf_, mem_, 0) != VK_SUCCESS) return fail(why, "vkBindBufferMemory failed");
    if (mapm(dev_, mem_, 0, VK_WHOLE_SIZE, 0, &host) != VK_SUCCESS || !host) return fail(why, "vkMapMemory failed");
    mapped_ = true;
    cudaExternalMemoryHandleDesc hd{};
#ifdef _WIN32
    if (!gethandle) return fail(why, "vkGetMemoryWin32HandleKHR missing");
    VkMemoryGetWin32HandleInfoKHR_ gi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR, nullptr, mem_, ht};
    if (gethandle(dev_, &gi, &handle_) != VK_SUCCESS || !handle_) return fail(why, "vkGetMemoryWin32HandleKHR failed");
    hd.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    hd.handle.win32.handle = handle_;
#else
    if (!getfd) return fail(why, "vkGetMemoryFdKHR missing");
    VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, nullptr, mem_, ht};
    int fd = -1;
    if (getfd(dev_, &gi, &fd) != VK_SUCCESS || fd < 0) return fail(why, "vkGetMemoryFdKHR failed");
    hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
    hd.handle.fd = fd;  // CUDA owns the descriptor after a successful import
#endif
    hd.size = mr.size;
    hd.flags = cudaExternalMemoryDedicated;
    cudaError_t e = cudaImportExternalMemory(&ext_, &hd);
    if (e != cudaSuccess) return fail(why, std::string("cudaImportExternalMemory: ") + cudaGetErrorString(e));
    cudaExternalMemoryBufferDesc bd{};
    bd.offset = 0;
    bd.size = want_bytes;
    e = cudaExternalMemoryGetMappedBuffer(&dev, ext_, &bd);
    if (e != cudaSuccess) return fail(why, std::string("cudaExternalMemoryGetMappedBuffer: ") + cudaGetErrorString(e));
    bytes = want_bytes;
    return true;
  }

  void close() {
    if (ext_) {
      (void)cudaDestroyExternalMemory(ext_);
      ext_ = nullptr;
    }
    dev = nullptr;
#ifdef _WIN32
    if (handle_) {
      CloseHandle(handle_);
      handle_ = nullptr;
    }
#endif
    if (mapped_ && vkUnmapMemory_) vkUnmapMemory_(dev_, mem_);
    mapped_ = false;
    host = nullptr;
    if (mem_ && vkFreeMemory_) vkFreeMemory_(dev_, mem_, nullptr);
    mem_ = VK_NULL_HANDLE;
    if (buf_ && vkDestroyBuffer_) vkDestroyBuffer_(dev_, buf_, nullptr);
    buf_ = VK_NULL_HANDLE;
    if (dev_ && vkDestroyDevice_) vkDestroyDevice_(dev_, nullptr);
    dev_ = VK_NULL_HANDLE;
    if (inst_ && vkDestroyInstance_) vkDestroyInstance_(inst_, nullptr);
    inst_ = VK_NULL_HANDLE;
  }

 private:
  bool fail(std::string& why, const std::string& w) {
    why = w;
    close();
    return false;
  }
  bool load(std::string& why) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("vulkan-1.dll");
    if (!h) return fail(why, "vulkan-1.dll not found");
    vkGetInstanceProcAddr_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(h, "vkGetInstanceProcAddr")));
#else
    void* h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) return fail(why, "libvulkan.so.1 not found (WSL2 has no NVIDIA Vulkan driver)");
    vkGetInstanceProcAddr_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(h, "vkGetInstanceProcAddr"));
#endif
    if (!vkGetInstanceProcAddr_) return fail(why, "vkGetInstanceProcAddr missing");
    vkCreateInstance_ = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr_(nullptr, "vkCreateInstance"));
    if (!vkCreateInstance_) return fail(why, "vkCreateInstance missing");
    return true;
  }

  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_ = nullptr;
  PFN_vkCreateInstance vkCreateInstance_ = nullptr;
  PFN_vkDestroyInstance vkDestroyInstance_ = nullptr;
  PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr_ = nullptr;
  PFN_vkDestroyDevice vkDestroyDevice_ = nullptr;
  PFN_vkDestroyBuffer vkDestroyBuffer_ = nullptr;
  PFN_vkFreeMemory vkFreeMemory_ = nullptr;
  PFN_vkUnmapMemory vkUnmapMemory_ = nullptr;
  VkInstance inst_ = VK_NULL_HANDLE;
  VkPhysicalDevice pd_ = VK_NULL_HANDLE;
  VkDevice dev_ = VK_NULL_HANDLE;
  VkBuffer buf_ = VK_NULL_HANDLE;
  VkDeviceMemory mem_ = VK_NULL_HANDLE;
  bool mapped_ = false;
  cudaExternalMemory_t ext_ = nullptr;
#ifdef _WIN32
  HANDLE handle_ = nullptr;
#endif
};

}  // namespace e16
