// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// vk_probe.cpp: the Vulkan half of gfx_probe (P-G0). Measures each bus building block "without" (the usual way)
// and "with" (the Pelias way) on the same GPU, in the same run:
//   submit   : record and submit every job        vs  replay one pre-recorded list of the same jobs
//   sync     : fence round trip per hand-off      vs  doorbell (a resident shader answering a CPU-written flag)
//   upload   : staging buffer plus copy command   vs  direct CPU write into GPU memory through the BAR
//   overlap  : copy, then compute                 vs  copy and compute on separate queues at once
// Vulkan is loaded at run time from the driver (vulkan-1.dll / libvulkan.so.1), so no Vulkan SDK is needed.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <atomic>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "probe.h"
#include "shaders_spv.h"

namespace gp {
namespace {

#define GP_VK_INSTANCE_FUNCS(X)                  \
    X(vkEnumeratePhysicalDevices)                \
    X(vkGetPhysicalDeviceProperties)             \
    X(vkGetPhysicalDeviceQueueFamilyProperties)  \
    X(vkGetPhysicalDeviceMemoryProperties)       \
    X(vkGetPhysicalDeviceFeatures2)              \
    X(vkCreateDevice)                            \
    X(vkGetDeviceProcAddr)                       \
    X(vkDestroyInstance)

#define GP_VK_DEVICE_FUNCS(X)                                                                                     \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkMapMemory) X(vkCreateShaderModule)             \
    X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout)                       \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateComputePipelines) X(vkDestroyPipeline)         \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)    \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkResetCommandPool) X(vkAllocateCommandBuffers)              \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets)                 \
    X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkQueueSubmit)            \
    X(vkQueueWaitIdle) X(vkDeviceWaitIdle) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences)                  \
    X(vkResetFences)

struct Fn {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = nullptr;
#define GP_DECL(n) PFN_##n n = nullptr;
    GP_VK_INSTANCE_FUNCS(GP_DECL)
    GP_VK_DEVICE_FUNCS(GP_DECL)
#undef GP_DECL
};

void ck(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed: VkResult " + std::to_string(static_cast<int>(r)));
}

std::string version_text(uint32_t v) {
    return std::to_string(VK_API_VERSION_MAJOR(v)) + "." + std::to_string(VK_API_VERSION_MINOR(v)) + "." +
           std::to_string(VK_API_VERSION_PATCH(v));
}

// NVIDIA packs its driver version as 10.8.8.6 bits; other vendors use the Vulkan packing.
std::string driver_text(uint32_t vendor, uint32_t v) {
    if (vendor == 0x10DE) {
        char b[32];
        std::snprintf(b, sizeof b, "%u.%02u", (v >> 22) & 0x3FF, (v >> 14) & 0xFF);
        return b;
    }
    return version_text(v);
}

std::string flags_text(VkMemoryPropertyFlags f) {
    std::string s;
    auto add = [&](VkMemoryPropertyFlags bit, const char* n) { if (f & bit) s += (s.empty() ? "" : "|") + std::string(n); };
    add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL");
    add(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE");
    add(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT");
    add(VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED");
    return s.empty() ? "NONE" : s;
}

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* map = nullptr;
    VkDeviceSize size = 0;
    uint32_t type = 0;
    VkDeviceSize heap_size = 0;
    VkMemoryPropertyFlags flags = 0;
};

struct Queue {
    uint32_t family = 0;
    uint32_t index = 0;
    VkQueue q = VK_NULL_HANDLE;
};

constexpr VkMemoryPropertyFlags kDev = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
constexpr VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr uint64_t kWaitNs = 10ull * 1000 * 1000 * 1000;  // 10 s: any wait longer than this is a failure

class Vk {
public:
    Fn f;
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem{};
    std::vector<VkQueueFamilyProperties> families;
    VkDevice dev = VK_NULL_HANDLE;
    Queue main_q, compute_q, transfer_q;
    std::vector<uint32_t> used_families;

    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipeline pipe_tiny = VK_NULL_HANDLE, pipe_doorbell = VK_NULL_HANDLE, pipe_busy = VK_NULL_HANDLE;
    VkPipeline pipe_door_atomic = VK_NULL_HANDLE, pipe_door_mm = VK_NULL_HANDLE;
    bool mm_ok = false;           // vulkanMemoryModel + vulkanMemoryModelDeviceScope supported and enabled
    std::string mm_error;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    std::vector<Buffer> buffers;
    std::vector<VkCommandPool> pools;
    std::vector<VkFence> fences;
    std::vector<VkShaderModule> modules;

    ~Vk() { cleanup(); }

    void load() {
#ifdef _WIN32
        HMODULE h = LoadLibraryA("vulkan-1.dll");
        if (!h) throw std::runtime_error("vulkan-1.dll not found (the GPU driver installs it)");
        f.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(h, "vkGetInstanceProcAddr")));
#else
        void* h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!h) throw std::runtime_error("libvulkan.so.1 not found");
        f.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(h, "vkGetInstanceProcAddr"));
#endif
        if (!f.vkGetInstanceProcAddr) throw std::runtime_error("vkGetInstanceProcAddr missing");
        f.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(f.vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
        f.vkEnumerateInstanceVersion =
            reinterpret_cast<PFN_vkEnumerateInstanceVersion>(f.vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
        if (!f.vkCreateInstance || !f.vkEnumerateInstanceVersion) throw std::runtime_error("Vulkan 1.1 loader required");
    }

    void create_instance() {
        uint32_t v = 0;
        ck(f.vkEnumerateInstanceVersion(&v), "vkEnumerateInstanceVersion");
        if (v < VK_API_VERSION_1_2) throw std::runtime_error("Vulkan 1.2 loader required, found " + version_text(v));
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "gfx_probe";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        ck(f.vkCreateInstance(&ci, nullptr, &inst), "vkCreateInstance");
#define GP_LOAD(n) f.n = reinterpret_cast<PFN_##n>(f.vkGetInstanceProcAddr(inst, #n)); if (!f.n) throw std::runtime_error(#n " missing");
        GP_VK_INSTANCE_FUNCS(GP_LOAD)
#undef GP_LOAD
    }

    void pick_device(int index) {
        uint32_t n = 0;
        ck(f.vkEnumeratePhysicalDevices(inst, &n, nullptr), "vkEnumeratePhysicalDevices");
        if (n == 0) throw std::runtime_error("no Vulkan device");
        std::vector<VkPhysicalDevice> all(n);
        ck(f.vkEnumeratePhysicalDevices(inst, &n, all.data()), "vkEnumeratePhysicalDevices");
        if (index >= 0) {
            if (static_cast<uint32_t>(index) >= n) throw std::runtime_error("--vk-device out of range");
            phys = all[index];
        } else {
            phys = all[0];
            for (auto p : all) {
                VkPhysicalDeviceProperties pr{};
                f.vkGetPhysicalDeviceProperties(p, &pr);
                if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = p; break; }
            }
        }
        f.vkGetPhysicalDeviceProperties(phys, &props);
        f.vkGetPhysicalDeviceMemoryProperties(phys, &mem);
        if (props.apiVersion < VK_API_VERSION_1_2) throw std::runtime_error("device supports Vulkan " + version_text(props.apiVersion) + "; 1.2 needed");
        uint32_t fc = 0;
        f.vkGetPhysicalDeviceQueueFamilyProperties(phys, &fc, nullptr);
        families.resize(fc);
        f.vkGetPhysicalDeviceQueueFamilyProperties(phys, &fc, families.data());
    }

    void create_device() {
        auto has = [&](uint32_t i, VkQueueFlags b) { return (families[i].queueFlags & b) != 0; };
        uint32_t main_f = UINT32_MAX, compute_f = UINT32_MAX, transfer_f = UINT32_MAX;
        for (uint32_t i = 0; i < families.size(); ++i)
            if (has(i, VK_QUEUE_COMPUTE_BIT) && has(i, VK_QUEUE_GRAPHICS_BIT) && main_f == UINT32_MAX) main_f = i;
        for (uint32_t i = 0; i < families.size() && main_f == UINT32_MAX; ++i)
            if (has(i, VK_QUEUE_COMPUTE_BIT)) main_f = i;
        if (main_f == UINT32_MAX) throw std::runtime_error("no compute queue");
        for (uint32_t i = 0; i < families.size(); ++i) {
            if (compute_f == UINT32_MAX && has(i, VK_QUEUE_COMPUTE_BIT) && !has(i, VK_QUEUE_GRAPHICS_BIT)) compute_f = i;
            if (transfer_f == UINT32_MAX && has(i, VK_QUEUE_TRANSFER_BIT) && !has(i, VK_QUEUE_COMPUTE_BIT) && !has(i, VK_QUEUE_GRAPHICS_BIT))
                transfer_f = i;
        }
        // Hand out queue indices family by family; when a family runs out, the last queue is shared.
        std::map<uint32_t, uint32_t> next;
        auto take = [&](uint32_t fam) {
            uint32_t idx = next[fam];
            if (idx >= families[fam].queueCount) return Queue{fam, families[fam].queueCount - 1};
            next[fam] = idx + 1;
            return Queue{fam, idx};
        };
        main_q = take(main_f);
        compute_q = take(compute_f != UINT32_MAX ? compute_f : main_f);
        transfer_q = take(transfer_f != UINT32_MAX ? transfer_f : compute_q.family);

        std::vector<VkDeviceQueueCreateInfo> qci;
        std::vector<std::vector<float>> prio;
        for (auto& [fam, count] : next) {
            prio.emplace_back(count, 1.0f);
            VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            q.queueFamilyIndex = fam;
            q.queueCount = count;
            q.pQueuePriorities = prio.back().data();
            qci.push_back(q);
            used_families.push_back(fam);
        }
        VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        have.pNext = &have12;
        f.vkGetPhysicalDeviceFeatures2(phys, &have);
        mm_ok = have12.vulkanMemoryModel && have12.vulkanMemoryModelDeviceScope;
        VkPhysicalDeviceVulkan12Features want12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        want12.vulkanMemoryModel = mm_ok ? VK_TRUE : VK_FALSE;
        want12.vulkanMemoryModelDeviceScope = mm_ok ? VK_TRUE : VK_FALSE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &want12;
        dci.queueCreateInfoCount = static_cast<uint32_t>(qci.size());
        dci.pQueueCreateInfos = qci.data();
        ck(f.vkCreateDevice(phys, &dci, nullptr, &dev), "vkCreateDevice");
#define GP_LOAD(n) f.n = reinterpret_cast<PFN_##n>(f.vkGetDeviceProcAddr(dev, #n)); if (!f.n) throw std::runtime_error(#n " missing");
        GP_VK_DEVICE_FUNCS(GP_LOAD)
#undef GP_LOAD
        for (Queue* q : {&main_q, &compute_q, &transfer_q}) f.vkGetDeviceQueue(dev, q->family, q->index, &q->q);
    }

    Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags need, VkMemoryPropertyFlags avoid,
                       VkMemoryPropertyFlags prefer = 0) {
        Buffer b;
        b.size = size;
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = size;
        ci.usage = usage;
        if (used_families.size() > 1) {
            ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = static_cast<uint32_t>(used_families.size());
            ci.pQueueFamilyIndices = used_families.data();
        }
        ck(f.vkCreateBuffer(dev, &ci, nullptr, &b.buf), "vkCreateBuffer");
        VkMemoryRequirements req{};
        f.vkGetBufferMemoryRequirements(dev, b.buf, &req);
        int pick = -1;
        for (int pass = 0; pass < 2 && pick < 0; ++pass) {
            for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
                VkMemoryPropertyFlags fl = mem.memoryTypes[i].propertyFlags;
                if (!(req.memoryTypeBits & (1u << i)) || (fl & need) != need || (fl & avoid)) continue;
                if (pass == 0 && prefer && (fl & prefer) != prefer) continue;
                pick = static_cast<int>(i);
                break;
            }
        }
        if (pick < 0) {
            f.vkDestroyBuffer(dev, b.buf, nullptr);
            throw std::runtime_error("no memory type with " + flags_text(need));
        }
        b.type = static_cast<uint32_t>(pick);
        b.flags = mem.memoryTypes[pick].propertyFlags;
        b.heap_size = mem.memoryHeaps[mem.memoryTypes[pick].heapIndex].size;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = b.type;
        VkResult r = f.vkAllocateMemory(dev, &ai, nullptr, &b.mem);
        if (r != VK_SUCCESS) {
            f.vkDestroyBuffer(dev, b.buf, nullptr);
            ck(r, "vkAllocateMemory");
        }
        ck(f.vkBindBufferMemory(dev, b.buf, b.mem, 0), "vkBindBufferMemory");
        if (b.flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ck(f.vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
        buffers.push_back(b);
        return b;
    }

    VkPipeline make_pipeline(const uint32_t* code, size_t bytes) {
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = bytes;
        sm.pCode = code;
        VkShaderModule mod = VK_NULL_HANDLE;
        ck(f.vkCreateShaderModule(dev, &sm, nullptr, &mod), "vkCreateShaderModule");
        modules.push_back(mod);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = pipe_layout;
        VkPipeline p = VK_NULL_HANDLE;
        ck(f.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p), "vkCreateComputePipelines");
        return p;
    }

    void create_pipelines() {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 1;
        lci.pBindings = &b;
        ck(f.vkCreateDescriptorSetLayout(dev, &lci, nullptr, &set_layout), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pci.setLayoutCount = 1;
        pci.pSetLayouts = &set_layout;
        pci.pushConstantRangeCount = 1;
        pci.pPushConstantRanges = &pc;
        ck(f.vkCreatePipelineLayout(dev, &pci, nullptr, &pipe_layout), "vkCreatePipelineLayout");
        pipe_tiny = make_pipeline(kSpv_tiny, kSpv_tiny_size);
        pipe_doorbell = make_pipeline(kSpv_doorbell, kSpv_doorbell_size);
        pipe_busy = make_pipeline(kSpv_busy, kSpv_busy_size);
        pipe_door_atomic = make_pipeline(kSpv_doorbell_atomic, kSpv_doorbell_atomic_size);
        if (mm_ok) {
            try {
                pipe_door_mm = make_pipeline(kSpv_doorbell_mm, kSpv_doorbell_mm_size);
            } catch (const std::exception& e) {
                mm_error = e.what();
            }
        } else {
            mm_error = "vulkanMemoryModel / vulkanMemoryModelDeviceScope not supported";
        }
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dp.maxSets = 8;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &ps;
        ck(f.vkCreateDescriptorPool(dev, &dp, nullptr, &desc_pool), "vkCreateDescriptorPool");
    }

    VkDescriptorSet make_set(const Buffer& b) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &set_layout;
        VkDescriptorSet s = VK_NULL_HANDLE;
        ck(f.vkAllocateDescriptorSets(dev, &ai, &s), "vkAllocateDescriptorSets");
        VkDescriptorBufferInfo bi{b.buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = s;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bi;
        f.vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
        return s;
    }

    VkCommandPool make_pool(uint32_t family, VkCommandPoolCreateFlags flags = 0) {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.queueFamilyIndex = family;
        ci.flags = flags;
        VkCommandPool p = VK_NULL_HANDLE;
        ck(f.vkCreateCommandPool(dev, &ci, nullptr, &p), "vkCreateCommandPool");
        pools.push_back(p);
        return p;
    }

    std::vector<VkCommandBuffer> alloc(VkCommandPool pool, uint32_t n) {
        std::vector<VkCommandBuffer> v(n);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = n;
        ck(f.vkAllocateCommandBuffers(dev, &ai, v.data()), "vkAllocateCommandBuffers");
        return v;
    }

    VkFence make_fence() {
        VkFenceCreateInfo ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fe = VK_NULL_HANDLE;
        ck(f.vkCreateFence(dev, &ci, nullptr, &fe), "vkCreateFence");
        fences.push_back(fe);
        return fe;
    }

    void begin(VkCommandBuffer c, VkCommandBufferUsageFlags flags = 0) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = flags;
        ck(f.vkBeginCommandBuffer(c, &bi), "vkBeginCommandBuffer");
    }
    void end(VkCommandBuffer c) { ck(f.vkEndCommandBuffer(c), "vkEndCommandBuffer"); }

    void submit(const Queue& q, VkCommandBuffer c, VkFence fe) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c;
        ck(f.vkQueueSubmit(q.q, 1, &si, fe), "vkQueueSubmit");
    }
    void wait(VkFence fe) {
        ck(f.vkWaitForFences(dev, 1, &fe, VK_TRUE, kWaitNs), "vkWaitForFences");
        ck(f.vkResetFences(dev, 1, &fe), "vkResetFences");
    }
    double run_once(const Queue& q, VkCommandBuffer c, VkFence fe) {
        double t0 = now_us();
        submit(q, c, fe);
        wait(fe);
        return now_us() - t0;
    }

    // One job = a compute-to-compute barrier plus one tiny dispatch, the same in every submit variant.
    void record_tiny(VkCommandBuffer c, VkDescriptorSet set) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        f.vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        f.vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_tiny);
        f.vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &set, 0, nullptr);
        f.vkCmdDispatch(c, 1, 1, 1);
    }

    void cleanup() {
        if (dev) {
            f.vkDeviceWaitIdle(dev);
            for (auto fe : fences) f.vkDestroyFence(dev, fe, nullptr);
            for (auto p : pools) f.vkDestroyCommandPool(dev, p, nullptr);
            if (desc_pool) f.vkDestroyDescriptorPool(dev, desc_pool, nullptr);
            for (auto p : {pipe_tiny, pipe_doorbell, pipe_busy, pipe_door_atomic, pipe_door_mm}) if (p) f.vkDestroyPipeline(dev, p, nullptr);
            for (auto m : modules) f.vkDestroyShaderModule(dev, m, nullptr);
            if (pipe_layout) f.vkDestroyPipelineLayout(dev, pipe_layout, nullptr);
            if (set_layout) f.vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
            for (auto& b : buffers) {
                f.vkDestroyBuffer(dev, b.buf, nullptr);
                f.vkFreeMemory(dev, b.mem, nullptr);
            }
            f.vkDestroyDevice(dev, nullptr);
            dev = VK_NULL_HANDLE;
        }
        if (inst) {
            f.vkDestroyInstance(inst, nullptr);
            inst = VK_NULL_HANDLE;
        }
    }
};

std::string device_json(Vk& vk) {
    Json j;
    j.str("name", vk.props.deviceName);
    j.str("api", version_text(vk.props.apiVersion));
    j.str("driver", driver_text(vk.props.vendorID, vk.props.driverVersion));
    j.integer("vendor_id", vk.props.vendorID);
    j.integer("device_id", vk.props.deviceID);
    std::string heaps = "[";
    for (uint32_t i = 0; i < vk.mem.memoryHeapCount; ++i) {
        Json h;
        h.num("size_mib", vk.mem.memoryHeaps[i].size / 1048576.0);
        h.boolean("device_local", (vk.mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0);
        heaps += (i ? ", " : "") + h.text();
    }
    j.raw("memory_heaps", heaps + "]");
    std::string types = "[";
    for (uint32_t i = 0; i < vk.mem.memoryTypeCount; ++i) {
        Json t;
        t.str("flags", flags_text(vk.mem.memoryTypes[i].propertyFlags));
        t.integer("heap", vk.mem.memoryTypes[i].heapIndex);
        types += (i ? ", " : "") + t.text();
    }
    j.raw("memory_types", types + "]");
    auto qtext = [](const Queue& q) { return std::to_string(q.family) + ":" + std::to_string(q.index); };
    j.str("queue_main", qtext(vk.main_q));
    j.str("queue_compute", qtext(vk.compute_q));
    j.str("queue_transfer", qtext(vk.transfer_q));
    return j.text();
}

// submit: per-job cost of three ways to issue the same N jobs.
std::string test_submit(Vk& vk, const Settings& s, VkDescriptorSet out_set) {
    const uint32_t n = static_cast<uint32_t>(s.submit_jobs);
    VkCommandPool pool_each = vk.make_pool(vk.main_q.family);
    VkCommandPool pool_pre = vk.make_pool(vk.main_q.family);
    VkCommandPool pool_replay = vk.make_pool(vk.main_q.family);
    auto each = vk.alloc(pool_each, n);
    auto pre = vk.alloc(pool_pre, n);
    auto replay = vk.alloc(pool_replay, 1);
    for (auto c : pre) { vk.begin(c); vk.record_tiny(c, out_set); vk.end(c); }
    vk.begin(replay[0]);
    for (uint32_t i = 0; i < n; ++i) vk.record_tiny(replay[0], out_set);
    vk.end(replay[0]);

    auto run_each = [&] {
        ck(vk.f.vkResetCommandPool(vk.dev, pool_each, 0), "vkResetCommandPool");
        double t0 = now_us();
        for (auto c : each) {
            vk.begin(c, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            vk.record_tiny(c, out_set);
            vk.end(c);
            vk.submit(vk.main_q, c, VK_NULL_HANDLE);
        }
        ck(vk.f.vkQueueWaitIdle(vk.main_q.q), "vkQueueWaitIdle");
        return (now_us() - t0) / n;
    };
    auto run_pre = [&] {
        double t0 = now_us();
        for (auto c : pre) vk.submit(vk.main_q, c, VK_NULL_HANDLE);
        ck(vk.f.vkQueueWaitIdle(vk.main_q.q), "vkQueueWaitIdle");
        return (now_us() - t0) / n;
    };
    auto run_replay = [&] {
        double t0 = now_us();
        vk.submit(vk.main_q, replay[0], VK_NULL_HANDLE);
        ck(vk.f.vkQueueWaitIdle(vk.main_q.q), "vkQueueWaitIdle");
        return (now_us() - t0) / n;
    };
    run_each(); run_pre(); run_replay();  // warm-up, not recorded
    std::vector<double> a, b, c;
    for (int t = 0; t < s.submit_trials; ++t) { a.push_back(run_each()); b.push_back(run_pre()); c.push_back(run_replay()); }
    Stats sa = stats(a), sb = stats(b), sc = stats(c);
    Json j;
    j.integer("jobs", n);
    j.stats("without_record_and_submit_each_us_per_job", sa);
    j.stats("submit_each_prerecorded_us_per_job", sb);
    j.stats("with_replay_one_list_us_per_job", sc);
    j.num("gain_x", sa.median / sc.median);
    std::printf("  submit   without %.3f us/job, prerecorded %.3f, with replay %.3f  -> %.2fx\n", sa.median, sb.median, sc.median,
                sa.median / sc.median);
    return j.text();
}

// One doorbell run in the given memory. Returns its JSON; *median_out gets the median round trip (0 if it failed).
std::string doorbell_run(Vk& vk, const Settings& s, const Buffer& b, VkDescriptorSet set, VkPipeline pipe, const char* label,
                         const char* read, double* median_out) {
    *median_out = 0;
    VkCommandPool pool = vk.make_pool(vk.main_q.family);
    VkCommandBuffer c = vk.alloc(pool, 1)[0];
    VkFence fe = vk.make_fence();
    struct PC { uint32_t rounds, max_polls; } pc{static_cast<uint32_t>(s.sync_rounds), static_cast<uint32_t>(s.doorbell_max_polls)};
    vk.begin(c);
    vk.f.vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk.f.vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pipe_layout, 0, 1, &set, 0, nullptr);
    vk.f.vkCmdPushConstants(c, vk.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vk.f.vkCmdDispatch(c, 1, 1, 1);
    vk.end(c);

    volatile uint32_t* m = static_cast<volatile uint32_t*>(b.map);
    m[0] = 0; m[16] = 0; m[32] = 0;
    publish();
    vk.submit(vk.main_q, c, fe);

    bool started = false;
    double t_start = now_us();
    while (now_us() - t_start < 3e6) {
        if (m[32] == 1u) { started = true; break; }
    }
    std::vector<double> rt;
    bool cap_hit = false, cpu_timeout = false;
    if (started) {
        for (uint32_t k = 1; k <= pc.rounds; ++k) {
            double t0 = now_us();
            m[0] = k;
            publish();
            uint32_t a = 0;
            uint64_t spins = 0;
            while ((a = m[16]) != k) {
                if (a == 0xFFFFFFFFu) { cap_hit = true; break; }
                if ((++spins & 4095u) == 0 && now_us() - t0 > 1e6) { cpu_timeout = true; break; }
            }
            if (cap_hit || cpu_timeout) break;
            rt.push_back(now_us() - t0);
        }
    }
    if (!started || cpu_timeout) {
        // Stop the shader: write values it can never reach so its per-round poll cap ends it.
        m[0] = 0xFFFFFFF0u;
        publish();
    }
    VkResult wr = vk.f.vkWaitForFences(vk.dev, 1, &fe, VK_TRUE, kWaitNs);
    Json j;
    j.str("memory", label);
    j.str("read", read);
    j.str("memory_flags", flags_text(b.flags));
    j.boolean("started", started);
    j.integer("rounds_done", static_cast<long long>(rt.size()));
    j.boolean("gpu_poll_cap_hit", cap_hit);
    j.boolean("cpu_timeout", cpu_timeout);
    j.boolean("shader_finished", wr == VK_SUCCESS);
    // A variant counts only if every round completed; partial runs keep their timings but are not a result.
    const bool complete = rt.size() == static_cast<size_t>(pc.rounds);
    j.boolean("complete", complete);
    if (!rt.empty()) {
        Stats st = stats(rt);
        j.stats(complete ? "round_trip_us" : "partial_round_trip_us", st);
        if (complete) *median_out = st.median;
    }
    if (wr != VK_SUCCESS) throw std::runtime_error("doorbell shader did not finish");
    return j.text();
}

// sync: one CPU<->GPU hand-off, the usual way (submit + fence wait) against the doorbell.
std::string test_sync(Vk& vk, const Settings& s, VkDescriptorSet out_set) {
    VkCommandPool pool = vk.make_pool(vk.main_q.family);
    VkCommandBuffer c = vk.alloc(pool, 1)[0];
    VkFence fe = vk.make_fence();
    vk.begin(c);
    vk.record_tiny(c, out_set);
    vk.end(c);
    for (int i = 0; i < 20; ++i) vk.run_once(vk.main_q, c, fe);  // warm-up
    std::vector<double> fence_rt;
    for (int i = 0; i < s.sync_rounds; ++i) fence_rt.push_back(vk.run_once(vk.main_q, c, fe));
    Stats sf = stats(fence_rt);

    Json j;
    j.stats("without_fence_round_trip_us", sf);
    std::string runs = "[";
    double best = 0;
    std::string best_label;
    // P-G0 + P-G0b: three memories x three ways of reading the flag.
    struct Mem { const char* label; VkMemoryPropertyFlags need, avoid, prefer; };
    const Mem mems[] = {
        {"system_ram_cached", kHost | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, kDev, 0},
        {"system_ram_uncached", kHost, kDev | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0},
        {"vram_bar", kHost | kDev, 0, 0},
    };
    struct Read { const char* label; VkPipeline pipe; };
    const Read reads[] = {{"plain", vk.pipe_doorbell}, {"atomic", vk.pipe_door_atomic}, {"memory_model", vk.pipe_door_mm}};
    bool first = true;
    auto add = [&](const std::string& text) { runs += (first ? "" : ", ") + text; first = false; };
    for (const auto& mm : mems) {
        Buffer b;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::string mem_error;
        try {
            b = vk.make_buffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mm.need, mm.avoid, mm.prefer);
            set = vk.make_set(b);
        } catch (const std::exception& e) {
            mem_error = e.what();
        }
        for (const auto& rd : reads) {
            std::string name = std::string(mm.label) + " / " + rd.label;
            Json r;
            r.str("memory", mm.label);
            r.str("read", rd.label);
            if (!mem_error.empty() || rd.pipe == VK_NULL_HANDLE) {
                r.str("error", !mem_error.empty() ? mem_error : vk.mm_error);
                add(r.text());
                std::printf("  sync     doorbell %-34s skipped: %s\n", name.c_str(), (!mem_error.empty() ? mem_error : vk.mm_error).c_str());
                continue;
            }
            try {
                double med = 0;
                add(doorbell_run(vk, s, b, set, rd.pipe, mm.label, rd.label, &med));
                if (med > 0 && (best == 0 || med < best)) { best = med; best_label = name; }
                std::printf("  sync     doorbell %-34s %s\n", name.c_str(), med > 0 ? (std::to_string(med) + " us").c_str() : "did not complete (see rounds_done in the JSON)");
            } catch (const std::exception& e) {
                r.str("error", e.what());
                add(r.text());
                std::printf("  sync     doorbell %-34s error: %s\n", name.c_str(), e.what());
            }
        }
    }
    j.raw("with_doorbell", runs + "]");
    if (best > 0) {
        j.str("best_doorbell_memory", best_label);
        j.num("gain_x", sf.median / best);
        std::printf("  sync     without fence %.2f us, with doorbell %.2f us (%s)  -> %.2fx\n", sf.median, best, best_label.c_str(), sf.median / best);
    } else {
        std::printf("  sync     without fence %.2f us, doorbell did not complete\n", sf.median);
    }
    return j.text();
}

// upload: getting an asset from application memory into GPU memory.
std::string test_upload(Vk& vk, const Settings& s) {
    const VkDeviceSize n = static_cast<VkDeviceSize>(s.upload_mib) * 1048576;
    std::vector<uint8_t> src(n);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i * 131u + 7u);

    Buffer staging = vk.make_buffer(n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost, kDev);
    Buffer dst = vk.make_buffer(n, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kDev, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkCommandPool pool = vk.make_pool(vk.main_q.family);
    VkCommandBuffer c = vk.alloc(pool, 1)[0];
    VkFence fe = vk.make_fence();
    VkBufferCopy region{0, 0, n};
    vk.begin(c);
    vk.f.vkCmdCopyBuffer(c, staging.buf, dst.buf, 1, &region);
    vk.end(c);

    auto gbs = [&](double us) { return static_cast<double>(n) / (us * 1e-6) / 1e9; };
    auto without = [&] {
        double t0 = now_us();
        std::memcpy(staging.map, src.data(), n);
        vk.submit(vk.main_q, c, fe);
        vk.wait(fe);
        return gbs(now_us() - t0);
    };
    without();
    std::vector<double> a;
    for (int t = 0; t < s.upload_trials; ++t) a.push_back(without());
    Stats sa = stats(a);
    Json j;
    j.integer("bytes", static_cast<long long>(n));
    j.stats("without_staging_copy_gbs", sa);
    try {
        Buffer bar = vk.make_buffer(n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost | kDev, 0);
        auto with = [&] {
            double t0 = now_us();
            std::memcpy(bar.map, src.data(), n);
            publish();
            return gbs(now_us() - t0);
        };
        with();
        std::vector<double> b;
        for (int t = 0; t < s.upload_trials; ++t) b.push_back(with());
        Stats sb = stats(b);
        bool same = std::memcmp(bar.map, src.data(), 4096) == 0 &&
                    std::memcmp(static_cast<uint8_t*>(bar.map) + n - 4096, src.data() + n - 4096, 4096) == 0;
        j.stats("with_direct_bar_write_gbs", sb);
        j.num("bar_heap_mib", bar.heap_size / 1048576.0);
        j.boolean("rebar_likely", bar.heap_size > 256ull * 1048576);
        j.boolean("spot_check_ok", same);
        j.num("gain_x", sb.median / sa.median);
        std::printf("  upload   without staging %.2f GB/s, with BAR write %.2f GB/s (BAR heap %.0f MiB)  -> %.2fx\n", sa.median, sb.median,
                    bar.heap_size / 1048576.0, sb.median / sa.median);
    } catch (const std::exception& e) {
        j.str("with_error", e.what());
        std::printf("  upload   without staging %.2f GB/s, BAR write unavailable: %s\n", sa.median, e.what());
    }
    return j.text();
}

// overlap: a big copy and a compute job of about the same length, one after the other vs at the same time.
std::string test_overlap(Vk& vk, const Settings& s, VkDescriptorSet out_set) {
    Json j;
    if (vk.transfer_q.q == vk.compute_q.q) {
        j.str("error", "only one queue available for copy and compute");
        std::printf("  overlap  unavailable: one queue\n");
        return j.text();
    }
    const VkDeviceSize n = static_cast<VkDeviceSize>(s.overlap_mib) * 1048576;
    Buffer src = vk.make_buffer(n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost, kDev);
    Buffer dst = vk.make_buffer(n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, kDev, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    std::memset(src.map, 0x5A, n);
    VkCommandPool cp = vk.make_pool(vk.transfer_q.family);
    VkCommandPool kp = vk.make_pool(vk.compute_q.family, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    VkCommandBuffer copy = vk.alloc(cp, 1)[0];
    VkCommandBuffer busy = vk.alloc(kp, 1)[0];
    VkFence fc = vk.make_fence(), fk = vk.make_fence();
    VkBufferCopy region{0, 0, n};
    vk.begin(copy);
    vk.f.vkCmdCopyBuffer(copy, src.buf, dst.buf, 1, &region);
    vk.end(copy);
    const uint32_t groups = 1024;
    auto record_busy = [&](uint32_t iters) {
        vk.begin(busy);
        vk.f.vkCmdBindPipeline(busy, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pipe_busy);
        vk.f.vkCmdBindDescriptorSets(busy, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pipe_layout, 0, 1, &out_set, 0, nullptr);
        uint32_t pc[2] = {iters, 0};
        vk.f.vkCmdPushConstants(busy, vk.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        vk.f.vkCmdDispatch(busy, groups, 1, 1);
        vk.end(busy);
    };
    auto median_of = [&](auto fn) {
        fn();
        std::vector<double> v;
        for (int t = 0; t < s.upload_trials; ++t) v.push_back(fn());
        return stats(v).median;
    };
    double t_copy = median_of([&] { return vk.run_once(vk.transfer_q, copy, fc); });
    // Size the compute job to match the copy.
    double iters = 2048;
    record_busy(static_cast<uint32_t>(iters));
    double t_b = median_of([&] { return vk.run_once(vk.compute_q, busy, fk); });
    while (t_b < 1000 && iters < (1 << 26)) {
        iters *= 8;
        record_busy(static_cast<uint32_t>(iters));
        t_b = median_of([&] { return vk.run_once(vk.compute_q, busy, fk); });
    }
    iters = std::min(std::max(iters * t_copy / t_b, 1.0), static_cast<double>(1u << 30));
    record_busy(static_cast<uint32_t>(iters));
    double t_busy = median_of([&] { return vk.run_once(vk.compute_q, busy, fk); });
    double t_both = median_of([&] {
        double t0 = now_us();
        vk.submit(vk.transfer_q, copy, fc);
        vk.submit(vk.compute_q, busy, fk);
        vk.wait(fc);
        vk.wait(fk);
        return now_us() - t0;
    });
    double serial = t_copy + t_busy;
    j.integer("copy_bytes", static_cast<long long>(n));
    j.integer("busy_iters", static_cast<long long>(iters));
    j.num("copy_alone_us", t_copy);
    j.num("compute_alone_us", t_busy);
    j.num("without_one_after_other_us", serial);
    j.num("with_concurrent_us", t_both);
    j.num("gain_x", serial / t_both);
    j.num("ideal_x", serial / std::max(t_copy, t_busy));
    std::printf("  overlap  without %.0f us, with %.0f us  -> %.2fx (ideal %.2fx)\n", serial, t_both, serial / t_both, serial / std::max(t_copy, t_busy));
    return j.text();
}

}  // namespace

std::string run_vulkan(const Settings& s, int device_index) {
    Json j;
    std::printf("Vulkan\n");
    try {
        Vk vk;
        vk.load();
        vk.create_instance();
        vk.pick_device(device_index);
        vk.create_device();
        vk.create_pipelines();
        std::printf("  device   %s, driver %s, Vulkan %s\n", vk.props.deviceName, driver_text(vk.props.vendorID, vk.props.driverVersion).c_str(),
                    version_text(vk.props.apiVersion).c_str());
        j.boolean("available", true);
        j.raw("device", device_json(vk));
        Buffer out = vk.make_buffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kDev, 0);
        VkDescriptorSet out_set = vk.make_set(out);
        auto guarded = [&](const char* name, auto fn) {
            try {
                j.raw(name, fn());
            } catch (const std::exception& e) {
                Json err;
                err.str("error", e.what());
                j.raw(name, err.text());
                std::printf("  %-8s error: %s\n", name, e.what());
            }
        };
        guarded("submit", [&] { return test_submit(vk, s, out_set); });
        guarded("sync", [&] { return test_sync(vk, s, out_set); });
        guarded("upload", [&] { return test_upload(vk, s); });
        guarded("overlap", [&] { return test_overlap(vk, s, out_set); });
    } catch (const std::exception& e) {
        j.boolean("available", false);
        j.str("error", e.what());
        std::printf("  unavailable: %s\n", e.what());
    }
    return j.text();
}

}  // namespace gp
