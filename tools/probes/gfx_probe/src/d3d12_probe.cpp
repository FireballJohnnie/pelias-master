// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// d3d12_probe.cpp: the Direct3D 12 half of gfx_probe (P-G0). The same four tests as the Vulkan half:
//   submit   : record and execute every job          vs  execute one pre-recorded list of the same jobs
//   sync     : fence round trip per hand-off         vs  doorbell (a resident shader answering a CPU-written flag)
//   upload   : upload heap plus copy command         vs  direct CPU write into a GPU upload heap (VRAM through the BAR)
//   overlap  : copy, then compute                    vs  copy queue and compute queue at once
// Shaders are compiled at run time by the Windows shader compiler (d3dcompiler_47.dll, part of Windows).
// Direct3D 12 declarations come from Microsoft's DirectX-Headers (for the GPU upload heap), DXGI from the Windows SDK.
#include "probe.h"

#ifndef _WIN32

namespace gp {
std::string run_d3d12(const Settings&, int) {
    std::printf("Direct3D 12\n  unavailable: Windows only\n");
    Json j;
    j.boolean("available", false);
    j.str("error", "Direct3D 12 is Windows only");
    return j.text();
}
}  // namespace gp

#else

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <directx/d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gp {
namespace {

using Microsoft::WRL::ComPtr;

void hr_ck(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        char b[256];
        std::snprintf(b, sizeof b, "%s failed: HRESULT 0x%08lx", what, static_cast<unsigned long>(hr));
        throw std::runtime_error(b);
    }
}

std::string narrow(const wchar_t* w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

// The shaders, in HLSL. They match shaders/*.comp one for one.
const char* kHlslTiny = R"(
RWStructuredBuffer<uint> o : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) { o[g.x & 1023] = g.x + 1; }
)";

const char* kHlslDoorbell = R"(
globallycoherent RWStructuredBuffer<uint> m : register(u0);
cbuffer C : register(b0) { uint rounds; uint max_polls; };
[numthreads(1, 1, 1)]
void main() {
    m[32] = 1;
    DeviceMemoryBarrier();
    [loop] for (uint k = 1; k <= rounds; ++k) {
        uint polls = 0;
        [loop] [allow_uav_condition] while (m[0] != k) {
            polls += 1;
            if (polls > max_polls) {
                m[16] = 0xFFFFFFFF;
                DeviceMemoryBarrier();
                return;
            }
        }
        m[16] = k;
        DeviceMemoryBarrier();
    }
}
)";

// P-G0b variant: the flag is read with an atomic (add 0), which must go to the memory that holds it.
const char* kHlslDoorbellAtomic = R"(
globallycoherent RWStructuredBuffer<uint> m : register(u0);
cbuffer C : register(b0) { uint rounds; uint max_polls; };
[numthreads(1, 1, 1)]
void main() {
    m[32] = 1;
    DeviceMemoryBarrier();
    [loop] for (uint k = 1; k <= rounds; ++k) {
        uint polls = 0;
        uint v = 0;
        InterlockedAdd(m[0], 0, v);
        [loop] [allow_uav_condition] while (v != k) {
            polls += 1;
            if (polls > max_polls) {
                m[16] = 0xFFFFFFFF;
                DeviceMemoryBarrier();
                return;
            }
            InterlockedAdd(m[0], 0, v);
        }
        m[16] = k;
        DeviceMemoryBarrier();
    }
}
)";

const char* kHlslBusy = R"(
RWStructuredBuffer<float> o : register(u0);
cbuffer C : register(b0) { uint iters; uint unused; };
[numthreads(256, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    float x = float(t.x) * 1e-6 + 1.0;
    [loop] for (uint i = 0; i < iters; ++i) { x = x * 1.0000001 + 1e-7; }
    if (x == 12345.678) o[t.x & 1023] = x;
}
)";

using PFN_Compile = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR, LPCSTR, UINT, UINT,
                                     ID3DBlob**, ID3DBlob**);

constexpr double kWaitUs = 10e6;  // any GPU wait longer than 10 s is a failure

struct Fence {
    ComPtr<ID3D12Fence> f;
    UINT64 v = 0;
};

class D3D {
public:
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 desc{};
    std::string driver;
    ComPtr<ID3D12Device> dev;
    ComPtr<ID3D12CommandQueue> direct, compute, copy;
    Fence f_direct, f_compute, f_copy;
    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> pso_tiny, pso_door, pso_door_atomic, pso_busy;
    HANDLE event = nullptr;
    bool gpu_upload_heap = false;

    ~D3D() {
        if (event) CloseHandle(event);
    }

    void open(int index) {
        ComPtr<IDXGIFactory1> factory;
        hr_ck(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> a;
        SIZE_T best = 0;
        for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d{};
            a->GetDesc1(&d);
            if (index >= 0) {
                if (static_cast<UINT>(index) == i) { adapter = a; desc = d; }
                continue;
            }
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (!adapter || d.DedicatedVideoMemory > best) { adapter = a; desc = d; best = d.DedicatedVideoMemory; }
        }
        if (!adapter) throw std::runtime_error(index >= 0 ? "--adapter out of range" : "no hardware adapter");
        LARGE_INTEGER umd{};
        if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd))) {
            char b[64];
            std::snprintf(b, sizeof b, "%u.%u.%u.%u", HIWORD(umd.HighPart), LOWORD(umd.HighPart), HIWORD(umd.LowPart), LOWORD(umd.LowPart));
            driver = b;
        }
        hr_ck(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)), "D3D12CreateDevice");
        D3D12_FEATURE_DATA_D3D12_OPTIONS16 o16{};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &o16, sizeof o16))) gpu_upload_heap = o16.GPUUploadHeapSupported != FALSE;
        auto make_queue = [&](D3D12_COMMAND_LIST_TYPE type, ComPtr<ID3D12CommandQueue>& q, Fence& fe) {
            D3D12_COMMAND_QUEUE_DESC qd{};
            qd.Type = type;
            hr_ck(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)), "CreateCommandQueue");
            hr_ck(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fe.f)), "CreateFence");
        };
        make_queue(D3D12_COMMAND_LIST_TYPE_DIRECT, direct, f_direct);
        make_queue(D3D12_COMMAND_LIST_TYPE_COMPUTE, compute, f_compute);
        make_queue(D3D12_COMMAND_LIST_TYPE_COPY, copy, f_copy);
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("CreateEvent failed");
    }

    void create_pipelines() {
        D3D12_ROOT_PARAMETER p[2]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        p[0].Descriptor.ShaderRegister = 0;
        p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p[1].Constants.ShaderRegister = 0;
        p[1].Constants.Num32BitValues = 2;
        p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rd{};
        rd.NumParameters = 2;
        rd.pParameters = p;
        ComPtr<ID3DBlob> blob, err;
        HRESULT hr = D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
        if (FAILED(hr)) throw std::runtime_error(std::string("root signature: ") + (err ? static_cast<const char*>(err->GetBufferPointer()) : ""));
        hr_ck(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)), "CreateRootSignature");

        HMODULE dll = LoadLibraryW(L"d3dcompiler_47.dll");
        if (!dll) throw std::runtime_error("d3dcompiler_47.dll not found");
        auto compile_fn = reinterpret_cast<PFN_Compile>(reinterpret_cast<void*>(GetProcAddress(dll, "D3DCompile")));
        if (!compile_fn) throw std::runtime_error("D3DCompile missing");
        auto make = [&](const char* src, const char* name, ComPtr<ID3D12PipelineState>& pso) {
            ComPtr<ID3DBlob> code, msg;
            HRESULT h = compile_fn(src, std::strlen(src), name, nullptr, nullptr, "main", "cs_5_1", 1u << 15 /* optimization level 3 */, 0,
                                   &code, &msg);
            if (FAILED(h)) throw std::runtime_error(std::string("compile ") + name + ": " + (msg ? static_cast<const char*>(msg->GetBufferPointer()) : ""));
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
            pd.pRootSignature = rs.Get();
            pd.CS = {code->GetBufferPointer(), code->GetBufferSize()};
            hr_ck(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "CreateComputePipelineState");
        };
        make(kHlslTiny, "tiny", pso_tiny);
        make(kHlslDoorbell, "doorbell", pso_door);
        make(kHlslDoorbellAtomic, "doorbell_atomic", pso_door_atomic);
        make(kHlslBusy, "busy", pso_busy);
    }

    // cpu_page != UNKNOWN makes a custom heap in system RAM (L0) with that CPU page property.
    ComPtr<ID3D12Resource> buffer(D3D12_HEAP_TYPE type, UINT64 size, bool uav, D3D12_RESOURCE_STATES state, void** map = nullptr,
                                  D3D12_CPU_PAGE_PROPERTY cpu_page = D3D12_CPU_PAGE_PROPERTY_UNKNOWN) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = type;
        if (cpu_page != D3D12_CPU_PAGE_PROPERTY_UNKNOWN) {
            hp.Type = D3D12_HEAP_TYPE_CUSTOM;
            hp.CPUPageProperty = cpu_page;
            hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        }
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = size;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> r;
        hr_ck(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r)), "CreateCommittedResource");
        if (map) {
            D3D12_RANGE none{0, 0};
            hr_ck(r->Map(0, &none, map), "Map");
        }
        return r;
    }

    struct List {
        ComPtr<ID3D12CommandAllocator> alloc;
        ComPtr<ID3D12GraphicsCommandList> list;
    };
    List make_list(D3D12_COMMAND_LIST_TYPE type) {
        List l;
        hr_ck(dev->CreateCommandAllocator(type, IID_PPV_ARGS(&l.alloc)), "CreateCommandAllocator");
        hr_ck(dev->CreateCommandList(0, type, l.alloc.Get(), nullptr, IID_PPV_ARGS(&l.list)), "CreateCommandList");
        return l;
    }

    void exec(ID3D12CommandQueue* q, ID3D12GraphicsCommandList* l) {
        ID3D12CommandList* a[] = {l};
        q->ExecuteCommandLists(1, a);
    }
    UINT64 signal(ID3D12CommandQueue* q, Fence& fe) {
        hr_ck(q->Signal(fe.f.Get(), ++fe.v), "Signal");
        return fe.v;
    }
    void spin(Fence& fe, UINT64 v) {
        double t0 = now_us();
        UINT64 c;
        while ((c = fe.f->GetCompletedValue()) < v) {
            if (now_us() - t0 > kWaitUs) throw std::runtime_error("GPU wait timed out");
        }
        if (c == UINT64_MAX) {
            char b[96];
            std::snprintf(b, sizeof b, "device removed: HRESULT 0x%08lx", static_cast<unsigned long>(dev->GetDeviceRemovedReason()));
            throw std::runtime_error(b);
        }
    }
    void block(Fence& fe, UINT64 v) {
        if (fe.f->GetCompletedValue() >= v) return;
        hr_ck(fe.f->SetEventOnCompletion(v, event), "SetEventOnCompletion");
        if (WaitForSingleObject(event, static_cast<DWORD>(kWaitUs / 1000)) != WAIT_OBJECT_0) throw std::runtime_error("GPU wait timed out");
    }
    double run_once(ID3D12CommandQueue* q, Fence& fe, ID3D12GraphicsCommandList* l) {
        double t0 = now_us();
        exec(q, l);
        spin(fe, signal(q, fe));
        return now_us() - t0;
    }

    // One job = a UAV barrier plus one tiny dispatch, the same in every submit variant.
    void record_tiny(ID3D12GraphicsCommandList* l, D3D12_GPU_VIRTUAL_ADDRESS out) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.UAV.pResource = nullptr;
        l->ResourceBarrier(1, &b);
        l->SetComputeRootSignature(rs.Get());
        l->SetPipelineState(pso_tiny.Get());
        l->SetComputeRootUnorderedAccessView(0, out);
        l->Dispatch(1, 1, 1);
    }
};

std::string os_build() {
    using PFN_RtlGetVersion = LONG(WINAPI*)(OSVERSIONINFOW*);
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    auto fn = nt ? reinterpret_cast<PFN_RtlGetVersion>(reinterpret_cast<void*>(GetProcAddress(nt, "RtlGetVersion"))) : nullptr;
    OSVERSIONINFOW v{};
    v.dwOSVersionInfoSize = sizeof v;
    if (!fn || fn(&v) != 0) return "unknown";
    return std::to_string(v.dwMajorVersion) + "." + std::to_string(v.dwMinorVersion) + "." + std::to_string(v.dwBuildNumber);
}

std::string device_json(D3D& d) {
    Json j;
    j.str("name", narrow(d.desc.Description));
    j.str("driver", d.driver);
    j.integer("vendor_id", d.desc.VendorId);
    j.integer("device_id", d.desc.DeviceId);
    j.num("dedicated_vram_mib", d.desc.DedicatedVideoMemory / 1048576.0);
    j.num("shared_system_mib", d.desc.SharedSystemMemory / 1048576.0);
    j.boolean("gpu_upload_heap_supported", d.gpu_upload_heap);
    j.str("windows_build", os_build());
    return j.text();
}

std::string test_submit(D3D& d, const Settings& s, D3D12_GPU_VIRTUAL_ADDRESS out) {
    const int n = s.submit_jobs;
    const auto T = D3D12_COMMAND_LIST_TYPE_DIRECT;
    std::vector<ComPtr<ID3D12CommandAllocator>> each_alloc(n);
    for (auto& a : each_alloc) hr_ck(d.dev->CreateCommandAllocator(T, IID_PPV_ARGS(&a)), "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList> each_list;
    hr_ck(d.dev->CreateCommandList(0, T, each_alloc[0].Get(), nullptr, IID_PPV_ARGS(&each_list)), "CreateCommandList");
    each_list->Close();
    std::vector<D3D::List> pre;
    for (int i = 0; i < n; ++i) {
        pre.push_back(d.make_list(T));
        d.record_tiny(pre.back().list.Get(), out);
        hr_ck(pre.back().list->Close(), "Close");
    }
    D3D::List replay = d.make_list(T);
    for (int i = 0; i < n; ++i) d.record_tiny(replay.list.Get(), out);
    hr_ck(replay.list->Close(), "Close");

    auto run_each = [&] {
        for (auto& a : each_alloc) hr_ck(a->Reset(), "allocator Reset");
        double t0 = now_us();
        for (int i = 0; i < n; ++i) {
            hr_ck(each_list->Reset(each_alloc[i].Get(), nullptr), "list Reset");
            d.record_tiny(each_list.Get(), out);
            hr_ck(each_list->Close(), "Close");
            d.exec(d.direct.Get(), each_list.Get());
        }
        d.spin(d.f_direct, d.signal(d.direct.Get(), d.f_direct));
        return (now_us() - t0) / n;
    };
    auto run_pre = [&] {
        double t0 = now_us();
        for (auto& l : pre) d.exec(d.direct.Get(), l.list.Get());
        d.spin(d.f_direct, d.signal(d.direct.Get(), d.f_direct));
        return (now_us() - t0) / n;
    };
    auto run_replay = [&] { return d.run_once(d.direct.Get(), d.f_direct, replay.list.Get()) / n; };
    run_each(); run_pre(); run_replay();  // warm-up, not recorded
    std::vector<double> a, b, c;
    for (int t = 0; t < s.submit_trials; ++t) { a.push_back(run_each()); b.push_back(run_pre()); c.push_back(run_replay()); }
    Stats sa = stats(a), sb = stats(b), sc = stats(c);
    Json j;
    j.integer("jobs", n);
    j.stats("without_record_and_execute_each_us_per_job", sa);
    j.stats("execute_each_prerecorded_us_per_job", sb);
    j.stats("with_replay_one_list_us_per_job", sc);
    j.num("gain_x", sa.median / sc.median);
    std::printf("  submit   without %.3f us/job, prerecorded %.3f, with replay %.3f  -> %.2fx\n", sa.median, sb.median, sc.median,
                sa.median / sc.median);
    return j.text();
}

std::string doorbell_run(D3D& d, const Settings& s, ID3D12Resource* res, void* map, ID3D12PipelineState* pso, const char* label,
                         const char* read, double* median_out) {
    *median_out = 0;
    D3D::List l = d.make_list(D3D12_COMMAND_LIST_TYPE_DIRECT);
    UINT pc[2] = {static_cast<UINT>(s.sync_rounds), static_cast<UINT>(s.doorbell_max_polls)};
    l.list->SetComputeRootSignature(d.rs.Get());
    l.list->SetPipelineState(pso);
    l.list->SetComputeRootUnorderedAccessView(0, res->GetGPUVirtualAddress());
    l.list->SetComputeRoot32BitConstants(1, 2, pc, 0);
    l.list->Dispatch(1, 1, 1);
    hr_ck(l.list->Close(), "Close");

    volatile uint32_t* m = static_cast<volatile uint32_t*>(map);
    m[0] = 0; m[16] = 0; m[32] = 0;
    publish();
    d.exec(d.direct.Get(), l.list.Get());
    UINT64 done = d.signal(d.direct.Get(), d.f_direct);

    bool started = false;
    double t_start = now_us();
    while (now_us() - t_start < 3e6) {
        if (m[32] == 1u) { started = true; break; }
    }
    std::vector<double> rt;
    bool cap_hit = false, cpu_timeout = false;
    if (started) {
        for (UINT k = 1; k <= pc[0]; ++k) {
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
        m[0] = 0xFFFFFFF0u;  // a value the shader never waits for, so its per-round poll cap ends it
        publish();
    }
    d.spin(d.f_direct, done);
    Json j;
    j.str("memory", label);
    j.str("read", read);
    j.boolean("started", started);
    j.integer("rounds_done", static_cast<long long>(rt.size()));
    j.boolean("gpu_poll_cap_hit", cap_hit);
    j.boolean("cpu_timeout", cpu_timeout);
    // A variant counts only if every round completed; partial runs keep their timings but are not a result.
    const bool complete = rt.size() == static_cast<size_t>(pc[0]);
    j.boolean("complete", complete);
    if (!rt.empty()) {
        Stats st = stats(rt);
        j.stats(complete ? "round_trip_us" : "partial_round_trip_us", st);
        if (complete) *median_out = st.median;
    }
    return j.text();
}

std::string test_sync(D3D& d, const Settings& s, D3D12_GPU_VIRTUAL_ADDRESS out) {
    D3D::List l = d.make_list(D3D12_COMMAND_LIST_TYPE_DIRECT);
    d.record_tiny(l.list.Get(), out);
    hr_ck(l.list->Close(), "Close");
    for (int i = 0; i < 20; ++i) d.run_once(d.direct.Get(), d.f_direct, l.list.Get());  // warm-up
    std::vector<double> spin_rt, event_rt;
    for (int i = 0; i < s.sync_rounds; ++i) spin_rt.push_back(d.run_once(d.direct.Get(), d.f_direct, l.list.Get()));
    for (int i = 0; i < s.sync_rounds; ++i) {
        double t0 = now_us();
        d.exec(d.direct.Get(), l.list.Get());
        d.block(d.f_direct, d.signal(d.direct.Get(), d.f_direct));
        event_rt.push_back(now_us() - t0);
    }
    Stats ss = stats(spin_rt), se = stats(event_rt);
    double without = std::min(ss.median, se.median);  // compare against the faster usual way
    Json j;
    j.stats("without_fence_spin_round_trip_us", ss);
    j.stats("without_fence_event_round_trip_us", se);
    std::string runs = "[";
    double best = 0;
    std::string best_label;
    // P-G0 + P-G0b: three memories x two ways of reading the flag.
    struct Mem { const char* label; D3D12_HEAP_TYPE type; D3D12_CPU_PAGE_PROPERTY page; };
    const Mem mems[] = {
        {"system_ram_writeback", D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK},
        {"system_ram_writecombine", D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE},
        {"vram_gpu_upload_heap", D3D12_HEAP_TYPE_GPU_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN},
    };
    struct Read { const char* label; ID3D12PipelineState* pso; };
    const Read reads[] = {{"plain", d.pso_door.Get()}, {"atomic", d.pso_door_atomic.Get()}};
    bool first = true;
    auto add = [&](const std::string& text) { runs += (first ? "" : ", ") + text; first = false; };
    for (const auto& mm : mems) {
        ComPtr<ID3D12Resource> res;
        void* map = nullptr;
        std::string mem_error;
        try {
            if (mm.type == D3D12_HEAP_TYPE_GPU_UPLOAD && !d.gpu_upload_heap)
                throw std::runtime_error("GPU upload heap not supported (needs Resizable BAR on and Windows 11 24H2 or later)");
            res = d.buffer(mm.type, 4096, true, D3D12_RESOURCE_STATE_COMMON, &map, mm.page);
        } catch (const std::exception& e) {
            mem_error = e.what();
        }
        for (const auto& rd : reads) {
            std::string name = std::string(mm.label) + " / " + rd.label;
            Json r;
            r.str("memory", mm.label);
            r.str("read", rd.label);
            if (!mem_error.empty()) {
                r.str("error", mem_error);
                add(r.text());
                std::printf("  sync     doorbell %-38s skipped: %s\n", name.c_str(), mem_error.c_str());
                continue;
            }
            try {
                double med = 0;
                add(doorbell_run(d, s, res.Get(), map, rd.pso, mm.label, rd.label, &med));
                if (med > 0 && (best == 0 || med < best)) { best = med; best_label = name; }
                std::printf("  sync     doorbell %-38s %s\n", name.c_str(), med > 0 ? (std::to_string(med) + " us").c_str() : "did not complete (see rounds_done in the JSON)");
            } catch (const std::exception& e) {
                r.str("error", e.what());
                add(r.text());
                std::printf("  sync     doorbell %-38s error: %s\n", name.c_str(), e.what());
            }
        }
    }
    j.raw("with_doorbell", runs + "]");
    if (best > 0) {
        j.str("best_doorbell_memory", best_label);
        j.num("gain_x", without / best);
        std::printf("  sync     without fence %.2f us (spin %.2f, event %.2f), with doorbell %.2f us (%s)  -> %.2fx\n", without, ss.median,
                    se.median, best, best_label.c_str(), without / best);
    } else {
        std::printf("  sync     without fence %.2f us, doorbell did not complete\n", without);
    }
    return j.text();
}

std::string test_upload(D3D& d, const Settings& s) {
    const UINT64 n = static_cast<UINT64>(s.upload_mib) * 1048576;
    std::vector<uint8_t> src(n);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i * 131u + 7u);
    void* up = nullptr;
    auto upload = d.buffer(D3D12_HEAP_TYPE_UPLOAD, n, false, D3D12_RESOURCE_STATE_GENERIC_READ, &up);
    auto dst = d.buffer(D3D12_HEAP_TYPE_DEFAULT, n, false, D3D12_RESOURCE_STATE_COMMON);
    D3D::List l = d.make_list(D3D12_COMMAND_LIST_TYPE_DIRECT);
    l.list->CopyBufferRegion(dst.Get(), 0, upload.Get(), 0, n);
    hr_ck(l.list->Close(), "Close");
    auto gbs = [&](double us) { return static_cast<double>(n) / (us * 1e-6) / 1e9; };
    auto without = [&] {
        double t0 = now_us();
        std::memcpy(up, src.data(), n);
        d.exec(d.direct.Get(), l.list.Get());
        d.spin(d.f_direct, d.signal(d.direct.Get(), d.f_direct));
        return gbs(now_us() - t0);
    };
    without();
    std::vector<double> a;
    for (int t = 0; t < s.upload_trials; ++t) a.push_back(without());
    Stats sa = stats(a);
    Json j;
    j.integer("bytes", static_cast<long long>(n));
    j.stats("without_upload_heap_copy_gbs", sa);
    try {
        if (!d.gpu_upload_heap) throw std::runtime_error("GPU upload heap not supported (needs Resizable BAR on and Windows 11 24H2 or later)");
        void* gm = nullptr;
        auto gpu_up = d.buffer(D3D12_HEAP_TYPE_GPU_UPLOAD, n, false, D3D12_RESOURCE_STATE_COMMON, &gm);
        auto with = [&] {
            double t0 = now_us();
            std::memcpy(gm, src.data(), n);
            publish();
            return gbs(now_us() - t0);
        };
        with();
        std::vector<double> b;
        for (int t = 0; t < s.upload_trials; ++t) b.push_back(with());
        Stats sb = stats(b);
        bool same = std::memcmp(gm, src.data(), 4096) == 0 && std::memcmp(static_cast<uint8_t*>(gm) + n - 4096, src.data() + n - 4096, 4096) == 0;
        j.stats("with_gpu_upload_heap_write_gbs", sb);
        j.boolean("spot_check_ok", same);
        j.num("gain_x", sb.median / sa.median);
        std::printf("  upload   without upload heap %.2f GB/s, with GPU upload heap %.2f GB/s  -> %.2fx\n", sa.median, sb.median, sb.median / sa.median);
    } catch (const std::exception& e) {
        j.str("with_error", e.what());
        std::printf("  upload   without upload heap %.2f GB/s, GPU upload heap unavailable: %s\n", sa.median, e.what());
    }
    return j.text();
}

std::string test_overlap(D3D& d, const Settings& s, D3D12_GPU_VIRTUAL_ADDRESS out) {
    const UINT64 n = static_cast<UINT64>(s.overlap_mib) * 1048576;
    void* up = nullptr;
    auto src = d.buffer(D3D12_HEAP_TYPE_UPLOAD, n, false, D3D12_RESOURCE_STATE_GENERIC_READ, &up);
    std::memset(up, 0x5A, n);
    auto dst = d.buffer(D3D12_HEAP_TYPE_DEFAULT, n, false, D3D12_RESOURCE_STATE_COMMON);
    D3D::List cl = d.make_list(D3D12_COMMAND_LIST_TYPE_COPY);
    cl.list->CopyBufferRegion(dst.Get(), 0, src.Get(), 0, n);
    hr_ck(cl.list->Close(), "Close");
    D3D::List kl = d.make_list(D3D12_COMMAND_LIST_TYPE_COMPUTE);
    hr_ck(kl.list->Close(), "Close");
    const UINT groups = 1024;
    auto record_busy = [&](UINT iters) {
        hr_ck(kl.alloc->Reset(), "allocator Reset");
        hr_ck(kl.list->Reset(kl.alloc.Get(), nullptr), "list Reset");
        UINT pc[2] = {iters, 0};
        kl.list->SetComputeRootSignature(d.rs.Get());
        kl.list->SetPipelineState(d.pso_busy.Get());
        kl.list->SetComputeRootUnorderedAccessView(0, out);
        kl.list->SetComputeRoot32BitConstants(1, 2, pc, 0);
        kl.list->Dispatch(groups, 1, 1);
        hr_ck(kl.list->Close(), "Close");
    };
    auto median_of = [&](auto fn) {
        fn();
        std::vector<double> v;
        for (int t = 0; t < s.upload_trials; ++t) v.push_back(fn());
        return stats(v).median;
    };
    double t_copy = median_of([&] { return d.run_once(d.copy.Get(), d.f_copy, cl.list.Get()); });
    double iters = 2048;
    record_busy(static_cast<UINT>(iters));
    double t_b = median_of([&] { return d.run_once(d.compute.Get(), d.f_compute, kl.list.Get()); });
    while (t_b < 1000 && iters < (1 << 26)) {
        iters *= 8;
        record_busy(static_cast<UINT>(iters));
        t_b = median_of([&] { return d.run_once(d.compute.Get(), d.f_compute, kl.list.Get()); });
    }
    iters = std::min(std::max(iters * t_copy / t_b, 1.0), static_cast<double>(1u << 30));
    record_busy(static_cast<UINT>(iters));
    double t_busy = median_of([&] { return d.run_once(d.compute.Get(), d.f_compute, kl.list.Get()); });
    double t_both = median_of([&] {
        double t0 = now_us();
        d.exec(d.copy.Get(), cl.list.Get());
        UINT64 vc = d.signal(d.copy.Get(), d.f_copy);
        d.exec(d.compute.Get(), kl.list.Get());
        UINT64 vk = d.signal(d.compute.Get(), d.f_compute);
        d.spin(d.f_copy, vc);
        d.spin(d.f_compute, vk);
        return now_us() - t0;
    });
    double serial = t_copy + t_busy;
    Json j;
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

std::string run_d3d12(const Settings& s, int adapter_index) {
    Json j;
    std::printf("Direct3D 12\n");
    try {
        D3D d;
        d.open(adapter_index);
        d.create_pipelines();
        std::printf("  device   %s, driver %s, GPU upload heap %s, Windows %s\n", narrow(d.desc.Description).c_str(), d.driver.c_str(),
                    d.gpu_upload_heap ? "yes" : "no", os_build().c_str());
        j.boolean("available", true);
        j.raw("device", device_json(d));
        auto out = d.buffer(D3D12_HEAP_TYPE_DEFAULT, 4096, true, D3D12_RESOURCE_STATE_COMMON);
        D3D12_GPU_VIRTUAL_ADDRESS va = out->GetGPUVirtualAddress();
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
        guarded("submit", [&] { return test_submit(d, s, va); });
        guarded("sync", [&] { return test_sync(d, s, va); });
        guarded("upload", [&] { return test_upload(d, s); });
        guarded("overlap", [&] { return test_overlap(d, s, va); });
    } catch (const std::exception& e) {
        j.boolean("available", false);
        j.str("error", e.what());
        std::printf("  unavailable: %s\n", e.what());
    }
    return j.text();
}

}  // namespace gp

#endif  // _WIN32
