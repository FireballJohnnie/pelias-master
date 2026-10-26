// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// main.cpp: gfx_probe (Pelias probe P-G0). Runs the Vulkan and Direct3D 12 halves and writes one JSON file.
//   gfx_probe [--only vulkan|d3d12] [--out FILE] [--jobs N] [--trials N] [--rounds N] [--max-polls N]
//             [--upload-mib N] [--overlap-mib N] [--vk-device N] [--adapter N]
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>

#include "probe.h"

namespace {

std::string utc_stamp(const char* fmt) {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char b[64];
    std::strftime(b, sizeof b, fmt, &tm);
    return b;
}

int usage() {
    std::printf(
        "gfx_probe: Direct3D 12 and Vulkan, each bus building block without and with.\n"
        "  --only vulkan|d3d12   run one half only\n"
        "  --out FILE            JSON results (default results/gfx_probe_<UTC time>.json)\n"
        "  --jobs N              jobs in the submit test (1000)\n"
        "  --trials N            repeats of the submit, upload and overlap tests (5)\n"
        "  --rounds N            CPU<->GPU round trips in the sync test (1000)\n"
        "  --max-polls N         doorbell poll cap per round (300000)\n"
        "  --upload-mib N        upload size (64)\n"
        "  --overlap-mib N       copy size in the overlap test (256)\n"
        "  --vk-device N         Vulkan device index (default: first discrete GPU)\n"
        "  --adapter N           DXGI adapter index (default: largest dedicated memory)\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    gp::Settings s;
    std::string only, out;
    int vk_device = -1, adapter = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::printf("missing value after %s\n", a.c_str()); std::exit(usage()); }
            return argv[++i];
        };
        auto next_int = [&](int lo) {
            std::string v = next();
            int n = std::atoi(v.c_str());
            if (n < lo) { std::printf("bad value for %s: %s\n", a.c_str(), v.c_str()); std::exit(usage()); }
            return n;
        };
        if (a == "--only") only = next();
        else if (a == "--out") out = next();
        else if (a == "--jobs") s.submit_jobs = next_int(1);
        else if (a == "--trials") s.submit_trials = s.upload_trials = next_int(1);
        else if (a == "--rounds") s.sync_rounds = next_int(1);
        else if (a == "--max-polls") s.doorbell_max_polls = next_int(1);
        else if (a == "--upload-mib") s.upload_mib = next_int(1);
        else if (a == "--overlap-mib") s.overlap_mib = next_int(1);
        else if (a == "--vk-device") vk_device = next_int(0);
        else if (a == "--adapter") adapter = next_int(0);
        else return usage();
    }
    if (!only.empty() && only != "vulkan" && only != "d3d12") return usage();
    if (out.empty()) out = "results/gfx_probe_" + utc_stamp("%Y%m%d_%H%M%SZ") + ".json";

    std::printf("gfx_probe 0.2.1 (Pelias P-G0 + P-G0b), started %s\n\n", utc_stamp("%Y-%m-%dT%H:%M:%SZ").c_str());
    gp::Json j;
    j.str("tool", "gfx_probe");
    j.str("version", "0.2.1");
    j.str("probe", "P-G0 + P-G0b");
    j.str("started_utc", utc_stamp("%Y-%m-%dT%H:%M:%SZ"));
#ifdef _WIN32
    j.str("os", "windows");
#else
    j.str("os", "linux");
#endif
    gp::Json sj;
    sj.integer("submit_jobs", s.submit_jobs);
    sj.integer("trials", s.submit_trials);
    sj.integer("sync_rounds", s.sync_rounds);
    sj.integer("doorbell_max_polls_per_round", s.doorbell_max_polls);
    sj.integer("upload_mib", s.upload_mib);
    sj.integer("overlap_mib", s.overlap_mib);
    j.raw("settings", sj.text());
    if (only.empty() || only == "vulkan") {
        j.raw("vulkan", gp::run_vulkan(s, vk_device));
        std::printf("\n");
    }
    if (only.empty() || only == "d3d12") {
        j.raw("d3d12", gp::run_d3d12(s, adapter));
        std::printf("\n");
    }
    j.str("finished_utc", utc_stamp("%Y-%m-%dT%H:%M:%SZ"));

    std::filesystem::path p(out);
    std::error_code ec;
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
    std::ofstream f(out, std::ios::binary);
    f << j.text() << "\n";
    if (!f) {
        std::printf("could not write %s\n", out.c_str());
        return 1;
    }
    std::printf("results written to %s\n", out.c_str());
    return 0;
}
