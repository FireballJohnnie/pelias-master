// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// common.h: timing, statistics and a tiny JSON writer shared by the Vulkan and Direct3D 12 halves of gfx_probe.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace gp {

using Clock = std::chrono::steady_clock;

inline double now_us() {
    return std::chrono::duration<double, std::micro>(Clock::now().time_since_epoch()).count();
}

struct Stats {
    int n = 0;
    double min = 0, median = 0, p99 = 0, max = 0, mean = 0;
};

inline Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = static_cast<int>(v.size());
    s.min = v.front();
    s.max = v.back();
    s.median = v[v.size() / 2];
    s.p99 = v[std::min(v.size() - 1, static_cast<size_t>(v.size() * 0.99))];
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    return s;
}

// Minimal JSON builder: an object is a list of "key": value strings joined at the end.
class Json {
public:
    void num(const std::string& k, double v) {
        char b[64];
        std::snprintf(b, sizeof b, "%.6g", v);
        add(k, b);
    }
    void integer(const std::string& k, long long v) { add(k, std::to_string(v)); }
    void boolean(const std::string& k, bool v) { add(k, v ? "true" : "false"); }
    void str(const std::string& k, const std::string& v) { add(k, quote(v)); }
    void raw(const std::string& k, const std::string& json) { add(k, json); }
    void stats(const std::string& k, const Stats& s) {
        Json j;
        j.integer("n", s.n);
        j.num("min", s.min);
        j.num("median", s.median);
        j.num("p99", s.p99);
        j.num("max", s.max);
        j.num("mean", s.mean);
        add(k, j.text());
    }
    std::string text() const {
        std::string out = "{";
        for (size_t i = 0; i < items_.size(); ++i) out += (i ? ", " : "") + items_[i];
        return out + "}";
    }
    static std::string quote(const std::string& v) {
        std::string o = "\"";
        for (char c : v) {
            if (c == '"' || c == '\\') { o += '\\'; o += c; }
            else if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
            else o += c;
        }
        return o + "\"";
    }

private:
    void add(const std::string& k, const std::string& v) { items_.push_back(quote(k) + ": " + v); }
    std::vector<std::string> items_;
};

inline void log(const char* fmt, const std::string& a = "") { std::printf(fmt, a.c_str()); std::fflush(stdout); }

// Make a CPU store to GPU-visible memory leave the CPU now: a full fence, plus SFENCE on x86 so write-combined
// (uncached) mappings flush their buffers too.
inline void publish() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
#if defined(_M_X64) || defined(__x86_64__)
    _mm_sfence();
#endif
}

// Settings shared by both halves.
struct Settings {
    int submit_jobs = 1000;        // jobs in the submit test
    int submit_trials = 5;         // repeats of the submit test
    int sync_rounds = 1000;        // CPU<->GPU round trips in the sync test
    int doorbell_max_polls = 300000;  // safety cap on GPU polls (keeps the polling shader well under the 2 s display timeout)
    int upload_mib = 64;           // upload test size
    int upload_trials = 5;
    int overlap_mib = 256;         // copy size in the overlap test
};

}  // namespace gp
