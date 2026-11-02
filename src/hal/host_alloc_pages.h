// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// host_alloc_pages.h - page-size advice for host memory that the GPU registers (v34, 2026-11-01). MEASURED 2026-11-01
// (v33b / v33c): with the model file in Linux's page cache as 2 MB folios (kernel 6.18 ext4 large folios; FileHugePages
// 21 GB), registering file ranges for the GPU mapped and split the same 2 MB page-table entries over and over
// (thp_file_mapped 54,251,448 vs thp_split_pmd 54,247,034) and the pin crawled for minutes. Asking for small pages on
// exactly the registered ranges keeps those mapped with 4 KB entries; the CPU's ranges keep their 2 MB mapping.
#pragma once

#include <cstddef>
#include <string>

namespace pelias::hal {

// Linux: madvise(MADV_NOHUGEPAGE) on [p, p + bytes) (page-aligned by the caller). Windows and other systems: nothing to
// do, returns true. Returns false if the OS refused the advice (the caller may still register the range).
bool advise_small_pages(void* p, std::size_t bytes);
// v45b (page-probe): madvise(MADV_HUGEPAGE) on [p, p + bytes). Elsewhere: nothing, returns true.
bool advise_huge_pages(void* p, std::size_t bytes);

// v43 (author 02:52 "lets pin and go"): bind the calling thread to one logical CPU (Linux: pthread_setaffinity_np). Kept in
// this file because it is one of the two places allowed OS headers (BC6). v58: Windows SetThreadAffinityMask (CPUs 0-63);
// other systems: nothing, returns false.
bool pin_this_thread(int cpu);
// v58 (author 15:09-15:11: Windows sees each core's threads): one line per physical core, "core N: CPUs a b" (Windows:
// GetLogicalProcessorInformationEx; Linux: /sys/devices/system/cpu/cpu*/topology/core_id, as the kernel reports it, which
// under WSL may not be the real pairing); empty if unknown. For choosing --pin-host / --pin-cpus.
std::string cpu_core_map();

// v46 arena probe (author 19:34 "probe it"; the author's WSL RAM dossiers, workaround 3A): an anonymous mapping of `bytes`
// (a multiple of 2 MB), asked for 2 MB pages, filled from the file with direct reads (O_DIRECT: the page cache is skipped,
// so the bytes are in RAM once), for the GPU to register. Linux only; elsewhere returns nullptr with *why set.
void* huge_arena_from_file(const char* path, unsigned long long offset, std::size_t bytes, bool* direct_used, std::string* why);
void huge_arena_free(void* p, std::size_t bytes);
// a line of /proc/self/smaps_rollup starting with key (e.g. "AnonHugePages:"), or "" when not available
std::string smaps_rollup_line(const char* key);

}  // namespace pelias::hal
