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

namespace pelias::hal {

// Linux: madvise(MADV_NOHUGEPAGE) on [p, p + bytes) (page-aligned by the caller). Windows and other systems: nothing to
// do, returns true. Returns false if the OS refused the advice (the caller may still register the range).
bool advise_small_pages(void* p, std::size_t bytes);

// v43 (author 02:52 "lets pin and go"): bind the calling thread to one logical CPU (Linux: pthread_setaffinity_np). Kept in
// this file because it is one of the two places allowed OS headers (BC6). Windows / other systems: nothing, returns false.
bool pin_this_thread(int cpu);

}  // namespace pelias::hal
