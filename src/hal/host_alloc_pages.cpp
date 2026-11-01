// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// host_alloc_pages.cpp - see host_alloc_pages.h. The only OS call here is madvise.
#include "hal/host_alloc_pages.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

namespace pelias::hal {

bool advise_small_pages(void* p, std::size_t bytes) {
#if defined(__linux__) && defined(MADV_NOHUGEPAGE)
  return madvise(p, bytes, MADV_NOHUGEPAGE) == 0;
#else
  (void)p;
  (void)bytes;
  return true;
#endif
}

bool advise_huge_pages(void* p, std::size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  return madvise(p, bytes, MADV_HUGEPAGE) == 0;
#else
  (void)p;
  (void)bytes;
  return true;
#endif
}

bool pin_this_thread(int cpu) {
#if defined(__linux__)
  if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<std::size_t>(cpu), &set);
  return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#else
  (void)cpu;
  return false;
#endif
}

}  // namespace pelias::hal
