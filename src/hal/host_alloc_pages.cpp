// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// host_alloc_pages.cpp - see host_alloc_pages.h. The only OS call here is madvise.
#include "hal/host_alloc_pages.h"

#if defined(__linux__)
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

}  // namespace pelias::hal
