// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "os/membw.h"
#include "pelias_test.h"

TEST(membw_small_sweep_reports_every_cell) {
  pelias::os::MemBwOptions opt;
  opt.bytes = std::size_t{64} << 20;
  opt.threads = {1, 2};
  opt.reps = 1;
  opt.passes = 1;
  pelias::os::MemBwResult r;
  std::string err;
  CHECK(pelias::os::membw_run(opt, r, err));
  CHECK_EQ(r.cells.size(), 8u);  // 2 thread counts x 2 page modes x prefetch off / on
  for (const auto& c : r.cells) CHECK(c.gbps > 0.0);
  CHECK(r.cells[0].work_ratio > 0.0);  // the plain cells carry the foreground numbers
  CHECK(r.idle_work_us > 0.0);
  CHECK(r.logical_cpus >= 1 && r.physical_cores >= 1);
}
