// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "hal/nvml.h"
#include "pelias_test.h"

using pelias::hal::NvmlPick;
using pelias::hal::nvml_pick;

TEST(nvml_pick_never_guesses_a_card) {
  CHECK(nvml_pick(true, 2) == NvmlPick::by_pci);
  CHECK(nvml_pick(true, 0) == NvmlPick::by_pci);
  CHECK(nvml_pick(false, 1) == NvmlPick::only_device);
  CHECK(nvml_pick(false, 2) == NvmlPick::none);  // two cards, no PCI match: do not read card 0 for card 1
  CHECK(nvml_pick(false, 0) == NvmlPick::none);
}
