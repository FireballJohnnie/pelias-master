// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// number_text.h - strict reading of numbers typed on the command line. The whole text must be the number: no sign,
// no trailing letters, no blank text. A refused text says why, so the program can stop with exit 1 instead of guessing.
#pragma once

#include <cstdint>
#include <string_view>

#include "pelias/status.h"

namespace pelias::io {

// A decimal number greater than zero and below `limit`, written as digits with an optional fraction ("7.5", "10",
// "0.25"). Refuses signs, exponents, "inf", "nan", hexadecimal and anything after the number.
Result<double> parse_positive_decimal(std::string_view text, double limit);

// A whole number from 0 up to and including `max` ("0", "3"). Refuses signs, fractions and anything after the digits.
Result<std::uint64_t> parse_whole(std::string_view text, std::uint64_t max);

}  // namespace pelias::io
