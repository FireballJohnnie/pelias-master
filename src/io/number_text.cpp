// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/number_text.h"

#include <string>

namespace pelias::io {
namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }

Status refused(std::string_view text, const char* why) {
  return Status::fail(Code::invalid_argument, "'" + std::string(text) + "' " + why);
}

}  // namespace

Result<double> parse_positive_decimal(std::string_view text, double limit) {
  // digits [ '.' digits ]: read by hand so no locale, sign, exponent or special value can slip through
  std::size_t i = 0;
  double whole = 0.0;
  std::size_t int_digits = 0;
  while (i < text.size() && is_digit(text[i])) {
    whole = whole * 10.0 + static_cast<double>(text[i] - '0');
    ++i;
    ++int_digits;
    if (whole >= limit) return Result<double>::fail(refused(text, "is too large"));
  }
  double frac = 0.0, scale = 1.0;
  std::size_t frac_digits = 0;
  if (i < text.size() && text[i] == '.') {
    ++i;
    while (i < text.size() && is_digit(text[i])) {
      scale /= 10.0;
      frac += static_cast<double>(text[i] - '0') * scale;
      ++i;
      ++frac_digits;
    }
    if (frac_digits == 0) return Result<double>::fail(refused(text, "is not a number (digits expected after the point)"));
  }
  if (int_digits == 0 && frac_digits == 0) return Result<double>::fail(refused(text, "is not a number"));
  if (i != text.size()) return Result<double>::fail(refused(text, "is not a plain positive number"));
  const double v = whole + frac;
  if (!(v > 0.0)) return Result<double>::fail(refused(text, "must be greater than zero"));
  if (v >= limit) return Result<double>::fail(refused(text, "is too large"));
  return Result<double>::good(v);
}

Result<std::uint64_t> parse_whole(std::string_view text, std::uint64_t max) {
  if (text.empty()) return Result<std::uint64_t>::fail(refused(text, "is not a whole number"));
  std::uint64_t v = 0;
  for (const char c : text) {
    if (!is_digit(c)) return Result<std::uint64_t>::fail(refused(text, "is not a whole number"));
    const auto d = static_cast<std::uint64_t>(c - '0');
    if (d > max || v > (max - d) / 10) return Result<std::uint64_t>::fail(refused(text, "is too large"));
    v = v * 10 + d;
  }
  return Result<std::uint64_t>::good(v);
}

}  // namespace pelias::io
