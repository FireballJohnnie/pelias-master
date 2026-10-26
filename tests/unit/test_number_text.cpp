// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/number_text.h"
#include "pelias_test.h"

using namespace pelias::io;

TEST(number_text_accepts_plain_decimals) {
  CHECK(parse_positive_decimal("7.5", 1e9).value == 7.5);
  CHECK(parse_positive_decimal("10", 1e9).value == 10.0);
  CHECK(parse_positive_decimal("0.25", 1e9).value == 0.25);
  CHECK(parse_positive_decimal(".5", 1e9).value == 0.5);
  CHECK_EQ(parse_whole("0", 100).value, 0u);
  CHECK_EQ(parse_whole("42", 100).value, 42u);
  CHECK_EQ(parse_whole("100", 100).value, 100u);
}

TEST(number_text_refuses_bad_text) {
  const char* bad_dec[] = {"", "abc", "-1", "+1", "1e3", "inf", "nan", "0x10", "7.5GiB", "1.", ".", "0", "0.0", " 1", "1 "};
  for (const char* b : bad_dec) {
    const auto r = parse_positive_decimal(b, 1e9);
    CHECK(!r.ok());
    if (r.ok()) std::printf("      accepted '%s'\n", b);
    else CHECK(r.status.code == pelias::Code::invalid_argument);
  }
  CHECK(!parse_positive_decimal("17179869184", 17179869184.0).ok());  // at the limit
  CHECK(parse_positive_decimal("17179869183", 17179869184.0).ok());
  const char* bad_whole[] = {"", "-1", "+1", "1.0", "abc", "1x", " 1"};
  for (const char* b : bad_whole) CHECK(!parse_whole(b, 100).ok());
  CHECK(!parse_whole("101", 100).ok());
  CHECK(!parse_whole("99999999999999999999999", 18446744073709551615ull).ok());
  CHECK_EQ(parse_whole("18446744073709551615", 18446744073709551615ull).value, 18446744073709551615ull);
  CHECK(!parse_whole("5", 3).ok());
}
