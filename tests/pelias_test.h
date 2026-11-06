// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// pelias_test.h - the project's own test harness (rule 1: no third-party test framework). A test is a function
// registered by TEST(name); CHECK records a failure with file and line and continues; the runner prints a summary and
// returns 1 on any failure. --data DIR names the test data directory; tests that need a file absent there report "skipped".
// v60 F2: --only SUBSTR runs the cases whose name contains SUBSTR (one case under compute-sanitizer, say); others are listed as
// "not selected" so the summary still accounts for every registered case.
#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace ptest {

struct Case {
  const char* name;
  std::function<void()> fn;
};
inline std::vector<Case>& cases() {
  static std::vector<Case> c;
  return c;
}
inline int& failures() {
  static int f = 0;
  return f;
}
inline int& skipped() {
  static int s = 0;
  return s;
}
inline std::string& data_dir() {
  static std::string d = "tests/data";
  return d;
}
struct Registrar {
  Registrar(const char* n, std::function<void()> f) { cases().push_back({n, std::move(f)}); }
};
inline void check(bool ok, const char* expr, const char* file, int line) {
  if (!ok) {
    ++failures();
    std::printf("    FAIL %s:%d  %s\n", file, line, expr);
  }
}
inline void skip(const char* why) {
  ++skipped();
  std::printf("    skipped: %s\n", why);
}
inline std::string& only() {
  static std::string o;
  return o;
}
inline int run(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--data") data_dir() = argv[i + 1];
    if (std::string(argv[i]) == "--only") only() = argv[i + 1];
  }
  int ran = 0, not_selected = 0;
  for (const Case& c : cases()) {
    if (!only().empty() && std::string(c.name).find(only()) == std::string::npos) {
      ++not_selected;
      continue;
    }
    const int before = failures();
    std::printf("  %s\n", c.name);
    c.fn();
    ++ran;
    if (failures() != before) std::printf("  -> %s FAILED\n", c.name);
  }
  if (not_selected) std::printf("  (%d cases not selected by --only %s)\n", not_selected, only().c_str());
  std::printf("%d tests, %d failures, %d skipped\n", ran, failures(), skipped());
  return failures() ? 1 : 0;
}

}  // namespace ptest

#define PTEST_CAT2(a, b) a##b
#define PTEST_CAT(a, b) PTEST_CAT2(a, b)
#define TEST(name)                                                      \
  static void PTEST_CAT(ptest_fn_, name)();                             \
  static ptest::Registrar PTEST_CAT(ptest_reg_, name)(#name, &PTEST_CAT(ptest_fn_, name)); \
  static void PTEST_CAT(ptest_fn_, name)()
#define CHECK(expr) ptest::check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)
#define CHECK_EQ(a, b) ptest::check((a) == (b), #a " == " #b, __FILE__, __LINE__)
