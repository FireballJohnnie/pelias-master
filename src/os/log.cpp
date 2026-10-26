// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "pelias/log.h"

#include <mutex>

namespace pelias::log {
namespace {
Level g_level = Level::info;
std::function<void(Level, std::string_view)> g_sink;
std::mutex g_mutex;
const char* tag(Level l) {
  switch (l) {
    case Level::error: return "error";
    case Level::warn: return "warn";
    case Level::info: return "info";
    case Level::debug: return "debug";
  }
  return "?";
}
}  // namespace

Level level() { return g_level; }
void set_level(Level l) { g_level = l; }
void set_sink(std::function<void(Level, std::string_view)> sink) {
  const std::lock_guard<std::mutex> lk(g_mutex);
  g_sink = std::move(sink);
}
void write(Level l, const std::string& line) {
  if (static_cast<int>(l) > static_cast<int>(g_level)) return;
  const std::lock_guard<std::mutex> lk(g_mutex);
  if (g_sink) {
    g_sink(l, line);
  } else {
    std::fprintf(stderr, "[%s] %s\n", tag(l), line.c_str());
  }
}

}  // namespace pelias::log
