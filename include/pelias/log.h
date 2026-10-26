// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// log.h - one logger for the whole program: levels, a sink the CLI can redirect, and the privacy rule enforced by
// convention (rule 7): callers pass counts, names, lengths, offsets and pass/fail, never file paths or dataset text.
#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <string_view>

namespace pelias::log {

enum class Level : int { error = 0, warn = 1, info = 2, debug = 3 };

Level level();
void set_level(Level l);
// The sink receives finished lines; the default writes to stderr. Set once at start-up.
void set_sink(std::function<void(Level, std::string_view)> sink);
void write(Level l, const std::string& line);

inline void error(const std::string& s) { write(Level::error, s); }
inline void warn(const std::string& s) { write(Level::warn, s); }
inline void info(const std::string& s) { write(Level::info, s); }
inline void debug(const std::string& s) { write(Level::debug, s); }

}  // namespace pelias::log
