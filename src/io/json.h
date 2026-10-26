// provenance: original (RFC 8259 is the specification; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// json.h - the one JSON reader and writer of the product (rule 1: no third-party JSON). It reads a complete document
// into a value tree (objects keep key order), refuses anything outside RFC 8259 with the byte offset, and writes a
// tree back with a fixed, deterministic formatting. Numbers are kept both as double and, when the text is an integer
// that fits, as int64/uint64, so tensor offsets in safetensors headers (up to 2^63) are exact.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "pelias/status.h"

namespace pelias::io {

class Json {
 public:
  enum class Kind { null, boolean, number, string, array, object };

  Json() = default;
  static Json null() { return Json(); }
  static Json boolean(bool b) { Json j; j.kind_ = Kind::boolean; j.b_ = b; return j; }
  static Json number(double d) { Json j; j.kind_ = Kind::number; j.d_ = d; j.i_ = static_cast<std::int64_t>(d); j.u_ = d < 0 ? 0 : static_cast<std::uint64_t>(d); j.is_int_ = false; return j; }
  static Json integer(std::int64_t i) { Json j; j.kind_ = Kind::number; j.i_ = i; j.u_ = i < 0 ? 0 : static_cast<std::uint64_t>(i); j.d_ = static_cast<double>(i); j.is_int_ = true; return j; }
  static Json unsigned_integer(std::uint64_t u) { Json j; j.kind_ = Kind::number; j.u_ = u; j.i_ = static_cast<std::int64_t>(u); j.d_ = static_cast<double>(u); j.is_int_ = true; return j; }
  static Json string(std::string s) { Json j; j.kind_ = Kind::string; j.s_ = std::move(s); return j; }
  static Json array() { Json j; j.kind_ = Kind::array; return j; }
  static Json object() { Json j; j.kind_ = Kind::object; return j; }

  [[nodiscard]] Kind kind() const { return kind_; }
  [[nodiscard]] bool is_null() const { return kind_ == Kind::null; }
  [[nodiscard]] bool is_bool() const { return kind_ == Kind::boolean; }
  [[nodiscard]] bool is_number() const { return kind_ == Kind::number; }
  [[nodiscard]] bool is_integer() const { return kind_ == Kind::number && is_int_; }
  [[nodiscard]] bool is_string() const { return kind_ == Kind::string; }
  [[nodiscard]] bool is_array() const { return kind_ == Kind::array; }
  [[nodiscard]] bool is_object() const { return kind_ == Kind::object; }

  [[nodiscard]] bool as_bool() const { return b_; }
  [[nodiscard]] double as_double() const { return d_; }
  [[nodiscard]] std::int64_t as_int() const { return i_; }
  [[nodiscard]] std::uint64_t as_uint() const { return u_; }
  [[nodiscard]] const std::string& as_string() const { return s_; }
  [[nodiscard]] const std::vector<Json>& items() const { return items_; }
  [[nodiscard]] const std::vector<std::pair<std::string, Json>>& members() const { return members_; }
  [[nodiscard]] std::size_t size() const { return kind_ == Kind::array ? items_.size() : members_.size(); }

  // Object lookup: nullptr when absent or not an object.
  [[nodiscard]] const Json* get(std::string_view key) const;
  // Array element: nullptr when out of range or not an array.
  [[nodiscard]] const Json* at(std::size_t i) const;

  Json& push(Json v);                        // array append; returns *this so calls chain
  Json& set(std::string key, Json v);        // object insert or replace (keeps first-insert order); returns *this

  // Parses a whole document (any value at the top level, as RFC 8259 allows). Depth is limited to 512.
  static Result<Json> parse(std::string_view text);
  // Writes compactly (indent 0) or pretty-printed with `indent` spaces per level; keys in stored order.
  [[nodiscard]] std::string dump(int indent = 0) const;

 private:
  void dump_to(std::string& out, int indent, int depth) const;
  Kind kind_ = Kind::null;
  bool b_ = false;
  bool is_int_ = false;
  double d_ = 0.0;
  std::int64_t i_ = 0;
  std::uint64_t u_ = 0;
  std::string s_;
  std::vector<Json> items_;
  std::vector<std::pair<std::string, Json>> members_;
};

}  // namespace pelias::io
