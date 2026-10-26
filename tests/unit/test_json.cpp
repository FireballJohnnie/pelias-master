// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/json.h"
#include "pelias_test.h"

using pelias::io::Json;

TEST(json_parse_and_access) {
  const auto r = Json::parse(R"({"a": 1, "b": [true, null, "x\u00e9\n"], "c": {"d": -2.5e3, "e": 18446744073709551615}, "f": -9223372036854775808})");
  CHECK(r.ok());
  const Json& j = r.value;
  CHECK(j.is_object());
  CHECK_EQ(j.get("a")->as_uint(), 1u);
  CHECK(j.get("a")->is_integer());
  CHECK_EQ(j.get("b")->size(), 3u);
  CHECK(j.get("b")->at(0)->as_bool());
  CHECK(j.get("b")->at(1)->is_null());
  CHECK_EQ(j.get("b")->at(2)->as_string(), std::string("x\xC3\xA9\n"));
  CHECK_EQ(j.get("c")->get("d")->as_double(), -2500.0);
  CHECK(!j.get("c")->get("d")->is_integer());
  CHECK_EQ(j.get("c")->get("e")->as_uint(), 18446744073709551615ull);
  CHECK_EQ(j.get("f")->as_int(), (-9223372036854775807LL - 1));
  CHECK(j.get("zz") == nullptr);
  CHECK(j.at(0) == nullptr);
}

TEST(json_rejects_malformed) {
  const char* bad[] = {"", "{", "[1,]", "{\"a\":}", "01", "1.", "-", "\"abc", "\"\\x\"", "{\"a\":1,}", "[1 2]", "tru", "\"\\ud800\"", "nul", "{\"a\" 1}", "1e", "\"\t\""};
  for (const char* b : bad) {
    const auto r = Json::parse(b);
    CHECK(!r.ok());
    if (r.ok()) std::printf("      accepted: %s\n", b);
  }
  std::string deep(600, '[');
  deep += std::string(600, ']');
  CHECK(!Json::parse(deep).ok());
}

TEST(json_dump_round_trip) {
  Json j = Json::object();
  j.set("name", Json::string("q\"uote\\back\n")).set("n", Json::unsigned_integer(12345678901234ull));  // set chains on the same object
  j.set("neg", Json::integer(-5));
  j.set("f", Json::number(0.1));
  Json arr = Json::array();
  arr.push(Json::boolean(false));
  arr.push(Json::null());
  j.set("arr", arr);
  const std::string compact = j.dump(0);
  CHECK_EQ(compact, std::string(R"({"name":"q\"uote\\back\n","n":12345678901234,"neg":-5,"f":0.10000000000000001,"arr":[false,null]})"));
  const auto back = Json::parse(compact);
  CHECK(back.ok());
  CHECK_EQ(back.value.dump(0), compact);
  const auto pretty = Json::parse(j.dump(2));
  CHECK(pretty.ok());
  CHECK_EQ(pretty.value.dump(0), compact);
  CHECK_EQ(Json::parse("  42 ").value.as_uint(), 42u);  // a scalar document is valid
}

TEST(json_refuses_duplicate_keys) {
  const auto r = Json::parse(R"({"a": 1, "b": 2, "a": 3})");
  CHECK(!r.ok());
  CHECK(r.status.code == pelias::Code::corrupt);
  CHECK_EQ(r.status.name, std::string("a"));
  CHECK_EQ(r.status.offset, 17u);  // the second "a"
  CHECK(Json::parse(R"({"a": {"x": 1}, "b": {"x": 2}})").ok());  // same key in different objects is fine
  // many keys: parses in linear time (the old insert scanned every earlier key)
  std::string big = "{";
  for (int i = 0; i < 50000; ++i) big += (i ? ",\"k" : "\"k") + std::to_string(i) + "\":" + std::to_string(i);
  big += "}";
  const auto b = Json::parse(big);
  CHECK(b.ok());
  CHECK_EQ(b.value.size(), 50000u);
  CHECK_EQ(b.value.members()[49999].first, std::string("k49999"));
}
