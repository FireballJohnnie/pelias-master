// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/text_file.h"

#include <cstdio>
#include <filesystem>

namespace pelias::io {

Result<std::string> read_text_file(const std::string& path, std::uint64_t limit) {
  using R = Result<std::string>;
  std::error_code ec;
  const std::filesystem::path p(path);
  if (!std::filesystem::is_regular_file(p, ec)) return R::fail(Status::fail(Code::not_found, "file not found", 0, p.filename().string()));
  const std::uint64_t size = std::filesystem::file_size(p, ec);
  if (ec) return R::fail(Status::fail(Code::io_error, "cannot read the file size", 0, p.filename().string()));
  if (size > limit) return R::fail(Status::fail(Code::invalid_argument, "file is larger than " + std::to_string(limit) + " bytes", 0, p.filename().string()));
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return R::fail(Status::fail(Code::io_error, "cannot open the file", 0, p.filename().string()));
  std::string s(static_cast<std::size_t>(size), '\0');
  const std::size_t got = size ? std::fread(s.data(), 1, s.size(), f) : 0;
  std::fclose(f);
  if (got != s.size()) return R::fail(Status::fail(Code::io_error, "short read", got, p.filename().string()));
  return R::good(std::move(s));
}

}  // namespace pelias::io
