// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// file_map.h - a read-only memory map of a whole file (model files, shards, profiles). The only place besides
// os/ that may include operating-system headers is hal/host_alloc_* (boundary check BC6). Reading a mapped byte past the
// file's end is impossible by construction: every reader checks offsets against size() first (io/gguf_reader).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "pelias/status.h"

namespace pelias::os {

class FileMap {
 public:
  FileMap() = default;
  FileMap(const FileMap&) = delete;
  FileMap& operator=(const FileMap&) = delete;
  FileMap(FileMap&& o) noexcept;
  FileMap& operator=(FileMap&& o) noexcept;
  ~FileMap();

  // Maps the whole file read-only. An empty file maps to size 0 with a null pointer and is not an error.
  // The message never carries the path (rule 7); the caller knows which file it asked for.
  Status open(const std::string& path);
  // Reads the whole file into private memory instead of mapping it (pelias generate --no-mmap). Costs RAM equal to the
  // file, read once; after that no access depends on the file system. Used where every token re-reads the weights and the
  // file sits on a mount that does not cache them (MEASURED 2026-10-29: WSL2's /mnt/c with a DAX bind mount kept about
  // 1 GB of a 22.7 GB model cached; llama-server decoded at 0.24 tok/s from it). progress, if set, gets bytes read so far.
  // A mount that refuses reads inside the file's last partial 2 MiB (MEASURED 2026-10-29: WSL2's DAX virtiofs bind mount
  // returns EINVAL for every read of a 22.7 GB model's final 1,562,912 bytes) is tolerated there only: those bytes are
  // left zero and unread_from() says where they start, so the caller can refuse if a tensor it uses lies there.
  Status load(const std::string& path, void (*progress)(std::uint64_t done, std::uint64_t total) = nullptr);
  [[nodiscard]] std::uint64_t unread_from() const { return unread_from_; }  // == size() when every byte was read
  void close();

  [[nodiscard]] const std::uint8_t* data() const { return data_; }
  [[nodiscard]] std::uint64_t size() const { return size_; }
  [[nodiscard]] bool is_open() const { return open_; }

  // Advises the OS that reads will be sequential (a hint; no effect on correctness).
  void advise_sequential();

 private:
  const std::uint8_t* data_ = nullptr;
  std::uint64_t size_ = 0;
  bool open_ = false;
  bool owned_ = false;  // data_ is private memory from load(), not a file view
  std::uint64_t unread_from_ = 0;
#ifdef _WIN32
  void* file_ = nullptr;     // HANDLE
  void* mapping_ = nullptr;  // HANDLE
#else
  int fd_ = -1;
#endif
};

// Size of a file without mapping it, or a Status if it cannot be read.
Result<std::uint64_t> file_size(const std::string& path);

}  // namespace pelias::os
