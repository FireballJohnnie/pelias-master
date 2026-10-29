// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "os/file_map.h"

#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#endif

namespace pelias::os {

FileMap::FileMap(FileMap&& o) noexcept { *this = std::move(o); }

FileMap& FileMap::operator=(FileMap&& o) noexcept {
  if (this != &o) {
    close();
    data_ = o.data_;
    size_ = o.size_;
    open_ = o.open_;
#ifdef _WIN32
    file_ = o.file_;
    mapping_ = o.mapping_;
    o.file_ = nullptr;
    o.mapping_ = nullptr;
#else
    fd_ = o.fd_;
    o.fd_ = -1;
#endif
    o.data_ = nullptr;
    o.size_ = 0;
    o.open_ = false;
  }
  return *this;
}

FileMap::~FileMap() { close(); }

#ifdef _WIN32

Status FileMap::open(const std::string& path) {
  close();
  HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    const DWORD e = GetLastError();
    return Status::fail(e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND ? Code::not_found : Code::io_error,
                        "cannot open file (Windows error " + std::to_string(e) + ")");
  }
  LARGE_INTEGER sz{};
  if (!GetFileSizeEx(h, &sz)) {
    const DWORD e = GetLastError();
    CloseHandle(h);
    return Status::fail(Code::io_error, "cannot read file size (Windows error " + std::to_string(e) + ")");
  }
  file_ = h;
  size_ = static_cast<std::uint64_t>(sz.QuadPart);
  open_ = true;
  if (size_ == 0) return Status::good();
  HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!m) {
    const DWORD e = GetLastError();
    close();
    return Status::fail(Code::io_error, "cannot create file mapping (Windows error " + std::to_string(e) + ")");
  }
  mapping_ = m;
  const void* v = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
  if (!v) {
    const DWORD e = GetLastError();
    close();
    return Status::fail(Code::io_error, "cannot map file (Windows error " + std::to_string(e) + ")");
  }
  data_ = static_cast<const std::uint8_t*>(v);
  return Status::good();
}

void FileMap::close() {
  if (data_) UnmapViewOfFile(data_);
  if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
  if (file_) CloseHandle(static_cast<HANDLE>(file_));
  data_ = nullptr;
  mapping_ = nullptr;
  file_ = nullptr;
  size_ = 0;
  open_ = false;
}

void FileMap::advise_sequential() {}  // Windows: FILE_FLAG_SEQUENTIAL_SCAN applies to unbuffered handles, not maps

Result<std::uint64_t> file_size(const std::string& path) {
  WIN32_FILE_ATTRIBUTE_DATA fa{};
  if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &fa)) {
    const DWORD e = GetLastError();
    return Result<std::uint64_t>::fail(Status::fail(
        e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND ? Code::not_found : Code::io_error, "cannot stat file (Windows error " + std::to_string(e) + ")"));
  }
  return Result<std::uint64_t>::good((static_cast<std::uint64_t>(fa.nFileSizeHigh) << 32) | fa.nFileSizeLow);
}

#else

Status FileMap::open(const std::string& path) {
  close();
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    const int e = errno;
    return Status::fail(e == ENOENT ? Code::not_found : Code::io_error, std::string("cannot open file: ") + std::strerror(e));
  }
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    const int e = errno;
    ::close(fd);
    return Status::fail(Code::io_error, std::string("cannot stat file: ") + std::strerror(e));
  }
  if (!S_ISREG(st.st_mode)) {
    ::close(fd);
    return Status::fail(Code::invalid_argument, "not a regular file");
  }
  fd_ = fd;
  size_ = static_cast<std::uint64_t>(st.st_size);
  open_ = true;
  if (size_ == 0) return Status::good();
  void* m = mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_PRIVATE, fd, 0);
  if (m == MAP_FAILED) {
    const int e = errno;
    close();
    return Status::fail(Code::io_error, std::string("cannot map file: ") + std::strerror(e));
  }
  data_ = static_cast<const std::uint8_t*>(m);
  return Status::good();
}

void FileMap::close() {
  if (data_) munmap(const_cast<std::uint8_t*>(data_), static_cast<size_t>(size_));
  if (fd_ >= 0) ::close(fd_);
  data_ = nullptr;
  fd_ = -1;
  size_ = 0;
  open_ = false;
}

void FileMap::advise_sequential() {
  if (data_ && size_) madvise(const_cast<std::uint8_t*>(data_), static_cast<size_t>(size_), MADV_SEQUENTIAL);
}

Result<std::uint64_t> file_size(const std::string& path) {
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) {
    const int e = errno;
    return Result<std::uint64_t>::fail(Status::fail(e == ENOENT ? Code::not_found : Code::io_error, std::string("cannot stat file: ") + std::strerror(e)));
  }
  return Result<std::uint64_t>::good(static_cast<std::uint64_t>(st.st_size));
}

#endif

}  // namespace pelias::os
