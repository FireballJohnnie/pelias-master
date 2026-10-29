// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "os/file_map.h"

#include <algorithm>
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
    owned_ = o.owned_;
    o.owned_ = false;
    unread_from_ = o.unread_from_;
    o.unread_from_ = 0;
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
  unread_from_ = size_;
  return Status::good();
}

Status FileMap::load(const std::string& path, void (*progress)(std::uint64_t, std::uint64_t)) {
  close();
  HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
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
  const std::uint64_t n = static_cast<std::uint64_t>(sz.QuadPart);
  void* buf = n ? VirtualAlloc(nullptr, static_cast<SIZE_T>(n), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) : nullptr;
  if (n && !buf) {
    CloseHandle(h);
    return Status::fail(Code::io_error, "not enough memory to load the file");
  }
  std::uint64_t done = 0;
  while (done < n) {
    const DWORD want = static_cast<DWORD>(std::min<std::uint64_t>(n - done, std::uint64_t{64} << 20));  // 64 MiB, as llama.cpp
    DWORD got = 0;
    if (!ReadFile(h, static_cast<std::uint8_t*>(buf) + done, want, &got, nullptr) || got == 0) {
      const DWORD e = GetLastError();
      VirtualFree(buf, 0, MEM_RELEASE);
      CloseHandle(h);
      return Status::fail(Code::io_error, "cannot read file (Windows error " + std::to_string(e) + ")");
    }
    done += got;
    if (progress) progress(done, n);
  }
  CloseHandle(h);
  data_ = static_cast<const std::uint8_t*>(buf);
  size_ = n;
  open_ = true;
  owned_ = true;
  unread_from_ = n;
  return Status::good();
}

void FileMap::close() {
  unread_from_ = 0;
  if (data_ && owned_) VirtualFree(const_cast<std::uint8_t*>(data_), 0, MEM_RELEASE);
  else if (data_) UnmapViewOfFile(data_);
  owned_ = false;
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
  unread_from_ = size_;
  return Status::good();
}

Status FileMap::load(const std::string& path, void (*progress)(std::uint64_t, std::uint64_t)) {
  close();
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    const int e = errno;
    return Status::fail(e == ENOENT ? Code::not_found : Code::io_error, std::string("cannot open file: ") + std::strerror(e));
  }
  struct stat st {};
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    ::close(fd);
    return Status::fail(Code::invalid_argument, "not a regular file");
  }
  const std::uint64_t n = static_cast<std::uint64_t>(st.st_size);
  void* buf = nullptr;
  if (n) {
    buf = mmap(nullptr, static_cast<size_t>(n), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
      ::close(fd);
      return Status::fail(Code::io_error, "not enough memory to load the file");
    }
#ifdef MADV_HUGEPAGE
    madvise(buf, static_cast<size_t>(n), MADV_HUGEPAGE);  // a hint: 2 MB pages for the weights
#endif
  }
  // 64 MiB per read, as llama.cpp's reader uses; some mounts refuse large reads with EINVAL (MEASURED 2026-10-29: 1 GiB
  // reads from a DAX virtiofs bind mount under WSL2), so a refused size is halved down to 1 MiB before giving up
  std::uint64_t done = 0;
  std::size_t chunk = std::size_t{64} << 20;
  while (done < n) {
    const size_t want = static_cast<size_t>(std::min<std::uint64_t>(n - done, chunk));
    const ssize_t got = ::read(fd, static_cast<std::uint8_t*>(buf) + done, want);
    if (got < 0 && errno == EINTR) continue;
    if (got < 0 && errno == EINVAL && chunk > (std::size_t{1} << 20)) {
      chunk /= 2;
      continue;
    }
    if (got < 0 && errno == EINVAL && n - done < (std::uint64_t{2} << 20) && done % (std::uint64_t{2} << 20) == 0) {
      break;  // the last partial 2 MiB window refused (see file_map.h): left zero, reported by unread_from()
    }
    if (got <= 0) {
      const int e = got < 0 ? errno : EIO;
      munmap(buf, static_cast<size_t>(n));
      ::close(fd);
      return Status::fail(Code::io_error, std::string("cannot read file: ") + std::strerror(e));
    }
    done += static_cast<std::uint64_t>(got);
    if (progress) progress(done, n);
  }
  ::close(fd);
  data_ = static_cast<const std::uint8_t*>(buf);
  size_ = n;
  open_ = true;
  owned_ = true;
  unread_from_ = done;
  return Status::good();
}

void FileMap::close() {
  unread_from_ = 0;
  if (data_) munmap(const_cast<std::uint8_t*>(data_), static_cast<size_t>(size_));
  owned_ = false;
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
