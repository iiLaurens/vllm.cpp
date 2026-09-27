// VT_CUDA_ALLOC_TRACE: the per-allocation instrument.
//
// A first-forward OOM names one failing size and nothing else, which is not
// enough to tell which allocation sequence produced the pressure on a 24 GiB
// card. This gate pins the three lines the instrument adds: a large allocation
// with its size, the matching free, and the failing request with its error. It
// also asserts its OWN capture with a sentinel, so a redirect that silently
// caught nothing fails here instead of reading as "the trace printed nothing".
//
// The flag is read ONCE per process (a function-static in CudaBackend), so this
// binary turns it on in a global initializer, before main and thus before any
// TEST_CASE. The flag-OFF path is the same code with the guard false and is
// covered by every CUDA test in the tree, none of which sets the variable.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#define VLLM_ALLOC_TRACE_CAPTURE 1
#endif

#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

struct EnableAllocTrace {
  EnableAllocTrace() { ::setenv("VT_CUDA_ALLOC_TRACE", "1", 1); }
};
const EnableAllocTrace g_enable_alloc_trace;

bool HasCuda() {
  try {
    vt::GetBackend(vt::DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

#ifdef VLLM_ALLOC_TRACE_CAPTURE
// Everything written to stderr while `body` runs, as a string.
template <typename F>
std::string CaptureStderr(F&& body) {
  std::fflush(stderr);
  int saved = ::dup(2);
  char path[] = "/tmp/vllm_alloctrace_XXXXXX";
  int fd = ::mkstemp(path);
  REQUIRE(saved >= 0);
  REQUIRE(fd >= 0);
  ::dup2(fd, 2);
  body();
  std::fflush(stderr);
  ::dup2(saved, 2);
  ::close(saved);
  ::lseek(fd, 0, SEEK_SET);
  std::string out;
  char buf[4096];
  ssize_t n = 0;
  while ((n = ::read(fd, buf, sizeof(buf))) > 0) out.append(buf, static_cast<size_t>(n));
  ::close(fd);
  ::unlink(path);
  return out;
}
#endif

}  // namespace

TEST_CASE("cuda alloc trace: a large allocation, its free, and the failing request") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  const size_t big = size_t{32} << 20;  // 32 MiB, above the 16 MiB report floor

  void* p = nullptr;
  std::string alloc_line;
#ifdef VLLM_ALLOC_TRACE_CAPTURE
  alloc_line = CaptureStderr([&] {
    std::fputs("VLLM_ALLOC_TRACE_SENTINEL\n", stderr);
    p = gpu.Alloc(big);
  });
  // The capture caught what we wrote. Without this the assertions below could
  // read an empty string and report "the trace printed nothing" as a pass.
  CHECK(alloc_line.find("VLLM_ALLOC_TRACE_SENTINEL") != std::string::npos);
#else
  p = gpu.Alloc(big);
#endif
  REQUIRE(p != nullptr);
  CHECK(alloc_line.find("[cuda-alloc]") != std::string::npos);
  CHECK(alloc_line.find("size=32.0 MiB") != std::string::npos);

#ifdef VLLM_ALLOC_TRACE_CAPTURE
  const std::string free_line = CaptureStderr([&] { gpu.Free(p); });
  CHECK(free_line.find("[cuda-free]") != std::string::npos);
  CHECK(free_line.find("size=32.0 MiB") != std::string::npos);

  // A request that cannot fit anywhere: the failure names its own size and the
  // free memory at that instant, and it still throws.
  const std::string failed_line = CaptureStderr([&] {
    CHECK_THROWS_AS(gpu.Alloc(size_t{1} << 50), std::runtime_error);
  });
  CHECK(failed_line.find("FAILED") != std::string::npos);
  CHECK(failed_line.find("size=") != std::string::npos);
  CHECK(failed_line.find("err=") != std::string::npos);
#else
  gpu.Free(p);
#endif
}
