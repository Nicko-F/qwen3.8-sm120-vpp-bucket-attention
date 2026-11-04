// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <memory>
#include <vector>

namespace vpp {

struct KvCopyResult {
  std::uint64_t job_id, generation, bytes, segments;
  bool cancelled;
  double device_ms;
  std::uint64_t first_issue_ns, last_done_ns;
};

// Native CopyRun descriptors target the EXISTING registered canonical payload.
// No payload allocation, CPU memcpy, GPU peer access, scheduler or allocator.
// One bounded CE progress thread, two streams, one outstanding segment/job.
class KvCopy {
 public:
  KvCopy(c10::DeviceIndex device, std::vector<at::Tensor> gpu, at::Tensor host,
         std::uint64_t segment_bytes = 8ULL<<20, std::uint32_t max_jobs = 8,
         std::uint32_t max_spans = 1U<<20, bool pitched = true);
  ~KvCopy() noexcept;
  // CPU int64[N,7]: tensor index, GPU/host byte offsets, fragment bytes,
  // fragment count, GPU/host strides. Preserve native compact CopyRuns.
  // Admission copies metadata only. IDs increase; false means bounded pressure.
  // Caller certifies stable source ranges and native destination block ownership.
  bool submit(std::uint64_t id, std::uint64_t generation, const at::Tensor& spans,
              bool store, c10::cuda::CUDAStream producer,
              const at::Tensor& gates = at::Tensor());
  // Shared CPU int64 flags: release-publish a complete canonical page after
  // D2H, acquire before H2D. Gate rows are [flag index, positive row epoch].
  // No payload mirror, GPU polling kernel or Python segment progress.
  void set_ready(const at::Tensor& flags);
  bool cancel(std::uint64_t id, std::uint64_t generation);
  std::vector<KvCopyResult> poll();
  std::vector<KvCopyResult> snapshot(); // cold diagnostic; does not harvest jobs
  void wait(const std::vector<std::uint64_t>& ids, std::uint32_t timeout_ms = 30000);
  void drain(std::uint32_t timeout_ms = 30000);
  void close(); // drain and harvest before dropping registered payload owners
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace vpp
