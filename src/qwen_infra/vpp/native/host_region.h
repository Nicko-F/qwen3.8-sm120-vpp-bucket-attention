// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace vpp {

// Cold control-plane identity, no process-local pointer and no payload copy.
struct HostRegionHandle {
  std::uint64_t bytes, device, inode, process_birth;
  int owner_pid, owner_fd, numa_node;
};

// Linux shared backing only. CUDA registration is a separate owning layer;
// mlock/NUMA success must never be reported as CUDA-pinned DMA qualification.
// The role owner keeps this object alive until all peer mappings are retired.
class HostRegion {
 public:
  // -1 inherits placement. An optional deployment reserve is separate from ABI.
  static std::shared_ptr<HostRegion> create(std::uint64_t bytes, int numa_node = -1,
                                           std::uint64_t minimum_available_bytes = 0);
  static std::shared_ptr<HostRegion> attach(const HostRegionHandle&);
  ~HostRegion() noexcept;
  HostRegion(const HostRegion&) = delete;
  HostRegion& operator=(const HostRegion&) = delete;
  const HostRegionHandle& handle() const { return handle_; }
  void* range(std::uint64_t offset, std::uint64_t bytes) const;
  std::string numa_receipt() const;
  void verify_numa() const;

 private:
  HostRegion() = default;
  void map();
  HostRegionHandle handle_{};
  int fd_ = -1;
  void* base_ = nullptr;
  bool locked_ = false;
};

}  // namespace vpp
