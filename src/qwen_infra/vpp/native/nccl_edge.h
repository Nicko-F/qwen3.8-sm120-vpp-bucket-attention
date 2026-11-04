// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "device_pool.h"
#include <nccl.h>
#include <memory>
#include <string>
#include <tuple>

namespace vpp {

// Cold bootstrap bytes travel through the existing CPU control plane.
std::string nccl_unique_id();
std::tuple<int, int, std::string> nccl_library_info();

// One two-rank communicator per directed edge; source=0, destination=1.
// Call init/close in common edge order, outside any external NCCL group.
// No Python callback, host polling thread or hot-path device synchronization.
class NcclEdge {
 public:
  NcclEdge(std::shared_ptr<DeviceSlotPool>, const std::string& unique_id);
  ~NcclEdge() noexcept;
  NcclEdge(const NcclEdge&) = delete;
  NcclEdge& operator=(const NcclEdge&) = delete;
  // Wait only on exhausted ring credit; no Python hot polling or global fence.
  SlotTicket reserve(const WireBoundary&, std::uint32_t timeout_ms);
  void submit(const SlotTicket&);
  // Optional external measurement marker; no observer on the production path.
  void mark(c10::cuda::CUDAEvent&);
  std::uint32_t progress();
  void close();  // Cold, requires full resource drain at both endpoints.
  void abort() noexcept { abort_noexcept(); }  // Fatal role shutdown only.

 private:
  void check_async();
  void abort_noexcept() noexcept;
  std::shared_ptr<DeviceSlotPool> pool_;
  ncclComm_t comm_ = nullptr;
  std::uint64_t next_sequence_ = 0;
};

}  // namespace vpp
