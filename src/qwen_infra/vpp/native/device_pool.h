// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "slot_pool.h"
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAEvent.h>
#include <atomic>
#include <memory>
#include <thread>
#include <utility>

namespace vpp {

class DeviceBufferArena {
 public:
  DeviceBufferArena(SlotRole, BoundaryLayout, c10::DeviceIndex);
  std::uint32_t live() const { return leases_.live(); }
  std::uint32_t peak_live() const { return leases_.peak_live(); }
  std::uint64_t stalls() const { return leases_.stalls(); }
  std::uint64_t payload_bytes() const { return layout_.slot_bytes() * layout_.slot_count; }
  void close();
 private:
  friend class DeviceSlotPool;
  SlotRole role_;
  BoundaryLayout layout_;
  c10::DeviceIndex device_;
  std::thread::id owner_;
  PhysicalSlotLeases leases_;
  at::Tensor storage_;
  c10::cuda::CUDAEvent allocation_ready_;
};

// One serialized host owner, eager boundary operations outside model graphs.
// Returned destination views are leases: no use after record_consumer_done.
class DeviceSlotPool : public std::enable_shared_from_this<DeviceSlotPool> {
 public:
  DeviceSlotPool(SlotRole, BoundaryLayout, std::uint64_t epoch,
                 std::uint32_t src_stage, c10::DeviceIndex device,
                 std::shared_ptr<DeviceBufferArena> arena = nullptr);
  DeviceSlotPool(const DeviceSlotPool&) = delete;
  DeviceSlotPool& operator=(const DeviceSlotPool&) = delete;
  std::optional<SlotTicket> reserve(const WireBoundary&);
  void publish(const SlotTicket&, const at::Tensor& hidden, const at::Tensor& residual);
  // Borrow once before enqueueing the compiled producer, outside capture. The
  // same compute stream must finish both writers before publish_written.
  std::pair<at::Tensor, at::Tensor> producer_views(const SlotTicket&);
  void publish_written(const SlotTicket&);
  // A CPU capability Tensor owns the pool/ticket through its DataPtr context.
  // It travels as an ordinary compiled input, not a pointer baked into FX.
  at::Tensor publication_token(const SlotTicket&);
  static void record_plane(const at::Tensor& plane, const at::Tensor& token, std::uint32_t plane_id);
  static float publication_elapsed_ms(const at::Tensor& token); // diagnostic, after GPU completion
  // A real same-device DMA test of the binding, NOT a cross-rank backend.
  void loopback(const SlotTicket&, DeviceSlotPool& destination, const SlotTicket&);
  std::pair<at::Tensor, at::Tensor> consume(const SlotTicket&);
  void record_consumer_done(const SlotTicket&);
  void discard_received(const SlotTicket&);
  std::uint32_t progress();  // Bounded current-ticket queries, never a spin loop.
  void close();             // Requires drain; no hidden host/device synchronize.
  std::uint32_t live() const { return metadata_.live(); }
  std::uint32_t peak_live() const { return metadata_.peak_live(); }
  std::uint64_t payload_bytes() const { return arena_->payload_bytes(); }
  std::uint32_t physical_slot(const SlotTicket&) const;
  BoundaryLayout layout() const { return layout_; }

 private:
  friend class NcclEdge;
  friend class DmaEdge;
  friend class DmaProgress;
  struct Events {
    c10::cuda::CUDAEvent ready, transfer_done, consumer_done;
    std::optional<SlotTicket> ticket;
    std::uint32_t physical = 0;
    bool consumer_recorded = false, discard = false;
    std::optional<c10::cuda::CUDAStream> consumer_stream;
    std::optional<c10::cuda::CUDAStream> producer_stream;
    std::array<c10::cuda::CUDAEvent,2> plane_ready{
        c10::cuda::CUDAEvent(cudaEventDefault), c10::cuda::CUDAEvent(cudaEventDefault)};
    std::atomic<std::uint32_t> published{0};
    bool token_borrowed = false;
  };
  void check_owner() const;
  c10::cuda::CUDAStream current_stream() const;
  c10::cuda::CUDAStream transfer_stream();
  void keep_storage(const at::Tensor&, const c10::cuda::CUDAStream&);
  void require_tensor(const at::Tensor&, const BoundaryDesc&) const;
  std::pair<at::Tensor, at::Tensor> views(const SlotTicket&) const;
  void poison() noexcept { failed_ = true; metadata_.stop_admission(); }
  SlotRole role_;
  BoundaryLayout layout_;
  SlotPool metadata_;
  c10::DeviceIndex device_;
  std::thread::id owner_;
  at::Tensor storage_;
  bool owns_arena_;
  std::shared_ptr<DeviceBufferArena> arena_;
  std::optional<c10::cuda::CUDAStream> transfer_stream_;
  std::array<Events, 7> events_{};
  bool failed_ = false, closed_ = false;
  bool backend_claimed_ = false, transport_attached_ = false;
};

}  // namespace vpp
