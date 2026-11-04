// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "interface.h"
#include <optional>

namespace vpp {

enum class SlotRole { Source, Destination };
enum class SlotState { Free, Reserved, Produced, Transferring, Consuming };
struct SlotTicket {
  std::uint64_t pool_id, sequence, generation;
  std::uint32_t slot_id;
};

// Metadata only. One serialized host owner per pool, no GPU/host allocation.
// Tickets stay local. Source and destination credits are independent.
class SlotPool {
 public:
  SlotPool(SlotRole, BoundaryLayout, std::uint64_t epoch, std::uint32_t src_stage);
  SlotPool(const SlotPool&) = delete;
  SlotPool& operator=(const SlotPool&) = delete;
  std::optional<SlotTicket> try_acquire(const BoundaryDesc&);
  const BoundaryDesc& descriptor(const SlotTicket&) const;
  SlotState state(const SlotTicket&) const;

  // The CUDA adapter must RECORD the event before publishing these transitions.
  void producer_recorded(const SlotTicket&);        // Source only.
  void transfer_recorded(const SlotTicket&);        // Both endpoints.
  // Queue a compute-stream wait on the recorded receive event first. This does
  // NOT require a CPU receive-completion wait and preserves asynchronous enqueue.
  void consumer_recorded(const SlotTicket&);        // Destination only.

  // Call only after the corresponding current-generation event query succeeds.
  void source_read_complete(const SlotTicket&);     // Source send/D2H done.
  void consumer_complete(const SlotTicket&);        // Last destination use done.
  void discard_received(const SlotTicket&);         // Cancelled batch, receive done.

  // Fatal role shutdown, not per-request cancellation. Never resume this epoch:
  // skipping an acquired sequence would leave a gap in the peer's receive order.
  void stop_admission() noexcept { stopped_ = true; }
  void release_unsubmitted_on_stop(const SlotTicket&);
  std::uint32_t live() const noexcept { return live_; }
  std::uint32_t peak_live() const noexcept { return peak_live_; }
  std::uint64_t next_sequence() const noexcept { return next_sequence_; }

 private:
  struct Entry { BoundaryDesc desc{}; SlotState state = SlotState::Free; };
  const Entry& entry(const SlotTicket&) const;
  void transition(const SlotTicket&, SlotRole, SlotState from, SlotState to);
  SlotRole role_;
  BoundaryLayout layout_;
  std::uint64_t epoch_, pool_id_, next_sequence_ = 0;
  std::uint32_t src_stage_, live_ = 0, peak_live_ = 0;
  bool stopped_ = false;
  std::array<Entry, 7> entries_{};
};

}  // namespace vpp
