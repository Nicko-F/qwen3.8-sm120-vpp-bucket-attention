// SPDX-License-Identifier: Apache-2.0
#include "slot_pool.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace vpp {
namespace {
std::atomic<std::uint64_t> next_pool_id{1};
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
}  // namespace

PhysicalSlotLeases::PhysicalSlotLeases(std::uint32_t slots) : slots_(slots) {
  require(slots > 0 && slots <= owners_.size(), "invalid physical arena capacity");
}
std::uint32_t PhysicalSlotLeases::acquire(const SlotTicket& ticket) {
  require(available(), "physical arena has no free lease");
  for (const auto& owner : owners_)
    require(!owner || owner->pool_id != ticket.pool_id || owner->slot_id != ticket.slot_id,
            "logical ticket already has a physical lease");
  for (std::uint32_t i = 0; i < slots_; ++i) if (!owners_[i]) {
    owners_[i] = ticket;
    peak_live_ = std::max(peak_live_, ++live_);
    return i;
  }
  throw std::logic_error("physical arena accounting mismatch");
}
void PhysicalSlotLeases::release(std::uint32_t physical, const SlotTicket& ticket) {
  require(physical < slots_, "physical lease outside arena");
  const auto& owner = owners_[physical];
  require(owner && owner->pool_id == ticket.pool_id && owner->sequence == ticket.sequence &&
          owner->generation == ticket.generation && owner->slot_id == ticket.slot_id,
          "physical lease completion belongs to another ticket/generation");
  owners_[physical].reset();
  --live_;
}

SlotPool::SlotPool(SlotRole role, BoundaryLayout layout, std::uint64_t epoch,
                   std::uint32_t src_stage)
    : role_(role), layout_(layout), epoch_(epoch),
      pool_id_(next_pool_id.fetch_add(1, std::memory_order_relaxed)),
      src_stage_(src_stage) {
  make_boundary(layout_, epoch_, 0, 0, src_stage_, 0);
  require(pool_id_ != 0, "pool identity exhausted");
  require(role == SlotRole::Source || role == SlotRole::Destination, "invalid pool role");
}

std::optional<SlotTicket> SlotPool::try_acquire(const BoundaryDesc& d) {
  require(!stopped_, "role stopped; create a new role epoch, do not resume");
  validate_boundary(layout_, d);
  require(d.epoch == epoch_ && d.src_stage == src_stage_, "wrong role epoch or edge");
  require(d.edge_sequence == next_sequence_, "out-of-order boundary admission");
  auto& e = entries_[d.slot_id];
  if (e.state != SlotState::Free) return std::nullopt;
  e.desc = d;
  e.state = SlotState::Reserved;
  ++next_sequence_;
  peak_live_ = std::max(peak_live_, ++live_);
  return SlotTicket{pool_id_, d.edge_sequence, d.generation, d.slot_id};
}

const SlotPool::Entry& SlotPool::entry(const SlotTicket& ticket) const {
  require(ticket.pool_id == pool_id_ && ticket.slot_id < layout_.slot_count,
          "ticket belongs to another pool");
  const auto& e = entries_[ticket.slot_id];
  require(e.state != SlotState::Free && e.desc.edge_sequence == ticket.sequence &&
          e.desc.generation == ticket.generation, "stale completion or retired ticket");
  return e;
}
const BoundaryDesc& SlotPool::descriptor(const SlotTicket& ticket) const {
  return entry(ticket).desc;
}
SlotState SlotPool::state(const SlotTicket& ticket) const { return entry(ticket).state; }
void SlotPool::transition(const SlotTicket& ticket, SlotRole role,
                          SlotState from, SlotState to) {
  require(role_ == role && entry(ticket).state == from, "invalid slot transition");
  entries_[ticket.slot_id].state = to;
  if (to == SlotState::Free) --live_;
}
void SlotPool::producer_recorded(const SlotTicket& t) {
  transition(t, SlotRole::Source, SlotState::Reserved, SlotState::Produced);
}
void SlotPool::transfer_recorded(const SlotTicket& t) {
  transition(t, role_, role_ == SlotRole::Source ? SlotState::Produced : SlotState::Reserved,
             SlotState::Transferring);
}
void SlotPool::consumer_recorded(const SlotTicket& t) {
  transition(t, SlotRole::Destination, SlotState::Transferring, SlotState::Consuming);
}
void SlotPool::source_read_complete(const SlotTicket& t) {
  transition(t, SlotRole::Source, SlotState::Transferring, SlotState::Free);
}
void SlotPool::consumer_complete(const SlotTicket& t) {
  transition(t, SlotRole::Destination, SlotState::Consuming, SlotState::Free);
}
void SlotPool::discard_received(const SlotTicket& t) {
  transition(t, SlotRole::Destination, SlotState::Transferring, SlotState::Free);
}
void SlotPool::release_unsubmitted_on_stop(const SlotTicket& t) {
  require(stopped_, "unsent sequence can only be abandoned at fatal role shutdown");
  transition(t, role_, SlotState::Reserved, SlotState::Free);
}
}  // namespace vpp
