// SPDX-License-Identifier: Apache-2.0
#include "nccl_edge.h"

#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <cstring>
#include <dlfcn.h>

namespace vpp {
namespace {
void check_nccl(ncclResult_t result, const char* operation) {
  TORCH_CHECK(result == ncclSuccess, operation, ": ", ncclGetErrorString(result));
}
}  // namespace

std::string nccl_unique_id() {
  ncclUniqueId id;
  check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
  return std::string(id.internal, sizeof(id.internal));
}
std::tuple<int, int, std::string> nccl_library_info() {
  int runtime = 0;
  check_nccl(ncclGetVersion(&runtime), "ncclGetVersion");
  Dl_info info{};
  TORCH_CHECK(dladdr(reinterpret_cast<void*>(&ncclGetVersion), &info) && info.dli_fname,
              "cannot locate loaded NCCL library");
  return {NCCL_VERSION_CODE, runtime, info.dli_fname};
}

NcclEdge::NcclEdge(std::shared_ptr<DeviceSlotPool> pool, const std::string& uid)
    : pool_(std::move(pool)) {
  TORCH_CHECK(pool_, "NCCL endpoint needs a CUDA slot pool");
  pool_->current_stream();  // Owner, health and capture checks before side effects.
  TORCH_CHECK(uid.size() == NCCL_UNIQUE_ID_BYTES, "expected NCCL128-byte unique ID");
  TORCH_CHECK(!pool_->backend_claimed_ && pool_->live() == 0 && pool_->peak_live() == 0,
              "attach exactly one transport to a fresh unused pool");
  const c10::cuda::CUDAGuard guard(pool_->device_);
  ncclUniqueId id;
  std::memcpy(id.internal, uid.data(), sizeof(id.internal));
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;  // GroupEnd must finish enqueue before recording completion.
  pool_->backend_claimed_ = true;
  try {
    check_nccl(ncclCommInitRankConfig(&comm_, 2, id,
        pool_->role_ == SlotRole::Source ? 0 : 1, &config), "ncclCommInitRankConfig");
    pool_->transport_attached_ = true;
  } catch (...) { pool_->poison(); abort_noexcept(); throw; }
}

void NcclEdge::check_async() {
  TORCH_CHECK(comm_, "NCCL endpoint is closed");
  ncclResult_t async = ncclSuccess;
  check_nccl(ncclCommGetAsyncError(comm_, &async), "ncclCommGetAsyncError");
  check_nccl(async, "NCCL asynchronous operation");
}

SlotTicket NcclEdge::reserve(const WireBoundary& wire, std::uint32_t timeout_ms) {
  pool_->current_stream();
  TORCH_CHECK(timeout_ms > 0 && timeout_ms <= 30000, "bounded credit timeout required");
  using Clock = std::chrono::steady_clock;
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  try {
    if (auto ticket = pool_->reserve(wire)) return *ticket;
    do {
      progress();  // Includes the existing asynchronous NCCL error check.
      if (auto ticket = pool_->reserve(wire)) return *ticket;
      TORCH_CHECK(Clock::now() < deadline, "VPP ring credit deadline expired");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (true);
  } catch (...) { pool_->poison(); throw; }
}

void NcclEdge::submit(const SlotTicket& t) {
  pool_->current_stream();
  TORCH_CHECK(comm_, "NCCL endpoint is closed");
  const bool send = pool_->role_ == SlotRole::Source;
  const auto expected = send ? SlotState::Produced : SlotState::Reserved;
  TORCH_CHECK(pool_->metadata_.state(t) == expected && t.sequence == next_sequence_,
              "NCCL edge submissions must match reservation sequence and slot state");
  const auto& d = pool_->metadata_.descriptor(t);
  const c10::cuda::CUDAGuard guard(pool_->device_);
  const auto stream = pool_->transfer_stream();
  auto& e = pool_->events_[t.slot_id];
  try {
    check_async();
    pool_->keep_storage(pool_->storage_, stream);
    if (send) e.ready.block(stream);
    // Use capacity-based plane addresses even for zero-count messages. P2P
    // retains zero tasks in NCCL, preserving peer FIFO and our sequence.
    auto* base = static_cast<char*>(pool_->storage_.data_ptr()) + pool_->physical_slot(t) * pool_->layout_.slot_bytes();
    const std::size_t count = d.hidden.valid_bytes / sizeof(at::BFloat16);
    check_nccl(ncclGroupStart(), "ncclGroupStart");
    ncclResult_t operation = ncclSuccess;
    for (const auto offset : {d.hidden.offset_bytes, d.residual.offset_bytes}) {
      operation = send ? ncclSend(base + offset, count, ncclBfloat16, 1, comm_, stream.stream())
                       : ncclRecv(base + offset, count, ncclBfloat16, 0, comm_, stream.stream());
      if (operation != ncclSuccess) break;
    }
    // Always balance the group, including a failed first/second enqueue.
    const auto ended = ncclGroupEnd();
    check_nccl(operation, "ncclSend/ncclRecv");
    check_nccl(ended, "ncclGroupEnd");
    e.transfer_done.record(stream);
    pool_->metadata_.transfer_recorded(t);
    ++next_sequence_;
  } catch (...) { pool_->poison(); throw; }
}

std::uint32_t NcclEdge::progress() {
  pool_->check_owner();
  try { check_async(); return pool_->progress(); }
  catch (...) { pool_->poison(); throw; }
}

void NcclEdge::mark(c10::cuda::CUDAEvent& event) {
  pool_->current_stream();
  TORCH_CHECK(comm_, "NCCL endpoint is closed");
  event.record(pool_->transfer_stream());
}

void NcclEdge::close() {
  pool_->check_owner();
  TORCH_CHECK(comm_, "NCCL endpoint already closed");
  TORCH_CHECK(pool_->live() == 0, "drain send/receive/last-consumer events before close");
  const c10::cuda::CUDAGuard guard(pool_->device_);
  try {
    check_async();
    check_nccl(ncclCommFinalize(comm_), "ncclCommFinalize");
    // Destroy consumes the handle; never retry/abort a possibly freed handle.
    auto comm = comm_;
    comm_ = nullptr;
    const auto result = ncclCommDestroy(comm);
    pool_->transport_attached_ = false;
    check_nccl(result, "ncclCommDestroy");
  } catch (...) { pool_->poison(); throw; }
}

void NcclEdge::abort_noexcept() noexcept {
  if (!comm_) return;
  pool_->poison();  // Fatal epoch only: no unsent sequence/credit reuse.
  auto comm = comm_;
  comm_ = nullptr;
  try {
    const c10::cuda::CUDAGuard guard(pool_->device_);
    const auto result = ncclCommAbort(comm);
    if (result != ncclSuccess) TORCH_WARN("NCCL abort: ", ncclGetErrorString(result));
  } catch (...) { /* Owned supervisor still bounds fatal process cleanup. */ }
  pool_->transport_attached_ = false;
}
NcclEdge::~NcclEdge() noexcept { abort_noexcept(); }

}  // namespace vpp
