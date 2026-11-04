// SPDX-License-Identifier: Apache-2.0
#include "device_pool.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace vpp {
namespace {
void outside_capture(const c10::cuda::CUDAStream& stream) {
  cudaStreamCaptureStatus status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &status));
  TORCH_CHECK(status == cudaStreamCaptureStatusNone,
              "V1 boundary enqueue is outside model graph capture");
}
struct Publication {
  std::shared_ptr<DeviceSlotPool> pool;
  SlotTicket ticket;
  std::array<std::int64_t,6> words;
};
void delete_publication(void* context) { delete static_cast<Publication*>(context); }
Publication& publication(const at::Tensor& token) {
  TORCH_CHECK(token.device().is_cpu() && token.scalar_type()==at::kLong &&
      token.dim()==1 && token.numel()==6 && token.is_contiguous(), "expected CPU publication capability");
  const auto& data=token.storage().data_ptr();
  TORCH_CHECK(data.get_deleter()==&delete_publication && data.get_context(),
              "publication capability cannot be cloned or forged");
  auto& p=*static_cast<Publication*>(data.get_context());
  TORCH_CHECK(token.const_data_ptr()==p.words.data(),"publication capability offset changed");
  return p;
}
}  // namespace

DeviceBufferArena::DeviceBufferArena(SlotRole role, BoundaryLayout layout, c10::DeviceIndex device)
    : role_(role), layout_(layout), device_(device), owner_(std::this_thread::get_id()),
      leases_(layout.slot_count) {
  layout_.validate();
  const c10::cuda::CUDAGuard guard(device_);
  outside_capture(c10::cuda::getCurrentCUDAStream(device_));
  storage_ = at::empty({layout.slot_count, 2, layout.token_capacity, kHiddenSize},
      at::TensorOptions().device(c10::Device(c10::kCUDA, device_)).dtype(at::kBFloat16));
  // Caching-allocator storage may have pending work on its allocation stream.
  // recordStream protects destruction but does NOT establish this dependency.
  allocation_ready_.record(c10::cuda::getCurrentCUDAStream(device_));
}
void DeviceBufferArena::close() {
  TORCH_CHECK(owner_ == std::this_thread::get_id() && live() == 0,
              "drain physical leases on the arena owner before close");
  storage_ = at::Tensor();
}

DeviceSlotPool::DeviceSlotPool(SlotRole role, BoundaryLayout layout,
    std::uint64_t epoch, std::uint32_t src_stage, c10::DeviceIndex device,
    std::shared_ptr<DeviceBufferArena> arena)
    : role_(role), layout_(layout), metadata_(role, layout, epoch, src_stage),
      device_(device), owner_(std::this_thread::get_id()),
      owns_arena_(!arena),
      arena_(arena ? std::move(arena) : std::make_shared<DeviceBufferArena>(role, layout, device)) {
  TORCH_CHECK(arena_->role_ == role && arena_->device_ == device &&
      arena_->layout_.token_capacity == layout.token_capacity &&
      arena_->owner_ == owner_ && arena_->storage_.defined(), "incompatible physical arena");
  storage_ = arena_->storage_;
  const c10::cuda::CUDAGuard guard(device_);
  for (std::uint32_t i = 0; i < layout.slot_count; ++i) {
    events_[i].ready.create(device_);
    events_[i].transfer_done.create(device_);
    events_[i].consumer_done.create(device_);
    for(auto& event:events_[i].plane_ready) event.create(device_);
  }
}
c10::cuda::CUDAStream DeviceSlotPool::transfer_stream() {
  check_owner();
  if (!transfer_stream_) transfer_stream_ = c10::cuda::getStreamFromPool(false, device_);
  return *transfer_stream_;
}

void DeviceSlotPool::check_owner() const {
  TORCH_CHECK(std::this_thread::get_id() == owner_, "one serialized host owner per pool");
  TORCH_CHECK(!failed_ && !closed_, "pool failed/closed; do not resume this role epoch");
}
c10::cuda::CUDAStream DeviceSlotPool::current_stream() const {
  check_owner();
  const auto stream = c10::cuda::getCurrentCUDAStream(device_);
  outside_capture(stream);
  return stream;
}
void DeviceSlotPool::keep_storage(const at::Tensor& tensor,
                                 const c10::cuda::CUDAStream& stream) {
  if (tensor.storage().data_ptr().get() == storage_.storage().data_ptr().get())
    arena_->allocation_ready_.block(stream);
  // Reuse PyTorch allocator's asynchronous destruction contract, including
  // input temporaries. No extra tensor clone or retention until remote receipt.
  c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}
void DeviceSlotPool::require_tensor(const at::Tensor& value, const BoundaryDesc& d) const {
  TORCH_CHECK(value.is_cuda() && value.get_device() == device_ &&
      value.scalar_type() == at::kBFloat16 && value.is_contiguous() &&
      value.dim() == 2 && value.size(0) == d.valid_tokens && value.size(1) == kHiddenSize,
      "expected contiguous CUDA BF16 [valid_tokens,5120] on the pool device");
  TORCH_CHECK(value.storage().data_ptr().get() != storage_.storage().data_ptr().get(),
      "source publication cannot alias its own fixed pool");
}
std::pair<at::Tensor, at::Tensor> DeviceSlotPool::views(const SlotTicket& t) const {
  const auto& d = metadata_.descriptor(t);
  auto slot = storage_.select(0, physical_slot(t));
  return {slot.select(0, 0).narrow(0, 0, d.valid_tokens),
          slot.select(0, 1).narrow(0, 0, d.valid_tokens)};
}
std::uint32_t DeviceSlotPool::physical_slot(const SlotTicket& t) const {
  metadata_.descriptor(t);  // Reject stale/wrong-pool tickets before indexing.
  return events_[t.slot_id].physical;
}
std::optional<SlotTicket> DeviceSlotPool::reserve(const WireBoundary& wire) {
  check_owner();
  TORCH_CHECK(!backend_claimed_ || transport_attached_, "NCCL endpoint already closed");
  const auto desc = decode_boundary(layout_, wire);
  if (!arena_->leases_.available()) {
    arena_->leases_.stalled();
    return std::nullopt;
  }
  const auto ticket = metadata_.try_acquire(desc);
  if (ticket) {
    auto& e = events_[ticket->slot_id];
    e.ticket = ticket;
    e.physical = arena_->leases_.acquire(*ticket);
    e.consumer_recorded = false;
    e.discard = false;
    e.consumer_stream.reset();
    e.producer_stream.reset();
    e.published.store(0,std::memory_order_release);
    e.token_borrowed=false;
  }
  return ticket;
}
void DeviceSlotPool::publish(const SlotTicket& t, const at::Tensor& h, const at::Tensor& r) {
  const auto stream = current_stream();
  const c10::cuda::CUDAGuard guard(device_);
  TORCH_CHECK(role_ == SlotRole::Source && metadata_.state(t) == SlotState::Reserved,
              "publish requires a reserved source ticket");
  TORCH_CHECK(!events_[t.slot_id].producer_stream,
              "borrowed producer views require publish_written, not a second writer");
  const auto& d = metadata_.descriptor(t);
  require_tensor(h, d);
  require_tensor(r, d);
  const auto out = views(t);
  try {
    keep_storage(storage_, stream);
    keep_storage(h, stream);
    keep_storage(r, stream);
    if (d.hidden.valid_bytes) {
      C10_CUDA_CHECK(cudaMemcpyAsync(out.first.data_ptr(), h.const_data_ptr(),
          d.hidden.valid_bytes, cudaMemcpyDeviceToDevice, stream.stream()));
      C10_CUDA_CHECK(cudaMemcpyAsync(out.second.data_ptr(), r.const_data_ptr(),
          d.residual.valid_bytes, cudaMemcpyDeviceToDevice, stream.stream()));
    }
    events_[t.slot_id].ready.record(stream);
    metadata_.producer_recorded(t);
  } catch (...) { poison(); throw; }
}
std::pair<at::Tensor, at::Tensor> DeviceSlotPool::producer_views(const SlotTicket& t) {
  const auto stream = current_stream();
  TORCH_CHECK(role_ == SlotRole::Source && metadata_.state(t) == SlotState::Reserved,
              "producer views require a reserved source ticket");
  auto& e = events_[t.slot_id];
  TORCH_CHECK(!e.producer_stream, "borrow source views only once per ticket");
  try {
    keep_storage(storage_, stream);
    e.producer_stream = stream;
    return views(t);
  } catch (...) { poison(); throw; }
}
void DeviceSlotPool::publish_written(const SlotTicket& t) {
  const auto stream = current_stream();
  TORCH_CHECK(role_ == SlotRole::Source && metadata_.state(t) == SlotState::Reserved,
              "written publication requires a reserved source ticket");
  auto& e = events_[t.slot_id];
  TORCH_CHECK(e.producer_stream && *e.producer_stream == stream,
              "publish once on the bound producer stream after BOTH plane writers");
  TORCH_CHECK(!e.token_borrowed || e.published.load(std::memory_order_acquire)==3,
              "compiled publication must record both current-generation planes");
  try {
    e.ready.record(stream);
    metadata_.producer_recorded(t);
  } catch (...) { poison(); throw; }
}
at::Tensor DeviceSlotPool::publication_token(const SlotTicket& t) {
  const auto stream=current_stream();
  TORCH_CHECK(role_==SlotRole::Source && metadata_.state(t)==SlotState::Reserved,
              "publication token requires reserved source views");
  auto& e=events_[t.slot_id];
  TORCH_CHECK(e.producer_stream && *e.producer_stream==stream && !e.token_borrowed,
              "borrow one publication token on the producer stream");
  auto* p=new Publication{shared_from_this(),t,
      {std::int64_t(t.pool_id),std::int64_t(t.sequence),std::int64_t(t.generation),t.slot_id,0,0}};
  auto token=at::for_blob(p->words.data(),{6}).context(p,&delete_publication)
      .options(at::TensorOptions().device(at::kCPU).dtype(at::kLong)).make_tensor();
  e.token_borrowed=true;
  return token;
}
void DeviceSlotPool::record_plane(const at::Tensor& value,const at::Tensor& token,std::uint32_t plane) {
  auto& p=publication(token);
  auto& pool=*p.pool;
  const auto stream=pool.current_stream();
  TORCH_CHECK(plane<2 && pool.role_==SlotRole::Source &&
              pool.metadata_.state(p.ticket)==SlotState::Reserved,"plane record requires a live reserved source");
  auto& e=pool.events_[p.ticket.slot_id];
  TORCH_CHECK(e.token_borrowed && e.producer_stream && *e.producer_stream==stream,
              "record only on the bound producer stream");
  const auto expected=plane==1?0u:2u;
  TORCH_CHECK(e.published.load(std::memory_order_acquire)==expected,
              "record residual then hidden exactly once for THIS generation");
  const auto pair=pool.views(p.ticket);
  const auto& output=plane==0?pair.first:pair.second;
  TORCH_CHECK(value.is_cuda() && value.get_device()==pool.device_ && value.scalar_type()==at::kBFloat16 &&
      value.is_contiguous() && value.sizes()==output.sizes() && value.const_data_ptr()==output.const_data_ptr(),
      "plane-ready must refer to this ticket's exact output view");
  try {
    e.plane_ready[plane].record(stream);
    // A never-recorded CUDA event can query complete: publish the generation
    // bit only AFTER Record, and keep the source credit until both DMA reads.
    p.words[4]=expected|(1u<<plane);
    e.published.store(std::uint32_t(p.words[4]),std::memory_order_release);
  } catch (...) { pool.poison(); throw; }
}
float DeviceSlotPool::publication_elapsed_ms(const at::Tensor& token) {
  auto& p=publication(token); auto& pool=*p.pool;
  pool.check_owner();
  const auto state=pool.metadata_.state(p.ticket);
  TORCH_CHECK((state==SlotState::Produced || state==SlotState::Transferring) &&
      pool.events_[p.ticket.slot_id].published.load(std::memory_order_acquire)==3,
      "diagnose only a fully published live generation before retirement");
  auto& e=pool.events_[p.ticket.slot_id];
  return e.plane_ready[1].elapsed_time(e.plane_ready[0]);
}
void DeviceSlotPool::loopback(const SlotTicket& t, DeviceSlotPool& dst, const SlotTicket& dt) {
  check_owner();
  dst.check_owner();
  TORCH_CHECK(!backend_claimed_ && !dst.backend_claimed_, "cannot mix loopback and NCCL endpoints");
  const c10::cuda::CUDAGuard guard(device_);
  const auto stream = transfer_stream();
  outside_capture(stream);
  TORCH_CHECK(role_ == SlotRole::Source && dst.role_ == SlotRole::Destination &&
              device_ == dst.device_ && layout_.slot_count == dst.layout_.slot_count &&
              layout_.token_capacity == dst.layout_.token_capacity,
              "loopback needs matched source/destination layouts on one device");
  TORCH_CHECK(metadata_.state(t) == SlotState::Produced &&
              dst.metadata_.state(dt) == SlotState::Reserved,
              "loopback needs produced source and reserved destination");
  const auto& d = metadata_.descriptor(t);
  TORCH_CHECK(encode_boundary(layout_, d) == encode_boundary(dst.layout_, dst.metadata_.descriptor(dt)),
              "loopback endpoints disagree on the wire boundary");
  const auto in = views(t), out = dst.views(dt);
  try {
    keep_storage(storage_, stream);
    dst.keep_storage(dst.storage_, stream);
    events_[t.slot_id].ready.block(stream);
    if (d.hidden.valid_bytes) {
      C10_CUDA_CHECK(cudaMemcpyAsync(out.first.data_ptr(), in.first.const_data_ptr(),
          d.hidden.valid_bytes, cudaMemcpyDeviceToDevice, stream.stream()));
      C10_CUDA_CHECK(cudaMemcpyAsync(out.second.data_ptr(), in.second.const_data_ptr(),
          d.residual.valid_bytes, cudaMemcpyDeviceToDevice, stream.stream()));
    }
    events_[t.slot_id].transfer_done.record(stream);
    dst.events_[dt.slot_id].transfer_done.record(stream);
    metadata_.transfer_recorded(t);
    dst.metadata_.transfer_recorded(dt);
  } catch (...) { poison(); dst.poison(); throw; }
}
std::pair<at::Tensor, at::Tensor> DeviceSlotPool::consume(const SlotTicket& t) {
  const auto stream = current_stream();
  TORCH_CHECK(role_ == SlotRole::Destination && metadata_.state(t) == SlotState::Transferring,
              "consume requires a submitted destination receive");
  auto& e = events_[t.slot_id];
  TORCH_CHECK(!e.discard, "cancelled boundary cannot acquire a consumer");
  try {
    keep_storage(storage_, stream);
    e.transfer_done.block(stream);  // Wait for THIS recorded generation, not host completion.
    e.consumer_stream = stream;
    metadata_.consumer_recorded(t);
    return views(t);
  } catch (...) { poison(); throw; }
}
void DeviceSlotPool::record_consumer_done(const SlotTicket& t) {
  const auto stream = current_stream();
  TORCH_CHECK(role_ == SlotRole::Destination && metadata_.state(t) == SlotState::Consuming,
              "last use requires a bound destination consumer");
  auto& e = events_[t.slot_id];
  TORCH_CHECK(!e.consumer_recorded && e.consumer_stream && *e.consumer_stream == stream,
              "record once on the bound consumer stream, after ALL reads/forwarding copies");
  try {
    e.consumer_done.record(stream);
    e.consumer_recorded = true;
  } catch (...) { poison(); throw; }
}
void DeviceSlotPool::discard_received(const SlotTicket& t) {
  check_owner();
  TORCH_CHECK(role_ == SlotRole::Destination && metadata_.state(t) == SlotState::Transferring,
              "discard requires submitted, unconsumed receive");
  events_[t.slot_id].discard = true;
}
std::uint32_t DeviceSlotPool::progress() {
  check_owner();
  std::uint32_t retired = 0;
  try {
    for (std::uint32_t i = 0; i < layout_.slot_count; ++i) {
      auto& e = events_[i];
      if (!e.ticket) continue;
      const auto t = *e.ticket;
      const auto state = metadata_.state(t);
      if (role_ == SlotRole::Source && state == SlotState::Transferring && e.transfer_done.query()) {
        metadata_.source_read_complete(t);
      } else if (role_ == SlotRole::Destination && state == SlotState::Consuming &&
                 e.consumer_recorded && e.consumer_done.query()) {
        metadata_.consumer_complete(t);
      } else if (role_ == SlotRole::Destination && state == SlotState::Transferring &&
                 e.discard && e.transfer_done.query()) {
        metadata_.discard_received(t);
      } else { continue; }
      arena_->leases_.release(e.physical, t);
      e.ticket.reset();
      e.consumer_stream.reset();
      e.producer_stream.reset();
      ++retired;
    }
  } catch (...) { poison(); throw; }
  return retired;
}
void DeviceSlotPool::close() {
  check_owner();
  TORCH_CHECK(!transport_attached_, "close the drained communicator before its pool");
  TORCH_CHECK(metadata_.live() == 0, "drain all resource-specific completion events before close");
  metadata_.stop_admission();
  storage_ = at::Tensor();
  if (owns_arena_) arena_->close();
  closed_ = true;
}
}  // namespace vpp
