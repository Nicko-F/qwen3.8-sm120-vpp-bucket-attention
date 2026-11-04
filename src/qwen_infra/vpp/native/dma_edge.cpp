// SPDX-License-Identifier: Apache-2.0
#include "dma_edge.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <linux/futex.h>
#include <limits>
#include <mutex>
#include <new>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace vpp {
namespace {
constexpr std::uint64_t kControlBytes = 4096;
constexpr std::uint64_t kMagic = 0x565050444d413031ULL;
struct alignas(64) SegmentControl {
  WireBoundary wire{};
  std::uint64_t offset = 0, bytes = 0;
  alignas(64) std::uint32_t ready = 0;
  alignas(64) std::uint32_t ack = 0;
};
struct alignas(64) Control {
  std::uint64_t magic = kMagic, payload_bytes = 0;
  std::uint64_t segment_bytes = 0;
  std::uint32_t capacity = 0, slots = 0;
  WireBoundary wire{};
  alignas(64) std::uint32_t ready = 0;
  alignas(64) std::uint32_t ack = 0;
  alignas(64) std::uint32_t failed = 0;
  std::array<SegmentControl,DmaEdge::kSegments> segments{};
};
static_assert(sizeof(Control) <= kControlBytes);
static_assert(__atomic_always_lock_free(sizeof(std::uint32_t), nullptr));
std::uint32_t acquire(std::uint32_t* value) { return __atomic_load_n(value, __ATOMIC_ACQUIRE); }
void publish(std::uint32_t* value, std::uint32_t sequence) noexcept {
  __atomic_store_n(value, sequence, __ATOMIC_RELEASE);
  // Shared futex (not FUTEX_PRIVATE), no CPU spins or per-segment Python.
  (void)::syscall(SYS_futex, value, FUTEX_WAKE, 1, nullptr, nullptr, 0);
}
std::uint64_t region_bytes(BoundaryLayout layout, std::uint64_t segment_bytes = 0) {
  layout.validate();
  TORCH_CHECK(segment_bytes==0 || segment_bytes==4*1024*1024 ||
      segment_bytes==8*1024*1024 || segment_bytes==16*1024*1024,"unsupported DMA segment size");
  const auto bytes = segment_bytes ? segment_bytes*DmaEdge::kSegments : layout.slot_bytes();
  TORCH_CHECK(::sysconf(_SC_PAGESIZE) == kControlBytes, "DMA control layout requires4KiB pages");
  return kControlBytes + ((bytes+kControlBytes-1)/kControlBytes)*kControlBytes;
}
void check_stream(cudaStream_t stream) {
  const auto status = cudaStreamQuery(stream);
  if (status != cudaErrorNotReady) C10_CUDA_CHECK(status);
}
}  // namespace

struct DmaEdge::State {
  struct Job {
    enum Phase { Empty, Queued, Recorded } phase = Empty;
    SlotTicket ticket{};
    BoundaryDesc desc{};
    WireBoundary wire{};
    char* device = nullptr;
    cudaEvent_t ready = nullptr, done = nullptr;
    std::array<cudaEvent_t,2> plane_ready{};
    const std::atomic<std::uint32_t>* published = nullptr;
    bool independent = false, plane_started = false;
    std::uint32_t plane = 0;
    std::uint64_t position = 0;
    bool started = false;
    std::chrono::steady_clock::time_point admitted;
  };
  std::shared_ptr<DeviceSlotPool> pool;
  std::shared_ptr<HostRegion> region;
  c10::cuda::CUDAStream stream;
  Control* control;
  void* payload;
  cudaEvent_t local = nullptr, peer = nullptr;
  std::array<cudaEvent_t,kSegments> local_segments{}, peer_segments{};
  std::array<Job,7> jobs{};
  DmaProgress* engine = nullptr;
  std::uint64_t segment_bytes = 0, next_segment = 0, next_job = 0;
  bool source = false;
  bool registered = false;
  State(std::shared_ptr<DeviceSlotPool> p, std::shared_ptr<HostRegion> r,
        c10::cuda::CUDAStream s) : pool(std::move(p)), region(std::move(r)), stream(s),
        control(static_cast<Control*>(region->range(0,kControlBytes))),
        payload(region->range(kControlBytes,region->handle().bytes-kControlBytes)) {}
  ~State() noexcept {
    // Only cold construction rollback or explicit, already-drained close.
    const c10::cuda::OptionalCUDAGuard guard(stream.device_index());
    if (peer) C10_CUDA_CHECK_WARN(cudaEventDestroy(peer));
    if (local) C10_CUDA_CHECK_WARN(cudaEventDestroy(local));
    for(auto e:peer_segments) if(e) C10_CUDA_CHECK_WARN(cudaEventDestroy(e));
    for(auto e:local_segments) if(e) C10_CUDA_CHECK_WARN(cudaEventDestroy(e));
    if (registered) C10_CUDA_CHECK_WARN(cudaHostUnregister(payload));
  }
};

// Pattern reference: NCCL v2.26.2-1 src/transport/shm.cc,
// shm{Send,Recv}ProxyProgress: bounded ring, nonblocking copy/event progress.
// Adaptation, not NCCL's internal GPU FIFO or a vendored NCCL transport.
// Here IPC Record publication allows the receiver to queue a device wait;
// host-slot reuse additionally requires the peer's H2D completion.
struct DmaProgress::Impl {
  using State = DmaEdge::State;
  using Job = State::Job;
  explicit Impl(c10::DeviceIndex d) : device(d) {}
  c10::DeviceIndex device;
  std::array<State*,4> edges{};
  std::size_t edge_count = 0;
  std::mutex mutex;
  std::condition_variable work, completed;
  std::thread worker;
  bool started = false, stopping = false;
  std::exception_ptr error;
  Stats counters;

  static bool event_ready(cudaEvent_t event) {
    const auto result = cudaEventQuery(event);
    if(result==cudaErrorNotReady) return false;
    C10_CUDA_CHECK(result); return true;
  }
  static bool queued(const State& s) {
    return std::any_of(s.jobs.begin(),s.jobs.end(),[](const Job& j){return j.phase==Job::Queued;});
  }
  void check() const {
    if(error) std::rethrow_exception(error);
    TORCH_CHECK(started && !stopping,"DMA progress engine is not running");
  }
  bool advance(State& s, Job& j) {
    TORCH_CHECK(!acquire(&s.control->failed),"segmented DMA peer aborted");
    TORCH_CHECK(std::chrono::steady_clock::now()-j.admitted<std::chrono::seconds(30),
                "segmented DMA job deadline expired");
    const auto plane=j.independent?1-j.plane:j.plane;
    if(s.source) {
      if(j.independent && !j.plane_started) {
        if(!(j.published->load(std::memory_order_acquire)&(1u<<plane)) ||
            !event_ready(j.plane_ready[plane])) return false;
        j.plane_started=true;
      } else if(!j.independent && !j.started && !event_ready(j.ready)) return false;
    }
    TORCH_CHECK(s.next_segment<std::numeric_limits<std::uint32_t>::max(),
                "DMA segment sequence exhausted; start a new drained epoch");
    const auto index=s.next_segment%DmaEdge::kSegments;
    const auto sequence=static_cast<std::uint32_t>(s.next_segment+1);
    auto& control=s.control->segments[index];
    if(s.source) {
      if(s.next_segment>=DmaEdge::kSegments) {
        const auto expected=sequence-DmaEdge::kSegments, seen=acquire(&control.ack);
        TORCH_CHECK(seen<=expected,"DMA segment ack overtook its generation");
        if(seen!=expected || !event_ready(s.peer_segments[index])) return false;
      }
    } else {
      const auto seen=acquire(&control.ready);
      TORCH_CHECK(seen<=sequence,"DMA segment ready overtook its generation");
      if(seen!=sequence) return false;
    }
    const auto& span=plane==0?j.desc.hidden:j.desc.residual;
    const auto bytes=std::min(s.segment_bytes,span.valid_bytes-j.position);
    const auto offset=span.offset_bytes+j.position;
    auto* host=static_cast<char*>(s.payload)+index*s.segment_bytes;
    if(!s.source) {
      TORCH_CHECK(control.wire==j.wire && control.offset==offset && control.bytes==bytes,
                  "DMA segment boundary/offset/length mismatch");
      // ready acquire proves this event's Record was submitted. The producer
      // cannot re-record it until our ack AND this generation's H2D completion.
      C10_CUDA_CHECK(cudaStreamWaitEvent(s.stream.stream(),s.peer_segments[index],0));
    }
    if(bytes) C10_CUDA_CHECK(cudaMemcpyAsync(s.source?host:j.device+offset,
        s.source?j.device+offset:host,bytes,
        s.source?cudaMemcpyDeviceToHost:cudaMemcpyHostToDevice,s.stream.stream()));
    C10_CUDA_CHECK(cudaEventRecord(s.local_segments[index],s.stream.stream()));
    if(s.source) { control.wire=j.wire; control.offset=offset; control.bytes=bytes; }
    publish(s.source?&control.ready:&control.ack,sequence);
    ++s.next_segment; ++counters.segments;
    j.started=true; j.position+=bytes;
    // Two noncontiguous valid prefixes; never copy capacity padding. A zero
    // boundary still consumes one ordered control/event segment.
    if(j.position==span.valid_bytes) {
      ++j.plane; j.position=0; j.plane_started=false;
      if(!j.independent && j.plane==1 && !j.desc.residual.valid_bytes) ++j.plane;
    }
    if(j.plane==2) {
      C10_CUDA_CHECK(cudaEventRecord(j.done,s.stream.stream()));
      j.phase=Job::Recorded; ++s.next_job;
      completed.notify_all();
    }
    return true;
  }
  void loop() noexcept {
    timespec start{}, finish{};
    (void)clock_gettime(CLOCK_THREAD_CPUTIME_ID,&start);
    try {
      const c10::cuda::CUDAGuard guard(device);
      counters.tid=static_cast<int>(::syscall(SYS_gettid));
      for(;;) {
        // One segment per eligible edge per pass; no head-of-line blocking on
        // another edge's GPU producer/remote credit. Only CUDA handles/pointers,
        // never DeviceSlotPool metadata, are used on this thread.
        std::unique_lock lock(mutex);
        if(stopping) break;
        bool pending=false, advanced=false;
        for(std::size_t i=0;i<edge_count;++i) {
          auto& s=*edges[i];
          for(auto& j:s.jobs) if(j.phase==Job::Queued && j.ticket.sequence==s.next_job) {
            pending=true; advanced=advance(s,j)||advanced; break;
          }
        }
        if(!advanced) {
          ++counters.idle_waits;
          if(pending) work.wait_for(lock,std::chrono::microseconds(50));
          else work.wait(lock,[&]{
            if(stopping) return true;
            for(std::size_t i=0;i<edge_count;++i) if(queued(*edges[i])) return true;
            return false;
          });
        }
      }
    } catch(...) {
      std::lock_guard lock(mutex);
      error=std::current_exception(); stopping=true;
      for(std::size_t i=0;i<edge_count;++i) publish(&edges[i]->control->failed,1);
      completed.notify_all();
    }
    (void)clock_gettime(CLOCK_THREAD_CPUTIME_ID,&finish);
    counters.cpu_seconds=(finish.tv_sec-start.tv_sec)+(finish.tv_nsec-start.tv_nsec)/1e9;
  }
};

DmaProgress::DmaProgress(c10::DeviceIndex device) : impl_(std::make_unique<Impl>(device)) {}
DmaProgress::~DmaProgress() noexcept { stop(); }
void DmaProgress::attach(DmaEdge::State* state) {
  std::lock_guard lock(impl_->mutex);
  TORCH_CHECK(!impl_->started && impl_->edge_count<impl_->edges.size() &&
      state->stream.device_index()==impl_->device,"attach at most four local edges before start");
  impl_->edges[impl_->edge_count++]=state;
}
void DmaProgress::start() {
  std::lock_guard lock(impl_->mutex);
  TORCH_CHECK(!impl_->started && !impl_->stopping && impl_->edge_count,"start DMA progress exactly once");
  for(std::size_t i=0;i<impl_->edge_count;++i)
    for(auto e:impl_->edges[i]->peer_segments) TORCH_CHECK(e,"connect all segment events before start");
  impl_->started=true;
  impl_->worker=std::thread([this]{impl_->loop();});
}
void DmaProgress::stop() noexcept {
  { std::lock_guard lock(impl_->mutex); impl_->stopping=true; }
  impl_->work.notify_all(); impl_->completed.notify_all();
  if(impl_->worker.joinable()) impl_->worker.join();
}
DmaProgress::Stats DmaProgress::stats() const {
  TORCH_CHECK(!impl_->worker.joinable(),"read progress counters after stop");
  if(impl_->error) std::rethrow_exception(impl_->error);
  return impl_->counters;
}
void DmaProgress::submit(DmaEdge::State& s,const SlotTicket& t,const BoundaryDesc& d,
    const WireBoundary& wire,char* device,cudaEvent_t ready,cudaEvent_t done,
    bool independent,std::array<cudaEvent_t,2> planes,const std::atomic<std::uint32_t>* published) {
  std::lock_guard lock(impl_->mutex); impl_->check();
  auto& j=s.jobs.at(t.slot_id);
  TORCH_CHECK(j.phase==Impl::Job::Empty,"DMA fixed job slot has not been harvested");
  j={}; j.ticket=t; j.desc=d; j.wire=wire; j.device=device; j.ready=ready; j.done=done;
  j.independent=independent; j.plane_ready=planes; j.published=published;
  TORCH_CHECK(!independent || !s.source || published,"independent producer needs generation-ready flags");
  j.admitted=std::chrono::steady_clock::now(); j.phase=Impl::Job::Queued;
  std::uint64_t jobs=0;
  for(std::size_t i=0;i<impl_->edge_count;++i)
    for(const auto& job:impl_->edges[i]->jobs) jobs+=job.phase!=Impl::Job::Empty;
  impl_->counters.peak_jobs=std::max(impl_->counters.peak_jobs,jobs);
  impl_->work.notify_one();
}
std::array<std::optional<SlotTicket>,7> DmaProgress::collect(DmaEdge::State& s,bool wait,
                                                          const SlotTicket* target,std::uint32_t timeout_ms) {
  std::unique_lock lock(impl_->mutex); impl_->check();
  if(wait) {
    const auto ready=impl_->completed.wait_for(lock,std::chrono::milliseconds(timeout_ms),[&]{
      if(impl_->error || impl_->stopping) return true;
      if(!target) return !Impl::queued(s);
      const auto& j=s.jobs.at(target->slot_id);
      TORCH_CHECK(j.ticket.pool_id==target->pool_id && j.ticket.sequence==target->sequence &&
                  j.ticket.generation==target->generation,"record-ready job identity mismatch");
      return j.phase==Impl::Job::Recorded;
    });
    impl_->check(); TORCH_CHECK(ready,"DMA completion-publication deadline expired");
  }
  std::array<std::optional<SlotTicket>,7> tickets{};
  for(std::size_t i=0;i<s.jobs.size();++i) if(s.jobs[i].phase==Impl::Job::Recorded) {
    tickets[i]=s.jobs[i].ticket; s.jobs[i].phase=Impl::Job::Empty;
  }
  return tickets;
}

std::shared_ptr<HostRegion> DmaEdge::create_region(BoundaryLayout layout, int node,
                                                 std::uint64_t segment_bytes,
                                                 std::uint64_t minimum_available_bytes) {
  return HostRegion::create(region_bytes(layout,segment_bytes),node,minimum_available_bytes);
}
DmaEdge::DmaEdge(std::shared_ptr<DeviceSlotPool> pool, std::shared_ptr<HostRegion> region,
                 c10::cuda::CUDAStream stream, std::uint64_t segment_bytes, DmaProgress* engine) {
  TORCH_CHECK(pool && region, "DMA needs a fixed pool and shared backing");
  pool->current_stream();
  TORCH_CHECK(!pool->backend_claimed_ && pool->live()==0 && pool->peak_live()==0,
              "attach DMA exactly once to a fresh fixed pool");
  TORCH_CHECK(stream.device_index()==pool->device_ &&
              region->handle().bytes==region_bytes(pool->layout_,segment_bytes), "DMA layout/device mismatch");
  TORCH_CHECK((segment_bytes!=0)==(engine!=nullptr),"segmented DMA requires one rank progress engine");
  const c10::cuda::CUDAGuard guard(pool->device_);
  int ipc = 0, host = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&ipc,cudaDevAttrIpcEventSupport,pool->device_));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&host,cudaDevAttrHostRegisterSupported,pool->device_));
  TORCH_CHECK(ipc && host, "CUDA IPC events/host registration unavailable; no pageable fallback");
  state_ = std::make_unique<State>(pool,std::move(region),stream);
  auto& s=*state_;
  s.source=pool->role_==SlotRole::Source;
  s.segment_bytes=segment_bytes; s.engine=engine;
  if(pool->role_==SlotRole::Source) {
    TORCH_CHECK(s.region->handle().owner_pid==::getpid(),"source must own its shared region");
    new(s.control) Control;
    s.control->payload_bytes=pool->layout_.slot_bytes();
    s.control->capacity=pool->layout_.token_capacity;
    s.control->slots=pool->layout_.slot_count;
    s.control->segment_bytes=segment_bytes;
  } else {
    TORCH_CHECK(s.control->magic==kMagic && s.control->payload_bytes==pool->layout_.slot_bytes() &&
        s.control->capacity==pool->layout_.token_capacity && s.control->slots==pool->layout_.slot_count &&
        s.control->segment_bytes==segment_bytes,
        "shared DMA control is not this fixed layout");
  }
  C10_CUDA_CHECK(cudaHostRegister(s.payload,s.region->handle().bytes-kControlBytes,
                                 cudaHostRegisterPortable));
  s.registered=true;
  // Registration faults the peer mapping too. NUMA placement is reported by
  // numa_receipt(); exact locality is a deployment/performance choice.
  if(engine) {
    for(auto& e:s.local_segments)
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&e,cudaEventDisableTiming|cudaEventInterprocess));
    engine->attach(&s);
  } else C10_CUDA_CHECK(cudaEventCreateWithFlags(&s.local,cudaEventDisableTiming|cudaEventInterprocess));
  pool->backend_claimed_=pool->transport_attached_=true;
}
void DmaEdge::check() const {
  TORCH_CHECK(state_ && !failed_ && !disconnected_, "DMA edge failed/disconnected/closed");
  state_->pool->current_stream();
  TORCH_CHECK(!acquire(&state_->control->failed), "DMA peer aborted this role epoch");
}
std::string DmaEdge::export_event() const {
  check(); cudaIpcEventHandle_t handle{};
  if(state_->engine) {
    std::string bytes;
    for(auto e:state_->local_segments) {
      C10_CUDA_CHECK(cudaIpcGetEventHandle(&handle,e));
      bytes.append(reinterpret_cast<const char*>(&handle),sizeof(handle));
    }
    return bytes;
  }
  C10_CUDA_CHECK(cudaIpcGetEventHandle(&handle,state_->local));
  return std::string(reinterpret_cast<const char*>(&handle),sizeof(handle));
}
void DmaEdge::connect(const std::string& bytes) {
  check(); const auto count=state_->engine?kSegments:1;
  TORCH_CHECK(!connected_ && bytes.size()==count*sizeof(cudaIpcEventHandle_t),
                      "expected matching peer CUDA event handles");
  const c10::cuda::CUDAGuard guard(state_->pool->device_);
  for(std::uint32_t i=0;i<count;++i) {
    cudaIpcEventHandle_t handle{}; std::memcpy(&handle,bytes.data()+i*sizeof(handle),sizeof(handle));
    C10_CUDA_CHECK(cudaIpcOpenEventHandle(state_->engine?&state_->peer_segments[i]:&state_->peer,handle));
  }
  connected_=true;
}
void DmaEdge::wait_sequence(std::uint32_t* value,std::uint32_t wanted) {
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
  for (;;) {
    check();
    const auto seen=acquire(value);
    TORCH_CHECK(seen<=wanted,"DMA peer sequence overtook the expected generation");
    if(seen==wanted) return;
    check_stream(state_->stream.stream());
    TORCH_CHECK(std::chrono::steady_clock::now()<deadline,"DMA record publication deadline expired");
    const timespec timeout{0,50000000};
    const auto rc=::syscall(SYS_futex,value,FUTEX_WAIT,seen,&timeout,nullptr,0);
    TORCH_CHECK(rc==0 || errno==EAGAIN || errno==EINTR || errno==ETIMEDOUT,
                "shared futex wait failed: ",std::strerror(errno));
  }
}
SlotTicket DmaEdge::reserve(const WireBoundary& wire,std::uint32_t timeout_ms) {
  check(); TORCH_CHECK(timeout_ms>0 && timeout_ms<=30000,"bounded DMA credit timeout required");
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
  for (;;) {
    progress();
    if(auto t=state_->pool->reserve(wire)) return *t;
    TORCH_CHECK(std::chrono::steady_clock::now()<deadline,"DMA GPU slot credit deadline expired");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
void DmaEdge::submit(const SlotTicket& t,bool independent_ready) {
  check(); TORCH_CHECK(connected_,"connect both DMA endpoints before submit");
  auto& s=*state_; auto& pool=*s.pool;
  const bool source=pool.role_==SlotRole::Source;
  const auto status=pool.metadata_.state(t);
  TORCH_CHECK(t.sequence==next_sequence_ && t.sequence<std::numeric_limits<std::uint32_t>::max() &&
      (source ? status==SlotState::Produced || (independent_ready && status==SlotState::Reserved)
              : status==SlotState::Reserved),
      "DMA submission sequence/state mismatch");
  TORCH_CHECK(!independent_ready || (s.engine && (!source ||
      (pool.events_[t.slot_id].token_borrowed && pool.events_[t.slot_id].producer_stream))),
      "independent ready requires segmented DMA and a bound producer capability");
  const auto& d=pool.metadata_.descriptor(t);
  const auto wire=encode_boundary(pool.layout_,d);
  const c10::cuda::CUDAGuard guard(pool.device_);
  try {
    pool.keep_storage(pool.storage_,s.stream);
    if(s.engine) {
      auto* device=static_cast<char*>(pool.storage_.data_ptr())+t.slot_id*pool.layout_.slot_bytes();
      s.engine->submit(s,t,d,wire,device,pool.events_[t.slot_id].ready.event(),
                       pool.events_[t.slot_id].transfer_done.event(),independent_ready,
                       {pool.events_[t.slot_id].plane_ready[0].event(),pool.events_[t.slot_id].plane_ready[1].event()},
                       independent_ready && source?&pool.events_[t.slot_id].published:nullptr);
      ++next_sequence_; return;
    }
    if(source) {
      if(t.sequence) {
        wait_sequence(&s.control->ack,static_cast<std::uint32_t>(t.sequence));
        C10_CUDA_CHECK(cudaStreamWaitEvent(s.stream.stream(),s.peer,0));
      }
      pool.events_[t.slot_id].ready.block(s.stream);
    } else {
      wait_sequence(&s.control->ready,static_cast<std::uint32_t>(t.sequence+1));
      TORCH_CHECK(s.control->wire==wire,"DMA peers disagree on the96B boundary");
      C10_CUDA_CHECK(cudaStreamWaitEvent(s.stream.stream(),s.peer,0));
    }
    auto* device=static_cast<char*>(pool.storage_.data_ptr())+t.slot_id*pool.layout_.slot_bytes();
    auto* host=static_cast<char*>(s.payload);
    for(const auto& span:{d.hidden,d.residual}) if(span.valid_bytes) {
      C10_CUDA_CHECK(cudaMemcpyAsync(source?host+span.offset_bytes:device+span.offset_bytes,
          source?device+span.offset_bytes:host+span.offset_bytes,span.valid_bytes,
          source?cudaMemcpyDeviceToHost:cudaMemcpyHostToDevice,s.stream.stream()));
    }
    pool.events_[t.slot_id].transfer_done.record(s.stream);
    C10_CUDA_CHECK(cudaEventRecord(s.local,s.stream.stream()));
    pool.metadata_.transfer_recorded(t);
    if(source) s.control->wire=wire;
    publish(source?&s.control->ready:&s.control->ack,static_cast<std::uint32_t>(t.sequence+1));
    ++next_sequence_;
  } catch (...) { abort(); throw; }
}
void DmaEdge::harvest(bool wait) {
  if(!state_->engine) return;
  for(const auto& t:state_->engine->collect(*state_,wait))
    if(t) state_->pool->metadata_.transfer_recorded(*t);
}
void DmaEdge::wait_recorded(const SlotTicket& ticket,std::uint32_t timeout_ms) {
  check(); auto& s=*state_;
  TORCH_CHECK(timeout_ms>0 && timeout_ms<=30000 && s.pool->role_==SlotRole::Destination,
              "bounded destination record-ready wait required");
  const auto status=s.pool->metadata_.state(ticket);
  if(status==SlotState::Transferring) return;
  TORCH_CHECK(s.engine && status==SlotState::Reserved,"submit this receive before record-ready wait");
  for(const auto& t:s.engine->collect(s,true,&ticket,timeout_ms))
    if(t) s.pool->metadata_.transfer_recorded(*t);
  TORCH_CHECK(s.pool->metadata_.state(ticket)==SlotState::Transferring,"receive event was not recorded");
}
void DmaEdge::set_receive_stream(c10::cuda::CUDAStream stream) {
  check(); auto& s=*state_;
  TORCH_CHECK(s.engine && !s.source && s.pool->live()==0 &&
              stream.device_index()==s.pool->device_, "rebind only a drained segmented receiver");
  const c10::cuda::CUDAGuard guard(s.pool->device_);
  std::lock_guard lock(s.engine->impl_->mutex);
  s.engine->impl_->check();
  for(const auto& j:s.jobs) TORCH_CHECK(j.phase==State::Job::Empty,"harvest jobs before stream rebind");
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&capture));
  TORCH_CHECK(capture==cudaStreamCaptureStatusNone,"receive policy change is outside capture");
  C10_CUDA_CHECK(cudaStreamQuery(s.stream.stream()));
  C10_CUDA_CHECK(cudaStreamQuery(stream.stream()));
  s.stream=stream;
}
void DmaEdge::mark(c10::cuda::CUDAEvent& event) { check(); harvest(true); event.record(state_->stream); }
std::uint32_t DmaEdge::progress() {
  check(); harvest(false); check_stream(state_->stream.stream()); return state_->pool->progress();
}
std::uint64_t DmaEdge::host_bytes() const { check(); return state_->region->handle().bytes; }
std::string DmaEdge::numa_receipt() const { check(); return state_->region->numa_receipt(); }
void DmaEdge::disconnect() {
  check(); auto& s=*state_;
  TORCH_CHECK(connected_ && s.pool->live()==0,"drain GPU slots before DMA disconnect");
  const c10::cuda::CUDAGuard guard(s.pool->device_);
  if(s.engine) {
    TORCH_CHECK(!s.engine->impl_->worker.joinable(),
                "stop the rank progress engine before disconnecting any attached edge");
    for(std::uint32_t i=0;i<kSegments;++i) {
      // Both peers have drained before this cold phase. Query recorded events only.
      if(s.next_segment>i) {
        C10_CUDA_CHECK(cudaEventQuery(s.local_segments[i]));
        C10_CUDA_CHECK(cudaEventQuery(s.peer_segments[i]));
      }
      C10_CUDA_CHECK(cudaEventDestroy(s.peer_segments[i])); s.peer_segments[i]=nullptr;
    }
    disconnected_=true; return;
  }
  if(next_sequence_) {
    if(s.pool->role_==SlotRole::Source) wait_sequence(&s.control->ack,static_cast<std::uint32_t>(next_sequence_));
    C10_CUDA_CHECK(cudaEventQuery(s.local));
    C10_CUDA_CHECK(cudaEventQuery(s.peer));
  }
  C10_CUDA_CHECK(cudaEventDestroy(s.peer)); s.peer=nullptr; disconnected_=true;
}
void DmaEdge::close() {
  TORCH_CHECK(state_ && disconnected_ && !failed_,"disconnect peers before closing DMA exports");
  state_->pool->check_owner();
  TORCH_CHECK(state_->pool->live()==0,"cannot close live DMA slots");
  const c10::cuda::CUDAGuard guard(state_->pool->device_);
  if(state_->local) { C10_CUDA_CHECK(cudaEventDestroy(state_->local)); state_->local=nullptr; }
  for(auto& e:state_->local_segments) if(e) { C10_CUDA_CHECK(cudaEventDestroy(e)); e=nullptr; }
  C10_CUDA_CHECK(cudaHostUnregister(state_->payload)); state_->registered=false;
  state_->pool->transport_attached_=false;
  state_.reset();
}
void DmaEdge::abort() noexcept {
  if(!state_) return;
  failed_=true;
  publish(&state_->control->failed,1);
  (void)::syscall(SYS_futex,&state_->control->ready,FUTEX_WAKE,1,nullptr,nullptr,0);
  (void)::syscall(SYS_futex,&state_->control->ack,FUTEX_WAKE,1,nullptr,nullptr,0);
  state_->pool->poison();
}
DmaEdge::~DmaEdge() noexcept {
  if(!state_) return;
  if(state_->engine) state_->engine->stop();
  abort();
  // A failed role cannot continue. Keep potentially in-flight host/GPU storage
  // and exported events alive until the owned supervisor terminates the process.
  // Normal operation MUST call disconnect/barrier/close and does not quarantine.
  if(connected_) { (void)state_.release(); TORCH_WARN("fatal DMA epoch: resources retained until process exit"); }
}

}  // namespace vpp
