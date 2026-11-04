// SPDX-License-Identifier: Apache-2.0
#include "kv_copy.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <dlfcn.h>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace vpp {
namespace {
// Same driver ABI and source-order contract as vLLM a9fe30e9,
// csrc/cache_kernels.cu::swap_blocks_batch (Apache-2.0, vLLM contributors).
// Resolve the 12.8 signature even with CUDA13 headers (_v2 is a different ABI).
using BatchFn = CUresult (*)(CUdeviceptr*, CUdeviceptr*, size_t*, size_t,
                            CUmemcpyAttributes*, size_t*, size_t, size_t*, CUstream);
using Batch3DFn = CUresult (*)(size_t, CUDA_MEMCPY3D_BATCH_OP*, unsigned long long, CUstream);
Batch3DFn pitched_function() {
  static Batch3DFn result=[] {
    void* library=::dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    TORCH_CHECK(library,"CUDA driver library unavailable");
    using GetProc=CUresult (*)(const char*,void**,int,cuuint64_t,CUdriverProcAddressQueryResult*);
    auto get=reinterpret_cast<GetProc>(::dlsym(library,"cuGetProcAddress_v2"));
    TORCH_CHECK(get,"CUDA driver entry-point resolver unavailable");
    void* fn=nullptr; CUdriverProcAddressQueryResult status{};
    // CUDA13 removed failIdx from this API. Resolve its explicit13.0 ABI,
    // never call the12.8 five-argument entry through a four-argument pointer.
    TORCH_CHECK(get("cuMemcpy3DBatchAsync",&fn,13000,CU_GET_PROC_ADDRESS_DEFAULT,&status)==CUDA_SUCCESS && fn,
                "CUDA13 strided batch DMA API unavailable");
    return reinterpret_cast<Batch3DFn>(fn);
  }();
  return result;
}
BatchFn batch_function() {
  static BatchFn result = [] {
    void* library = ::dlopen("libcuda.so.1", RTLD_NOW|RTLD_LOCAL);
    TORCH_CHECK(library, "CUDA driver library unavailable");
    using GetProc = CUresult (*)(const char*, void**, int, cuuint64_t,
                                 CUdriverProcAddressQueryResult*);
    auto get = reinterpret_cast<GetProc>(::dlsym(library, "cuGetProcAddress_v2"));
    TORCH_CHECK(get, "CUDA driver entry-point resolver unavailable");
    void* fn = nullptr;
    CUdriverProcAddressQueryResult status{};
    TORCH_CHECK(get("cuMemcpyBatchAsync", &fn, 12080, CU_GET_PROC_ADDRESS_DEFAULT,
                    &status)==CUDA_SUCCESS && fn, "Native batch DMA API unavailable");
    // Process-lifetime function pointer; do not unload the driver's library.
    return reinterpret_cast<BatchFn>(fn);
  }();
  return result;
}
bool ready(cudaEvent_t event) {
  const auto status = cudaEventQuery(event);
  if(status==cudaErrorNotReady) return false;
  C10_CUDA_CHECK(status); return true;
}
} // namespace

struct KvCopy::Impl {
  struct Span { std::uint64_t tensor, gpu, host, bytes, fragments, gpu_stride, host_stride; };
  struct Job {
    enum Phase { Empty, Queued, Copying, Done } phase = Empty;
    std::uint64_t id=0, generation=0, bytes=0, segments=0;
    struct Unit {
      std::vector<Span> spans;
      std::size_t cursor=0;
      std::uint64_t fragment=0, offset=0, bytes=0, done=0;
      std::int64_t gate=-1, epoch=0;
    };
    std::vector<Unit> units;
    std::vector<std::pair<std::size_t,std::uint64_t>> issued;
    std::uint64_t issued_bytes=0, first_issue_ns=0, last_done_ns=0;
    bool store=false, cancelled=false, started=false;
    cudaEvent_t dependency=nullptr, begin=nullptr, segment=nullptr;
    // Remain valid until this segment's event completes; no descriptor UAF.
    std::vector<CUdeviceptr> source, destination;
    std::vector<size_t> sizes;
    std::vector<CUDA_MEMCPY3D_BATCH_OP> boxes;
    std::chrono::steady_clock::time_point admitted;
  };
  c10::DeviceIndex device;
  std::vector<at::Tensor> gpu;
  at::Tensor host, flags;
  c10::cuda::CUDAStream store_stream, load_stream;
  std::uint64_t segment_bytes, last_id=0;
  std::uint32_t max_spans;
  bool pitched;
  std::vector<Job> jobs;
  std::mutex mutex;
  std::condition_variable work, completed;
  std::thread worker;
  std::exception_ptr error;
  bool stopping=false, closed=false;

  Impl(c10::DeviceIndex d, std::vector<at::Tensor> g, at::Tensor h,
       std::uint64_t segment, std::uint32_t count, std::uint32_t spans,bool use_pitched)
      : device(d), gpu(std::move(g)), host(std::move(h)),
        store_stream(c10::cuda::getStreamFromPool(false,d)),
        load_stream(c10::cuda::getStreamFromPool(false,d)),
        segment_bytes(segment), max_spans(spans), pitched(use_pitched), jobs(count) {}
  void check() const {
    if(error) std::rethrow_exception(error);
    TORCH_CHECK(!closed && !stopping, "KV copy engine is closed/stopping");
  }
  void finish(Job& job) { job.phase=Job::Done; completed.notify_all(); }
  static std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  bool eligible(const Job::Unit& unit, bool store) const {
    return unit.cursor<unit.spans.size() && (store || unit.gate<0 ||
        __atomic_load_n(flags.const_data_ptr<std::int64_t>()+unit.gate,__ATOMIC_ACQUIRE)==unit.epoch);
  }
  bool advance(Job& j) {
    TORCH_CHECK(std::chrono::steady_clock::now()-j.admitted<std::chrono::seconds(120),
                "KV copy job deadline expired");
    if(j.phase==Job::Copying) {
      if(!ready(j.segment)) return false;
      j.bytes+=j.issued_bytes; ++j.segments; j.phase=Job::Queued;
      j.last_done_ns=now_ns();
      for(const auto& [index,bytes]:j.issued) {
        auto& u=j.units[index]; u.done+=bytes;
        TORCH_CHECK(u.done<=u.bytes,"Canonical page completion overflow");
        if(j.store && u.gate>=0 && u.done==u.bytes)
          __atomic_store_n(flags.mutable_data_ptr<std::int64_t>()+u.gate,u.epoch,__ATOMIC_RELEASE);
      }
    }
    if(j.phase!=Job::Queued) return false;
    // A cancellation only stops work that was NOT submitted. In-flight copies
    // must finish before this slot/host row/native GPU blocks may retire.
    if(j.cancelled || std::all_of(j.units.begin(),j.units.end(),[](const auto& u) {
          return u.cursor==u.spans.size();
        })) { finish(j); return true; }
    if(!ready(j.dependency)) return false;
    if(std::none_of(j.units.begin(),j.units.end(),[&](const auto& u) {
          return eligible(u,j.store);
        })) return false;
    const auto stream=(j.store?store_stream:load_stream).stream();
    j.source.clear(); j.destination.clear(); j.sizes.clear(); j.boxes.clear();
    j.issued.clear();
    std::uint64_t budget=segment_bytes;
    while(budget && (pitched?j.boxes.size():j.sizes.size())<max_spans) {
      // Ready FA pages may bypass a pending final GDN unit. A single cursor
      // over native groups would otherwise serialize all H2D behind GDN.
      auto found=std::find_if(j.units.begin(),j.units.end(),[&](const auto& u) {
        return eligible(u,j.store);
      });
      if(found==j.units.end()) break;
      auto& unit=*found;
      const auto index=std::size_t(found-j.units.begin());
      const auto& span=unit.spans[unit.cursor];
      const auto width=std::min(budget,span.bytes-unit.offset);
      const auto rows=(pitched && unit.offset==0 && budget>=span.bytes)
          ?std::min(span.fragments-unit.fragment,budget/span.bytes):1;
      const auto bytes=width*rows;
      const auto device_ptr=reinterpret_cast<CUdeviceptr>(gpu[span.tensor].data_ptr())+
          span.gpu+unit.fragment*span.gpu_stride+unit.offset;
      const auto host_ptr=reinterpret_cast<CUdeviceptr>(host.data_ptr())+
          span.host+unit.fragment*span.host_stride+unit.offset;
      if(pitched) {
        CUDA_MEMCPY3D_BATCH_OP op{};
        op.src.type=op.dst.type=CU_MEMCPY_OPERAND_TYPE_POINTER;
        op.src.op.ptr.ptr=j.store?device_ptr:host_ptr;
        op.dst.op.ptr.ptr=j.store?host_ptr:device_ptr;
        op.src.op.ptr.rowLength=rows>1?(j.store?span.gpu_stride:span.host_stride):0;
        op.dst.op.ptr.rowLength=rows>1?(j.store?span.host_stride:span.gpu_stride):0;
        op.extent={width,rows,1};
        op.srcAccessOrder=j.store?CU_MEMCPY_SRC_ACCESS_ORDER_STREAM:CU_MEMCPY_SRC_ACCESS_ORDER_ANY;
        j.boxes.push_back(op);
      } else {
        j.source.push_back(j.store?device_ptr:host_ptr);
        j.destination.push_back(j.store?host_ptr:device_ptr);
        j.sizes.push_back(bytes);
      }
      if(!j.issued.empty() && j.issued.back().first==index) j.issued.back().second+=bytes;
      else j.issued.emplace_back(index,bytes);
      unit.offset+=width; budget-=bytes;
      if(unit.offset==span.bytes) {
        unit.fragment+=rows; unit.offset=0;
        if(unit.fragment==span.fragments) { ++unit.cursor; unit.fragment=0; }
      }
    }
    if(!j.started) {
      C10_CUDA_CHECK(cudaEventRecord(j.begin,stream)); j.started=true; j.first_issue_ns=now_ns();
    }
    CUresult status;
    size_t failure=0;
    if(pitched) status=pitched_function()(j.boxes.size(),j.boxes.data(),0,static_cast<CUstream>(stream));
    else {
      CUmemcpyAttributes attr{};
      attr.srcAccessOrder=j.store?CU_MEMCPY_SRC_ACCESS_ORDER_STREAM:CU_MEMCPY_SRC_ACCESS_ORDER_ANY;
      size_t attr_index=0;
      status=batch_function()(j.destination.data(),j.source.data(),j.sizes.data(),
          j.sizes.size(),&attr,&attr_index,1,&failure,static_cast<CUstream>(stream));
    }
    TORCH_CHECK(status==CUDA_SUCCESS,"Segment batch DMA failed at descriptor ",failure,
                " driver status ",status);
    C10_CUDA_CHECK(cudaEventRecord(j.segment,stream));
    j.issued_bytes=segment_bytes-budget; j.phase=Job::Copying; return true;
  }
  void loop() noexcept {
    try {
      const c10::cuda::CUDAGuard guard(device);
      std::unique_lock lock(mutex);
      for(;;) {
        if(stopping) break;
        bool pending=false, progressed=false;
        for(auto& job:jobs) if(job.phase==Job::Queued || job.phase==Job::Copying) {
          pending=true; progressed=advance(job)||progressed;
        }
        if(!progressed) {
          if(pending) work.wait_for(lock,std::chrono::microseconds(50));
          else work.wait(lock,[&] {
            return stopping || std::any_of(jobs.begin(),jobs.end(),[](const Job& j) {
              return j.phase==Job::Queued || j.phase==Job::Copying;
            });
          });
        }
      }
    } catch(...) {
      std::lock_guard lock(mutex); error=std::current_exception(); stopping=true;
      completed.notify_all(); work.notify_all();
    }
  }
  void destroy_events() noexcept {
    const c10::cuda::OptionalCUDAGuard guard(device);
    for(auto& j:jobs) for(auto* e:{&j.dependency,&j.begin,&j.segment}) if(*e) {
      C10_CUDA_CHECK_WARN(cudaEventDestroy(*e)); *e=nullptr;
    }
  }
};

KvCopy::KvCopy(c10::DeviceIndex device,std::vector<at::Tensor> gpu,at::Tensor host,
    std::uint64_t segment,std::uint32_t count,std::uint32_t max_spans,bool pitched) {
  TORCH_CHECK((segment==4ULL<<20 || segment==8ULL<<20 || segment==16ULL<<20) &&
      count>0 && count<=8 && max_spans>0 && max_spans<=1U<<20,"Bounded KV copy resources required");
  TORCH_CHECK(!gpu.empty() && gpu.size()<=64 && host.device().is_cpu() &&
      host.scalar_type()==at::kChar && host.is_contiguous(),"Expected registered canonical int8 host payload");
  for(const auto& t:gpu) TORCH_CHECK(t.is_cuda() && t.get_device()==device && t.is_contiguous(),
                                   "Expected contiguous native byte views on the selected device");
  const c10::cuda::CUDAGuard guard(device);
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes,host.data_ptr()));
  TORCH_CHECK(attributes.type==cudaMemoryTypeHost,"No pageable host fallback for KV DMA");
  (void)batch_function();
  if(pitched) (void)pitched_function();
  impl_=std::make_unique<Impl>(device,std::move(gpu),std::move(host),segment,count,max_spans,pitched);
  try {
    for(auto& j:impl_->jobs) {
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&j.dependency,cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventCreate(&j.begin));
      C10_CUDA_CHECK(cudaEventCreate(&j.segment));
    }
    impl_->worker=std::thread([this]{impl_->loop();});
  } catch(...) { impl_->destroy_events(); throw; }
}

bool KvCopy::submit(std::uint64_t id,std::uint64_t generation,const at::Tensor& plan,
                    bool store,c10::cuda::CUDAStream producer,const at::Tensor& gates) {
  auto& s=*impl_;
  TORCH_CHECK(plan.device().is_cpu() && plan.scalar_type()==at::kLong && plan.is_contiguous() &&
      plan.dim()==2 && plan.size(1)==7 && plan.size(0)<=s.max_spans,"Expected bounded CPU int64[N,7] CopyRuns");
  TORCH_CHECK(producer.device_index()==s.device,"Producer belongs to another device");
  std::lock_guard lock(s.mutex); s.check();
  TORCH_CHECK(id>s.last_id,"KV copy IDs must increase; a retired job cannot be resubmitted");
  auto found=std::find_if(s.jobs.begin(),s.jobs.end(),[](const Impl::Job& j){return j.phase==Impl::Job::Empty;});
  if(found==s.jobs.end()) return false;
  auto& j=*found;
  if(gates.defined()) TORCH_CHECK(s.flags.defined() && gates.device().is_cpu() &&
      gates.scalar_type()==at::kLong && gates.is_contiguous() && gates.dim()==2 &&
      gates.size(0)==plan.size(0) && gates.size(1)==2,"Expected per-CopyRun int64[N,2] page gates");
  std::vector<Impl::Job::Unit> units;
  std::unordered_map<std::int64_t,std::size_t> by_gate;
  const auto* gate_rows=gates.defined()?gates.const_data_ptr<std::int64_t>():nullptr;
  const auto* rows=plan.const_data_ptr<std::int64_t>();
  for(std::int64_t i=0;i<plan.size(0);++i) {
    const auto* row=rows+7*i;
    TORCH_CHECK(row[0]>=0 && std::uint64_t(row[0])<s.gpu.size() && row[1]>=0 && row[2]>=0 &&
                row[3]>0 && row[4]>0 && row[5]>=0 && row[6]>=0,
                "Invalid canonical span index/offset/length");
    const auto gpu_bytes=s.gpu[row[0]].nbytes(), host_bytes=s.host.nbytes();
    const auto offset=std::uint64_t(row[1]), host_offset=std::uint64_t(row[2]), bytes=std::uint64_t(row[3]);
    TORCH_CHECK(offset<=gpu_bytes && bytes<=gpu_bytes-offset && host_offset<=host_bytes &&
                bytes<=host_bytes-host_offset,"Canonical span escapes its native storage/pool");
    const auto fragments=std::uint64_t(row[4]), gs=std::uint64_t(row[5]), hs=std::uint64_t(row[6]);
    TORCH_CHECK(fragments==1 || (gs>=bytes && hs>=bytes &&
                fragments-1<=(gpu_bytes-offset-bytes)/gs &&
                fragments-1<=(host_bytes-host_offset-bytes)/hs),"CopyRun stride/count escapes native storage/pool");
    // Contiguous native runs collapse without a host payload copy. Strided TP
    // heads remain compact here and expand only into one bounded DMA segment.
    Impl::Span span{std::uint64_t(row[0]),offset,host_offset,bytes,fragments,gs,hs};
    if(gs==bytes && hs==bytes) { span.bytes*=fragments; span.fragments=1; }
    const auto gate=gate_rows?gate_rows[i*2]:-1, epoch=gate_rows?gate_rows[i*2+1]:0;
    TORCH_CHECK(!gate_rows || (gate>=0 && gate<s.flags.numel() && epoch>0),"Invalid page gate/epoch");
    auto [entry,inserted]=by_gate.emplace(gate,units.size());
    if(inserted) { units.emplace_back(); units.back().gate=gate; units.back().epoch=epoch; }
    auto& unit=units[entry->second];
    TORCH_CHECK(unit.epoch==epoch,"One page cannot mix row epochs");
    unit.bytes+=span.bytes*span.fragments;
    auto& spans=unit.spans;
    if(span.fragments==1 && !spans.empty() && spans.back().fragments==1 &&
        spans.back().tensor==std::uint64_t(row[0]) &&
        spans.back().gpu+spans.back().bytes==offset && spans.back().host+spans.back().bytes==host_offset)
      spans.back().bytes+=span.bytes;
    else spans.push_back(span);
  }
  const c10::cuda::CUDAGuard guard(s.device);
  C10_CUDA_CHECK(cudaEventRecord(j.dependency,producer.stream()));
  j.id=id; j.generation=generation; j.store=store; j.units=std::move(units);
  j.bytes=0; j.segments=0; j.issued_bytes=0; j.first_issue_ns=0; j.last_done_ns=0;
  j.cancelled=false; j.started=false; j.admitted=std::chrono::steady_clock::now();
  j.phase=Impl::Job::Queued; s.last_id=id; s.work.notify_one(); return true;
}
void KvCopy::set_ready(const at::Tensor& flags) {
  auto& s=*impl_; std::lock_guard lock(s.mutex); s.check();
  TORCH_CHECK(!s.flags.defined() && !s.last_id && flags.device().is_cpu() &&
      flags.scalar_type()==at::kLong && flags.is_contiguous() && flags.dim()==1 &&
      flags.numel()>0 && flags.numel()<=64*4096 &&
      reinterpret_cast<std::uintptr_t>(flags.data_ptr())%alignof(std::int64_t)==0,
      "Ready control must be bounded, aligned and attached before submission");
  s.flags=flags;
}
bool KvCopy::cancel(std::uint64_t id,std::uint64_t generation) {
  auto& s=*impl_; std::lock_guard lock(s.mutex); s.check();
  for(auto& j:s.jobs) if(j.phase!=Impl::Job::Empty && j.id==id && j.generation==generation) {
    if(j.phase!=Impl::Job::Done) j.cancelled=true;
    s.work.notify_one(); return true;
  }
  return false;
}
std::vector<KvCopyResult> KvCopy::poll() {
  auto& s=*impl_; std::lock_guard lock(s.mutex); s.check();
  const c10::cuda::CUDAGuard guard(s.device);
  std::vector<KvCopyResult> result;
  for(auto& j:s.jobs) if(j.phase==Impl::Job::Done) {
    float ms=0;
    if(j.started) C10_CUDA_CHECK(cudaEventElapsedTime(&ms,j.begin,j.segment));
    result.push_back({j.id,j.generation,j.bytes,j.segments,j.cancelled,ms,j.first_issue_ns,j.last_done_ns});
    j.phase=Impl::Job::Empty; j.units.clear(); j.issued.clear();
  }
  return result;
}
std::vector<KvCopyResult> KvCopy::snapshot() {
  auto& s=*impl_; std::lock_guard lock(s.mutex); s.check();
  std::vector<KvCopyResult> result;
  for(const auto& j:s.jobs) if(j.phase!=Impl::Job::Empty)
    result.push_back({j.id,j.generation,j.bytes,j.segments,j.cancelled,0,j.first_issue_ns,j.last_done_ns});
  return result;
}
void KvCopy::wait(const std::vector<std::uint64_t>& ids,std::uint32_t timeout) {
  TORCH_CHECK(timeout>0 && timeout<=120000,"Bounded KV wait timeout required");
  auto& s=*impl_; std::unique_lock lock(s.mutex); s.check();
  const bool done=s.completed.wait_for(lock,std::chrono::milliseconds(timeout),[&] {
    return s.error || std::all_of(ids.begin(),ids.end(),[&](std::uint64_t id) {
      return std::none_of(s.jobs.begin(),s.jobs.end(),[&](const Impl::Job& j) {
        return j.id==id && (j.phase==Impl::Job::Queued || j.phase==Impl::Job::Copying);
      });
    });
  });
  s.check(); TORCH_CHECK(done,"KV copy selected-job wait deadline expired");
}
void KvCopy::drain(std::uint32_t timeout) {
  TORCH_CHECK(timeout>0 && timeout<=120000,"Bounded KV drain timeout required");
  auto& s=*impl_; std::unique_lock lock(s.mutex); s.check();
  const bool done=s.completed.wait_for(lock,std::chrono::milliseconds(timeout),[&] {
    return s.error || std::all_of(s.jobs.begin(),s.jobs.end(),[](const Impl::Job& j) {
      return j.phase==Impl::Job::Empty || j.phase==Impl::Job::Done;
    });
  });
  s.check(); TORCH_CHECK(done,"KV copy drain deadline expired");
}
void KvCopy::close() {
  auto& s=*impl_;
  { std::lock_guard lock(s.mutex); s.check();
    TORCH_CHECK(std::all_of(s.jobs.begin(),s.jobs.end(),[](const Impl::Job& j) {
      return j.phase==Impl::Job::Empty;
    }),"Drain and harvest all KV jobs before close");
    s.stopping=true;
  }
  s.work.notify_all(); if(s.worker.joinable()) s.worker.join();
  s.destroy_events(); s.gpu.clear(); s.host=at::Tensor(); s.flags=at::Tensor(); s.closed=true;
}
KvCopy::~KvCopy() noexcept {
  if(!impl_ || impl_->closed) return;
  auto& s=*impl_;
  { std::lock_guard lock(s.mutex); s.stopping=true; }
  s.work.notify_all(); if(s.worker.joinable()) s.worker.join();
  // Failure is terminal for this owned experiment. Never unmap/unpin a payload
  // that an already-issued DMA might still reference. Supervisor ends the PG.
  if(s.error || std::any_of(s.jobs.begin(),s.jobs.end(),[](const Impl::Job& j) {
        return j.phase==Impl::Job::Copying;
      })) {
    (void)impl_.release(); TORCH_WARN("Fatal KV DMA epoch retained until owned process exit");
  } else s.destroy_events();
}
} // namespace vpp
