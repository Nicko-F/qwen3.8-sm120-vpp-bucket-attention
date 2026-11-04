// SPDX-License-Identifier: Apache-2.0
// Explicit compact-task ownership around the Apache-licensed native XQA body.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef NVFP4_XQA_PERSISTENT
#define NVFP4_XQA_PERSISTENT 0
#endif
#ifndef NVFP4_XQA_TAIL
#define NVFP4_XQA_TAIL 0
#endif

struct XQATask { int request, begin, end, stride; }; // Native256-token CTA tiles.
struct XQARange { int begin, count; };
struct XQABucket { int begin, end; }; // Compact task-slot interval, not KV tiles.
struct NativeTaskContext {
  uint32_t request, slot, begin_tile, end_tile, stride;
  uint32_t partial_begin, partial_count, total_slots;
  float qk_scale, v_scale;
  uint32_t reuse_barriers;
  const __nv_bfloat16* gate;
  const float* output_scale;
  uint8_t* fp8_output;
};
struct XQAParams {
  const __nv_bfloat16* q;
  const uint8_t *k, *v, *ks, *vs;
  const int *table, *lengths;
  XQATask* tasks;
  XQARange* requests;
  XQABucket* buckets;
  int *header, *semaphores;
  uint8_t* scratch;
  __nv_bfloat16* output;
  int batch, ctas, table_width, mode; // 0: intervals, 1: native cyclic, 2: request-aligned.
  int64_t kv_page, kv_token, sf_page, sf_token;
  float qscale, vscale;
  const __nv_bfloat16* gate;
  const float* output_scale;
  uint8_t* fp8_output;
};

#if NVFP4_XQA_TAIL
// Final-result epilogue only. Native partials and merge stay BF16/FP32.
// One existing BF16 grain plus gate/scalar scale; no new shared tile or barrier.
template<int Count>
__device__ __forceinline__ void store_final_fp8(const NativeTaskContext& task,
    uint32_t head,uint32_t col,const __nv_bfloat16* values) {
  const uint32_t first=(task.request*6+head)*256+col;
  const float scale=*task.output_scale;
  static_assert(Count==8,"The native BF16 final output grain contains eight elements");
  uint64_t packed=0;
  #pragma unroll
  for(int i=0;i<Count;i++) {
    const float gate=__bfloat162float(task.gate[first+i]);
    const float sigmoid=__bfloat162float(__float2bfloat16_rn(1.f/(1.f+expf(-gate))));
    const float gated=__bfloat162float(__float2bfloat16_rn(__bfloat162float(values[i])*sigmoid));
    const __nv_fp8_e4m3 value(gated/scale);
    packed|=uint64_t(value.__x)<<(8*i);
  }
  // Native grain columns are multiples of8; each lane issues one aligned8B
  // store instead of eight strided byte stores across the warp.
  *reinterpret_cast<uint64_t*>(task.fp8_output+first)=packed;
}
#endif

// Generated from installed native source with audited ownership/stride changes.
// Includes the original copyright/license and its complete QK/PV/merge helpers.
#include "xqa_task_impl.cuh"

#if NVFP4_XQA_PERSISTENT
static_assert(ctaSize==256 && ctaShapeInWarps.x==4 && ctaShapeInWarps.y==1 &&
              ctaShapeInWarps.z==2 && grpLoadV && nbVBuffers==2 &&
              gemm1NbWarpGrps==1 && nbHeadSplits==1 && headElems==256 &&
              SPEC_DEC==0 && BEAM_WIDTH==1 && USE_CUSTOM_BARRIER==1,
              "Persistent barrier participants are audited only for the TP4 q1/D256 core");
static_assert(sizeof(SharedMem::Barrier)==8 &&
              (sizeof(SharedMem::xBarriers)/sizeof(CtaBarrierPair))*2+
              ctaShapeInWarps.y+SharedMem::nbOtherBarriers==12);
#endif

namespace xqa_task {
constexpr int MaxB=32, Tile=256;
// Match Python: native physical-page rounding is independent of live KV length.
// Keep width*32+Tile-1 and all32 dense table rows representable as signed int.
constexpr int MaxTableWidth=(0x7fffffff-(Tile-1))/32;
__device__ __forceinline__ int ceildiv(int a,int b) { return (a+b-1)/b; }

__global__ void assign(XQAParams p) {
  __shared__ int prefix[MaxB+1],base[MaxB+1],first[MaxB];
  __shared__ int budget,splits,slots,used,max_parts;
  if(threadIdx.x==0) {
    prefix[0]=base[0]=0;
    for(int r=0;r<p.batch;r++)prefix[r+1]=prefix[r]+ceildiv(max(0,p.lengths[r]),Tile);
    budget=max(2,ceildiv(prefix[p.batch],p.ctas));
    if(p.mode==2) {
      int lo=1,hi=1;
      for(int r=0;r<p.batch;r++)hi=max(hi,prefix[r+1]-prefix[r]);
      while(lo<hi) {
        const int mid=(lo+hi)/2;
        int count=0;
        for(int r=0;r<p.batch;r++)count+=ceildiv(prefix[r+1]-prefix[r],mid);
        if(count<=p.ctas)hi=mid;else lo=mid+1;
      }
      budget=lo;
    }
    splits=min(max(1,p.ctas/p.batch),ceildiv(p.table_width*32,Tile));
    used=p.mode?0:ceildiv(prefix[p.batch],budget);max_parts=0;
    for(int r=0;r<p.batch;r++) {
      int n=prefix[r+1]-prefix[r];
      first[r]=prefix[r]/budget;
      int count=p.mode==1?splits:(p.mode==2?ceildiv(n,budget):
                  (n?(prefix[r+1]-1)/budget-first[r]+1:0));
      p.requests[r]={base[r],count};base[r+1]=base[r]+count;
      max_parts=max(max_parts,count);
    }
    slots=base[p.batch];if(p.mode)used=slots;
    p.header[0]=p.mode==1?splits:budget;p.header[1]=slots;
    p.header[2]=used;p.header[3]=max_parts;
  }
  __syncthreads();
  for(int slot=threadIdx.x;slot<slots;slot+=blockDim.x) {
    int r=0;while(r+1<p.batch && base[r+1]<=slot)r++;
    int ordinal=slot-base[r],n=prefix[r+1]-prefix[r];
    if(p.mode==1)p.tasks[slot]={r,ordinal,n,splits};
    else if(p.mode==2) {
      const int count=base[r+1]-base[r];
      p.tasks[slot]={r,ordinal*n/count,(ordinal+1)*n/count,1};
    }
    else {
      int bucket=first[r]+ordinal;
      int lo=max(prefix[r],bucket*budget)-prefix[r];
      int hi=min(prefix[r+1],(bucket+1)*budget)-prefix[r];
      p.tasks[slot]={r,lo,hi,1};
    }
  }
  // Request-major compact slots are also monotonic in global bucket. Build
  // disjoint begin/end intervals directly from shared prefix/base metadata;
  // no task-table reads, atomics or additional synchronization are required.
  for(int bucket=threadIdx.x;bucket<p.ctas;bucket+=blockDim.x) {
    if(p.mode==2) {
      p.buckets[bucket]=bucket<slots?XQABucket{bucket,bucket+1}:XQABucket{0,0};
      continue;
    }
    const int lo=bucket*budget,hi=min(prefix[p.batch],lo+budget);
    if(p.mode==1 || lo>=hi) {p.buckets[bucket]={0,0};continue;}
    int first_req=0;
    while(first_req+1<p.batch && prefix[first_req+1]<=lo)first_req++;
    int last_req=first_req;
    while(last_req+1<p.batch && prefix[last_req+1]<hi)last_req++;
    p.buckets[bucket]={base[first_req]+bucket-first[first_req],
                      base[last_req]+bucket-first[last_req]+1};
  }
}

__global__ __launch_bounds__(256,1) void compute(XQAParams p) {
  // Native PDL acquire must precede EVERY task/header/length read. The native
  // body no longer repeats it after metadata has already been dereferenced.
#if ENABLE_PDL
  preExit();acqBulk();
#endif
  const int tid=threadIdx.x+128*threadIdx.z;
  const int owner=blockIdx.x;
  if(owner<p.batch && p.lengths[owner]==0) {
    for(int i=tid;i<6*256;i+=256) {
#if NVFP4_XQA_TAIL
      p.fp8_output[owner*6*256+i]=0;
#else
      p.output[owner*6*256+i]=__float2bfloat16_rn(0.f);
#endif
    }
    if(tid==0)p.semaphores[owner]=0;
  }
  const int count=p.header[1];
#if NVFP4_XQA_PERSISTENT
  const XQABucket span=p.buckets[owner];
  bool reused=false;
  for(int slot=span.begin;slot<span.end;slot++) {
#else
  const int slot=owner;
  if(slot>=count)return; // Whole-CTA retirement, including unused Graph slots.
#endif
  const XQATask task=p.tasks[slot];
  if(p.lengths[task.request]==0) {
#if NVFP4_XQA_PERSISTENT
    continue;
#else
    return; // Zero-row owner above; no partials published.
#endif
  }
  const XQARange range=p.requests[task.request];
  NativeTaskContext context{uint32_t(task.request),uint32_t(slot),uint32_t(task.begin),
    uint32_t(task.end),uint32_t(task.stride),uint32_t(range.begin),uint32_t(range.count),
    uint32_t(count),p.qscale,p.vscale,
#if NVFP4_XQA_PERSISTENT
    uint32_t(reused)
#else
    0
#endif
    ,p.gate,p.output_scale,p.fp8_output
  };
  KVCacheList<true> cache{reinterpret_cast<GMemCacheHead*>(const_cast<uint8_t*>(p.k)),
    reinterpret_cast<GMemCacheHead*>(const_cast<uint8_t*>(p.v)),
    reinterpret_cast<GMemCacheHeadSf*>(const_cast<uint8_t*>(p.ks)),
    reinterpret_cast<GMemCacheHeadSf*>(const_cast<uint8_t*>(p.vs)),
    reinterpret_cast<const KVCachePageIndex*>(p.table),
    reinterpret_cast<const uint32_t*>(p.lengths),uint32_t(p.table_width)};
  // HeadPtr arithmetic is in units of packed GMemCacheHead / SF head objects,
  // NOT logical256-element heads; preserve actual native views' byte strides.
  kernel_mha_task_impl(context,1,1.f,nullptr,reinterpret_cast<OutputHead*>(p.output),
    reinterpret_cast<const IOHead*>(p.q),nullptr,cache,p.batch,1.f,nullptr,
    uint32_t(p.kv_page/sizeof(GMemCacheHead)),uint32_t(p.kv_token/sizeof(GMemCacheHead)),1,
    uint32_t(p.sf_page/sizeof(GMemCacheHeadSf)),uint32_t(p.sf_token/sizeof(GMemCacheHeadSf)),1,
    reinterpret_cast<uint32_t*>(p.semaphores),p.scratch);
#if NVFP4_XQA_PERSISTENT
  // QK has already drained and handed off through named barrier2. PV's named
  // barrier1 tail preserved every partial-publication/merge collective.
  // Close any merge-copy groups before the full256 task-lifetime join.
  ldgsts::commitGroup();ldgsts::waitGroup<0>();
  __syncthreads();
  reused=true;
  }
#else
  // End of entry: retain native producer retirement for the flat control.
#endif
}
} // namespace xqa_task

extern "C" int nvfp4_xqa_assign(XQAParams p,cudaStream_t stream) {
  if(p.batch<1 || p.batch>32 || p.ctas<p.batch || p.ctas>340 ||
     p.table_width<1 || p.table_width>xqa_task::MaxTableWidth || p.mode<0 || p.mode>2)
    return int(cudaErrorInvalidValue);
#if NVFP4_XQA_PERSISTENT
  if(p.mode!=0)return int(cudaErrorInvalidValue);
#endif
  xqa_task::assign<<<1,128,0,stream>>>(p);return int(cudaPeekAtLastError());
}
extern "C" int nvfp4_xqa_compute(XQAParams p,cudaStream_t stream) {
#if NVFP4_XQA_TAIL
  if(!p.gate || !p.output_scale || !p.fp8_output)return int(cudaErrorInvalidValue);
#endif
  if(p.batch<1 || p.batch>32 || p.ctas<p.batch || p.ctas>340 ||
     p.table_width<1 || p.table_width>xqa_task::MaxTableWidth || p.mode<0 || p.mode>2)
    return int(cudaErrorInvalidValue);
#if NVFP4_XQA_PERSISTENT
  if(p.mode!=0)return int(cudaErrorInvalidValue);
#endif
  if(p.kv_page%sizeof(GMemCacheHead) || p.kv_token%sizeof(GMemCacheHead) ||
     p.sf_page%sizeof(GMemCacheHeadSf) || p.sf_token%sizeof(GMemCacheHeadSf))
    return int(cudaErrorInvalidValue);
  const int splits=std::min(std::max(1,p.ctas/p.batch),(p.table_width*32+255)/256);
  const int grid=(NVFP4_XQA_PERSISTENT || p.mode==2)?p.ctas:
                 (p.mode==1?p.batch*splits:p.ctas+p.batch-1);
  cudaLaunchConfig_t config=makeLaunchConfig(dim3(grid),dim3(128,1,2),
                                            sizeof(SharedMem),stream,ENABLE_PDL!=0);
  cudaLaunchKernelEx(&config,xqa_task::compute,p);
  return int(cudaPeekAtLastError());
}
extern "C" int nvfp4_xqa_resources(int* values) {
  cudaFuncAttributes a;
  cudaError_t error=cudaFuncSetAttribute(xqa_task::compute,
      cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(SharedMem));
  if(error!=cudaSuccess)return int(error);
  error=cudaFuncGetAttributes(&a,xqa_task::compute);if(error!=cudaSuccess)return int(error);
  int resident=0;
  error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,xqa_task::compute,256,
                                                    sizeof(SharedMem));
  if(error!=cudaSuccess)return int(error);
  values[0]=sizeof(XQAParams);values[1]=a.numRegs;values[2]=sizeof(SharedMem);
  values[3]=a.localSizeBytes;values[4]=resident;values[5]=sizeof(SMemWarpRowMax);
  values[6]=6*256*2;values[7]=ENABLE_PDL;
  values[8]=sizeof(GMemCacheHead);values[9]=sizeof(GMemCacheHeadSf);
  values[10]=sizeof(XQATask);values[11]=sizeof(XQARange);
  values[12]=sizeof(XQABucket);
  return 0;
}
extern "C" uint64_t nvfp4_xqa_scratch_bytes(int capacity) {
  if(capacity<1 || capacity>371)return 0;
  Segmenter<> layout;
  layout.newSeg<SMemWarpRowMax>(capacity,sizeof(SMemWarpRowMax));
  layout.newSeg<SMemWarpRowMax>(capacity,sizeof(SMemWarpRowMax));
  using ScratchBuf=Array2D<LdGrain,nbValidRows,SharedMem::XSmemBuffer::cols>;
  using Slices=Vec<ScratchBuf,gemm1WarpsPerGrp*nbHeadSplits>;
  layout.newSeg<Slices>(capacity,sizeof(Slices));
  return layout.getEndOffset();
}
