// SPDX-License-Identifier: Apache-2.0
// Qwen Gemma QK norm/NeoX interleaved MRoPE/gate + NVFP4 paged append.
// Math/layout reference: installed vLLM fused_qk_norm_rope.py and
// FlashInfer page.cuh::NVFP4QuantizeAppendPagedKVCacheWithSlotMappingKernel.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <cmath>

struct Front {
  const __nv_bfloat16 *qkv,*qw,*kw;
  const void *cos_sin,*positions,*slots;
  const float *k_scale,*v_scale;
  uint8_t *k,*v,*ks,*vs;
  __nv_bfloat16 *q,*gate;
  int rows,qheads,kvheads,rotary,slot_rows,pages,page_size;
  int cos_float,pos_long,slot_long,mrope_h,mrope_w;
  int64_t input_row,cos_row,pos_m,pos_t;
  int64_t data_page,data_row,data_head,sf_page,sf_row,sf_head;
  float epsilon;
};

__device__ __forceinline__ float bf(float x) {return __bfloat162float(__float2bfloat16_rn(x));}
__device__ __forceinline__ float warp_sum(float x) {
  #pragma unroll
  for(int d=16;d;d>>=1)x+=__shfl_down_sync(0xffffffff,x,d);
  return x;
}
__device__ __forceinline__ float group_max(float x,int width) {
  #pragma unroll
  for(int d=1;d<16;d*=2)if(d<width)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));
  return x;
}
__device__ __forceinline__ int64_t index_value(const void* ptr,int64_t i,int is_long) {
  return is_long?static_cast<const int64_t*>(ptr)[i]:static_cast<const int32_t*>(ptr)[i];
}
__device__ __forceinline__ float trig(const Front& p,int64_t pos,int d) {
  int64_t offset=pos*p.cos_row+d;
  return p.cos_float?static_cast<const float*>(p.cos_sin)[offset]:
    __bfloat162float(static_cast<const __nv_bfloat16*>(p.cos_sin)[offset]);
}

__device__ __forceinline__ void quantize_pair(float x,float y,int d0,int d1,
    int half,float global,uint8_t* out,uint8_t* sf) {
  int tid=threadIdx.x;
  if(tid<half) {
    float sx=group_max(fabsf(x),16),sy=group_max(fabsf(y),16);
    __nv_fp8_e4m3 fx(sx/(6.f*global)),fy(sy/(6.f*global));
    if((tid&15)==0){sf[d0/16]=fx.__x;sf[d1/16]=fy.__x;}
    float ix=float(fx)>0.f?1.f/(float(fx)*global):0.f;
    float iy=float(fy)>0.f?1.f/(float(fy)*global):0.f;
    float next_x=__shfl_xor_sync(0xffffffff,x,1),next_y=__shfl_xor_sync(0xffffffff,y,1);
    if((tid&1)==0) {
      out[d0/2]=__nv_cvt_float2_to_fp4x2(make_float2(x*ix,next_x*ix),__NV_E2M1,cudaRoundNearest);
      out[d1/2]=__nv_cvt_float2_to_fp4x2(make_float2(y*iy,next_y*iy),__NV_E2M1,cudaRoundNearest);
    }
  } else {
    float scale=group_max(fmaxf(fabsf(x),fabsf(y)),8);
    __nv_fp8_e4m3 f(scale/(6.f*global));
    if((tid&7)==0)sf[d0/16]=f.__x;
    float inverse=float(f)>0.f?1.f/(float(f)*global):0.f;
    out[d0/2]=__nv_cvt_float2_to_fp4x2(make_float2(x*inverse,y*inverse),__NV_E2M1,cudaRoundNearest);
  }
}

__global__ __launch_bounds__(128) void qk_rope_append(Front p) {
  const int row=blockIdx.x,head=blockIdx.y,tid=threadIdx.x,half=p.rotary/2;
  const bool key=head>=p.qheads;
  const int local=key?head-p.qheads:head;
  const int d0=tid<half?tid:tid*2,d1=tid<half?tid+half:tid*2+1;
  const int64_t base=int64_t(row)*p.input_row+(key?p.qheads*512+local*256:local*512);
  float x=__bfloat162float(p.qkv[base+d0]),y=__bfloat162float(p.qkv[base+d1]);
  float sum=warp_sum(x*x+y*y);
  __shared__ float parts[4],inverse;
  if((tid&31)==0)parts[tid/32]=sum;
  __syncthreads();
  if(tid<32) {
    sum=warp_sum(tid<4?parts[tid]:0.f);
    if(tid==0)inverse=rsqrtf(sum/256.f+p.epsilon);
  }
  __syncthreads();
  const __nv_bfloat16* weight=key?p.kw:p.qw;
  x=bf((x*inverse)*(__bfloat162float(weight[d0])+1.f));
  y=bf((y*inverse)*(__bfloat162float(weight[d1])+1.f));
  if(tid<half) {
    int axis=(tid%3==1 && tid<3*p.mrope_h)?1:
             ((tid%3==2 && tid<3*p.mrope_w)?2:0);
    int64_t pos=index_value(p.positions,int64_t(row)*p.pos_t+axis*p.pos_m,p.pos_long);
    float c=trig(p,pos,tid),s=trig(p,pos,half+tid);
    float a=x,b=y;x=bf(a*c-b*s);y=bf(b*c+a*s);
  }
  if(!key) {
    int64_t out=(int64_t(row)*p.qheads+local)*256;
    p.q[out+d0]=__float2bfloat16_rn(x);p.q[out+d1]=__float2bfloat16_rn(y);
    p.gate[out+d0]=p.qkv[base+256+d0];p.gate[out+d1]=p.qkv[base+256+d1];
    return;
  }
  if(row>=p.slot_rows)return;
  int64_t slot=index_value(p.slots,row,p.slot_long);
  if(slot<0 || slot>=int64_t(p.pages)*p.page_size)return;
  const float kg=*p.k_scale,vg=*p.v_scale;
  if(!(isfinite(kg)&&kg>0.f&&isfinite(vg)&&vg>0.f)){asm volatile("trap;");return;}
  int64_t page=slot/p.page_size,entry=slot%p.page_size;
  int64_t data=page*p.data_page+entry*p.data_row+local*p.data_head;
  int64_t sf=page*p.sf_page+entry*p.sf_row+local*p.sf_head;
  quantize_pair(x,y,d0,d1,half,kg,p.k+data,p.ks+sf);
  int64_t vb=int64_t(row)*p.input_row+p.qheads*512+p.kvheads*256+local*256;
  x=__bfloat162float(p.qkv[vb+d0]);y=__bfloat162float(p.qkv[vb+d1]);
  quantize_pair(x,y,d0,d1,half,vg,p.v+data,p.vs+sf);
}

extern "C" int nvfp4_attention_front(Front p,cudaStream_t stream) {
  if(p.rows<1 || p.qheads<1 || p.kvheads<1 || p.rotary<64 || p.rotary>256 || p.rotary%64)
    return int(cudaErrorInvalidValue);
  qk_rope_append<<<dim3(p.rows,p.qheads+p.kvheads),128,0,stream>>>(p);
  return int(cudaPeekAtLastError());
}
extern "C" int nvfp4_attention_front_resources(int* out) {
  cudaFuncAttributes a;int occupancy=0;
  auto e=cudaFuncGetAttributes(&a,qk_rope_append);if(e!=cudaSuccess)return int(e);
  e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,qk_rope_append,128,0);
  if(e!=cudaSuccess)return int(e);
  out[0]=sizeof(Front);out[1]=a.numRegs;out[2]=a.sharedSizeBytes;out[3]=a.localSizeBytes;out[4]=occupancy;
  return 0;
}
