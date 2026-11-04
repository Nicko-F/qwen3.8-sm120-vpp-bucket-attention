// Fixed H5120 Gemma add/RMS -> native NVFP4 group16/E4M3 swizzle.
// Raw gamma is BF16, but (float(gamma)+1) and normalization stay FP32.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace gemma_fp4 {
constexpr int H=5120, Groups=H/16, Threads=320;
struct alignas(16) B8 { __nv_bfloat16 v[8]; };

__global__ void pack(const B8* __restrict__ x, const B8* __restrict__ residual,
                     const B8* __restrict__ gamma, const float* global_inv,
                     uint64_t* __restrict__ q, uint8_t* __restrict__ scales,
                     B8* __restrict__ updated) {
  int row=blockIdx.x, group=threadIdx.x, lane=group&31, warp=group>>5;
  B8 xl=x[row*640+group*2], xh=x[row*640+group*2+1];
  B8 rl=residual[row*640+group*2], rh=residual[row*640+group*2+1];
  B8 wl=gamma[group*2], wh=gamma[group*2+1], ol, oh;
  float h[16], y[16], sum=0.f;
#pragma unroll
  for(int j=0;j<8;j++) {
    h[j]=__bfloat162float(xl.v[j])+__bfloat162float(rl.v[j]);
    h[j+8]=__bfloat162float(xh.v[j])+__bfloat162float(rh.v[j]);
    sum+=h[j]*h[j]; sum+=h[j+8]*h[j+8];
    ol.v[j]=__float2bfloat16_rn(h[j]); oh.v[j]=__float2bfloat16_rn(h[j+8]);
  }
  updated[row*640+group*2]=ol; updated[row*640+group*2+1]=oh;
#pragma unroll
  for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
  __shared__ float partial[10], inverse_rms;
  if(lane==0)partial[warp]=sum;
  __syncthreads();
  if(warp==0) {
    sum=lane<10?partial[lane]:0.f;
#pragma unroll
    for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(lane==0)inverse_rms=rsqrtf(sum/float(H)+1e-6f);
  }
  __syncthreads();
  float amax=0.f;
#pragma unroll
  for(int j=0;j<8;j++) {
    // Preserve the actual BF16 producer boundary before FP4 quantization.
    y[j]=__bfloat162float(__float2bfloat16_rn(
        (h[j]*inverse_rms)*(__bfloat162float(wl.v[j])+1.f)));
    y[j+8]=__bfloat162float(__float2bfloat16_rn(
        (h[j+8]*inverse_rms)*(__bfloat162float(wh.v[j])+1.f)));
    amax=fmaxf(amax,fabsf(y[j])); amax=fmaxf(amax,fabsf(y[j+8]));
  }
  float global=*global_inv;
  __nv_fp8_e4m3 sf(fminf(amax*global*(1.f/6.f),448.f));
  float scale=float(sf), inverse=scale>0.f?global/scale:0.f;
  int offset=(row/128)*40960+(group/4)*512+(row%32)*16+((row%128)/32)*4+group%4;
  scales[offset]=sf.__x;
  uint64_t packed=0;
#pragma unroll
  for(int j=0;j<8;j++) {
    auto pair=__nv_cvt_float2_to_fp4x2(make_float2(y[2*j]*inverse,y[2*j+1]*inverse),
                                      __NV_E2M1,cudaRoundNearest);
    packed|=uint64_t(pair)<<(8*j);
  }
  q[row*Groups+group]=packed;
}
}

extern "C" int nvfp4_norm_pack_launch(const void* x,const void* r,const void* w,
    const void* global,void* q,void* scale,void* updated,int rows,cudaStream_t stream) {
  if(rows<1||rows>8192||!x||!r||!w||!global||!q||!scale||!updated)
    return int(cudaErrorInvalidValue);
  gemma_fp4::pack<<<rows,320,0,stream>>>(static_cast<const gemma_fp4::B8*>(x),
      static_cast<const gemma_fp4::B8*>(r),static_cast<const gemma_fp4::B8*>(w),
      static_cast<const float*>(global),static_cast<uint64_t*>(q),
      static_cast<uint8_t*>(scale),static_cast<gemma_fp4::B8*>(updated));
  return int(cudaPeekAtLastError());
}

extern "C" int nvfp4_norm_pack_metadata(int* out) {
  if(!out)return int(cudaErrorInvalidValue);
  cudaFuncAttributes attr{};auto status=cudaFuncGetAttributes(&attr,gemma_fp4::pack);
  if(status!=cudaSuccess)return int(status);
  out[0]=attr.numRegs;out[1]=int(attr.sharedSizeBytes);out[2]=int(attr.localSizeBytes);
  return 0;
}
