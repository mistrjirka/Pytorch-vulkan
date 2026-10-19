// Generic FP32 NCHW implicit GEMM. Groups never share a tile.
#include "conv2d_tile_config.glsl"
layout(set=0,binding=0) readonly buffer Input { float x[]; };
layout(set=0,binding=1) readonly buffer Weight { float w[]; };
layout(set=0,binding=2) readonly buffer Bias { float bias[]; };
layout(set=0,binding=3) writeonly buffer Output { float y[]; };
layout(push_constant) uniform Params {
  uint n,cin,cout,h,w,kh,kw,oh,ow;
  uint sh,sw,ph,pw,dh,dw,groups;
} p;
#ifdef CONV_FIXED_3X3
const uint KERNEL_H=3u, KERNEL_W=3u;
#else
#define KERNEL_H p.kh
#define KERNEL_W p.kw
#endif
shared float a[TM*TK];
shared float b[TK*TN];
void store_result(uint m,uint spatial,uint group,float value) {
  uint co=p.cout/p.groups;
  if(m<co && spatial<p.n*p.oh*p.ow) {
    uint batch=spatial/(p.oh*p.ow),pos=spatial%(p.oh*p.ow),oc=group*co+m;
    y[(batch*p.cout+oc)*p.oh*p.ow+pos]=value+bias[oc];
  }
}
void main() {
  uint lx=gl_LocalInvocationID.x, ly=gl_LocalInvocationID.y;
  uint tid=ly*WG_N+lx;
  uint co=p.cout/p.groups, ci=p.cin/p.groups;
  uint tiles=(co+TM-1)/TM;
  uint group=gl_WorkGroupID.y/tiles;
  uint m0=(gl_WorkGroupID.y%tiles)*TM;
  uint n0=gl_WorkGroupID.x*TN;
  uint size_n=p.n*p.oh*p.ow, total=ci*KERNEL_H*KERNEL_W;
#if CONV_RM == 2
  vec4 acc=vec4(0.0);
#else
  float acc=0.0;
#endif
  for(uint k0=0;k0<total;k0+=TK) {
    // Linear cooperative loads work for every tile aspect ratio.
    for(uint i=tid;i<TM*TK;i+=THREADS) {
      uint m=m0+i/TK, k=k0+i%TK;
      a[i]=(m<co && k<total) ?
          w[(group*co+m)*total+k] : 0.0;
    }
    for(uint i=tid;i<TK*TN;i+=THREADS) {
      uint k=k0+i/TN, pos=n0+i%TN;
      float v=0.0;
      if(pos<size_n && k<total) {
        uint batch=pos/(p.oh*p.ow), outpos=pos%(p.oh*p.ow);
        uint inch=group*ci+k/(KERNEL_H*KERNEL_W), rem=k%(KERNEL_H*KERNEL_W);
        int iy=int((outpos/p.ow)*p.sh)-int(p.ph)+int((rem/KERNEL_W)*p.dh);
        int ix=int((outpos%p.ow)*p.sw)-int(p.pw)+int((rem%KERNEL_W)*p.dw);
        if(iy>=0 && iy<int(p.h) && ix>=0 && ix<int(p.w))
          v=x[((batch*p.cin+inch)*p.h+uint(iy))*p.w+uint(ix)];
      }
      b[i]=v;
    }
    barrier();
    #include "conv2d_tile_accumulate.glsl"
    barrier();
  }
  #include "conv2d_tile_store.glsl"
}
