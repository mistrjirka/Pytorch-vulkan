// dW = dY * im2col(X)^T, gathered a shared tile at a time.
// Adaptive spatial Split-K reduction with bounded temporary storage.
// Retain 256-position splits where they fit; extend each split otherwise.
#include "conv2d_tile_config.glsl"
layout(set=0,binding=0) readonly buffer Input { float x[]; };
layout(set=0,binding=1) readonly buffer GradOut { float go[]; };
layout(set=0,binding=2) writeonly buffer GradWeight { float out_data[]; };
layout(push_constant) uniform Params {
  uint n,cin,cout,h,w,kh,kw,oh,ow;
  uint sh,sw,ph,pw,dh,dw,groups,split_span;
} p;
#ifdef CONV_FIXED_3X3
const uint KERNEL_H=3u, KERNEL_W=3u;
#else
#define KERNEL_H p.kh
#define KERNEL_W p.kw
#endif
shared float a[TM*TK];
shared float b[TK*TN];
void store_result(uint m,uint col,uint group,float value) {
  uint co=p.cout/p.groups,cols=(p.cin/p.groups)*KERNEL_H*KERNEL_W;
  if(m<co && col<cols) {
    uint oc=group*co+m,split=gl_WorkGroupID.z;
    out_data[(split*p.cout+oc)*cols+col]=value;
  }
}
void main() {
  uint lx=gl_LocalInvocationID.x, ly=gl_LocalInvocationID.y, tid=ly*WG_N+lx;
  uint ci=p.cin/p.groups, co=p.cout/p.groups;
  uint cols=ci*KERNEL_H*KERNEL_W, tiles=(co+TM-1)/TM;
  uint group=gl_WorkGroupID.y/tiles, m0=(gl_WorkGroupID.y%tiles)*TM;
  uint n0=gl_WorkGroupID.x*TN;
  uint spatial=p.oh*p.ow, total=p.n*spatial, split=gl_WorkGroupID.z;
#if CONV_RM == 2
  vec4 acc=vec4(0.0);
#else
  float acc=0.0;
#endif
  // The launch grid and scratch layout use the *same* bounded split plan.
  // A fixed 256-span is fastest when memory permits; otherwise each split
  // processes more positions without using additional scratch.
  uint first=split*p.split_span;
  uint end=first+min(p.split_span,total-first);
  for(uint k0=first;k0<end;) {
    for(uint i=tid;i<TM*TK;i+=THREADS) {
      uint m=m0+i/TK, k=k0+i%TK, oc=group*co+m;
      a[i]=(m<co && k<total) ?
          go[((k/spatial)*p.cout+oc)*spatial+k%spatial] : 0.0;
    }
    for(uint i=tid;i<TK*TN;i+=THREADS) {
      uint k=k0+i/TN, col=n0+i%TN;
      float v=0.0;
      if(col<cols && k<total) {
        uint ic=group*ci+col/(KERNEL_H*KERNEL_W), ky=(col/KERNEL_W)%KERNEL_H, kx=col%KERNEL_W;
        uint pos=k%spatial;
        int iy=int((pos/p.ow)*p.sh)-int(p.ph)+int(ky*p.dh);
        int ix=int((pos%p.ow)*p.sw)-int(p.pw)+int(kx*p.dw);
        if(iy>=0 && iy<int(p.h) && ix>=0 && ix<int(p.w))
          v=x[(((k/spatial)*p.cin+ic)*p.h+uint(iy))*p.w+uint(ix)];
      }
      b[i]=v;
    }
    barrier();
    #include "conv2d_tile_accumulate.glsl"
    barrier();
    // Avoid a uint32 wraparound on extreme but otherwise valid geometries.
    if(end-k0<=TK) break;
    k0+=TK;
  }
  #include "conv2d_tile_store.glsl"
}
