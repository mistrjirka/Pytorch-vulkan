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
shared float a_tile[TM*TK];
shared float b_tile[TK*TN];
void main() {
  uint lx=gl_LocalInvocationID.x, ly=gl_LocalInvocationID.y;
  uint tid=ly*TN+lx;
  uint co=p.cout/p.groups, ci=p.cin/p.groups;
  uint tiles=(co+TM-1)/TM;
  uint group=gl_WorkGroupID.y/tiles;
  uint m0=(gl_WorkGroupID.y%tiles)*TM;
  uint n0=gl_WorkGroupID.x*TN;
  uint spatial=n0+lx, oc=group*co+m0+ly;
  uint size_n=p.n*p.oh*p.ow, total=ci*p.kh*p.kw;
  float acc=0.0;
  for(uint k0=0;k0<total;k0+=TK) {
    // Linear cooperative loads work for every tile aspect ratio.
    for(uint i=tid;i<TM*TK;i+=TM*TN) {
      uint m=m0+i/TK, k=k0+i%TK;
      a_tile[i]=(m<co && k<total) ?
          w[(group*co+m)*total+k] : 0.0;
    }
    for(uint i=tid;i<TK*TN;i+=TM*TN) {
      uint k=k0+i/TN, pos=n0+i%TN;
      float v=0.0;
      if(pos<size_n && k<total) {
        uint batch=pos/(p.oh*p.ow), outpos=pos%(p.oh*p.ow);
        uint inch=group*ci+k/(p.kh*p.kw), rem=k%(p.kh*p.kw);
        int iy=int((outpos/p.ow)*p.sh)-int(p.ph)+int((rem/p.kw)*p.dh);
        int ix=int((outpos%p.ow)*p.sw)-int(p.pw)+int((rem%p.kw)*p.dw);
        if(iy>=0 && iy<int(p.h) && ix>=0 && ix<int(p.w))
          v=x[((batch*p.cin+inch)*p.h+uint(iy))*p.w+uint(ix)];
      }
      b_tile[i]=v;
    }
    barrier();
    for(uint k=0;k<TK;++k) acc+=a_tile[ly*TK+k]*b_tile[k*TN+lx];
    barrier();
  }
  if(m0+ly<co && spatial<size_n) {
    uint batch=spatial/(p.oh*p.ow), pos=spatial%(p.oh*p.ow);
    y[(batch*p.cout+oc)*p.oh*p.ow+pos]=acc+bias[oc];
  }
}
