// dX: [Cin/group, Cout/group*KH*KW] times gathered dY.
// No zero-insertion or im2col workspace; stride/dilation stay runtime values.
#include "conv2d_tile_config.glsl"
layout(set=0,binding=0) readonly buffer GradOut { float go[]; };
layout(set=0,binding=1) readonly buffer Weight { float weight[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float dx[]; };
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
void store_result(uint c,uint spatial,uint group,float value) {
  uint ci=p.cin/p.groups;
  if(c<ci && spatial<p.n*p.h*p.w) {
    uint batch=spatial/(p.h*p.w),pos=spatial%(p.h*p.w);
    dx[(batch*p.cin+group*ci+c)*p.h*p.w+pos]=value;
  }
}
void main() {
  uint lx=gl_LocalInvocationID.x, ly=gl_LocalInvocationID.y, tid=ly*WG_N+lx;
  uint ci=p.cin/p.groups, co=p.cout/p.groups;
  uint tiles=(ci+TM-1)/TM, group=gl_WorkGroupID.y/tiles;
  uint m0=(gl_WorkGroupID.y%tiles)*TM, n0=gl_WorkGroupID.x*TN;
  uint total=co*KERNEL_H*KERNEL_W, size_n=p.n*p.h*p.w;
#ifdef CONV_HOIST_SPATIAL
  // N-column indexing does not depend on the reduction coordinate K.
  // Hoist spatial quotient/remainder calculations outside the K loop.
  uint lane_spatial=n0+tid%TN;
  uint lane_batch=0u, lane_y=0u, lane_x=0u;
  if(lane_spatial<size_n) {
    uint pos=lane_spatial%(p.h*p.w);
    lane_batch=lane_spatial/(p.h*p.w);
    lane_y=pos/p.w;
    lane_x=pos%p.w;
  }
#endif
#if CONV_RM == 2
  vec4 acc=vec4(0.0);
#else
  float acc=0.0;
#endif
  for(uint k0=0;k0<total;k0+=TK) {
    for(uint i=tid;i<TM*TK;i+=THREADS) {
      uint c=m0+i/TK, k=k0+i%TK;
      float v=0.0;
      if(c<ci && k<total) {
        uint oc=group*co+k/(KERNEL_H*KERNEL_W);
        v=weight[(oc*ci+c)*KERNEL_H*KERNEL_W+k%(KERNEL_H*KERNEL_W)];
      }
      a[i]=v;
    }
    for(uint i=tid;i<TK*TN;i+=THREADS) {
      uint k=k0+i/TN, spatial=n0+i%TN;
      float v=0.0;
      if(spatial<size_n && k<total) {
#ifdef CONV_HOIST_SPATIAL
        uint oc=group*co+k/(KERNEL_H*KERNEL_W), rem=k%(KERNEL_H*KERNEL_W);
        int yh=int(lane_y)+int(p.ph)-int((rem/KERNEL_W)*p.dh);
        int xw=int(lane_x)+int(p.pw)-int((rem%KERNEL_W)*p.dw);
        uint batch=lane_batch;
#else
        uint batch=spatial/(p.h*p.w), pos=spatial%(p.h*p.w);
        uint oc=group*co+k/(KERNEL_H*KERNEL_W), rem=k%(KERNEL_H*KERNEL_W);
        int yh=int(pos/p.w)+int(p.ph)-int((rem/KERNEL_W)*p.dh);
        int xw=int(pos%p.w)+int(p.pw)-int((rem%KERNEL_W)*p.dw);
#endif
        if(yh>=0 && xw>=0) {
          uint oy=uint(yh), ox=uint(xw);
          bool aligned=true;
          // Unit stride needs neither divisibility checks nor coordinate
          // division. This uniform geometry guard applies to every model,
          // filter size, dilation and group count; other strides stay general.
          if(p.sh!=1u || p.sw!=1u) {
            aligned=(oy%p.sh==0u && ox%p.sw==0u);
            oy/=p.sh; ox/=p.sw;
          }
          if(aligned && oy<p.oh && ox<p.ow)
            v=go[((batch*p.cout+oc)*p.oh+oy)*p.ow+ox];
        }
      }
      b[i]=v;
    }
    barrier();
    #include "conv2d_tile_accumulate.glsl"
    barrier();
  }
  #include "conv2d_tile_store.glsl"
}
