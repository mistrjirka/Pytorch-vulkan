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
  uint total=co*p.kh*p.kw, size_n=p.n*p.h*p.w;
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
        uint oc=group*co+k/(p.kh*p.kw);
        v=weight[(oc*ci+c)*p.kh*p.kw+k%(p.kh*p.kw)];
      }
      a[i]=v;
    }
    for(uint i=tid;i<TK*TN;i+=THREADS) {
      uint k=k0+i/TN, spatial=n0+i%TN;
      float v=0.0;
      if(spatial<size_n && k<total) {
        uint batch=spatial/(p.h*p.w), pos=spatial%(p.h*p.w);
        uint oc=group*co+k/(p.kh*p.kw), rem=k%(p.kh*p.kw);
        int yh=int(pos/p.w)+int(p.ph)-int((rem/p.kw)*p.dh);
        int xw=int(pos%p.w)+int(p.pw)-int((rem%p.kw)*p.dw);
        if(yh>=0 && xw>=0 && yh%int(p.sh)==0 && xw%int(p.sw)==0) {
          uint oy=uint(yh)/p.sh, ox=uint(xw)/p.sw;
          if(oy<p.oh && ox<p.ow)
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
