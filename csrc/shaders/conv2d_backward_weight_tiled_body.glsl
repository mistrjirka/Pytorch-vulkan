// dW = dY * im2col(X)^T, gathered a shared tile at a time.
// Fixed 256-element splits preserve the existing deterministic final reduction.
#include "conv2d_tile_config.glsl"
layout(set=0,binding=0) readonly buffer Input { float x[]; };
layout(set=0,binding=1) readonly buffer GradOut { float go[]; };
layout(set=0,binding=2) writeonly buffer GradWeight { float out_data[]; };
layout(push_constant) uniform Params {
  uint n,cin,cout,h,w,kh,kw,oh,ow;
  uint sh,sw,ph,pw,dh,dw,groups;
} p;
shared float a[TM*TK];
shared float b[TK*TN];
void main() {
  uint lx=gl_LocalInvocationID.x, ly=gl_LocalInvocationID.y, tid=ly*TN+lx;
  uint ci=p.cin/p.groups, co=p.cout/p.groups;
  uint cols=ci*p.kh*p.kw, tiles=(co+TM-1)/TM;
  uint group=gl_WorkGroupID.y/tiles, m0=(gl_WorkGroupID.y%tiles)*TM;
  uint n0=gl_WorkGroupID.x*TN;
  uint spatial=p.oh*p.ow, total=p.n*spatial, split=gl_WorkGroupID.z;
  float acc=0.0;
  for(uint k0=split*256;k0<min(total,(split+1)*256);k0+=TK) {
    for(uint i=tid;i<TM*TK;i+=TM*TN) {
      uint m=m0+i/TK, k=k0+i%TK, oc=group*co+m;
      a[i]=(m<co && k<total) ?
          go[((k/spatial)*p.cout+oc)*spatial+k%spatial] : 0.0;
    }
    for(uint i=tid;i<TK*TN;i+=TM*TN) {
      uint k=k0+i/TN, col=n0+i%TN;
      float v=0.0;
      if(col<cols && k<total) {
        uint ic=group*ci+col/(p.kh*p.kw), ky=(col/p.kw)%p.kh, kx=col%p.kw;
        uint pos=k%spatial;
        int iy=int((pos/p.ow)*p.sh)-int(p.ph)+int(ky*p.dh);
        int ix=int((pos%p.ow)*p.sw)-int(p.pw)+int(kx*p.dw);
        if(iy>=0 && iy<int(p.h) && ix>=0 && ix<int(p.w))
          v=x[(((k/spatial)*p.cin+ic)*p.h+uint(iy))*p.w+uint(ix)];
      }
      b[i]=v;
    }
    barrier();
    for(uint k=0;k<TK;++k) acc+=a[ly*TK+k]*b[k*TN+lx];
    barrier();
  }
  uint m=m0+ly, col=n0+lx, oc=group*co+m;
  if(m<co && col<cols) out_data[(split*p.cout+oc)*cols+col]=acc;
}
