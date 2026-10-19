#ifndef TORCH_VULKAN_FP32_TRIG_GLSL
#define TORCH_VULKAN_FP32_TRIG_GLSL
// Full-range FP32 argument reduction. 2/pi is stored as 256 fixed-point bits;
// 24x256-bit multiplication uses only portable uint32 operations.
vec2 portable_sincos(float argument){uint original=floatBitsToUint(argument),bits=original&0x7fffffffu;if(bits>=0x7f800000u)return vec2(uintBitsToFloat(0x7fc00000u));float a=abs(argument),r=a;uint quadrant=0u;
 if(a>.7853981633974483){if(a<8192.){float k=roundEven(a*.6366197723675813);precise float reduced=a-k*1.5703125;reduced-=k*.0004837512969970703125;reduced-=k*7.549789954891882e-8;r=reduced;quadrant=uint(k)&3u;}
 else{const uint c[8]=uint[8](0xdebbc561u,0xfe5163abu,0x3c439041u,0xdb629599u,0xf534ddc0u,0xfc2757d1u,0x4e441529u,0xa2f9836eu);uint product[9];uint carry=0u,m=(bits&0x7fffffu)|0x800000u;for(uint i=0u;i<8u;i++){uint hi,lo;umulExtended(c[i],m,hi,lo);uint value=lo+carry;product[i]=value;carry=hi+uint(value<lo);}product[8]=carry;uint shift=279u-uint(int(bits>>23)-127);quadrant=((product[shift/32u]>>(shift&31u))&1u)|(((product[(shift+1u)/32u]>>((shift+1u)&31u))&1u)<<1u);bool upper=((product[(shift-1u)/32u]>>((shift-1u)&31u))&1u)!=0u;if(upper){quadrant=(quadrant+1u)&3u;carry=1u;for(uint i=0u;i<9u;i++){uint value=~product[i]+carry;carry=uint(carry!=0u&&value==0u);product[i]=value;}}
 float fraction=0.,weight=.5;for(uint i=1u;i<=64u;i++){uint bit=shift-i;fraction+=float((product[bit/32u]>>(bit&31u))&1u)*weight;weight*=.5;}r=(upper?-fraction:fraction)*1.5707963267948966;}}
 float z=r*r;float sine=r*(1.+z*(-1./6.+z*(1./120.+z*(-1./5040.+z*(1./362880.+z*(-1./39916800.+z/6227020800.))))));float cosine=1.+z*(-.5+z*(1./24.+z*(-1./720.+z*(1./40320.+z*(-1./3628800.+z/479001600.)))));vec2 result=quadrant==0u?vec2(sine,cosine):quadrant==1u?vec2(cosine,-sine):quadrant==2u?vec2(-sine,-cosine):vec2(-cosine,sine);if((original&0x80000000u)!=0u)result.x=-result.x;return result;}
float portable_atan(float v){if((floatBitsToUint(v)&0x7fffffffu)<0x39800000u)return v;float a=abs(v);bool invert=a>1.;float z=invert?1./a:a;bool shift=z>.4142135623730951;if(shift)z=(z-1.)/(z+1.);float t=z*z;float poly=1./21.;poly=-1./19.+t*poly;poly=1./17.+t*poly;poly=-1./15.+t*poly;poly=1./13.+t*poly;poly=-1./11.+t*poly;poly=1./9.+t*poly;poly=-1./7.+t*poly;poly=1./5.+t*poly;poly=-1./3.+t*poly;float r=z*(1.+t*poly);if(shift)r+=.7853981633974483;if(invert)r=1.5707963267948966-r;return (floatBitsToUint(v)&0x80000000u)!=0u?-r:r;}
float portable_sin(float a){return portable_sincos(a).x;}
float portable_cos(float a){return portable_sincos(a).y;}
#endif
