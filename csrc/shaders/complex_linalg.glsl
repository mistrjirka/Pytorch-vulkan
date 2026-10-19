// Shared complex64 arithmetic for LU factorization and triangular solves.
#include "fp32_math.glsl"
#include "fp32_scale.glsl"
vec2 complex_multiply(vec2 a,vec2 b){return vec2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
vec2 complex_divide(vec2 a,vec2 b){
 // Normalize the divisor before Smith division to avoid overflow in its sum
 // and device-dependent flushing of subnormal divisors.
 uint largest=max(floatBitsToUint(b.x)&0x7fffffffu,floatBitsToUint(b.y)&0x7fffffffu);
 if(largest!=0u&&largest<0x7f800000u){
  int exponent=int(largest>>23)-126;if((largest>>23)==0u)exponent=findMSB(largest)-148;
  a=vec2(scale_pow2_fp32(a.x,-exponent),scale_pow2_fp32(a.y,-exponent));
  b=vec2(scale_pow2_fp32(b.x,-exponent),scale_pow2_fp32(b.y,-exponent));
 }
 if(abs(b.x)>=abs(b.y)){
  float r=divide_refined(b.y,b.x),den=b.x+b.y*r;
  return vec2(divide_refined(a.x+a.y*r,den),divide_refined(a.y-a.x*r,den));
 }
 float r=divide_refined(b.x,b.y),den=b.y+b.x*r;
 return vec2(divide_refined(a.x*r+a.y,den),divide_refined(a.y*r-a.x,den));
}
