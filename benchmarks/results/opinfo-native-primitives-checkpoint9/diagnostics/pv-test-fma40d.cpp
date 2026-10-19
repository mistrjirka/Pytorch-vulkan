#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
using uint=uint32_t;
struct uvec3 {uint x,y,z;uvec3(uint a=0):x(a),y(a),z(a){}uvec3(uint a,uint b,uint c):x(a),y(b),z(c){}uint& operator[](uint i){return i==0?x:i==1?y:z;}uint operator[](uint i)const{return i==0?x:i==1?y:z;}};
bool any(uvec3 v){return v.x||v.y||v.z;}bool all(uvec3 v){return v.x&&v.y&&v.z;}
uvec3 equal(uvec3 a,uvec3 b){return {a.x==b.x,a.y==b.y,a.z==b.z};}
uvec3 notEqual(uvec3 a,uvec3 b){return {a.x!=b.x,a.y!=b.y,a.z!=b.z};}
uint floatBitsToUint(float x){return std::bit_cast<uint>(x);}float uintBitsToFloat(uint x){return std::bit_cast<float>(x);}
int findMSB(uint x){return x?int(std::bit_width(x))-1:-1;}void umulExtended(uint a,uint b,uint& hi,uint& lo){uint64_t x=uint64_t(a)*b;hi=x>>32;lo=x;}
using std::max;using std::fma;
// A portable, single-rounded binary32 a*b+c using only uint32 arithmetic.
// GLSL fma accuracy can depend on the driver. Keep 96 significant bits with
// a sticky low bit; that covers the 48-bit product and cancellation exactly
// whenever cancellation can affect binary32 rounding. No shaderFloat64/Int64.
uvec3 fma_shift_left(uvec3 v,uint shift){
 if(shift>=96u)return uvec3(0u);
 uint words=shift/32u,bits=shift%32u;uvec3 r=uvec3(0u);
 for(uint i=0u;i<3u;i++)if(i>=words){uint j=i-words;r[i]=v[j]<<bits;if(bits!=0u&&j>0u)r[i]|=v[j-1u]>>(32u-bits);}
 return r;
}
uvec3 fma_shift_right(uvec3 v,uint shift){
 if(shift>=96u)return uvec3(0u);
 uint words=shift/32u,bits=shift%32u;uvec3 r=uvec3(0u);
 for(uint i=0u;i<3u;i++){uint j=i+words;if(j<3u){r[i]=v[j]>>bits;if(bits!=0u&&j+1u<3u)r[i]|=v[j+1u]<<(32u-bits);}}
 return r;
}
bool fma_low_bits(uvec3 v,uint count){
 if(count>=96u)return any(notEqual(v,uvec3(0u)));
 uint words=count/32u,bits=count%32u;
 for(uint i=0u;i<words;i++)if(v[i]!=0u)return true;
 return bits!=0u&&(v[words]&((1u<<bits)-1u))!=0u;
}
uvec3 fma_align(uvec3 v,int shift){
 if(shift>=0)return fma_shift_left(v,uint(shift));
 uint right=uint(-shift);uvec3 r=fma_shift_right(v,right);r.x|=uint(fma_low_bits(v,right));return r;
}
uvec3 fma_add_words(uvec3 a,uvec3 b){
 uint lo=a.x+b.x,carry=uint(lo<a.x),mid=a.y+b.y,next=uint(mid<a.y);uint with_carry=mid+carry;next+=uint(with_carry<mid);return uvec3(lo,with_carry,a.z+b.z+next);
}
uvec3 fma_sub_words(uvec3 a,uvec3 b){
 uint borrow=uint(a.x<b.x),mid=a.y-b.y,next=uint(a.y<b.y);uint with_borrow=mid-borrow;next+=uint(mid<borrow);return uvec3(a.x-b.x,with_borrow,a.z-b.z-next);
}
bool fma_less_words(uvec3 a,uvec3 b){return a.z<b.z||(a.z==b.z&&(a.y<b.y||(a.y==b.y&&a.x<b.x)));}
uint fma_round_words(uvec3 value,int shift){
 if(shift<=0)return fma_shift_left(value,uint(-shift)).x;
 uint s=uint(shift),q=fma_shift_right(value,s).x;
 bool guard=s<=96u&&((value[(s-1u)/32u]>>((s-1u)%32u))&1u)!=0u;
 if(guard&&(fma_low_bits(value,s-1u)||(q&1u)!=0u))q++;
 return q;
}
void fma_decode(uint raw,uint& mantissa,int& exponent){
 int e=int(raw>>23);mantissa=raw&0x7fffffu;
 if(e==0){int shift=23-findMSB(mantissa);mantissa<<=shift;exponent=-126-shift;}else{mantissa|=0x800000u;exponent=e-127;}
}
float fma_refined(float a,float b,float c){
 uint ua=floatBitsToUint(a),ub=floatBitsToUint(b),uc=floatBitsToUint(c);
 uint aa=ua&0x7fffffffu,bb=ub&0x7fffffffu,cc=uc&0x7fffffffu;
 if(aa>=0x7f800000u||bb>=0x7f800000u||cc>=0x7f800000u)return fma(a,b,c);
 uint product_sign=(ua^ub)&0x80000000u,addend_sign=uc&0x80000000u;
 if(aa==0u||bb==0u){if(cc!=0u)return c;return uintBitsToFloat(product_sign&addend_sign);}
 uint ma,mb,mc=0u;int ea,eb,ec=-1000;fma_decode(aa,ma,ea);fma_decode(bb,mb,eb);if(cc!=0u)fma_decode(cc,mc,ec);
 int product_exponent=ea+eb-46,addend_exponent=ec-23;
 int base_exponent=max(product_exponent+47,addend_exponent+23)-94;
 uint hi,lo;umulExtended(ma,mb,hi,lo);
 uvec3 product=fma_align(uvec3(lo,hi,0u),product_exponent-base_exponent);
 uvec3 addend=fma_align(uvec3(mc,0u,0u),addend_exponent-base_exponent),result;uint sign=product_sign;
 if(product_sign==addend_sign)result=fma_add_words(product,addend);
 else if(fma_less_words(product,addend)){result=fma_sub_words(addend,product);sign=addend_sign;}else result=fma_sub_words(product,addend);
 if(all(equal(result,uvec3(0u))))return 0.0;
 int top=result.z!=0u?64+findMSB(result.z):result.y!=0u?32+findMSB(result.y):findMSB(result.x);
 int exponent=base_exponent+top;
 if(exponent< -126)return uintBitsToFloat(sign|fma_round_words(result,-149-base_exponent));
 uint q=fma_round_words(result,top-23);if(q>=0x1000000u){q>>=1;exponent++;}
 if(exponent>127)return uintBitsToFloat(sign|0x7f800000u);
 return uintBitsToFloat(sign|(uint(exponent+127)<<23)|(q&0x7fffffu));
}
int main(){std::mt19937 rng(8841);for(int i=0;i<2000000;i++){float a=uintBitsToFloat(rng()),b=uintBitsToFloat(rng()),c=uintBitsToFloat(rng());float actual=fma_refined(a,b,c),expected=std::fma(a,b,c);if(!(std::isnan(actual)&&std::isnan(expected))&&floatBitsToUint(actual)!=floatBitsToUint(expected)){std::cout<<std::hex<<floatBitsToUint(a)<<" "<<floatBitsToUint(b)<<" "<<floatBitsToUint(c)<<" got "<<floatBitsToUint(actual)<<" expected "<<floatBitsToUint(expected)<<"\n";return 1;}}std::cout<<"2000000 exact FMA bit comparisons passed\n";}
