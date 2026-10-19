from pathlib import Path
s=Path('/tmp/pv-fma40d.glsl').read_text().replace('out uint mantissa,out int exponent','uint& mantissa,int& exponent')
prefix='''#include <algorithm>
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
'''
suffix='''int main(){std::mt19937 rng(8841);for(int i=0;i<2000000;i++){float a=uintBitsToFloat(rng()),b=uintBitsToFloat(rng()),c=uintBitsToFloat(rng());float actual=fma_refined(a,b,c),expected=std::fma(a,b,c);if(!(std::isnan(actual)&&std::isnan(expected))&&floatBitsToUint(actual)!=floatBitsToUint(expected)){std::cout<<std::hex<<floatBitsToUint(a)<<" "<<floatBitsToUint(b)<<" "<<floatBitsToUint(c)<<" got "<<floatBitsToUint(actual)<<" expected "<<floatBitsToUint(expected)<<"\\n";return 1;}}std::cout<<"2000000 exact FMA bit comparisons passed\\n";}
'''
Path('/tmp/pv-test-fma40d.cpp').write_text(prefix+s+suffix)
