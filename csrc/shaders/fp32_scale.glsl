// Correctly rounded binary32 power-of-two scaling, including subnormals.
// Integer manipulation avoids device-dependent flush-to-zero arithmetic.
float scale_pow2_fp32(float x,int shift){
 uint raw=floatBitsToUint(x),sign=raw&0x80000000u,a=raw&0x7fffffffu;
 if(a==0u||a>=0x7f800000u)return x;
 int e=int(a>>23);uint m=a&0x7fffffu;
 if(e==0){int adjust=23-findMSB(m);m<<=adjust;e=1-adjust;}else m|=0x800000u;
 e+=shift;if(e>=255)return uintBitsToFloat(sign|0x7f800000u);
 if(e>0)return uintBitsToFloat(sign|(uint(e)<<23)|(m&0x7fffffu));
 uint right=uint(1-e);if(right>24u)return uintBitsToFloat(sign);
 uint q=m>>right,remainder=m&((1u<<right)-1u),halfway=1u<<(right-1u);
 if(remainder>halfway||(remainder==halfway&&(q&1u)!=0u))q++;
 return uintBitsToFloat(sign|q);
}
