// IEEE binary32 division with round-to-nearest-even. A hardware quotient is
// only an estimate: integer mantissa products establish the exact rounding.
// Uses 32-bit integers on all devices; no float64, int64, or fused-FMA contract.
uint divide_significand(uint a,uint b,int fraction_bits){
 uint q=1u,r=a-b;
 for(int i=0;i<fraction_bits;i++){r<<=1;q<<=1;if(r>=b){r-=b;q++;}}
 uint twice=r<<1;if(twice>b||(twice==b&&(q&1u)!=0))q++;
 return q;
}
float divide_refined(float a,float b){
 uint ua=floatBitsToUint(a),ub=floatBitsToUint(b),sign=(ua^ub)&0x80000000u;
 uint aa=ua&0x7fffffffu,bb=ub&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0&&bb==0)||(aa==0x7f800000u&&bb==0x7f800000u))return uintBitsToFloat(0x7fc00000u);
 if(bb==0||aa==0x7f800000u)return uintBitsToFloat(sign|0x7f800000u);
 if(aa==0||bb==0x7f800000u)return uintBitsToFloat(sign);
 int ea=int(aa>>23),eb=int(bb>>23);uint ma=aa&0x7fffffu,mb=bb&0x7fffffu;
 if(ea==0){int shift=23-findMSB(ma);ma<<=shift;ea=1-shift;}else ma|=0x800000u;
 if(eb==0){int shift=23-findMSB(mb);mb<<=shift;eb=1-shift;}else mb|=0x800000u;
 int exponent=ea-eb+127;if(ma<mb){ma<<=1;exponent--;}
 if(exponent>=255)return uintBitsToFloat(sign|0x7f800000u);
 if(exponent<=0){if(exponent< -23)return uintBitsToFloat(sign);if(exponent== -23)return uintBitsToFloat(sign|uint(ma>mb));return uintBitsToFloat(sign|divide_significand(ma,mb,exponent+22));}
 uint estimate=floatBitsToUint(a/b)&0x7fffffffu;int qe=int(estimate>>23);
 uint q=qe==exponent?((estimate&0x7fffffu)|0x800000u):qe>exponent?0x1000000u:0x800000u;
 uint hi,lo;umulExtended(q,mb,hi,lo);uvec2 product=uvec2(lo,hi),numerator=uvec2(ma<<23,ma>>9);
 bool above=numerator.y>product.y||(numerator.y==product.y&&numerator.x>=product.x);
 uvec2 big=above?numerator:product,small=above?product:numerator;
 uvec2 diff=uvec2(big.x-small.x,big.y-small.y-uint(big.x<small.x));
 if(diff.y!=0)q=divide_significand(ma,mb,23);
 else{uint whole=diff.x/mb,remainder=diff.x-whole*mb;uint candidate=above?q+whole:q-whole;uint twice=remainder<<1;if(twice>mb||(twice==mb&&(candidate&1u)!=0))whole++;q=above?q+whole:q-whole;}
 if(q>=0x1000000u){q>>=1;exponent++;}
 if(exponent>=255)return uintBitsToFloat(sign|0x7f800000u);
 return uintBitsToFloat(sign|(uint(exponent)<<23)|(q&0x7fffffu));
}
