// Scale by integer exponent fields so subnormals and huge magnitudes do not
// depend on the device's floating-point denormal/reciprocal behavior.
float portable_hypot(float a,float b){
 uint aa=floatBitsToUint(a)&0x7fffffffu,bb=floatBitsToUint(b)&0x7fffffffu;
 if(aa==0x7f800000u||bb==0x7f800000u)return uintBitsToFloat(0x7f800000u);
 if(aa>0x7f800000u||bb>0x7f800000u)return uintBitsToFloat(0x7fc00000u);
 uint hi=max(aa,bb),lo=min(aa,bb);if(hi==0u)return 0.;
 int exponent=int(hi>>23)-127;uint mantissa=hi&0x7fffffu;
 if(exponent==-127){int shift=23-findMSB(mantissa);mantissa<<=shift;exponent=-126-shift;}else mantissa|=0x800000u;
 float normalized=uintBitsToFloat(0x3f800000u|(mantissa&0x7fffffu));float ratio=divide_refined(uintBitsToFloat(lo),uintBitsToFloat(hi));
 float product=normalized*sqrt(1.+ratio*ratio);uint value=floatBitsToUint(product);int result_exp=int(value>>23)-127+exponent;uint significand=(value&0x7fffffu)|0x800000u;
 if(result_exp>127)return uintBitsToFloat(0x7f800000u);
 if(result_exp>=-126)return uintBitsToFloat((uint(result_exp+127)<<23)|(significand&0x7fffffu));
 uint shift=uint(-126-result_exp);if(shift>24u)return 0.;uint outv=significand>>shift,rem=significand&((1u<<shift)-1u),midpoint=1u<<(shift-1u);if(rem>midpoint||(rem==midpoint&&(outv&1u)!=0u))outv++;return uintBitsToFloat(outv);
}
