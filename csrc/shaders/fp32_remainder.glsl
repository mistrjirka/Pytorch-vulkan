// Binary32 remainder by exact significand reduction. Uses only uint32 and
// preserves signed zero, subnormals and IEEE invalid cases.
float exact_fmod(float a,float b){
 uint sign=floatBitsToUint(a)&0x80000000u,ax=floatBitsToUint(a)&0x7fffffffu,by=floatBitsToUint(b)&0x7fffffffu;
 if(by==0u||ax>=0x7f800000u||by>0x7f800000u)return uintBitsToFloat(0x7fc00000u);
 if(ax<by)return a;if(ax==by)return uintBitsToFloat(sign);
 int ex=int(ax>>23)-127,ey=int(by>>23)-127;uint mx=ax&0x7fffffu,my=by&0x7fffffu;
 if(ex==-127){int shift=23-findMSB(mx);mx<<=shift;ex=-126-shift;}else mx|=0x800000u;
 if(ey==-127){int shift=23-findMSB(my);my<<=shift;ey=-126-shift;}else my|=0x800000u;
 for(int i=ex-ey;i>=0;i--){if(mx>=my)mx-=my;if(i>0)mx<<=1;}
 if(mx==0u)return uintBitsToFloat(sign);int shift=23-findMSB(mx),exponent=ey-shift;mx<<=shift;
 uint bits=exponent>=-126?(uint(exponent+127)<<23)|(mx&0x7fffffu):mx>>uint(-126-exponent);
 return uintBitsToFloat(sign|bits);
}

uint shift_jam(uint a,uint d){if(d==0)return a;if(d>=32)return uint(a!=0u);return (a>>d)|uint((a<<uint(32-d))!=0u);}
// Correctly rounded nonnegative subtraction, including subnormal inputs.
uint positive_difference(uint a,uint b){if(b==0u)return a;if(a==b)return 0u;int ea=int(a>>23)-127,eb=int(b>>23)-127;uint ma=a&0x7fffffu,mb=b&0x7fffffu;if(ea==-127){ea=-126;}else ma|=0x800000u;if(eb==-127){eb=-126;}else mb|=0x800000u;uint m=(ma<<3)-shift_jam(mb<<3,uint(ea-eb));if(m==0u)return 0u;int shift=26-findMSB(m);m<<=shift;ea-=shift;if(ea< -126){m=shift_jam(m,uint(-126-ea));ea=-126;}uint lo=m&7u,hi=m>>3;if(lo>4u||(lo==4u&&(hi&1u)!=0u))hi++;if(hi>=0x1000000u){hi>>=1;ea++;}return hi<0x800000u?hi:(uint(ea+127)<<23)|(hi&0x7fffffu);}
float remainder_value(float a,float b){float mod=exact_fmod(a,b);uint m=floatBitsToUint(mod),v=floatBitsToUint(b);if((m&0x7fffffffu)!=0u&&(m&0x7fffffffu)<0x7f800000u&&((m^v)&0x80000000u)!=0u){if((v&0x7fffffffu)==0x7f800000u)return b;return uintBitsToFloat((v&0x80000000u)|positive_difference(v&0x7fffffffu,m&0x7fffffffu));}return mod;}
float floor_quotient(float a,float b){uint aa=floatBitsToUint(a),bb=floatBitsToUint(b);if((bb&0x7fffffffu)==0u)return divide_refined(a,b);float mod=exact_fmod(a,b);uint mm=floatBitsToUint(mod);if((mm&0x7fffffffu)>0x7f800000u)return mod;float numerator=uintBitsToFloat((aa&0x80000000u)|positive_difference(aa&0x7fffffffu,mm&0x7fffffffu));float q=divide_refined(numerator,b);if((mm&0x7fffffffu)!=0u&&((bb^mm)&0x80000000u)!=0u)q-=1.;if(q!=0.){float f=floor(q);if(q-f>.5)f+=1.;return f;}return uintBitsToFloat((aa^bb)&0x80000000u);}
