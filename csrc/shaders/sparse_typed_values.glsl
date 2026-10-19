// Exact storage access for boolean, signed integer and complex64 COO values.
// Subword destinations use disjoint atomic masks so adjacent entries cannot
// overwrite one another. Integer accumulation wraps at the tensor's width.
uvec2 sparse_read(uint i){
 if(p.bytes==8u)return uvec2(v[2u*i],v[2u*i+1u]);
 if(p.bytes==4u)return uvec2(v[i],0u);
 uint bit=(i*p.bytes%4u)*8u,mask=(1u<<(p.bytes*8u))-1u;
 return uvec2((v[i*p.bytes/4u]>>bit)&mask,0u);
}
uvec2 sparse_add(uvec2 a,uvec2 b){
 if(p.kind==1u)return uvec2(uint(a.x!=0u||b.x!=0u),0u);
 if(p.kind==4u)return floatBitsToUint(uintBitsToFloat(a)+uintBitsToFloat(b));
 uint lo=a.x+b.x;return uvec2(lo,a.y+b.y+uint(lo<a.x));
}
void sparse_write(uint i,uvec2 value){
 if(p.bytes==8u){result_data[2u*i]=value.x;result_data[2u*i+1u]=value.y;return;}
 if(p.bytes==4u){result_data[i]=value.x;return;}
 uint at=i*p.bytes/4u,bit=(i*p.bytes%4u)*8u,mask=((1u<<(p.bytes*8u))-1u)<<bit;
 atomicAnd(result_data[at],~mask);atomicOr(result_data[at],(value.x<<bit)&mask);
}
