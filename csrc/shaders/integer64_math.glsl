// Portable exact arithmetic modulo 2^64 using pairs of uint32 words.
uvec2 plus64(uvec2 a,uvec2 b){uint lo=a.x+b.x;return uvec2(lo,a.y+b.y+uint(lo<a.x));}
uvec2 neg64(uvec2 a){return plus64(~a,uvec2(1,0));}
uvec2 minus64(uvec2 a,uvec2 b){return plus64(a,neg64(b));}
uvec2 times64(uvec2 a,uvec2 b){uint hi,lo;umulExtended(a.x,b.x,hi,lo);return uvec2(lo,hi+a.x*b.y+a.y*b.x);}
bool less(uvec2 a,uvec2 b){return a.y==b.y?a.x<b.x:a.y<b.y;}
uvec2 quotient(uvec2 a,uvec2 b,out uvec2 remainder){uvec2 q=uvec2(0);remainder=uvec2(0);for(int bit=63;bit>=0;bit--){bool carry=(remainder.y&0x80000000u)!=0u;remainder=uvec2((remainder.x<<1)|((bit>=32?a.y:a.x)>>(uint(bit)&31u)&1u),(remainder.y<<1)|(remainder.x>>31));if(carry||!less(remainder,b)){remainder=minus64(remainder,b);if(bit>=32)q.y|=1u<<(bit-32);else q.x|=1u<<bit;}}return q;}
uvec2 divide_trunc64(uvec2 a,uvec2 b){bool an=int(a.y)<0,bn=int(b.y)<0;uvec2 remainder;uvec2 q=quotient(an?neg64(a):a,bn?neg64(b):b,remainder);return an!=bn?neg64(q):q;}
