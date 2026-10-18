// Original GLSL implementation of the Philox4x32-10 counter transformation.
// Integer multiplication uses core GLSL extended multiply; shaderInt64 is unnecessary.
uvec4 philox_counter(uvec2 seed,uvec4 c){
 uvec2 key=seed;
 for(int round=0;round<10;round++){
  uint hi0,lo0,hi1,lo1;umulExtended(0xD2511F53u,c.x,hi0,lo0);umulExtended(0xCD9E8D57u,c.z,hi1,lo1);
  c=uvec4(hi1^c.y^key.x,lo1,hi0^c.w^key.y,lo0);key+=uvec2(0x9E3779B9u,0xBB67AE85u);
 }
 return c;
}
uvec4 philox(uvec2 seed,uvec2 offset,uint block){uint low=offset.x+block;return philox_counter(seed,uvec4(low,offset.y+uint(low<offset.x),0,0));}
float uniform01(uint value){return float(value>>8)*0.000000059604644775390625;}
