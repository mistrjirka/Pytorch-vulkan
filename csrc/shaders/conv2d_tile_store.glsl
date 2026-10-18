// Each invocation owns either one output or a fixed 2x2 register block.
// The phase-specific store function guards the exact runtime boundaries.
#if CONV_RM == 2
store_result(m0+ly,n0+lx,group,acc.x);
store_result(m0+ly,n0+lx+WG_N,group,acc.y);
store_result(m0+ly+WG_M,n0+lx,group,acc.z);
store_result(m0+ly+WG_M,n0+lx+WG_N,group,acc.w);
#else
store_result(m0+ly,n0+lx,group,acc);
#endif
