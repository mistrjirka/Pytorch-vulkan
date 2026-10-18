// Included inside the reduction loop after its shared-memory barrier.
#if CONV_RM == 2
// Four explicit scalar lanes: no dynamic accumulator indexing. The reduction
// is statically expanded so each shared load pair feeds four FP32 products.
#define CONV_ACCUMULATE(K) { \
  float a0=a[ly*TK+(K)],a1=a[(ly+WG_M)*TK+(K)]; \
  float b0=b[(K)*TN+lx],b1=b[(K)*TN+lx+WG_N]; \
  acc+=vec4(a0*b0,a0*b1,a1*b0,a1*b1); \
}
CONV_ACCUMULATE(0u)
CONV_ACCUMULATE(1u)
CONV_ACCUMULATE(2u)
CONV_ACCUMULATE(3u)
CONV_ACCUMULATE(4u)
CONV_ACCUMULATE(5u)
CONV_ACCUMULATE(6u)
CONV_ACCUMULATE(7u)
CONV_ACCUMULATE(8u)
CONV_ACCUMULATE(9u)
CONV_ACCUMULATE(10u)
CONV_ACCUMULATE(11u)
CONV_ACCUMULATE(12u)
CONV_ACCUMULATE(13u)
CONV_ACCUMULATE(14u)
CONV_ACCUMULATE(15u)
#undef CONV_ACCUMULATE
#else
for(uint k=0;k<TK;++k) acc+=a[ly*TK+k]*b[k*TN+lx];
#endif
