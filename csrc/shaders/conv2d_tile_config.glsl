// Tile parameters affect only work partitioning, never convolution semantics.
#ifndef CONV_TM
#define CONV_TM 8
#endif
#ifndef CONV_TN
#define CONV_TN 16
#endif
#ifndef CONV_RM
#define CONV_RM 1
#endif
#ifndef CONV_RN
#define CONV_RN 1
#endif
#if !((CONV_RM == 1 && CONV_RN == 1) || (CONV_RM == 2 && CONV_RN == 2))
#error Unsupported convolution register block
#endif
#if CONV_TM % CONV_RM != 0 || CONV_TN % CONV_RN != 0
#error Register block must divide the tile
#endif
#define CONV_TK 16
#define CONV_WG_M (CONV_TM / CONV_RM)
#define CONV_WG_N (CONV_TN / CONV_RN)
#if defined(CONV_HOIST_SPATIAL) && ((CONV_WG_M*CONV_WG_N)%CONV_TN != 0)
#error Hoisted spatial address requires THREADS divisible by TN
#endif
layout(local_size_x=CONV_WG_N,local_size_y=CONV_WG_M,local_size_z=1) in;
const uint TM=CONV_TM;
const uint TN=CONV_TN;
const uint TK=CONV_TK;
const uint WG_M=CONV_WG_M;
const uint WG_N=CONV_WG_N;
const uint THREADS=CONV_WG_M*CONV_WG_N;
