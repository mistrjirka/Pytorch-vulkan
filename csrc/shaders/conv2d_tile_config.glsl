// Tile parameters affect only work partitioning, never convolution semantics.
#ifndef CONV_TM
#define CONV_TM 8
#endif
#ifndef CONV_TN
#define CONV_TN 16
#endif
#define CONV_TK 16
layout(local_size_x=CONV_TN,local_size_y=CONV_TM,local_size_z=1) in;
const uint TM=CONV_TM;
const uint TN=CONV_TN;
const uint TK=CONV_TK;
