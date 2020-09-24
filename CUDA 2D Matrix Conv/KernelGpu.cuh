#pragma once
#include "CpuGpuMat.h"

#ifdef __cplusplus									
extern "C"
#endif // __cplusplus


void gpuMatrixConvulation(struct CpuGpuMat* Mat1, struct CpuGpuMat* Mat2, struct CpuGpuMat* Mat3);