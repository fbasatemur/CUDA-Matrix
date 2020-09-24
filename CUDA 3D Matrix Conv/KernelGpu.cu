#pragma once
#include "device_launch_parameters.h"
#include "CpuGpuMat.h"
#include "KernelGpu.cuh"
#include <math.h>


__global__ void gpuMatrixConv3D(float* image, float* mask, float* result, int imageRows, int imageCols, int maskRC, int maskDepth, int resultRows, int resultCols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0;

	if (row < resultRows && col < resultCols)
	{
		int imageRowsCols = imageRows * imageCols;

		for (int maskRow = 0; maskRow < maskRC; maskRow++) {
			for (int maskCol = 0; maskCol < maskRC; maskCol++) {
				for (int dep = 0; dep < maskDepth; dep++)

					sum += image[(row + maskRow) * imageCols + col + maskCol + dep * imageRowsCols] * mask[maskRow * maskRC + maskCol + dep * maskDepth];
			}
		}
		result[row * resultCols + col] = sum;
	}
}


void gpuMatrixConvulation3D(struct CpuGpuMat* image, struct CpuGpuMat* mask, struct CpuGpuMat* result)
{
	//vscc
	int threadsPerBlock = 32;

	int gridCols = ceil(float(result->Cols) / float(threadsPerBlock));
	int gridRows = ceil(float(result->Rows) / float(threadsPerBlock));

	dim3 gridDim(gridCols, gridRows);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);		// total 32*32 = 1024 threads


	gpuMatrixConv3D << < gridDim, blockDim >> > ((float*)image->gpuP, (float*)mask->gpuP, (float*)result->gpuP, image->Rows, image->Cols, mask->Rows, mask->Depth, result->Rows, result->Cols);
}