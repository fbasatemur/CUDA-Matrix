#pragma once
#include "device_launch_parameters.h"
#include "CpuGpuMat.h"
#include "KernelGpu.cuh"
#include <math.h>


__global__ void gpuMatrixConv(float* gpuMat1, float* gpuMat2, float* gpuMat3, int m1Rows, int m1Cols, int mRowsCols, int m3Rows, int m3Cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;

	if (row < m3Rows && col < m3Cols) {
		for (int maskRow = 0; maskRow < mRowsCols; maskRow++) {
			for (int maskCol = 0; maskCol < mRowsCols; maskCol++) {
				sum += gpuMat1[(row + maskRow) * m1Cols + col + maskCol] * gpuMat2[maskRow * mRowsCols + maskCol];
			}
		}
		gpuMat3[row * m3Cols + col] = sum;
	}
}


void gpuMatrixConvulation(struct CpuGpuMat* Mat1, struct CpuGpuMat* Mat2, struct CpuGpuMat* Mat3)
{
	//vscc
	int threadsPerBlock = 32;

	int gridCols = ceil(double(Mat3->Cols) / double(threadsPerBlock));
	int gridRows = ceil(double(Mat3->Rows) / double(threadsPerBlock));

	dim3 gridDim(gridCols, gridRows);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);	// total 32x32=1024 threads

	//nvcc
	gpuMatrixConv << < gridDim, blockDim >> > ((float*)Mat1->gpuP, (float*)Mat2->gpuP, (float*)Mat3->gpuP, Mat1->Rows, Mat1->Cols, Mat2->Rows, Mat3->Rows, Mat3->Cols);
}