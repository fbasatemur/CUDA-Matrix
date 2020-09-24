#pragma once
#include "CpuGpuMat.h"
#include "device_launch_parameters.h"
#include "KernelGpu.cuh"
#include <math.h>


__global__ void gpuMatrixMult(float* gpuMat1, float* gpuMat2, float* gpuMat3, int m1Rows, int m1Cols, int m2Cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;

	if (row < m1Rows && col < m2Cols) {
		for (int i = 0; i < m1Cols; i++) {

			sum += gpuMat1[row * m1Cols + i] * gpuMat2[i * m2Cols + col];
		}
		gpuMat3[row * m2Cols + col] = sum;
	}
}

void gpuMatrixMultiplication(struct CpuGpuMat* Mat1, struct CpuGpuMat* Mat2, struct CpuGpuMat* Mat3)
{
	//vscc
	int threadsPerBlock = 32;

	int gridCols = ceil(double(Mat2->Cols) / double(threadsPerBlock));
	int gridRows = ceil(double(Mat1->Rows) / double(threadsPerBlock));

	dim3 gridDim(gridCols, gridRows);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);

	//nvcc
	gpuMatrixMult << < gridDim, blockDim >> > ((float*)Mat1->gpuP, (float*)Mat2->gpuP, (float*)Mat3->gpuP, Mat1->Rows, Mat1->Cols, Mat2->Cols);
}
