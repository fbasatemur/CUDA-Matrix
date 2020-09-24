#include "CpuGpuMat.h"
#include "KernelGpu.cuh"
#include <cuda_runtime_api.h>						// cudaDeviceSynchronize()
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>


void cpuMatrixMultiplication(float* cpuMat1, float* cpuMat2, float* cpuMat3, int m1Rows, int m1Cols, int m2Cols)
{
	float sum = 0.0;
	for (int row = 0; row < m1Rows; row++)
	{
		for (int col = 0; col < m2Cols; col++)
		{
			sum = 0.0;
			for (int i = 0; i < m1Cols; i++)
			{
				sum += cpuMat1[row * m1Cols + i] * cpuMat2[i * m2Cols + col];
			}
			cpuMat3[row * m2Cols + col] = sum;
		}
	}
}


int main()
{
	struct CpuGpuMat Mat1;
	struct CpuGpuMat Mat2;
	struct CpuGpuMat Mat3;

	// matrix1
	Mat1.Rows = 1024;
	Mat1.Cols = 625;

	// matrix2
	Mat2.Rows = 625;
	Mat2.Cols = 5;

	// matrix3 (result matrix)
	Mat3.Rows = Mat1.Rows;
	Mat3.Cols = Mat2.Cols;

	Mat1.Size = Mat1.Rows * Mat1.Cols;			// matrix1 size
	Mat2.Size = Mat2.Rows * Mat2.Cols;			// matrix2 size
	Mat3.Size = Mat3.Rows * Mat3.Cols;			// result matrix3 size


	// cpu and gpu memory allocation
	Mat1.cpuP = (void*)malloc(Mat1.Size * sizeof(float));
	Mat2.cpuP = (void*)malloc(Mat2.Size * sizeof(float));
	Mat3.cpuP = (void*)malloc(Mat3.Size * sizeof(float));

	cudaError_t result1 = cudaMalloc(&Mat1.gpuP, Mat1.Size * sizeof(float));
	cudaError_t result2 = cudaMalloc(&Mat2.gpuP, Mat2.Size * sizeof(float));
	cudaError_t result3 = cudaMalloc(&Mat3.gpuP, Mat3.Size * sizeof(float));
	assert(result1 == cudaSuccess || result2 == cudaSuccess || result3 == cudaSuccess);


	// set values to cpu memory
	float* cpuFloatP1 = (float*)Mat1.cpuP;
	float* cpuFloatP2 = (float*)Mat2.cpuP;

	for (int i = 0; i < Mat1.Size; i++)
		cpuFloatP1[i] = i;

	for (int i = 0; i < Mat2.Size; i++)
		cpuFloatP2[i] = 1;


	/*
		Host => ram
		Device => graphics memory
	*/

	// Host -> Device
	result1 = cudaMemcpy(Mat1.gpuP, Mat1.cpuP, Mat1.Size * sizeof(float), cudaMemcpyHostToDevice);
	result2 = cudaMemcpy(Mat2.gpuP, Mat2.cpuP, Mat2.Size * sizeof(float), cudaMemcpyHostToDevice);
	result3 = cudaMemcpy(Mat3.gpuP, Mat3.cpuP, Mat3.Size * sizeof(float), cudaMemcpyHostToDevice);
	assert(result1 == cudaSuccess || result2 == cudaSuccess || result3 == cudaSuccess);


	gpuMatrixMultiplication(&Mat1, &Mat2, &Mat3);
	// cpuMatrixMultiplication((float*)Mat1.cpuP, (float*)Mat2.cpuP, (float*)Mat3.cpuP, Mat1.Rows, Mat1.Cols, Mat2.Cols);


	// Device -> Host
	cudaError_t result = cudaMemcpy(Mat3.cpuP, Mat3.gpuP, Mat3.Size * sizeof(float), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	cudaDeviceSynchronize();


	// show result of matrix muliplication  
	float* cpuFloatP = (float*)Mat3.cpuP;

	for (int i = 0; i < Mat3.Size; i++)
		printf("%d \t %f \n", i, cpuFloatP[i]);


	// cpu and gpu memory free
	result1 = cudaFree(Mat1.gpuP);
	result2 = cudaFree(Mat2.gpuP);
	result3 = cudaFree(Mat3.gpuP);
	assert(result1 == cudaSuccess || result2 == cudaSuccess || result3 == cudaSuccess);

	free(Mat1.cpuP);
	free(Mat2.cpuP);
	free(Mat3.cpuP);

	return 0;
}
