#include "CpuGpuMat.h"
#include <stdlib.h>
#include <assert.h>
#include "KernelGpu.cuh"
#include <cuda_runtime_api.h>						// cudaDeviceSynchronize()
#include <iostream>

int main() {

	struct CpuGpuMat Mat1;
	struct CpuGpuMat Mat2;
	struct CpuGpuMat Mat3;
	int maskSize = 3;

	// matrix1
	Mat1.Rows = 4;
	Mat1.Cols = 4;

	// matrix2 (mask)
	Mat2.Rows = maskSize;
	Mat2.Cols = maskSize;

	// matrix3 (result matrix)
	Mat3.Rows = Mat1.Rows - maskSize + 1;
	Mat3.Cols = Mat1.Cols - maskSize + 1;

	Mat1.Size = Mat1.Rows * Mat1.Cols;			// matrix1 size
	Mat2.Size = Mat2.Rows * Mat2.Cols;			// matrix2 size
	Mat3.Size = Mat3.Rows * Mat3.Cols;			// matrix3 size


	// cpu and gpu memory allocation
	Mat1.cpuP = (void*)malloc(Mat1.Size * sizeof(float));
	Mat2.cpuP = new float[Mat2.Size]{ 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111 };		// mean filter
	Mat3.cpuP = (void*)malloc(Mat3.Size * sizeof(float));

	cudaError_t result1 = cudaMalloc(&Mat1.gpuP, Mat1.Size * sizeof(float));
	cudaError_t result2 = cudaMalloc(&Mat2.gpuP, Mat2.Size * sizeof(float));
	cudaError_t result3 = cudaMalloc(&Mat3.gpuP, Mat3.Size * sizeof(float));
	assert(result1 == cudaSuccess || result2 == cudaSuccess || result3 == cudaSuccess);

	// set values to cpu memory
	float* cpuFloatP = (float*)Mat1.cpuP;

	for (int i = 0; i < Mat1.Size; i++)
		cpuFloatP[i] = (float)i;


	//	Host => ram
	//	Device => graphics memory	

	// Host -> Device
	result1 = cudaMemcpy(Mat1.gpuP, Mat1.cpuP, Mat1.Size * sizeof(float), cudaMemcpyHostToDevice);
	result2 = cudaMemcpy(Mat2.gpuP, Mat2.cpuP, Mat2.Size * sizeof(float), cudaMemcpyHostToDevice);
	result3 = cudaMemcpy(Mat3.gpuP, Mat3.cpuP, Mat3.Size * sizeof(float), cudaMemcpyHostToDevice);
	assert(result1 == cudaSuccess || result2 == cudaSuccess || result3 == cudaSuccess);


	// parallel conv
	gpuMatrixConvulation(&Mat1, &Mat2, &Mat3);


	// Device -> Host
	cudaError_t result = cudaMemcpy(Mat3.cpuP, Mat3.gpuP, Mat3.Size * sizeof(float), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	cudaDeviceSynchronize();


	// show result
	cpuFloatP = (float*)Mat3.cpuP;
	for (size_t row = 0; row < Mat3.Rows; row++)
	{
		for (size_t col = 0; col < Mat3.Cols; col++)
		{
			std::cout << cpuFloatP[row * Mat3.Cols + col] << " ";
		}
		std::cout << std::endl;
	}


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