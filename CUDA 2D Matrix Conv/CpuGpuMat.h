#pragma once

struct CpuGpuMat {

	void* cpuP;		// ram pointer
	void* gpuP;		// graphic memory pointer
	int Rows;
	int Cols;
	int Size;		// rows * cols
};