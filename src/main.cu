#include <stdint.h>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <climits>
#include <bitset>
#include "include/kernel.cu"

int main() 
{
	const uint32_t a_rows = 256, a_cols = 256;
	const uint32_t b_rows = 256, b_cols = 256;

	uint32_t* h_A 		= new uint32_t[a_rows * a_cols / 4];
	uint32_t* h_B 		= new uint32_t[b_rows * b_cols / 4];
	uint32_t* h_B_tr 	= new uint32_t[b_rows * b_cols / 4];
	uint32_t* h_C 		= new uint32_t[a_rows * b_cols / 4];
	uint32_t* h_result 	= new uint32_t[a_rows * b_cols / 4];

	for (uint32_t i = 0; i < a_rows; i++)
		for (uint32_t j = 0; j < a_cols / UINT32_BIT_SIZE; j++)
			h_A[i * a_cols / UINT32_BIT_SIZE + j] = (uint32_t)rand();

	for (uint32_t i = 0; i < b_rows; i++)
		for (uint32_t j = 0; j < b_cols / UINT32_BIT_SIZE; j++)
			h_B[i * b_cols / UINT32_BIT_SIZE + j] = (uint32_t)rand();

	uint32_t *d_A = nullptr, *d_B = nullptr, *d_B_tr = nullptr, *d_C = nullptr;

	cudaMalloc((void **)&d_A, a_rows * a_cols);
	cudaMalloc((void **)&d_B, b_rows * b_cols);
	cudaMalloc((void **)&d_B_tr, b_rows * b_cols);
	cudaMalloc((void **)&d_C, a_rows * b_cols);

	cudaMemcpy((void *)d_A, (const void*)h_A, a_rows * a_cols, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_B, (const void*)h_B, b_rows * b_cols, cudaMemcpyHostToDevice);

	dim3 block_size(std::min(int(a_rows), 512));
	dim3 grid_size(a_rows / block_size.x);

	bool* d_precalc = nullptr;
	int precalc_size = cpu_log2(a_rows);
	cudaMalloc((void **)&d_precalc, (1 << precalc_size) * (1 << precalc_size));
	

	float gpu_time = 0;
	cudaEvent_t timer_start, timer_stop;
	cudaEventCreate(&timer_start); cudaEventCreate(&timer_stop);
	cudaEventRecord(timer_start, 0);

	gpu_transpose <<< 1, 1 >>> (d_B_tr, d_B, b_rows, b_cols);

	precalc<<<1, 1>>>(d_precalc, precalc_size);

	gpu_multiplication <<< grid_size, block_size >>> (d_A, a_rows, a_cols, d_B_tr, b_cols, b_rows, d_C, d_precalc);
	
	cudaEventRecord(timer_stop);
	cudaEventSynchronize(timer_stop);
	cudaEventElapsedTime(&gpu_time, timer_start, timer_stop);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		std::cout << cudaGetErrorString(error) << std::endl;
		return 1;
	}

	float cpu_time = clock();
	cpu_multiplication(h_A, a_rows, a_cols, h_B, b_rows, b_cols, h_C);
	cpu_time = clock() - cpu_time;
	
	std::cout << "GPU elapsed time: " << gpu_time << " ms" << std::endl;
	std::cout << "CPU elapsed time: " << cpu_time <<" ms" << std::endl;

	cudaMemcpy((void **)h_result, (const void**)d_C, a_rows * b_cols, cudaMemcpyDeviceToHost);

	bool success = true;
	for (uint32_t i = 0; i < a_rows * b_cols / UINT32_BIT_SIZE; i++)
		if (h_result[i] != h_C[i])
		{
			std::cout << "Error! h_result["<< i << "] != h_C[" << i << "]." << std::endl;
			std::cout << std::bitset<32>(h_result[i]) << " != " << std::bitset<32>(h_C[i]) << std::endl;
			success = false;
			break;
		}

	if (success)
		std::cout << "Test passed" << std::endl;
	else
		std::cout << "Error!" << std::endl;

	cudaFree((void *)d_A);
	cudaFree((void *)d_B);
	cudaFree((void *)d_C);
	cudaFree((void *)d_B_tr);
	cudaFree((void *)d_precalc);

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	delete[] h_B_tr;
	delete[] h_result;

	return 0;
}