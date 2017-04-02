#include <stdint.h>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <climits>

#define UINT32_BIT_SIZE 32

__device__ 
bool inline d_get_bit(const uint32_t* arr, const uint32_t n)
{
	uint32_t cell_idx = n / UINT32_BIT_SIZE;
	return arr[cell_idx] & (uint32_t(1) << (n % UINT32_BIT_SIZE));
}


__device__ 
void inline d_set_bit(uint32_t* arr, const uint32_t n, const uint32_t bit_value)
{
	uint32_t cell_idx = n / UINT32_BIT_SIZE;
	if(bit_value)
		arr[cell_idx] = arr[cell_idx] | (uint32_t(1) << (n % UINT32_BIT_SIZE));
	else
		arr[cell_idx] = arr[cell_idx] & ~(uint32_t(1) << (n % UINT32_BIT_SIZE));
}

__global__ 
void gpu_multiplication(
	const uint32_t* const A, const uint32_t a_rows, const uint32_t a_cols,
	const uint32_t* const B, const uint32_t b_rows, const uint32_t b_cols, uint32_t* const C) 
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	if(index < a_rows)
	{
		for (unsigned int i = 0; i < b_rows; i++)											 
		{
			bool sum = 0;
			for (unsigned int j = 0; j < a_cols; j++)
			{											
				sum ^= d_get_bit(A, index * a_cols + j) & d_get_bit(B, i * b_cols + j);
			}
			d_set_bit(C, index * b_rows + i, sum);
		}
	}
}

__global__
void gpu_transpose(uint32_t* dst, const uint32_t* const src, const uint32_t rows, const uint32_t cols)
{
	for(int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			d_set_bit(dst, i * cols + j, d_get_bit(src, j * rows + i));
}


bool inline h_get_bit(const uint32_t* arr, const uint32_t n)
{
	uint32_t cell_idx = n / UINT32_BIT_SIZE;
	return arr[cell_idx] & (uint32_t(1) << (n % UINT32_BIT_SIZE));
}


void inline h_set_bit(uint32_t* arr, const uint32_t n, const uint32_t bit_value)
{
	uint32_t cell_idx = n / UINT32_BIT_SIZE;
	if (bit_value)
		arr[cell_idx] = arr[cell_idx] | (uint32_t(1) << (n % UINT32_BIT_SIZE));
	else
		arr[cell_idx] = arr[cell_idx] & ~(uint32_t(1) << (n % UINT32_BIT_SIZE));
}

void cpu_multiplication(
	const uint32_t* const A, const uint32_t a_rows, const uint32_t a_cols,
	const uint32_t* const B, const uint32_t b_rows, const uint32_t b_cols, uint32_t* const C)
{
	for (unsigned int k = 0; k < a_rows; k++)
	{
		for (unsigned int i = 0; i < b_cols; i++)
		{
			bool sum = 0;
			for (uint32_t j = 0; j < a_cols; j++)										
				sum ^= h_get_bit(A, k * a_cols + j) & h_get_bit(B, j * b_cols + i);
			h_set_bit(C, k * b_cols + i, sum);
		}
	}
}

void cpu_transpose(uint32_t* dst, const uint32_t* const src, const uint32_t rows, const uint32_t cols)
{
	for(unsigned int i = 0; i < rows; ++i)
		for (unsigned int j = 0; j < cols; ++j)
			h_set_bit(dst, i * cols + j, h_get_bit(src, j * rows + i));
}


int main() 
{
	const uint32_t a_rows = 32, a_cols = 32;
	const uint32_t b_rows = 32, b_cols = 32;

	uint32_t* h_A = new uint32_t[a_rows * a_cols / 4];
	uint32_t* h_B = new uint32_t[b_rows * b_cols / 4];
	uint32_t* h_B_tr = new uint32_t[b_rows * b_cols / 4];
	uint32_t* h_C = new uint32_t[a_rows * b_cols / 4];
	uint32_t* h_result = new uint32_t[a_rows * b_cols / 4];

	for (uint32_t i = 0; i < a_rows; i++)
		for (uint32_t j = 0; j < a_cols / UINT32_BIT_SIZE; j++)
			h_A[i * a_cols / UINT32_BIT_SIZE + j] = (uint32_t)rand();

	for (uint32_t i = 0; i < b_rows; i++)
		for (uint32_t j = 0; j < b_cols / UINT32_BIT_SIZE; j++)
			h_B[i * b_cols / UINT32_BIT_SIZE + j] = (uint32_t)rand();

	uint32_t *d_A = 0, *d_B = 0, *d_B_tr, *d_C = 0;

	cudaMalloc((void **)&d_A, a_rows * a_cols);
	cudaMalloc((void **)&d_B, b_rows * b_cols);
	cudaMalloc((void **)&d_B_tr, b_rows * b_cols);
	cudaMalloc((void **)&d_C, a_rows * b_cols);

	cudaMemcpy((void *)d_A, (const void*)h_A, a_rows * a_cols, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_B, (const void*)h_B, b_rows * b_cols, cudaMemcpyHostToDevice);

	dim3 block_size(std::min(int(a_rows), 512));
	dim3 grid_size(a_rows / block_size.x);

	float gpu_time = 0;
	cudaEvent_t timer_start, timer_stop;
	cudaEventCreate(&timer_start); cudaEventCreate(&timer_stop);
	cudaEventRecord(timer_start, 0);

	gpu_transpose <<< 1, 1 >>> (d_B_tr, d_B, b_rows, b_cols);
	gpu_multiplication <<< grid_size, block_size >>> (d_A, a_rows, a_cols, d_B_tr, b_cols, b_rows, d_C);

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
			std::cout << int2bin(h_result[i]) << " " << int2bin(h_C[i]) << std::endl;
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

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	delete[] h_B_tr;
	delete[] h_result;

	return 0;
}