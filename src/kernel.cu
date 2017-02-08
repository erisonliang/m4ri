#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#define UINT32_BIT_SIZE 32

/**
	\brief Возвращает указанный бит из массива значений в памяти GPU
	\param[in] arr Массив значений
	\param[in] n   Порядковый номер бита
	\return Указанный бит (0 или 1)
 */
__device__ 
uint32_t inline d_get_bit(const uint32_t* arr, const uint32_t n)
{
	uint32_t cell_idx = n / UINT32_BIT_SIZE;
	return arr[cell_idx] & (uint32_t(1) << (n % UINT32_BIT_SIZE));
}


/**
	\brief Устанавливает значение указанного бита в указанное значение в памяти на GPU
	\param[in, out] arr Массив значений
	\param[in]      n   Порядковый номер бита
	\param[in]      bit_value Значение устанавливаемого бита
 */
__device__ 
void inline d_set_bit(uint32_t* arr, const uint32_t n, const uint32_t bit_value)
{
	uint32_t cell_idx = n / UINT32_BIT_SIZE;
	if(bit_value)
		arr[cell_idx] = arr[cell_idx] | (uint32_t(1) << (n % UINT32_BIT_SIZE));
	else
		arr[cell_idx] = arr[cell_idx] & ~(uint32_t(1) << (n % UINT32_BIT_SIZE));
}


/**
	\brief Функция для вычисления произведения битовых матриц на GPU

	Каждый CUDA thread вычисляет произведение вектора-строки матрицы A на вектор
	столбец матрицы B. В качестве операций сложения и умножений используется XOR и 
	битовое AND соответственно. Результат операции сохраняется в соответствующую 
	ячейку матрицы C.

	\param[in]  A Левый множитель в матричном произведении
	\param[in]  a_rows Количество строк в матрице А
	\param[in]  a_cols Количество столбцов в матрице А
	\param[in]  B Правый множитель в матричном произведении
	\param[in]  b_rows Количество строк в матрице B
	\param[in]  b_cols Количество столбцов в матрице B
	\param[out] C Результат матричного произведения
 */
__global__ 
void gpu_multiplication(
	const uint32_t* const A, const uint32_t a_rows, const uint32_t a_cols,
	const uint32_t* const B, const uint32_t b_rows, const uint32_t b_cols, uint32_t* const C) 
{
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t sum = 0;

	if(index < a_rows)
		for (uint32_t i = 0; i < b_cols; i++)											 
		{
			for (uint32_t j = 0; j < a_cols; j++)											
				sum ^= d_get_bit(A, index * a_cols + j) & d_get_bit(B, j * b_cols + i);
			d_set_bit(C, index * b_cols + i, sum);
		}
}


/**
	\brief Возвращает указанный бит из массива значений в памяти CPU
	\param[in] arr Массив значений
	\param[in] n   Порядковый номер бита
	\return Указанный бит (0 или 1)
 */
uint32_t inline h_get_bit(const uint32_t* arr, const uint32_t n)
{
	uint32_t cell_idx = n / UINT32_BIT_SIZE;
	return arr[cell_idx] & (uint32_t(1) << (n % UINT32_BIT_SIZE));
}


/**
	\brief Устанавливает значение указанного бита в указанное значение в памяти на CPU
	\param[in,out] arr Массив значений
	\param[in]      n   Порядковый номер бита
	\param[in]      bit_value Значение устанавливаемого бита
*/
void inline h_set_bit(uint32_t* arr, const uint32_t n, const uint32_t bit_value)
{
	uint32_t cell_idx = n / UINT32_BIT_SIZE;
	if (bit_value)
		arr[cell_idx] = arr[cell_idx] | (uint32_t(1) << (n % UINT32_BIT_SIZE));
	else
		arr[cell_idx] = arr[cell_idx] & ~(uint32_t(1) << (n % UINT32_BIT_SIZE));
}


/**
	\brief Функция для вычисления произведения битовых матриц на CPU

	В однопоточном режиме функция вычисляет произведение битовых матриц
	А и B. В качестве операций сложения и умножений используется XOR и 
	битовое AND соответственно. Результат операции сохраняется в матрицу С.

	\param[in]  A Левый множитель в матричном произведении
	\param[in]  a_rows Количество строк в матрице А
	\param[in]  a_cols Количество столбцов в матрице А
	\param[in]  B Правый множитель в матричном произведении
	\param[in]  b_rows Количество строк в матрице B
	\param[in]  b_cols Количество столбцов в матрице B
	\param[out] C Результат матричного произведения
 */
void cpu_multiplication(
	const uint32_t* const A, const uint32_t a_rows, const uint32_t a_cols,
	const uint32_t* const B, const uint32_t b_rows, const uint32_t b_cols, uint32_t* const C)
{
	for (uint32_t k = 0; k < a_rows; k++)
	{
		uint32_t sum = 0;
		for (uint32_t i = 0; i < b_cols; i++)
		{
			for (uint32_t j = 0; j < a_cols; j++)										
				sum ^= h_get_bit(A, k * a_cols + j) & h_get_bit(B, j * b_cols + i);
			h_set_bit(C, k * b_cols + i, sum);
		}
	}
}


int main() 
{
	uint32_t a_rows = 1024, a_cols = 1024;
	uint32_t b_rows = a_cols, b_cols = 2048;

	uint32_t* h_A      = (uint32_t *)calloc(a_rows * a_cols / 4, sizeof(uint32_t));
	uint32_t* h_B      = (uint32_t *)calloc(b_rows * b_cols / 4, sizeof(uint32_t));
	uint32_t* h_C      = (uint32_t *)calloc(a_rows * b_cols / 4, sizeof(uint32_t));
	uint32_t* h_result = (uint32_t *)calloc(a_rows * b_cols / 4, sizeof(uint32_t));

	if (!h_A || !h_B || !h_C || !h_result) 
	{
		printf("Allocation error!");
		exit(1);
	}

	for (uint32_t i = 0; i < a_rows; i++)
		for (uint32_t j = 0; j < a_cols / UINT32_BIT_SIZE; j++)
			h_A[i * a_cols / UINT32_BIT_SIZE + j] = (uint32_t)rand();

	for (uint32_t i = 0; i < b_rows; i++)
		for (uint32_t j = 0; j < b_cols / UINT32_BIT_SIZE; j++)
			h_B[i * b_cols / UINT32_BIT_SIZE + j] = (uint32_t)rand();

	uint32_t *d_A = 0, *d_B = 0, *d_C = 0;

	cudaMalloc((void **)&d_A, a_rows * a_cols);
	cudaMalloc((void **)&d_B, b_rows * b_cols);
	cudaMalloc((void **)&d_C, a_rows * b_cols);

	cudaMemcpy((void *)d_A, (const void*)h_A, a_rows * a_cols, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_B, (const void*)h_B, b_rows * b_cols, cudaMemcpyHostToDevice);

	dim3 block_size(std::min(int(a_rows), 512));
	dim3 grid_size(a_rows / block_size.x);

	float gpu_time = 0;
	cudaEvent_t timer_start, timer_stop;
	cudaEventCreate(&timer_start); cudaEventCreate(&timer_stop);
	cudaEventRecord(timer_start, 0);

	gpu_multiplication <<< grid_size, block_size >>> (d_A, a_rows, a_cols, d_B, b_rows, b_cols, d_C);

	cudaEventRecord(timer_stop);
	cudaEventSynchronize(timer_stop);
	cudaEventElapsedTime(&gpu_time, timer_start, timer_stop);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
		return 1;
	}

	float cpu_time = clock();
	cpu_multiplication(h_A, a_rows, a_cols, h_B, b_rows, b_cols, h_C);
	cpu_time = clock() - cpu_time;
	
	printf("GPU elapsed time: %.2f ms\nCPU elapsed time: %.2f ms\n", gpu_time, cpu_time);

	cudaMemcpy((void **)h_result, (const void**)d_C, a_rows * b_cols, cudaMemcpyDeviceToHost);

	bool success = true;
	for (uint32_t i = 0; i < a_rows * b_cols / UINT32_BIT_SIZE; i++)
		if (h_result[i] != h_C[i])
		{
			printf("Error! h_result[%d] != h_C[%d].\n", i, i);
			success = false;
			break;
		}

	if (success)
		printf("Test passed.\n");
	else
		printf("Error!\n");

	cudaFree((void *)d_A);
	cudaFree((void *)d_B);
	cudaFree((void *)d_C);
		
	free((void *)h_A);
	free((void *)h_B);
	free((void *)h_C);

	return 0;
}