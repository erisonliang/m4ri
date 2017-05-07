#include <stdint.h>

namespace x86 
{
	namespace gpu
	{

		#define UINT32_BIT_SIZE 32

		__device__ 
		bool inline get_bit(const uint32_t* arr, uint32_t n)
		{
			uint32_t cell_idx = n / UINT32_BIT_SIZE;
			return arr[cell_idx] & (1 << (n % UINT32_BIT_SIZE));
		}


		__device__ 
		void inline set_bit(uint32_t* arr, uint32_t n, uint32_t bit_value)
		{
			uint32_t cell_idx = n / UINT32_BIT_SIZE;
			if(bit_value)
				arr[cell_idx] = arr[cell_idx] | (1 << (n % UINT32_BIT_SIZE));
			else
				arr[cell_idx] = arr[cell_idx] & ~(1 << (n % UINT32_BIT_SIZE));
		}

		__global__ 
		void m4ri_multiply(
			const uint32_t* A, uint32_t a_rows, uint32_t a_cols,
			const uint32_t* B, uint32_t b_rows, uint32_t b_cols, 
			uint32_t* C, const bool* precalc_matrix, unsigned int k) 
		{
			// index - номер строки в матрице А и В
			unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

			if(index < a_rows)
			{
				for (unsigned int i = 0; i < b_rows; i++)											 
				{
					bool sum = 0;
					for(unsigned int j = 0; j < b_cols / k; j++)
					{
						// получаем нужный uint32_t элемент в массиве A и B
						unsigned int id_A = (index * a_cols + j * k) / UINT32_BIT_SIZE;
						unsigned int id_B = (i * b_cols + j * k) / UINT32_BIT_SIZE;

						// получаем смещение (должно быть кратно k)
						unsigned int offset_A = (index * a_cols + j * k) % UINT32_BIT_SIZE;
						unsigned int offset_B = (i * b_cols + j * k) % UINT32_BIT_SIZE;

						uint32_t k_vector_A = (A[id_A] >> offset_A) & ((1 << k) - 1);
						uint32_t k_vector_B = (B[id_B] >> offset_B) & ((1 << k) - 1);

						sum ^= precalc_matrix[k_vector_A * (1 << k) + k_vector_B];
					}
					set_bit(C, index * b_rows + i, sum);
				}
			}
		}

		__global__
		void transpose(uint32_t* dst, const uint32_t* src, uint32_t rows, uint32_t cols)
		{
			for(int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					set_bit(dst, i * cols + j, get_bit(src, j * rows + i));
		}


		__global__
		void m4ri_precalc(bool* precalc_matrix, uint32_t bits)
		{
			for(unsigned int i = 0; i < (1 << bits); i++)
			{
				for(unsigned int j = 0; j < (1 << bits); j++)
				{
					bool scalar_product = 0;
					for(unsigned int k = 0; k < bits; k++)
					{
						scalar_product ^= get_bit(&i, k) & get_bit(&j, k);
					}
					precalc_matrix[i * (1 << bits) + j] = scalar_product;
				}
			}
		}

		__global__
		void mar_multiply(const uint32_t* A, uint32_t a_rows, uint32_t a_cols,
						  const uint32_t* B, uint32_t b_rows, uint32_t b_cols, uint32_t* C)
		{
			unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

			for(unsigned int i = 0; i < a_cols; i++)
			{
				bool bit = get_bit(A, index * a_cols + i);
				for(unsigned int j = 0; j < b_cols / UINT32_BIT_SIZE; j++)
				{
					C[index * a_cols / UINT32_BIT_SIZE + j] ^= B[i * b_cols / UINT32_BIT_SIZE + j] * bit;
				}
			}
		}
	}
}