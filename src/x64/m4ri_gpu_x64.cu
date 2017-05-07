#include <stdint.h>

namespace x64
{
	namespace gpu 
	{

		#define UINT64_BIT_SIZE 64

		__device__ 
		bool inline get_bit(const uint64_t* arr, uint64_t n)
		{
			uint64_t cell_idx = n / UINT64_BIT_SIZE;
			return arr[cell_idx] & (1 << (n % UINT64_BIT_SIZE));
		}


		__device__ 
		void inline set_bit(uint64_t* arr, uint64_t n, uint64_t bit_value)
		{
			uint64_t cell_idx = n / UINT64_BIT_SIZE;
			if(bit_value)
				arr[cell_idx] = arr[cell_idx] | (1 << (n % UINT64_BIT_SIZE));
			else
				arr[cell_idx] = arr[cell_idx] & ~(1 << (n % UINT64_BIT_SIZE));
		}

		__global__ 
		void m4ri_multiply(
			const uint64_t* A, uint64_t a_rows, uint64_t a_cols,
			const uint64_t* B, uint64_t b_rows, uint64_t b_cols, 
			uint64_t* C, const bool* precalc_matrix, unsigned int k) 
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
						// получаем нужный uint64_t элемент в массиве A и B
						unsigned int id_A = (index * a_cols + j * k) / UINT64_BIT_SIZE;
						unsigned int id_B = (i * b_cols + j * k) / UINT64_BIT_SIZE;

						// получаем смещение (должно быть кратно k)
						unsigned int offset_A = (index * a_cols + j * k) % UINT64_BIT_SIZE;
						unsigned int offset_B = (i * b_cols + j * k) % UINT64_BIT_SIZE;

						uint64_t k_vector_A = (A[id_A] >> offset_A) & ((1 << k) - 1);
						uint64_t k_vector_B = (B[id_B] >> offset_B) & ((1 << k) - 1);

						sum ^= precalc_matrix[k_vector_A * (1 << k) + k_vector_B];
					}
					set_bit(C, index * b_rows + i, sum);
				}
			}
		}

		__global__
		void transpose(uint64_t* dst, const uint64_t* src, uint64_t rows, uint64_t cols)
		{
			for(int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					set_bit(dst, i * cols + j, get_bit(src, j * rows + i));
		}


		__global__
		void m4ri_precalc(bool* precalc_matrix, uint64_t bits)
		{
			for(unsigned long i = 0; i < (1 << bits); i++)
			{
				for(unsigned long j = 0; j < (1 << bits); j++)
				{
					bool scalar_product = 0;
					for(unsigned  long k = 0; k < bits; k++)
					{
						scalar_product ^= get_bit(&i, k) & get_bit(&j, k);
					}
					precalc_matrix[i * (1 << bits) + j] = scalar_product;
				}
			}
		}

		__global__
		void mar_multiply(const uint64_t* A, uint64_t a_rows, uint64_t a_cols,
						  const uint64_t* B, uint64_t b_rows, uint64_t b_cols, uint64_t* C)
		{
			unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

			for(unsigned long i = 0; i < a_cols; i++)
			{
				bool bit = get_bit(A, index * a_cols + i);
				for(unsigned long j = 0; j < b_cols / UINT64_BIT_SIZE; j++)
				{
					C[index * a_cols / UINT64_BIT_SIZE + j] ^= B[i * b_cols / UINT64_BIT_SIZE + j] * bit;
				}
			}
		}
	}
}