#pragma once

#include <stdint.h>

#define UINT64_BIT_SIZE 64

namespace x64
{
	namespace gpu 
	{
		const unsigned int SUBVECTOR_SIZE 	 = 8;
		const unsigned int M4RI_PRECALC_SIZE = (1 << SUBVECTOR_SIZE) * (1 << SUBVECTOR_SIZE);

		__device__ 
		bool inline get_bit(const uint64_t* arr, uint64_t n)
		{
			uint64_t cell_idx = n / UINT64_BIT_SIZE;
			return (arr[cell_idx] >> (n % UINT64_BIT_SIZE)) & uint64_t(1);
		}


		__device__ 
		void inline set_bit(uint64_t* arr, uint64_t n, uint64_t bit_value)
		{
			uint64_t cell_idx = n / UINT64_BIT_SIZE;
			if(bit_value)
				arr[cell_idx] = arr[cell_idx] | (uint64_t(1) << (n % UINT64_BIT_SIZE));
			else
				arr[cell_idx] = arr[cell_idx] & ~(uint64_t(1) << (n % UINT64_BIT_SIZE));
		}

		__global__ 
		void m4ri_multiply(
			const uint64_t* A, uint64_t a_rows, uint64_t a_cols,
			const uint64_t* B, uint64_t b_rows, uint64_t b_cols, 
			uint64_t* C, const uint64_t* precalc_matrix, unsigned int k) 
		{
			// index - номер строки в матрице А и В
			const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
			const unsigned int size = M4RI_PRECALC_SIZE / UINT64_BIT_SIZE;
			
			__shared__ uint64_t precalc_sh[size];
			
			const unsigned int elems_per_thread = size / blockDim.x;
			for(unsigned int i = 0; i < elems_per_thread; i++)
				precalc_sh[threadIdx.x * elems_per_thread + i] = precalc_matrix[threadIdx.x * elems_per_thread + i];  

			__syncthreads();

			for (unsigned int i = 0; i < b_rows; i++)											 
			{
				uint64_t sum = 0;
				for(unsigned int j = 0; j < b_cols / k; j++)
				{
					// получаем нужный uint64_t элемент в массиве A и B
					unsigned int id_A = (index * a_cols) / UINT64_BIT_SIZE + j * k / UINT64_BIT_SIZE;
					unsigned int id_B = (i * b_cols) / UINT64_BIT_SIZE + j * k / UINT64_BIT_SIZE;

					// получаем смещение (должно быть кратно k)
					unsigned int offset = (j * k) % UINT64_BIT_SIZE;

					uint64_t k_vector_A = (A[id_A] >> offset) & ((1 << k) - 1);
					uint64_t k_vector_B = (B[id_B] >> offset) & ((1 << k) - 1);

					sum ^= get_bit(precalc_sh, k_vector_A * (1 << k) + k_vector_B);
				}
				set_bit(C, index * b_rows + i, sum);
			}
		}

		__global__
		void transpose(uint64_t* dst, const uint64_t* src, uint64_t rows, uint64_t cols)
		{
			const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

			for (unsigned int j = 0; j < cols; ++j)
				set_bit(dst, index * cols + j, get_bit(src, j * rows + index));
		}


		__global__
		void m4ri_precalc(uint64_t* precalc_matrix, uint64_t bits)
		{
			const uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;

			for(uint64_t j = 0; j < (1 << bits); j++)
			{
				uint64_t scalar_product = 0;
				for(uint64_t k = 0; k < bits; k++)
				{
					scalar_product ^= get_bit(&index, k) & get_bit(&j, k);
				}
				set_bit(precalc_matrix, index * (1 << bits) + j, scalar_product);
			}
		}

		__global__
		void mar_multiply(const uint64_t* A, uint64_t a_rows, uint64_t a_cols,
						  const uint64_t* B, uint64_t b_rows, uint64_t b_cols, uint64_t* C)
		{
			const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

			for(unsigned long i = 0; i < a_cols; i++)
			{
				bool bit = get_bit(A, index * a_cols + i);
				for(unsigned long j = 0; j < b_cols / UINT64_BIT_SIZE; j++)
				{
					C[index * a_cols / UINT64_BIT_SIZE + j] ^= B[i * b_cols / UINT64_BIT_SIZE + j] * bit;
				}
			}
		}

		__global__
		void m4ri_opt_precalc(const uint64_t* B, size_t b_rows, uint64_t* precalc, unsigned int block_col_num)
		{
			const unsigned int table_num = threadIdx.x + blockIdx.x * blockDim.x;

			// в каждой таблице 16 строк по 128 битов, т.е. по 32 элемента типа uint64_t
			const unsigned int offset = table_num * 32;
			
			for(uint64_t counter = 0; counter < 16; counter++)
			{
				// 2 = 128 / UINT64_BIT_SIZE
				for(unsigned int i = 0; i < 2; i++)
				{
					// <offset> + <номер строки> + <номер столбца>
					precalc[offset + counter * 2 + i] = 
						(get_bit(&counter, 0) * B[(table_num * 4 + 0) * b_rows / UINT64_BIT_SIZE + block_col_num * 2 + i]) ^ 
						(get_bit(&counter, 1) * B[(table_num * 4 + 1) * b_rows / UINT64_BIT_SIZE + block_col_num * 2 + i]) ^ 
						(get_bit(&counter, 2) * B[(table_num * 4 + 2) * b_rows / UINT64_BIT_SIZE + block_col_num * 2 + i]) ^ 
						(get_bit(&counter, 3) * B[(table_num * 4 + 3) * b_rows / UINT64_BIT_SIZE + block_col_num * 2 + i]);
				}
			}
		}

		__global__
		void m4ri_opt_multiply(	const uint64_t* A, uint64_t a_rows, uint64_t a_cols,
						  		const uint64_t* B, uint64_t b_rows, uint64_t b_cols, 
						  		uint64_t* C, const uint64_t* precalc, unsigned int block_col_num)
		{
			const unsigned int row_idx = threadIdx.x + blockDim.x * blockIdx.x;

			const size_t precalc_table_size = 32;
			const unsigned int block_offset = block_col_num * 2;

			// по всем группам по 4 во всей строке матрицы А (по всем таблицам)
			for(unsigned long i = 0; i < a_rows / 4; i++)
			{
				uint64_t t_index = 
					(A[row_idx * a_cols / UINT64_BIT_SIZE + i * 4 / UINT64_BIT_SIZE] >> (i * 4 % UINT64_BIT_SIZE)) & 15;

				// суммируем строки
				for(unsigned int k = 0; k < 2; k++)
				{
					C[row_idx * a_cols / UINT64_BIT_SIZE + block_offset + k] ^= 
						precalc[precalc_table_size * i + t_index * 128 / UINT64_BIT_SIZE + k];
				}
			}
		}
	}	// namespace gpu
}	// namespace x86