#pragma once

#include <stdint.h>

namespace x86 
{
	namespace gpu
	{
		#define UINT32_BIT_SIZE 32

		__device__ 
		uint32_t inline get_bit(const uint32_t* arr, uint32_t n)
		{
			uint32_t cell_idx = n / UINT32_BIT_SIZE;
			return (arr[cell_idx] >> (n % UINT32_BIT_SIZE)) & 1;
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
			uint32_t* C, const uint32_t* precalc_matrix, unsigned int k) 
		{
			// index - номер строки в матрице А и В
			const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

			for (unsigned int i = 0; i < b_rows; i++)											 
			{
				uint32_t sum = 0;
				for(unsigned int j = 0; j < b_cols / k; j++)
				{
					// получаем нужный uint32_t элемент в массиве A и B
					unsigned int id_A = (index * a_cols) / UINT32_BIT_SIZE + j * k / UINT32_BIT_SIZE;
					unsigned int id_B = (i * b_cols) / UINT32_BIT_SIZE + j * k / UINT32_BIT_SIZE;

					// получаем смещение (должно быть кратно k)
					unsigned int offset = (j * k) % UINT32_BIT_SIZE;

					uint32_t k_vector_A = (A[id_A] >> offset) & ((1 << k) - 1);
					uint32_t k_vector_B = (B[id_B] >> offset) & ((1 << k) - 1);

					sum ^= get_bit(precalc_matrix, k_vector_A * (1 << k) + k_vector_B);
				}
				set_bit(C, index * b_rows + i, sum);
			}
		}

		__global__
		void transpose(uint32_t* dst, const uint32_t* src, unsigned int rows, unsigned int cols)
		{
			for(unsigned int i = 0; i < rows; ++i)
				for (unsigned int j = 0; j < cols; ++j)
					set_bit(dst, i * cols + j, get_bit(src, j * rows + i));
		}

		__global__
		void m4ri_precalc(uint32_t* precalc_matrix, uint32_t bits)
		{
			for(uint32_t i = 0; i < (1 << bits); i++)
			{
				for(uint32_t j = 0; j < (1 << bits); j++)
				{
					uint32_t scalar_product = 0;
					for(uint32_t k = 0; k < bits; k++)
					{
						scalar_product ^= get_bit(&i, k) & get_bit(&j, k);
					}
					set_bit(precalc_matrix, i * (1 << bits) + j, scalar_product);
				}
			}
		}

		__global__
		void mar_multiply(const uint32_t* A, uint32_t a_rows, uint32_t a_cols,
						  const uint32_t* B, uint32_t b_rows, uint32_t b_cols, uint32_t* C)
		{
			const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

			for(unsigned int i = 0; i < a_cols; i++)
			{
				bool bit = get_bit(A, index * a_cols + i);
				for(unsigned int j = 0; j < b_cols / UINT32_BIT_SIZE; j++)
				{
					C[index * a_cols / UINT32_BIT_SIZE + j] ^= B[i * b_cols / UINT32_BIT_SIZE + j] * bit;
				}
			}
		}

		// каждый поток вычисляет одну таблицу (из 32 * a_rows / 128 таблиц)
		// i-ый поток вычисляет все возможные суммы 4i, 4i+1, 4i+2, 4i+3 строк матрицы B
		__global__
		void m4ri_opt_precalc(const uint32_t* B, size_t b_rows, uint32_t* precalc, unsigned int block_col_num)
		{
			const unsigned int table_num = threadIdx.x + blockIdx.x * blockDim.x;

			// в каждой таблице 16 строк по 128 битов, т.е. по 64 элемента типа uint32_t
			const unsigned int offset = table_num * 64;
			
			for(uint32_t counter = 0; counter < 16; counter++)
			{
				// 4 = 128 / UINT32_BIT_SIZE
				for(unsigned int i = 0; i < 4; i++)
				{
					// <offset> + <номер строки> + <номер столбца>
					precalc[offset + counter * 4 + i] = 
						(get_bit(&counter, 0) * B[(table_num * 4 + 0) * b_rows / UINT32_BIT_SIZE + block_col_num * 4 + i]) ^ 
						(get_bit(&counter, 1) * B[(table_num * 4 + 1) * b_rows / UINT32_BIT_SIZE + block_col_num * 4 + i]) ^ 
						(get_bit(&counter, 2) * B[(table_num * 4 + 2) * b_rows / UINT32_BIT_SIZE + block_col_num * 4 + i]) ^ 
						(get_bit(&counter, 3) * B[(table_num * 4 + 3) * b_rows / UINT32_BIT_SIZE + block_col_num * 4 + i]);
				}
			}
		}

		__global__
		void m4ri_opt_multiply(	const uint32_t* A, uint32_t a_rows, uint32_t a_cols,
						  		const uint32_t* B, uint32_t b_rows, uint32_t b_cols, 
						  		uint32_t* C, const uint32_t* precalc, unsigned int block_col_num)
		{
			const unsigned int row_idx = threadIdx.x + blockDim.x * blockIdx.x;

			const size_t precalc_table_size = 64;
			const unsigned int block_offset = block_col_num * 4;

			// по всем группам по 4 во всей строке матрицы А (по всем таблицам)
			for(unsigned int i = 0; i < a_rows / 4; i++)
			{
				uint32_t t_index = 
					(A[row_idx * a_cols / UINT32_BIT_SIZE + i * 4 / UINT32_BIT_SIZE] >> (i * 4 % UINT32_BIT_SIZE)) & 15;

				// суммируем строки
				for(unsigned int k = 0; k < 4; k++)
				{
					C[row_idx * a_cols / UINT32_BIT_SIZE + block_offset + k] ^= 
						precalc[precalc_table_size * i + t_index * 128 / UINT32_BIT_SIZE + k];
				}
			}
		}


	} 	// namespace gpu
}	// namespace x86