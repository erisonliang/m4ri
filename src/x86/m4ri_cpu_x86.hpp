#pragma once

#include <cstdint>

#define UINT32_BIT_SIZE 32

namespace x86
{
	namespace cpu
	{
		const unsigned int SUBVECTOR_SIZE 	 = 8;
		const unsigned int M4RI_PRECALC_SIZE = (1 << SUBVECTOR_SIZE) * (1 << SUBVECTOR_SIZE);
		
		bool inline get_bit(const uint32_t* arr, uint32_t n)
		{
			uint32_t cell_idx = n / UINT32_BIT_SIZE;
			return arr[cell_idx] & (1 << (n % UINT32_BIT_SIZE));
		}


		void inline set_bit(uint32_t* arr, uint32_t n, uint32_t bit_value)
		{
			uint32_t cell_idx = n / UINT32_BIT_SIZE;
			if (bit_value)
				arr[cell_idx] = arr[cell_idx] | (1 << (n % UINT32_BIT_SIZE));
			else
				arr[cell_idx] = arr[cell_idx] & ~(1 << (n % UINT32_BIT_SIZE));
		}

		void multiply(
			const uint32_t* A, uint32_t a_rows, uint32_t a_cols,
			const uint32_t* B, uint32_t b_rows, uint32_t b_cols, uint32_t* C)
		{
			for (unsigned int k = 0; k < a_rows; k++)
			{
				for (unsigned int i = 0; i < b_cols; i++)
				{
					bool sum = 0;
					for (uint32_t j = 0; j < a_cols; j++)										
						sum ^= get_bit(A, k * a_cols + j) & get_bit(B, j * b_cols + i);
					set_bit(C, k * b_cols + i, sum);
				}
			}
		}

		void transpose(uint32_t* dst, const uint32_t* src, uint32_t rows, uint32_t cols)
		{
			for(unsigned int i = 0; i < rows; ++i)
				for (unsigned int j = 0; j < cols; ++j)
					set_bit(dst, i * cols + j, get_bit(src, j * rows + i));
		}

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

		void m4ri_multiply(
			const uint32_t* A, uint32_t a_rows, uint32_t a_cols,
			const uint32_t* B_tr, uint32_t b_rows, uint32_t b_cols, uint32_t* C, 
			const uint32_t* precalc_matrix, unsigned int k)
		{
			for(unsigned int index = 0; index < a_rows; ++index)
			{
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
						uint32_t k_vector_B = (B_tr[id_B] >> offset) & ((1 << k) - 1);
						
						sum ^= get_bit(precalc_matrix, k_vector_A * (1 << k) + k_vector_B);
					}
					set_bit(C, index * b_rows + i, sum);
				}
			}
		}
	}
}