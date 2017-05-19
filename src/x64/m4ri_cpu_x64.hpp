#pragma once

#include <cstdint>

#define UINT64_BIT_SIZE 64

namespace x64
{
	namespace cpu 
	{
		const unsigned int SUBVECTOR_SIZE 	 = 8;
		const unsigned int M4RI_PRECALC_SIZE = (1 << SUBVECTOR_SIZE) * (1 << SUBVECTOR_SIZE);

		bool inline get_bit(const uint64_t* arr, uint64_t n)
		{
			uint64_t cell_idx = n / UINT64_BIT_SIZE;
			return arr[cell_idx] & (uint64_t(1) << (n % UINT64_BIT_SIZE));
		}


		void inline set_bit(uint64_t* arr, uint64_t n, uint64_t bit_value)
		{
			uint64_t cell_idx = n / UINT64_BIT_SIZE;
			if (bit_value)
				arr[cell_idx] = arr[cell_idx] | (uint64_t(1) << (n % UINT64_BIT_SIZE));
			else
				arr[cell_idx] = arr[cell_idx] & ~(uint64_t(1) << (n % UINT64_BIT_SIZE));
		}

		void multiply(
			const uint64_t* A, uint64_t a_rows, uint64_t a_cols,
			const uint64_t* B, uint64_t b_rows, uint64_t b_cols, uint64_t* C)
		{
			for (uint64_t k = 0; k < a_rows; k++)
			{
				for (uint64_t i = 0; i < b_cols; i++)
				{
					bool sum = 0;
					for (uint64_t j = 0; j < a_cols; j++)										
						sum ^= get_bit(A, k * a_cols + j) & get_bit(B, j * b_cols + i);
					set_bit(C, k * b_cols + i, sum);
				}
			}
		}

		void transpose(uint64_t* dst, const uint64_t* src, uint64_t rows, uint64_t cols)
		{
			for(unsigned long i = 0; i < rows; ++i)
				for (unsigned long j = 0; j < cols; ++j)
					set_bit(dst, i * cols + j, get_bit(src, j * rows + i));
		}

		void m4ri_precalc(uint64_t* precalc_matrix, uint64_t bits)
		{
			for(uint64_t i = 0; i < (1 << bits); i++)
			{
				for(uint64_t j = 0; j < (1 << bits); j++)
				{
					uint64_t scalar_product = 0;
					for(uint64_t k = 0; k < bits; k++)
					{
						scalar_product ^= get_bit(&i, k) & get_bit(&j, k);
					}
					set_bit(precalc_matrix, i * (1 << bits) + j, scalar_product);
				}
			}
		}

		void m4ri_multiply(
			const uint64_t* A, uint64_t a_rows, uint64_t a_cols,
			const uint64_t* B_tr, uint64_t b_rows, uint64_t b_cols, uint64_t* C, 
			const uint64_t* precalc_matrix, unsigned int k)
		{
			for(unsigned int index = 0; index < a_rows; ++index)
			{
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
						uint64_t k_vector_B = (B_tr[id_B] >> offset) & ((1 << k) - 1);

						sum ^= get_bit(precalc_matrix, k_vector_A * (1 << k) + k_vector_B);
					}
					set_bit(C, index * b_rows + i, sum);
				}
			}
		}
	}	// namespace cpu
}	// namespace x86