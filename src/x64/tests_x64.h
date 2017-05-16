#pragma once

#include <vector>				// std::vector
#include <cstdint>				// uint64_t

#include "m4ri_cpu_x64.hpp"
#include "m4ri_gpu_x64.cu"

namespace x64
{
	#define UINT64_BIT_SIZE 64

	std::vector<bool>
	m4ri_cpu_test(const std::vector<unsigned int>& arr_sizes)
	{
		std::vector<bool> test_results;

		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			unsigned int k = 8;

			uint64_t* A 			 = (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* B 			 = (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* B_tr 			 = (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* cpu_result 	 = (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* cpu_check 	 = (uint64_t *)malloc((*size) * (*size) / 8);

			uint64_t* precalc_matrix = (uint64_t *)malloc((1 << k) * (1 << k) / 8);

			for(uint32_t i = 0; i < (*size) * (*size) / 32; i++)
				((uint32_t *)A)[i] = (uint32_t)rand();

			for (uint32_t i = 0; i < (*size) * (*size) / 32; i++)
				((uint32_t *)B)[i] = (uint32_t)rand();

			cpu::transpose(B_tr, B, *size, *size);
			cpu::m4ri_precalc(precalc_matrix, k);
			cpu::m4ri_multiply(A, *size, *size, B_tr, *size, *size, cpu_result, precalc_matrix, k);

			cpu::multiply(A, *size, *size, B, *size, *size, cpu_check);
			bool success = true;
			for (uint64_t i = 0; i < (*size) * (*size) / UINT64_BIT_SIZE; i++)
			{
				if (cpu_check[i] != cpu_result[i])
				{
					success = false;
					break;
				}
			}
			test_results.push_back(success);
				
			free(A);
			free(B);
			free(B_tr);
			free(cpu_result);
			free(cpu_check);
			free(precalc_matrix);
		}

		return test_results;
	}

	std::vector<bool>
	m4ri_gpu_test(const std::vector<unsigned int>& arr_sizes)
	{
		std::vector<bool> test_results;

		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			unsigned int k = 8;

			uint64_t* A 			= (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* B 			= (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* gpu_result 	= (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* cpu_result 	= (uint64_t *)malloc((*size) * (*size) / 8);

			for(uint32_t i = 0; i < (*size) * (*size) / 32; i++)
				((uint32_t *)A)[i] = (uint32_t)rand();

			for (uint32_t i = 0; i < (*size) * (*size) / 32; i++)
				((uint32_t *)B)[i] = (uint32_t)rand();

			uint64_t *d_A 	 = nullptr; 
			uint64_t *d_B 	 = nullptr; 
			uint64_t *d_B_tr = nullptr;
			uint64_t *d_C 	 = nullptr;
			uint64_t* d_precalc  = nullptr;

			cudaMalloc((void **)&d_A, 		(*size) * (*size) / 8);
			cudaMalloc((void **)&d_B, 		(*size) * (*size) / 8);
			cudaMalloc((void **)&d_B_tr, 	(*size) * (*size) / 8);
			cudaMalloc((void **)&d_C, 		(*size) * (*size) / 8);
			cudaMalloc((void **)&d_precalc, (1 << k) * (1 << k) / 8);

			cudaMemcpy((void *)d_A, (const void*)A, (*size) * (*size) / 8, cudaMemcpyHostToDevice);
			cudaMemcpy((void *)d_B, (const void*)B, (*size) * (*size) / 8, cudaMemcpyHostToDevice);

			dim3 block_size(std::min(int(*size), 512));
			dim3 grid_size(*size / block_size.x);

			gpu::transpose <<< 1, 1 >>> (d_B_tr, d_B, *size, *size);
			gpu::m4ri_precalc <<<1, 1>>> (d_precalc, k);
			gpu::m4ri_multiply <<< grid_size, block_size >>> (d_A, *size, *size, d_B_tr, *size, *size, d_C, d_precalc, k);
			cudaDeviceSynchronize();

			cudaMemcpy((void **)gpu_result, (const void**)d_C, (*size) * (*size) / 8, cudaMemcpyDeviceToHost);

			cpu::multiply(A, *size, *size, B, *size, *size, cpu_result);
			bool success = true;
			for (uint64_t i = 0; i < (*size) * (*size) / UINT64_BIT_SIZE; i++)
			{
				if (cpu_result[i] != gpu_result[i])
				{
					success = false;
					break;
				}
			}
			test_results.push_back(success);
			
			cudaFree((void *)d_A);
			cudaFree((void *)d_B);
			cudaFree((void *)d_C);
			cudaFree((void *)d_B_tr);
			cudaFree((void *)d_precalc);

			free(A);
			free(B);
			free(gpu_result);
			free(cpu_result);
		}

		return test_results;
	}

	std::vector<bool> 
	mar_gpu_test(const std::vector<unsigned int>& arr_sizes)
	{
		std::vector<bool> test_results;

		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			uint64_t* A 			= (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* B 			= (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* gpu_result	= (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* cpu_result	= (uint64_t *)malloc((*size) * (*size) / 8);

			for(uint32_t i = 0; i < (*size) * (*size) / 32; i++)
				((uint32_t *)A)[i] = (uint32_t)rand();

			for (uint32_t i = 0; i < (*size) * (*size) / 32; i++)
				((uint32_t *)B)[i] = (uint32_t)rand();

			uint64_t *d_A 		= nullptr; 
			uint64_t *d_B 		= nullptr; 
			uint64_t *d_C 		= nullptr;

			cudaMalloc((void **)&d_A, (*size) * (*size) / 8);
			cudaMalloc((void **)&d_B, (*size) * (*size) / 8);
			cudaMalloc((void **)&d_C, (*size) * (*size) / 8);

			cudaMemset(d_C, 0, (*size) * (*size) / 8);

			cudaMemcpy((void *)d_A, (const void*)A, (*size) * (*size) / 8, cudaMemcpyHostToDevice);
			cudaMemcpy((void *)d_B, (const void*)B, (*size) * (*size) / 8, cudaMemcpyHostToDevice);

			dim3 block_size(std::min(int(*size), 512));
			dim3 grid_size((*size) / block_size.x);	
			gpu::mar_multiply <<< grid_size, block_size >>> (d_A, *size, *size, d_B, *size, *size, d_C);
			cudaDeviceSynchronize();

			cudaMemcpy((void **)gpu_result, (const void**)d_C, (*size) * (*size) / 8, cudaMemcpyDeviceToHost);

			cpu::multiply(A, *size, *size, B, *size, *size, cpu_result);
			bool success = true;
			for (uint64_t i = 0; i < (*size) * (*size) / UINT64_BIT_SIZE; i++)
			{
				if (cpu_result[i] != gpu_result[i])
				{
					success = false;
					break;
				}
			}
			test_results.push_back(success);
			
			cudaFree((void *)d_A);
			cudaFree((void *)d_B);
			cudaFree((void *)d_C);

			free(gpu_result);
			free(cpu_result);
			free(B);
			free(A);
		}

		return test_results;
	}

	std::vector<bool>
	m4ri_opt_gpu_test(const std::vector<unsigned int>& arr_sizes)
	{
		std::vector<bool> test_results;

		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			uint64_t* A 			= (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* B 			= (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* gpu_result 	= (uint64_t *)malloc((*size) * (*size) / 8);
			uint64_t* cpu_result 	= (uint64_t *)malloc((*size) * (*size) / 8);

			for(uint32_t i = 0; i < (*size) * (*size) / 32; i++)
				((uint32_t *)A)[i] = (uint32_t)rand();

			for (uint32_t i = 0; i < (*size) * (*size) / 32; i++)
				((uint32_t *)B)[i] = (uint32_t)rand();

			uint64_t *d_A = nullptr; 
			uint64_t *d_B = nullptr; 
			uint64_t *d_C = nullptr;

			cudaMalloc((void **)&d_A, (*size) * (*size) / 8);
			cudaMalloc((void **)&d_B, (*size) * (*size) / 8);
			cudaMalloc((void **)&d_C, (*size) * (*size) / 8);

			cudaMemset(d_C, 0, (*size) * (*size) / 8);

			cudaMemcpy((void *)d_A, (const void*)A, (*size) * (*size) / 8, cudaMemcpyHostToDevice);
			cudaMemcpy((void *)d_B, (const void*)B, (*size) * (*size) / 8, cudaMemcpyHostToDevice);

			std::vector<cudaStream_t> cuda_streams(*size / 128);
			for(unsigned int i = 0; i < *size / 128; i++)
			{
				cudaStreamCreate(&cuda_streams[i]);
			}

			std::vector<uint64_t*> precalc_tables(*size / 128);
			for(unsigned int i = 0; i < *size / 128; i++)
			{
				cudaMalloc((void **)&precalc_tables[i], (*size) * 8 * sizeof(uint64_t));
			}

			for(unsigned int i = 0; i < *size / 128; i++)
			{
				// всего таблиц для 1 блока-столбца - 32 * (size / 128) таблиц
				dim3 block_size(std::min(int(*size / 4), 512));
				dim3 grid_size((*size) / block_size.x);	
				gpu::m4ri_opt_precalc <<<grid_size, block_size, 0, cuda_streams[i]>>> (d_B, *size, precalc_tables[i], i);

				block_size = dim3(std::min(int(*size), 512));
				grid_size = dim3((*size) / block_size.x);	
				gpu::m4ri_opt_multiply <<<grid_size, block_size, 0, cuda_streams[i]>>> 
					(d_A, *size, *size, d_B, *size, *size, d_C, precalc_tables[i], i);
			}

			cudaDeviceSynchronize();

			cudaMemcpy((void **)gpu_result, (const void**)d_C, (*size) * (*size) / 8, cudaMemcpyDeviceToHost);

			cpu::multiply(A, *size, *size, B, *size, *size, cpu_result);
			bool success = true;
			for (uint64_t i = 0; i < (*size) * (*size) / UINT64_BIT_SIZE; i++)
			{
				if (gpu_result[i] != cpu_result[i])
				{
					success = false;
					break;
				}
			}
			test_results.push_back(success);

			for(unsigned int i = 0; i < *size / 128; i++)
				cudaStreamDestroy(cuda_streams[i]);

			for(unsigned int i = 0; i < *size / 128; i++)
				cudaFree((void *)precalc_tables[i]);
			cudaFree((void *)d_A);
			cudaFree((void *)d_B);
			cudaFree((void *)d_C);

			free(cpu_result);
			free(gpu_result);
			free(B);
			free(A);

		}

		return test_results;
	}
}	// namespace x64