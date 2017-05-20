#pragma once

#include <vector>				// std::vector
#include <cstdint>				// uint32_t
#include <chrono>				// std::chrono::steady_clock

#include <iostream>

#include "m4ri_cpu_x86.hpp"
#include "m4ri_gpu_x86.cu"

namespace x86
{
	int log2(uint32_t a)
	{
		int value = 1;
		while(a / 2 != 1)
		{
			a /= 2;
			value++;
		}
		return value;
	}

	std::vector<std::vector<long long>> 
	simple_cpu_benchmark(const std::vector<unsigned int>& arr_sizes, unsigned int times)
	{
		std::vector<std::vector<long long>> bench_results(arr_sizes.size(), std::vector<long long>(3));
		unsigned int iter = 0;
		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			std::vector<std::vector<long long>> durations(times, std::vector<long long>(3));
			for(unsigned int rep = 0; rep < times; rep++)
			{
				uint32_t* A 		 = (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* B 		 = (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* cpu_result = (uint32_t *)malloc((*size) * (*size) / 8);

				for(uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
					A[i] = (uint32_t)rand();

				for (uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
					B[i] = (uint32_t)rand();

				auto cpu_time_begin = std::chrono::steady_clock::now();
				cpu::multiply(A, *size, *size, B, *size, *size, cpu_result);
				auto cpu_time_end = std::chrono::steady_clock::now();
				long long mult_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_time_end - cpu_time_begin).count();

				free(A);
				free(B);
				free(cpu_result);

				durations[rep] = {0, mult_time, 0};
			}

			for(unsigned int i = 0; i < 3; i++)
			{
				bench_results[iter][i] = std::accumulate(durations.cbegin(), durations.cend(), (long long)0, 
					[i](const long long a, const std::vector<long long> v2){
						return a + v2[i];
					}
				);				
			}

			iter++;
		}

		return bench_results;
	}
	std::vector<std::vector<long long>> 
	m4ri_cpu_benchmark(const std::vector<unsigned int>& arr_sizes, unsigned int times)
	{
		std::vector<std::vector<long long>> bench_results(arr_sizes.size(), std::vector<long long>(3));
		unsigned int iter = 0;
		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			std::vector<std::vector<long long>> durations(times, std::vector<long long>(3));
			for(unsigned int rep = 0; rep < times; rep++)
			{
				uint32_t* A 			 = (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* B 			 = (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* B_tr 			 = (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* cpu_result 	 = (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* precalc_matrix = (uint32_t *)malloc(cpu::M4RI_PRECALC_SIZE / 8);

				for(uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
					A[i] = (uint32_t)rand();

				for (uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
					B[i] = (uint32_t)rand();

				auto prep_begin = std::chrono::steady_clock::now();
				cpu::transpose(B_tr, B, *size, *size);
				cpu::m4ri_precalc(precalc_matrix, cpu::SUBVECTOR_SIZE);
				auto prep_end = std::chrono::steady_clock::now();
				long long prep_time = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_begin).count();

				auto cpu_time_begin = std::chrono::steady_clock::now();
				cpu::m4ri_multiply(A, *size, *size, B_tr, *size, *size, cpu_result, precalc_matrix, cpu::SUBVECTOR_SIZE);
				auto cpu_time_end = std::chrono::steady_clock::now();
				long long mult_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_time_end - cpu_time_begin).count();
						
				free(A);
				free(B);
				free(B_tr);
				free(cpu_result);
				free(precalc_matrix);

				durations[rep] = {prep_time, mult_time, 0};
			}

			for(unsigned int i = 0; i < 3; i++)
			{
				bench_results[iter][i] = std::accumulate(durations.cbegin(), durations.cend(), (long long)0, 
					[i](const long long a, const std::vector<long long> v2){
						return a + v2[i];
					}
				);				
			}

			iter++;
		}

		return bench_results;
	}

	std::vector<std::vector<long long>> 
	m4ri_gpu_benchmark(const std::vector<unsigned int>& arr_sizes, unsigned int times)
	{
		std::vector<std::vector<long long>> bench_results(arr_sizes.size(), std::vector<long long>(3));
		unsigned int iter = 0;
		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			std::vector<std::vector<long long>> durations(times, std::vector<long long>(3));
			for(unsigned int rep = 0; rep < times; rep++)
			{
				uint32_t* A 			= (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* B 			= (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* gpu_result 	= (uint32_t *)malloc((*size) * (*size) / 8);

				for(uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
						A[i] = (uint32_t)rand();

				for (uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
						B[i] = (uint32_t)rand();

				uint32_t *d_A 		= nullptr; 
				uint32_t *d_B 		= nullptr; 
				uint32_t *d_B_tr 	= nullptr;
				uint32_t *d_C 		= nullptr;
				uint32_t* d_precalc = nullptr;

				cudaMalloc((void **)&d_A, 		(*size) * (*size) / 8);
				cudaMalloc((void **)&d_B, 		(*size) * (*size) / 8);
				cudaMalloc((void **)&d_B_tr, 	(*size) * (*size) / 8);
				cudaMalloc((void **)&d_C, 		(*size) * (*size) / 8);
				cudaMalloc((void **)&d_precalc, gpu::M4RI_PRECALC_SIZE / 8);

				auto transfer_hd_begin = std::chrono::steady_clock::now();
				cudaMemcpy((void *)d_A, (const void*)A, (*size) * (*size) / 8, cudaMemcpyHostToDevice);
				cudaMemcpy((void *)d_B, (const void*)B, (*size) * (*size) / 8, cudaMemcpyHostToDevice);
				auto transfer_hd_end = std::chrono::steady_clock::now();

				dim3 block_size(std::min(1 << gpu::SUBVECTOR_SIZE, 512));
				dim3 grid_size((1 << gpu::SUBVECTOR_SIZE) / block_size.x);

				auto prep_begin = std::chrono::steady_clock::now();
				gpu::m4ri_precalc <<< grid_size, block_size >>> (d_precalc, gpu::SUBVECTOR_SIZE);

				block_size = dim3(std::min(int(*size), 512));
				grid_size  = dim3(*size / block_size.x);
				
				gpu::transpose <<< grid_size, block_size >>> (d_B_tr, d_B, *size, *size);
				cudaDeviceSynchronize();
				auto prep_end = std::chrono::steady_clock::now();
				long long prep_time = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_begin).count();

				auto gpu_mult_begin = std::chrono::steady_clock::now();
				gpu::m4ri_multiply <<< grid_size, block_size >>> 
					(d_A, *size, *size, d_B_tr, *size, *size, d_C, d_precalc, gpu::SUBVECTOR_SIZE);
				cudaDeviceSynchronize();
				auto gpu_mult_end = std::chrono::steady_clock::now();
				long long mult_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_mult_end - gpu_mult_begin).count();

				auto transfer_dh_begin = std::chrono::steady_clock::now();
				cudaMemcpy((void **)gpu_result, (const void**)d_C, (*size) * (*size) / 8, cudaMemcpyDeviceToHost);
				auto transfer_dh_end = std::chrono::steady_clock::now();
				long long transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(
					transfer_hd_end - transfer_hd_begin + transfer_dh_end - transfer_dh_begin).count();

				free(gpu_result);
				free(B);
				free(A);

				cudaFree((void *)d_A);
				cudaFree((void *)d_B);
				cudaFree((void *)d_C);
				cudaFree((void *)d_B_tr);
				cudaFree((void *)d_precalc);

				durations[rep] = {prep_time, mult_time, transfer_time};
			}

			for(unsigned int i = 0; i < 3; i++)
			{
				bench_results[iter][i] = std::accumulate(durations.cbegin(), durations.cend(), (long long)0, 
					[i](const long long a, const std::vector<long long> v2){
						return a + v2[i];
					}
				);				
			}

			iter++;
		}

		return bench_results;
	}

	std::vector<std::vector<long long>> 
	mar_gpu_benchmark(const std::vector<unsigned int>& arr_sizes, unsigned int times)
	{
		std::vector<std::vector<long long>> bench_results(arr_sizes.size(), std::vector<long long>(3));
		unsigned int iter = 0;
		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			std::vector<std::vector<long long>> durations(times, std::vector<long long>(3));
			for(unsigned int rep = 0; rep < times; rep++)
			{
				uint32_t* A 			= (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* B 			= (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* gpu_result 	= (uint32_t *)malloc((*size) * (*size) / 8);

				for(uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
						A[i] = (uint32_t)rand();

				for (uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
						B[i] = (uint32_t)rand();

				uint32_t *d_A = nullptr; 
				uint32_t *d_B = nullptr; 
				uint32_t *d_C = nullptr;

				cudaMalloc((void **)&d_A, (*size) * (*size) / 8);
				cudaMalloc((void **)&d_B, (*size) * (*size) / 8);
				cudaMalloc((void **)&d_C, (*size) * (*size) / 8);

				cudaMemset(d_C, 0, (*size) * (*size) / 8);

				auto transfer_hd_begin = std::chrono::steady_clock::now();
				cudaMemcpy((void *)d_A, (const void*)A, (*size) * (*size), cudaMemcpyHostToDevice);
				cudaMemcpy((void *)d_B, (const void*)B, (*size) * (*size), cudaMemcpyHostToDevice);
				auto transfer_hd_end = std::chrono::steady_clock::now();

				dim3 block_size(std::min(int(*size), 512));
				dim3 grid_size((*size) / block_size.x);	
				
				auto mult_begin = std::chrono::steady_clock::now();
				gpu::mar_multiply <<< grid_size, block_size >>> (d_A, *size, *size, d_B, *size, *size, d_C);
				cudaDeviceSynchronize();
				auto mult_end = std::chrono::steady_clock::now();
				long long mult_time = std::chrono::duration_cast<std::chrono::microseconds>(mult_end - mult_begin).count();

				auto transfer_dh_begin = std::chrono::steady_clock::now();
				cudaMemcpy((void **)gpu_result, (const void**)d_C, (*size) * (*size) / 8, cudaMemcpyDeviceToHost);
				auto transfer_dh_end = std::chrono::steady_clock::now();
				long long transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(
					transfer_hd_end - transfer_hd_begin + transfer_dh_end - transfer_dh_begin).count();

				cudaFree((void *)d_A);
				cudaFree((void *)d_B);
				cudaFree((void *)d_C);

				free(gpu_result);
				free(B);
				free(A);

				durations[rep] = {0, mult_time, transfer_time};
			}

			for(unsigned int i = 0; i < 3; i++)
			{
				bench_results[iter][i] = std::accumulate(durations.cbegin(), durations.cend(), (long long)0, 
					[i](const long long a, const std::vector<long long> v2){
						return a + v2[i];
					}
				);				
			}

			iter++;
		}

		return bench_results;
	}

	std::vector<std::vector<long long>>
	m4ri_opt_gpu_benchmark(const std::vector<unsigned int>& arr_sizes, unsigned int times)
	{
		std::vector<std::vector<long long>> bench_results(arr_sizes.size(), std::vector<long long>(3));
		unsigned int iter = 0;
		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			std::vector<std::vector<long long>> durations(times, std::vector<long long>(3));
			for(unsigned int rep = 0; rep < times; rep++)
			{
				uint32_t* A 		 = (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* B 		 = (uint32_t *)malloc((*size) * (*size) / 8);
				uint32_t* gpu_result = (uint32_t *)malloc((*size) * (*size) / 8);

				for(uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
						A[i] = (uint32_t)rand();

				for (uint32_t i = 0; i < (*size) * (*size) / UINT32_BIT_SIZE; i++)
						B[i] = (uint32_t)rand();

				uint32_t *d_A = nullptr; 
				uint32_t *d_B = nullptr; 
				uint32_t *d_C = nullptr;

				cudaMalloc((void **)&d_A, (*size) * (*size) / 8);
				cudaMalloc((void **)&d_B, (*size) * (*size) / 8);
				cudaMalloc((void **)&d_C, (*size) * (*size) / 8);

				cudaMemset(d_C, 0, (*size) * (*size) / 8);

				auto transfer_hd_begin = std::chrono::steady_clock::now();
				cudaMemcpy((void *)d_A, (const void*)A, (*size) * (*size) / 8, cudaMemcpyHostToDevice);
				cudaMemcpy((void *)d_B, (const void*)B, (*size) * (*size) / 8, cudaMemcpyHostToDevice);
				auto transfer_hd_end = std::chrono::steady_clock::now();

				std::vector<cudaStream_t> cuda_streams(*size / 128);
				for(unsigned int i = 0; i < *size / 128; i++)
				{
					cudaStreamCreate(&cuda_streams[i]);
				}

				std::vector<uint32_t*> precalc_tables(*size / 128);
				for(unsigned int i = 0; i < *size / 128; i++)
				{
					cudaMalloc((void **)&precalc_tables[i], (*size) * 16 * sizeof(uint32_t));
				}

				auto mult_begin = std::chrono::steady_clock::now();
				for(unsigned int i = 0; i < *size / 128; i++)
				{
					// всего таблиц для 1 блока-столбца - 32 * (size / 128), общий размер 32 * size / 128 * 64 * sizeof(uint32_t) 
					dim3 block_size(std::min(int(*size / 4), 512));
					dim3 grid_size((*size) / block_size.x);	
					gpu::m4ri_opt_precalc <<<grid_size, block_size, 0, cuda_streams[i]>>> (d_B, *size, precalc_tables[i], i);

					block_size = dim3(std::min(int(*size), 512));
					grid_size = dim3((*size) / block_size.x);	
					gpu::m4ri_opt_multiply <<<grid_size, block_size, 0, cuda_streams[i]>>> 
						(d_A, *size, *size, d_B, *size, *size, d_C, precalc_tables[i], i);
				}
				cudaDeviceSynchronize();

				auto mult_end = std::chrono::steady_clock::now();
				long long mult_time = std::chrono::duration_cast<std::chrono::microseconds>(mult_end - mult_begin).count();

				auto transfer_dh_begin = std::chrono::steady_clock::now();
				cudaMemcpy((void **)gpu_result, (const void**)d_C, (*size) * (*size) / 8, cudaMemcpyDeviceToHost);
				auto transfer_dh_end = std::chrono::steady_clock::now();
				long long transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(
					transfer_hd_end - transfer_hd_begin + transfer_dh_end - transfer_dh_begin).count();

				for(unsigned int i = 0; i < *size / 128; i++)
					cudaStreamDestroy(cuda_streams[i]);

				for(unsigned int i = 0; i < *size / 128; i++)
					cudaFree((void *)precalc_tables[i]);
				cudaFree((void *)d_A);
				cudaFree((void *)d_B);
				cudaFree((void *)d_C);

				free(gpu_result);
				free(B);
				free(A);

				durations[rep] = {0, mult_time, transfer_time};
			}

			for(unsigned int i = 0; i < 3; i++)
			{
				bench_results[iter][i] = std::accumulate(durations.cbegin(), durations.cend(), (long long)0, 
					[i](const long long a, const std::vector<long long> v2){
						return a + v2[i];
					}
				);				
			}

			iter++;
		}

		return bench_results;
	}
}	// namespace x86