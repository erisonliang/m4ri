#include <vector>				// std::vector
#include <cstdint>				// uint32_t
#include <chrono>				// std::chrono::steady_clock

#include <iostream>				// std::cout, std::endl
#include <bitset>				// std::bitset

#include "m4ri_cpu_x86.hpp"
#include "m4ri_gpu_x86.cu"

namespace x86
{

	#define UINT32_BIT_SIZE 32

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
	simple_cpu_benchmark(const std::vector<unsigned int>& arr_sizes)
	{
		std::vector<std::vector<long long>> bench_results;

		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
				uint32_t* A 			= new uint32_t[(*size) * (*size) / 4];
				uint32_t* B 			= new uint32_t[(*size) * (*size) / 4];
				uint32_t* cpu_result 	= new uint32_t[(*size) * (*size) / 4];

				for (uint32_t i = 0; i < *size; i++)
					for (uint32_t j = 0; j < *size / UINT32_BIT_SIZE; j++)
						A[i * (*size) / UINT32_BIT_SIZE + j] = (uint32_t)rand();

				for (uint32_t i = 0; i < *size; i++)
					for (uint32_t j = 0; j < *size / UINT32_BIT_SIZE; j++)
						B[i * (*size) / UINT32_BIT_SIZE + j] = (uint32_t)rand();

				auto cpu_time_begin = std::chrono::steady_clock::now();
				cpu::multiply(A, *size, *size, B, *size, *size, cpu_result);
				auto cpu_time_end = std::chrono::steady_clock::now();
				long long mult_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_time_end - cpu_time_begin).count();

				bench_results.push_back({0, mult_time, 0});

				delete[] A;
				delete[] B;
				delete[] cpu_result;
		}

		return bench_results;
	}

	std::vector<std::vector<long long>> 
	m4ri_cpu_benchmark(const std::vector<unsigned int>& arr_sizes)
	{
		std::vector<std::vector<long long>> bench_results;

		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
				unsigned int k = log2(*size);

				uint32_t* A 			= new uint32_t[(*size) * (*size) / 4];
				uint32_t* B 			= new uint32_t[(*size) * (*size) / 4];
				uint32_t* B_tr 			= new uint32_t[(*size) * (*size) / 4];
				uint32_t* cpu_result 	= new uint32_t[(*size) * (*size) / 4];
				bool* precalc_matrix 	= new bool[(1 << k) * (1 << k)];

				for (uint32_t i = 0; i < *size; i++)
					for (uint32_t j = 0; j < *size / UINT32_BIT_SIZE; j++)
						A[i * (*size) / UINT32_BIT_SIZE + j] = (uint32_t)rand();

				for (uint32_t i = 0; i < *size; i++)
					for (uint32_t j = 0; j < *size / UINT32_BIT_SIZE; j++)
						B[i * (*size) / UINT32_BIT_SIZE + j] = (uint32_t)rand();

				auto prep_begin = std::chrono::steady_clock::now();
				cpu::transpose(B_tr, B, *size, *size);
				cpu::m4ri_precalc(precalc_matrix, k);
				auto prep_end = std::chrono::steady_clock::now();
				long long prep_time = std::chrono::duration_cast<std::chrono::milliseconds>(prep_end - prep_begin).count();

				auto cpu_time_begin = std::chrono::steady_clock::now();
				cpu::m4ri_multiply(A, *size, *size, B_tr, *size, *size, cpu_result, precalc_matrix, k);
				auto cpu_time_end = std::chrono::steady_clock::now();
				long long mult_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_time_end - cpu_time_begin).count();

				bench_results.push_back({prep_time, mult_time, 0});
				
				delete[] A;
				delete[] B;
				delete[] B_tr;
				delete[] cpu_result;
				delete[] precalc_matrix;
		}

		return bench_results;
	}

	std::vector<std::vector<long long>> 
	m4ri_gpu_benchmark(const std::vector<unsigned int>& arr_sizes)
	{
		std::vector<std::vector<long long>> bench_results;

		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			uint32_t* A 			= new uint32_t[(*size) * (*size) / 4];
			uint32_t* B 			= new uint32_t[(*size) * (*size) / 4];
			uint32_t* gpu_result 	= new uint32_t[(*size) * (*size) / 4];

			unsigned int k = log2(*size);

			for (uint32_t i = 0; i < *size; i++)
				for (uint32_t j = 0; j < *size / UINT32_BIT_SIZE; j++)
					A[i * (*size) / UINT32_BIT_SIZE + j] = (uint32_t)rand();

			for (uint32_t i = 0; i < *size; i++)
				for (uint32_t j = 0; j < *size / UINT32_BIT_SIZE; j++)
					B[i * (*size) / UINT32_BIT_SIZE + j] = (uint32_t)rand();

			uint32_t *d_A = nullptr; 
			uint32_t *d_B = nullptr; 
			uint32_t *d_B_tr = nullptr;
			uint32_t *d_C = nullptr;
			bool* d_precalc = nullptr;

			cudaMalloc((void **)&d_A, (*size) * (*size));
			cudaMalloc((void **)&d_B, (*size) * (*size));
			cudaMalloc((void **)&d_B_tr, (*size) * (*size));
			cudaMalloc((void **)&d_C, (*size) * (*size));
			cudaMalloc((void **)&d_precalc, (1 << k) * (1 << k));

			auto transfer_hd_begin = std::chrono::steady_clock::now();
			cudaMemcpy((void *)d_A, (const void*)A, (*size) * (*size), cudaMemcpyHostToDevice);
			cudaMemcpy((void *)d_B, (const void*)B, (*size) * (*size), cudaMemcpyHostToDevice);
			auto transfer_hd_end = std::chrono::steady_clock::now();

			dim3 block_size(std::min(int(*size), 512));
			dim3 grid_size(*size / block_size.x);

			auto prep_begin = std::chrono::steady_clock::now();
			gpu::transpose <<< 1, 1 >>> (d_B_tr, d_B, *size, *size);
			gpu::m4ri_precalc <<<1, 1>>> (d_precalc, k);
			cudaDeviceSynchronize();
			auto prep_end = std::chrono::steady_clock::now();
			long long prep_time = std::chrono::duration_cast<std::chrono::milliseconds>(prep_end - prep_begin).count();


			auto gpu_mult_begin = std::chrono::steady_clock::now();
			gpu::m4ri_multiply <<< grid_size, block_size >>> (d_A, *size, *size, d_B_tr, *size, *size, d_C, d_precalc, k);
			cudaDeviceSynchronize();
			auto gpu_mult_end = std::chrono::steady_clock::now();
			long long mult_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_mult_end - gpu_mult_begin).count();

			auto transfer_dh_begin = std::chrono::steady_clock::now();
			cudaMemcpy((void **)gpu_result, (const void**)d_C, (*size) * (*size), cudaMemcpyDeviceToHost);
			auto transfer_dh_end = std::chrono::steady_clock::now();
			long long transfer_time = std::chrono::duration_cast<std::chrono::milliseconds>(
				transfer_hd_end - transfer_hd_begin + transfer_dh_end - transfer_dh_begin).count();

			bench_results.push_back({prep_time, mult_time, transfer_time});

			cudaFree((void *)d_A);
			cudaFree((void *)d_B);
			cudaFree((void *)d_C);
			cudaFree((void *)d_B_tr);
			cudaFree((void *)d_precalc);
		}

		return bench_results;
	}

	std::vector<std::vector<long long>> 
	mar_gpu_benchmark(const std::vector<unsigned int>& arr_sizes)
	{
		std::vector<std::vector<long long>> bench_results;

		for(auto size = arr_sizes.cbegin(); size != arr_sizes.cend(); size++)
		{
			uint32_t* A 			= new uint32_t[(*size) * (*size) / 4];
			uint32_t* B 			= new uint32_t[(*size) * (*size) / 4];
			uint32_t* C 			= new uint32_t[(*size) * (*size) / 4];
			uint32_t* gpu_result	= new uint32_t[(*size) * (*size) / 4];

			for (uint32_t i = 0; i < *size; i++)
				for (uint32_t j = 0; j < *size / UINT32_BIT_SIZE; j++)
					A[i * (*size) / UINT32_BIT_SIZE + j] = (uint32_t)rand();

			for (uint32_t i = 0; i < *size; i++)
				for (uint32_t j = 0; j < *size / UINT32_BIT_SIZE; j++)
					B[i * (*size) / UINT32_BIT_SIZE + j] = (uint32_t)rand();

			uint32_t *d_A 		= nullptr; 
			uint32_t *d_B 		= nullptr; 
			uint32_t *d_C 		= nullptr;

			cudaMalloc((void **)&d_A, (*size) * (*size));
			cudaMalloc((void **)&d_B, (*size) * (*size));
			cudaMalloc((void **)&d_C, (*size) * (*size));

			cudaMemset(d_C, 0, (*size) * (*size));

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
			long long mult_time = std::chrono::duration_cast<std::chrono::milliseconds>(mult_end - mult_begin).count();

			auto transfer_dh_begin = std::chrono::steady_clock::now();
			cudaMemcpy((void **)gpu_result, (const void**)d_C, (*size) * (*size), cudaMemcpyDeviceToHost);
			auto transfer_dh_end = std::chrono::steady_clock::now();
			long long transfer_time = std::chrono::duration_cast<std::chrono::milliseconds>(
				transfer_hd_end - transfer_hd_begin + transfer_dh_end - transfer_dh_begin).count();

			bench_results.push_back({0, mult_time, transfer_time});

			cudaFree((void *)d_A);
			cudaFree((void *)d_B);
			cudaFree((void *)d_C);

			delete[] gpu_result;
			delete[] C;
			delete[] B;
			delete[] A;
		}

		return bench_results;
	}
}