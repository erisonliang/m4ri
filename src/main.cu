#include <cstdint>				// uint32_t
#include <iostream>				// std::cout, std::endl
#include <vector>				// std::vector
#include <iomanip>				// std::setw
#include <string>				// std::to_string
#include <functional>			// std::plus
#include <numeric>				// std::accumulate

#include <benchmarks_x86.h>
#include <benchmarks_x64.h>

#include <tests_x86.h>
#include <tests_x64.h>

void print_bench(const std::string& title, 
				 const std::vector<unsigned int>& arr_sizes, 
				 const std::vector<std::vector<long long>>& bench)
{
	std::cout << title << std::endl;
	std::cout << std::setw(15) << "SIZE" << std::setw(15) << "PREP_DUR" << std::setw(15) << "MULT_DUR" 
		<< std::setw(15) << "TRANSF_DUR" << std::setw(15) << "TOTAL" << std::endl;

	for(unsigned int i = 0; i < arr_sizes.size(); i++)
	{
		std::cout << std::setw(15) << (std::to_string(arr_sizes[i]) + "x" + std::to_string(arr_sizes[i]));
		std::cout << std::setw(15) << (std::to_string(bench[i][0]) + "µm");
		std::cout << std::setw(15) << (std::to_string(bench[i][1]) + "µm");
		std::cout << std::setw(15) << (std::to_string(bench[i][2]) + "µm");
		std::cout << std::setw(15) << (std::to_string(std::accumulate(bench[i].begin(), bench[i].end(), 
			0, std::plus<long long>())) + "µm") << std::endl;
	}
	std::cout << std::endl;
}

bool print_test(const std::string& title, 
				const std::vector<unsigned int>& arr_sizes, 
				const std::vector<bool>& tests)
{
	std::cout << title << std::endl;
	std::cout << std::setw(10) << "SIZE" << std::setw(10) << "RESULT" << std::endl;

	bool success = true;

	for(unsigned int i = 0; i < arr_sizes.size(); i++)
	{
		std::cout << std::setw(10) << (std::to_string(arr_sizes[i]) + "x" + std::to_string(arr_sizes[i]));
		if(tests[i])
			std::cout << std::setw(10) << "PASSED" << std::endl;
		else
		{
			std::cout << std::setw(10) << "FAILED" << std::endl;
			success = false;
		}
	}
	std::cout << std::endl;
	return success;
}

int main() 
{
	const std::vector<unsigned int> arr_sizes{128, 256, 512, 1024, 2048};
	const unsigned int times = 1;

	bool success;
	
	// Каноничные версии алгоритмов - не тестируются
	auto simple_cpu_bench_results_x86 = x86::simple_cpu_benchmark(arr_sizes, times);
	print_bench("BENCH (x86)Simple CPU matrix multiplication:", arr_sizes, simple_cpu_bench_results_x86);

	auto simple_cpu_bench_results_x64 = x64::simple_cpu_benchmark(arr_sizes, times);
	print_bench("BENCH (x64)Simple CPU matrix multiplication:", arr_sizes, simple_cpu_bench_results_x64);

	std::cout << std::endl;
	
	//!-----------------------------------------------------------------------------------------------------
	
	auto m4ri_cpu_test_results_x86 = x86::m4ri_cpu_test(arr_sizes);
	success = print_test("TEST (x86)M4RI CPU matrix multiplication:", arr_sizes, m4ri_cpu_test_results_x86);
	if(success)
	{
		auto m4ri_cpu_bench_results_x86 = x86::m4ri_cpu_benchmark(arr_sizes, times);
		print_bench("BENCH (x86)M4RI CPU matrix multiplication:", arr_sizes, m4ri_cpu_bench_results_x86);
	}
	std::cout << std::endl;
	
	auto m4ri_cpu_test_results_x64 = x64::m4ri_cpu_test(arr_sizes);
	success = print_test("TEST (x64)M4RI CPU matrix multiplication:", arr_sizes, m4ri_cpu_test_results_x64);
	if(success)
	{
		auto m4ri_cpu_bench_results_x64 = x64::m4ri_cpu_benchmark(arr_sizes, times);
		print_bench("BENCH (x64)M4RI CPU matrix multiplication:", arr_sizes, m4ri_cpu_bench_results_x64);
	}
	std::cout << std::endl;
	
	//!-----------------------------------------------------------------------------------------------------
	
	auto m4ri_gpu_test_results_x86 = x86::m4ri_gpu_test(arr_sizes);
	success = print_test("TEST (x86)M4RI GPU matrix multiplication:", arr_sizes, m4ri_gpu_test_results_x86);
	if(success)
	{
		auto m4ri_gpu_bench_results_x86 = x86::m4ri_gpu_benchmark(arr_sizes, times);
		print_bench("BENCH (x86)M4RI GPU matrix multiplication:", arr_sizes, m4ri_gpu_bench_results_x86);
	}
	std::cout << std::endl;
	
	auto m4ri_gpu_test_results_x64 = x64::m4ri_gpu_test(arr_sizes);
	success = print_test("TEST (x64)M4RI GPU matrix multiplication:", arr_sizes, m4ri_gpu_test_results_x64);
	if(success)
	{
		auto m4ri_gpu_bench_results_x64 = x64::m4ri_gpu_benchmark(arr_sizes, times);
		print_bench("BENCH (x64)M4RI GPU matrix multiplication:", arr_sizes, m4ri_gpu_bench_results_x64);
	}
	std::cout << std::endl;
	
	//!-----------------------------------------------------------------------------------------------------
	
	auto mar_gpu_test_results_x86 = x86::mar_gpu_test(arr_sizes);
	success = print_test("TEST (x86)MAR GPU matrix multiplication:", arr_sizes, mar_gpu_test_results_x86);
	if(success)
	{
		auto mar_gpu_bench_results_x86 = x86::mar_gpu_benchmark(arr_sizes, times);
		print_bench("BENCH (x86)MAR GPU matrix multiplication:", arr_sizes, mar_gpu_bench_results_x86);
	}
	std::cout << std::endl;

	auto mar_gpu_test_results_x64 = x64::mar_gpu_test(arr_sizes);
	success = print_test("TEST (x64)MAR GPU matrix multiplication:", arr_sizes, mar_gpu_test_results_x64);
	if(success)
	{
		auto mar_gpu_bench_results_x64 = x64::mar_gpu_benchmark(arr_sizes, times);
		print_bench("BENCH (x64)MAR GPU matrix multiplication:", arr_sizes, mar_gpu_bench_results_x64);
	}
	std::cout << std::endl;
	
	//!-----------------------------------------------------------------------------------------------------
	
	auto m4ri_opt_gpu_test_results_x86 = x86::m4ri_opt_gpu_test(arr_sizes);
	success = print_test("TEST (x86)M4RI OPTIMIZED GPU matrix multiplication:", arr_sizes, m4ri_opt_gpu_test_results_x86);
	if(success)
	{
		auto m4ri_opt_gpu_bench_results_x86 = x86::m4ri_opt_gpu_benchmark(arr_sizes, times);
		print_bench("BENCH (x86)M4RI OPTIMIZED GPU matrix multiplication:", arr_sizes, m4ri_opt_gpu_bench_results_x86);
	}
	std::cout << std::endl;
	
	auto m4ri_opt_gpu_test_results_x64 = x64::m4ri_opt_gpu_test(arr_sizes);
	success = print_test("TEST (x64)M4RI OPTIMIZED GPU matrix multiplication:", arr_sizes, m4ri_opt_gpu_test_results_x64);
	if(success)
	{
		auto m4ri_opt_gpu_bench_results_x64 = x64::m4ri_opt_gpu_benchmark(arr_sizes, times);
		print_bench("BENCH (x64)M4RI OPTIMIZED GPU matrix multiplication:", arr_sizes, m4ri_opt_gpu_bench_results_x64);
	}
	std::cout << std::endl;

	//!-----------------------------------------------------------------------------------------------------
	
	return 0;
}
