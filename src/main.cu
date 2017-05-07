#include <cstdint>				// uint32_t
#include <iostream>				// std::cout, std::endl
#include <vector>				// std::vector
#include <iomanip>				// std::setw
#include <string>				// std::to_string
#include <functional>			// std::plus
#include <numeric>				// std::accumulate

#include <benchmarks_x86.h>
#include <benchmarks_x64.h>

void print_bench(const std::string& title, 
				 const std::vector<unsigned int>& arr_sizes, 
				 const std::vector<std::vector<long long>>& bench)
{
	std::cout << title << std::endl;
	std::cout << std::setw(10) << "SIZE" << std::setw(10) << "PREP_DUR" << std::setw(10) << "MULT_DUR" 
		<< std::setw(11) << "TRANSF_DUR" << std::setw(10) << "TOTAL" << std::endl;
	for(unsigned int i = 0; i < arr_sizes.size(); i++)
	{
		std::cout << std::setw(10) << (std::to_string(arr_sizes[i]) + "x" + std::to_string(arr_sizes[i]));
		std::cout << std::setw(10) << (std::to_string(bench[i][0]) + "ms");
		std::cout << std::setw(10) << (std::to_string(bench[i][1]) + "ms");
		std::cout << std::setw(10) << (std::to_string(bench[i][2]) + "ms");
		std::cout << std::setw(10) << (std::to_string(std::accumulate(bench[i].begin(), bench[i].end(), 
			0, std::plus<long long>())) + "ms") << std::endl;
	}
	std::cout << std::endl;
}

int main() 
{
	const std::vector<unsigned int> arr_sizes{64, 128, 256, 512, 1024};

	auto simple_cpu_bench_results_x86 = x86::simple_cpu_benchmark(arr_sizes);
	print_bench("(x86)Simple CPU matrix multiplication:", arr_sizes, simple_cpu_bench_results_x86);

	auto simple_cpu_bench_results_x64 = x64::simple_cpu_benchmark(arr_sizes);
	print_bench("(x64)Simple CPU matrix multiplication:", arr_sizes, simple_cpu_bench_results_x64);

	auto m4ri_cpu_bench_results_x86 = x86::m4ri_cpu_benchmark(arr_sizes);
	print_bench("(x86)M4RI CPU matrix multiplication:", arr_sizes, m4ri_cpu_bench_results_x86);

	auto m4ri_cpu_bench_results_x64 = x64::m4ri_cpu_benchmark(arr_sizes);
	print_bench("(x64)M4RI CPU matrix multiplication:", arr_sizes, m4ri_cpu_bench_results_x64);

	auto m4ri_gpu_bench_results_x86 = x86::m4ri_gpu_benchmark(arr_sizes);
	print_bench("(x86)M4RI GPU matrix multiplication:", arr_sizes, m4ri_gpu_bench_results_x86);

	auto m4ri_gpu_bench_resylts_x64 = x64::m4ri_gpu_benchmark(arr_sizes);
	print_bench("(x64)M4RI GPU matrix multiplication:", arr_sizes, m4ri_gpu_bench_resylts_x64);

	auto mar_gpu_bench_results_x86 = x86::mar_gpu_benchmark(arr_sizes);
	print_bench("(x86)MAR GPU matrix multiplication:", arr_sizes, mar_gpu_bench_results_x86);

	auto mar_gpu_bench_results_x64 = x64::mar_gpu_benchmark(arr_sizes);
	print_bench("(x64)MAR GPU matrix multiplication:", arr_sizes, mar_gpu_bench_results_x64);

	return 0;
}