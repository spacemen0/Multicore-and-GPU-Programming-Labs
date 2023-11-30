#define SKEPU_PRECOMPILED
#define SKEPU_OPENMP
#define SKEPU_OPENCL
#define SKEPU_CUDA
/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu>

/* SkePU user functions */

/*
float userfunction(...)
{
	// your code here
}

// more user functions...

*/


struct skepu_userfunction_skepu_skel_0sepReduce_lambda_uf_fa37JncCHr
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<float, float>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float a, float b)
{ return a + b;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{ return a + b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{ return a + b;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "dotproduct_precompiled_ReduceKernel_lambda_uf_fa37JncCHr.cu"
#include "dotproduct_precompiled_ReduceKernel_lambda_uf_fa37JncCHr_cl_source.inl"

struct skepu_userfunction_skepu_skel_1sepMap_lambda_uf_yDsbzayy4c
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float a, float b)
{ return a * b;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{ return a * b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{ return a * b;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "dotproduct_precompiled_MapKernel_lambda_uf_yDsbzayy4c.cu"
#include "dotproduct_precompiled_MapKernel_lambda_uf_yDsbzayy4c_arity_2_cl_source.inl"

struct skepu_userfunction_skepu_skel_2comb_lambda_uf_BWDxS22Jjz
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float a, float b)
{ return a * b;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{ return a * b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{ return a * b;
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_2comb_lambda_uf_hMaiRrV41m
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float a, float b)
{ return a + b;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{ return a + b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{ return a + b;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "dotproduct_precompiled_MapReduceKernel_lambda_uf_BWDxS22Jjz_lambda_uf_hMaiRrV41m.cu"
#include "dotproduct_precompiled_MapReduceKernel_lambda_uf_BWDxS22Jjz_lambda_uf_hMaiRrV41m_arity_2_cl_source.inl"
int main(int argc, const char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}

	const size_t size = std::stoul(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	//	spec.setCPUThreads(<integer value>);
	skepu::setGlobalBackendSpec(spec);
	skepu::backend::MapReduce<2, skepu_userfunction_skepu_skel_2comb_lambda_uf_BWDxS22Jjz, skepu_userfunction_skepu_skel_2comb_lambda_uf_hMaiRrV41m, decltype(&dotproduct_precompiled_MapReduceKernel_lambda_uf_BWDxS22Jjz_lambda_uf_hMaiRrV41m), decltype(&dotproduct_precompiled_MapReduceKernel_lambda_uf_BWDxS22Jjz_lambda_uf_hMaiRrV41m_ReduceOnly), CLWrapperClass_dotproduct_precompiled_MapReduceKernel_lambda_uf_BWDxS22Jjz_lambda_uf_hMaiRrV41m_arity_2> comb(dotproduct_precompiled_MapReduceKernel_lambda_uf_BWDxS22Jjz_lambda_uf_hMaiRrV41m, dotproduct_precompiled_MapReduceKernel_lambda_uf_BWDxS22Jjz_lambda_uf_hMaiRrV41m_ReduceOnly);
	skepu::backend::Map<2, skepu_userfunction_skepu_skel_1sepMap_lambda_uf_yDsbzayy4c, decltype(&dotproduct_precompiled_MapKernel_lambda_uf_yDsbzayy4c), CLWrapperClass_dotproduct_precompiled_MapKernel_lambda_uf_yDsbzayy4c_arity_2> sepMap(dotproduct_precompiled_MapKernel_lambda_uf_yDsbzayy4c);
	skepu::backend::Reduce1D<skepu_userfunction_skepu_skel_0sepReduce_lambda_uf_fa37JncCHr, decltype(&dotproduct_precompiled_ReduceKernel_lambda_uf_fa37JncCHr), CLWrapperClass_dotproduct_precompiled_ReduceKernel_lambda_uf_fa37JncCHr> sepReduce(dotproduct_precompiled_ReduceKernel_lambda_uf_fa37JncCHr);

		/* Skeleton instances */
		//	auto instance = skepu::Map(userfunction);
		// ...

		/* SkePU containers */
		skepu::Vector<float>
			v1(size, 1.0f),
		 v2(size, 2.0f);
	skepu::Vector<float> resMap(size);

	/* Compute and measure time */
	float resComb, resSep;

	auto timeComb = skepu::benchmark::measureExecTime([&]
													  { resComb = comb(v1, v2); });

	auto timeSep = skepu::benchmark::measureExecTime([&]
													 {
														 sepMap(resMap,v1,v2);
														 resSep = sepReduce(resMap); });

	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << (timeSep.count() / 10E6) << " seconds.\n";

	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep << "\n";

	return 0;
}
