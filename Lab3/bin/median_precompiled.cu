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

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu>

#include "support.h"

unsigned char median_kernel(skepu::Region2D<unsigned char> image, size_t elemPerPx)
{
	const size_t size = (2 * image.oi + 1) + (2* image.oj/elemPerPx + 1);
	unsigned char values [size];

	for (int y = -image.oi; y <= image.oi; ++y)
	{
		for (int x = -image.oj; x <= image.oj; x += elemPerPx)
		{
			values[(y+image.oi) * image.oj/elemPerPx + (x + image.oj)/elemPerPx] = image(y, x);
		}
	}

	for (size_t i = 0; i < size - 1; ++i)
	{
		for (size_t j = 0; j < size - i - 1; ++j)
		{
			if (values[j] > values[j + 1])
			{
				auto temp = values[j];
				values[j] = values[j + 1];
				values[j + 1] = temp;
			}
		}
	}

	unsigned char median;
	if (size % 2 == 0)
	{
		median = (values[size / 2 - 1] + values[size / 2]) / 2;
	}
	else
	{
		median = values[size / 2];
	}

	return median;
}


struct skepu_userfunction_skepu_skel_0calculateMedian_median_kernel
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<unsigned long>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ unsigned char CU(skepu::Region2D<unsigned char> image, unsigned long elemPerPx)
{
	const size_t size = (2 * image.oi + 1) + (2* image.oj/elemPerPx + 1);
	unsigned char values [size];

	for (int y = -image.oi; y <= image.oi; ++y)
	{
		for (int x = -image.oj; x <= image.oj; x += elemPerPx)
		{
			values[(y+image.oi) * image.oj/elemPerPx + (x + image.oj)/elemPerPx] = image(y, x);
		}
	}

	for (size_t i = 0; i < size - 1; ++i)
	{
		for (size_t j = 0; j < size - i - 1; ++j)
		{
			if (values[j] > values[j + 1])
			{
				auto temp = values[j];
				values[j] = values[j + 1];
				values[j + 1] = temp;
			}
		}
	}

	unsigned char median;
	if (size % 2 == 0)
	{
		median = (values[size / 2 - 1] + values[size / 2]) / 2;
	}
	else
	{
		median = values[size / 2];
	}

	return median;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region2D<unsigned char> image, unsigned long elemPerPx)
{
	const size_t size = (2 * image.oi + 1) + (2* image.oj/elemPerPx + 1);
	unsigned char values [size];

	for (int y = -image.oi; y <= image.oi; ++y)
	{
		for (int x = -image.oj; x <= image.oj; x += elemPerPx)
		{
			values[(y+image.oi) * image.oj/elemPerPx + (x + image.oj)/elemPerPx] = image(y, x);
		}
	}

	for (size_t i = 0; i < size - 1; ++i)
	{
		for (size_t j = 0; j < size - i - 1; ++j)
		{
			if (values[j] > values[j + 1])
			{
				auto temp = values[j];
				values[j] = values[j + 1];
				values[j + 1] = temp;
			}
		}
	}

	unsigned char median;
	if (size % 2 == 0)
	{
		median = (values[size / 2 - 1] + values[size / 2]) / 2;
	}
	else
	{
		median = values[size / 2];
	}

	return median;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region2D<unsigned char> image, unsigned long elemPerPx)
{
	const size_t size = (2 * image.oi + 1) + (2* image.oj/elemPerPx + 1);
	unsigned char values [size];

	for (int y = -image.oi; y <= image.oi; ++y)
	{
		for (int x = -image.oj; x <= image.oj; x += elemPerPx)
		{
			values[(y+image.oi) * image.oj/elemPerPx + (x + image.oj)/elemPerPx] = image(y, x);
		}
	}

	for (size_t i = 0; i < size - 1; ++i)
	{
		for (size_t j = 0; j < size - i - 1; ++j)
		{
			if (values[j] > values[j + 1])
			{
				auto temp = values[j];
				values[j] = values[j + 1];
				values[j + 1] = temp;
			}
		}
	}

	unsigned char median;
	if (size % 2 == 0)
	{
		median = (values[size / 2 - 1] + values[size / 2]) / 2;
	}
	else
	{
		median = values[size / 2];
	}

	return median;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "median_precompiled_Overlap2DKernel_median_kernel.cu"
#include "median_precompiled_Overlap2DKernel_median_kernel_cl_source.inl"
int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;
	
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}
	
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";
		
	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	
	// Skeleton instance
	skepu::backend::MapOverlap2D<skepu_userfunction_skepu_skel_0calculateMedian_median_kernel, decltype(&median_precompiled_Overlap2DKernel_median_kernel_conv_cuda_2D_kernel), CLWrapperClass_median_precompiled_Overlap2DKernel_median_kernel> calculateMedian(median_precompiled_Overlap2DKernel_median_kernel_conv_cuda_2D_kernel);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
	auto timeTaken = skepu::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	
	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";
	
	return 0;
}


