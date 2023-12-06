
class CLWrapperClass_median_precompiled_Overlap2DKernel_median_kernel
{
public:

	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
	}

	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;

		std::string source = skepu::backend::cl_helpers::replaceSizeT(R"###(
typedef struct {
	__local unsigned char *data;
	int oi, oj;
	size_t stride;
} skepu_region2d_unsigned__space__char;

static unsigned char skepu_region_access_2d_unsigned__space__char(skepu_region2d_unsigned__space__char r, int i, int j)
{ return r.data[i * r.stride + j]; }

#define SKEPU_USING_BACKEND_CL 1

typedef struct{
	size_t i;
} index1_t;

typedef struct {
	size_t row;
	size_t col;
} index2_t;

typedef struct {
	size_t i;
	size_t j;
	size_t k;
} index3_t;

typedef struct {
	size_t i;
	size_t j;
	size_t k;
	size_t l;
} index4_t;

static size_t get_device_id()
{
	return SKEPU_INTERNAL_DEVICE_ID;
}

#define VARIANT_OPENCL(block) block
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block)

// Size of basic integer types defined in OpenCL standard.
// Emulate stdint.h based on this.
typedef uchar    uint8_t;
typedef ushort   uint16_t;
typedef uint     uint32_t;
typedef ulong    uint64_t;

typedef char     int8_t;
typedef short    int16_t;
typedef int      int32_t;
typedef long     int64_t;

enum {
	SKEPU_EDGE_NONE = 0,
	SKEPU_EDGE_CYCLIC = 1,
	SKEPU_EDGE_DUPLICATE = 2,
	SKEPU_EDGE_PAD = 3,
};

static unsigned char median_kernel(skepu_region2d_unsigned__space__char image, unsigned long elemPerPx)
{
	const size_t size = (2 * image.oi + 1) * elemPerPx;
	unsigned char values[size];

	// Copy pixel values from the region to the array
	for (int y = -image.oi; y <= image.oi; ++y)
	{
		for (int x = -image.oj; x <= image.oj; x += elemPerPx)
		{
			values[y * elemPerPx + x + image.oi] = skepu_region_access_2d_unsigned__space__char(image,y, x);
		}
	}

	// Simple Bubble Sort for sorting the array
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

	// Find the median value
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


__kernel void median_precompiled_Overlap2DKernel_median_kernel( __global unsigned char* skepu_output, __global unsigned char* image, unsigned long elemPerPx, 
	size_t out_rows, size_t out_cols, size_t skepu_overlap_y, size_t skepu_overlap_x,
	size_t in_pitch, size_t sharedRows, size_t sharedCols,
	__local unsigned char* sdata)
{
	size_t xx = ((size_t)(get_global_id(0) / get_local_size(0))) * get_local_size(0);
	size_t yy = ((size_t)(get_global_id(1) / get_local_size(1))) * get_local_size(1);
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	skepu_region2d_unsigned__space__char skepu_region = { .data = &sdata[(get_local_id(1) + skepu_overlap_y) * sharedCols + (get_local_id(0) + skepu_overlap_x)], .oi = skepu_overlap_y, .oj = skepu_overlap_x, .stride = sharedCols };

	

	if (x < out_cols + skepu_overlap_x * 2 && y < out_rows + skepu_overlap_y * 2)
	{
		size_t sharedIdx = get_local_id(1) * sharedCols + get_local_id(0);
		sdata[sharedIdx]= image[y * in_pitch + x];

		size_t shared_x = get_local_id(0)+get_local_size(0);
		size_t shared_y = get_local_id(1);
		while (shared_y < sharedRows)
		{
			while (shared_x < sharedCols)
			{
				sharedIdx = shared_y * sharedCols + shared_x;
				sdata[sharedIdx] = image[(yy + shared_y) * in_pitch + xx + shared_x];
				shared_x = shared_x + get_local_size(0);
			}
			shared_x = get_local_id(0);
			shared_y = shared_y + get_local_size(1);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	

	if (x < out_cols && y < out_rows)
	{
#if !0
		skepu_output[y * out_cols + x] = median_kernel(skepu_region, elemPerPx);
#else
		skepu_multiple skepu_out_temp = median_kernel(skepu_region, elemPerPx);
		
#endif
	}
}
)###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "median_precompiled_Overlap2DKernel_median_kernel", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 2D kernel 'median_precompiled_Overlap2DKernel_median_kernel'");

			kernels(counter++, &kernel);
		}

		initialized = true;
	}

	static void mapOverlap2D
	(
		size_t deviceID, size_t localSize[2], size_t globalSize[2],
		 skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_output, skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_input, unsigned long elemPerPx, 
		size_t out_rows, size_t out_cols, size_t skepu_overlap_y, size_t skepu_overlap_x,
		size_t in_pitch, size_t sharedRows, size_t sharedCols,
		size_t sharedMemSize
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(kernels(deviceID),  skepu_output->getDeviceDataPointer(), skepu_input->getDeviceDataPointer(), elemPerPx, 
			out_rows, out_cols, skepu_overlap_y, skepu_overlap_x, in_pitch, sharedRows, sharedCols);
		clSetKernelArg(kernels(deviceID), 3 + 7, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernels(deviceID), 2, NULL, globalSize, localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 2D kernel");
	}
};
