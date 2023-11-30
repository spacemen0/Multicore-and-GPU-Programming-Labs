
__global__ void dotproduct_precompiled_MapKernel_lambda_uf_yDsbzayy4c(float* skepu_output, float *a, float *b,  size_t w2, size_t w3, size_t w4, size_t n, size_t base)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;

	while (i < n)
	{
		
		auto res = skepu_userfunction_skepu_skel_1sepMap_lambda_uf_yDsbzayy4c::CU(a[i], b[i]);
		skepu_output[i] = res;
		i += gridSize;
	}
}
