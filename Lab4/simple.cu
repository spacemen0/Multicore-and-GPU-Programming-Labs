// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.
// Update 2022: Changed to cudaDeviceSynchronize.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <math.h>

const int N = 100;
const int blocksize = 100;

__global__ void simple(float *c)
{
	c[threadIdx.x] = threadIdx.x;
}
__global__ void ssqrt(float *i)
{
	i[threadIdx.x] = sqrt((float)threadIdx.x);
}

int main()
{

	const int size = N * sizeof(float);
	dim3 dimBlock(blocksize, 1);
	dim3 dimGrid(1, 1);


	float *s = new float[N];
	float *sd;
	cudaMalloc((void **)&sd, size);

	ssqrt<<<dimGrid, dimBlock>>>(sd);

	cudaMemcpy(s, sd, size, cudaMemcpyDeviceToHost);

	cudaFree(sd);


	for (int i = 0; i < N; i++)
		printf("%f ", s[i]);
	printf("\n");
	delete[] s;
	printf("done\n");
	return EXIT_SUCCESS;
}
