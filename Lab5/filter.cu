// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4
// 2022-12-07: A correction for a deprecated function.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#else
#include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"
#include <iostream>

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10

__global__ void filter_naive(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
	// map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int dy, dx;
	unsigned int sumx, sumy, sumz;

	int divby = (2 * kernelsizex + 1) * (2 * kernelsizey + 1); // Works for box filters only!

	if (x < imagesizex && y < imagesizey) // If inside image
	{
		// Filter kernel (simple box filter)
		sumx = 0;
		sumy = 0;
		sumz = 0;
		for (dy = -kernelsizey; dy <= kernelsizey; dy++)
			for (dx = -kernelsizex; dx <= kernelsizex; dx++)
			{
				// Use max and min to avoid branching!
				int yy = min(max(y + dy, 0), imagesizey - 1);
				int xx = min(max(x + dx, 0), imagesizex - 1);

				sumx += image[((yy)*imagesizex + (xx)) * 3 + 0];
				sumy += image[((yy)*imagesizex + (xx)) * 3 + 1];
				sumz += image[((yy)*imagesizex + (xx)) * 3 + 2];
			}
		out[(y * imagesizex + x) * 3 + 0] = sumx / divby;
		out[(y * imagesizex + x) * 3 + 1] = sumy / divby;
		out[(y * imagesizex + x) * 3 + 2] = sumz / divby;
	}
}

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
	// Calculate the dimensions of the shared memory array
	// Including padding for the kernel radius
	__shared__ unsigned char sharedMem[(16 + 2 * maxKernelSizeY) * (16 + 2 * maxKernelSizeX) * 3];

	// Calculate the global coordinates of the thread
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate the local coordinates within the block
	int localX = threadIdx.x + kernelsizex;
	int localY = threadIdx.y + kernelsizey;

	// Load data into shared memory, handling the borders
	for (int dy = -kernelsizey; dy <= kernelsizey; dy++)
	{
		for (int dx = -kernelsizex; dx <= kernelsizex; dx++)
		{
			int globalX = min(max(x + dx, 0), imagesizex - 1);
			int globalY = min(max(y + dy, 0), imagesizey - 1);

			// Calculate the index in the global memory
			int globalIndex = (globalY * imagesizex + globalX) * 3;

			// Calculate the index in the shared memory
			int localIndex = ((localY + dy) * (blockDim.x + 2 * kernelsizex) + (localX + dx)) * 3;

			// Load data from global memory to shared memory
			sharedMem[localIndex + 0] = image[globalIndex + 0];
			sharedMem[localIndex + 1] = image[globalIndex + 1];
			sharedMem[localIndex + 2] = image[globalIndex + 2];
		}
	}

	// Synchronize threads to ensure shared memory is fully loaded
	__syncthreads();

	// Perform the convolution using shared memory
	if (x < imagesizex && y < imagesizey)
	{
		unsigned int sumx = 0, sumy = 0, sumz = 0;
		int divby = (2 * kernelsizex + 1) * (2 * kernelsizey + 1);

		for (int dy = -kernelsizey; dy <= kernelsizey; dy++)
		{
			for (int dx = -kernelsizex; dx <= kernelsizex; dx++)
			{
				int localIndex = ((localY + dy) * (blockDim.x + 2 * kernelsizex) + (localX + dx)) * 3;

				sumx += sharedMem[localIndex + 0];
				sumy += sharedMem[localIndex + 1];
				sumz += sharedMem[localIndex + 2];
			}
		}

		// Calculate the output pixel value and write it to global memory
		int outputIndex = (y * imagesizex + x) * 3;
		out[outputIndex + 0] = sumx / divby;
		out[outputIndex + 1] = sumy / divby;
		out[outputIndex + 2] = sumz / divby;
	}
}

__global__ void filter_horizontal(unsigned char *input, unsigned char *output, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsize)
{
	__shared__ unsigned char sharedMem[16 * (16 + 2 * maxKernelSizeX) * 3];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int localX = threadIdx.x + kernelsize;
	int localY = threadIdx.y;

	// Load data into shared memory
	for (int dx = -kernelsize; dx <= kernelsize; dx++)
	{
		int globalX = min(max(x + dx, 0), imagesizex - 1);
		int globalY = y;

		int globalIndex = (globalY * imagesizex + globalX) * 3;
		int localIndex = (localY * (blockDim.x + 2 * kernelsize) + (localX + dx)) * 3;

		sharedMem[localIndex + 0] = input[globalIndex + 0];
		sharedMem[localIndex + 1] = input[globalIndex + 1];
		sharedMem[localIndex + 2] = input[globalIndex + 2];
	}

	__syncthreads();

	if (x < imagesizex && y < imagesizey)
	{
		unsigned int sumx = 0, sumy = 0, sumz = 0;
		int divby = (2 * kernelsize + 1);

		for (int dx = -kernelsize; dx <= kernelsize; dx++)
		{
			int localIndex = (localY * (blockDim.x + 2 * kernelsize) + (localX + dx)) * 3;

			sumx += sharedMem[localIndex + 0];
			sumy += sharedMem[localIndex + 1];
			sumz += sharedMem[localIndex + 2];
		}

		int outputIndex = (y * imagesizex + x) * 3;
		output[outputIndex + 0] = sumx / divby;
		output[outputIndex + 1] = sumy / divby;
		output[outputIndex + 2] = sumz / divby;
	}
}

__global__ void filter_vertical(unsigned char *input, unsigned char *output, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsize)
{
	__shared__ unsigned char sharedMem[(16 + 2 * maxKernelSizeY) * (16) * 3];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int localX = threadIdx.x;
	int localY = threadIdx.y + kernelsize;

	// Load data into shared memory
	for (int dy = -kernelsize; dy <= kernelsize; dy++)
	{
		int globalX = x;
		int globalY = min(max(y + dy, 0), imagesizey - 1);

		int globalIndex = (globalY * imagesizex + globalX) * 3;
		int localIndex = ((localY + dy) * blockDim.x + localX) * 3;

		sharedMem[localIndex + 0] = input[globalIndex + 0];
		sharedMem[localIndex + 1] = input[globalIndex + 1];
		sharedMem[localIndex + 2] = input[globalIndex + 2];
	}

	__syncthreads();

	if (x < imagesizex && y < imagesizey)
	{
		unsigned int sumx = 0, sumy = 0, sumz = 0;
		int divby = (2 * kernelsize + 1);

		for (int dy = -kernelsize; dy <= kernelsize; dy++)
		{
			int localIndex = ((localY + dy) * blockDim.x + localX) * 3;

			sumx += sharedMem[localIndex + 0];
			sumy += sharedMem[localIndex + 1];
			sumz += sharedMem[localIndex + 2];
		}

		int outputIndex = (y * imagesizex + x) * 3;
		output[outputIndex + 0] = sumx / divby;
		output[outputIndex + 1] = sumy / divby;
		output[outputIndex + 2] = sumz / divby;
	}
}

__global__ void filter_horizontal_gaussian(unsigned char *input, unsigned char *output, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsize, const float *weights)
{
	__shared__ unsigned char sharedMem[16 * (16 + 2 * maxKernelSizeX) * 3];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int localX = threadIdx.x + kernelsize;
	int localY = threadIdx.y;

	// Load data into shared memory
	for (int dx = -kernelsize; dx <= kernelsize; dx++)
	{
		int globalX = min(max(x + dx, 0), imagesizex - 1);
		int globalY = y;

		int globalIndex = (globalY * imagesizex + globalX) * 3;
		int localIndex = (localY * (blockDim.x + 2 * kernelsize) + (localX + dx)) * 3;

		sharedMem[localIndex + 0] = input[globalIndex + 0];
		sharedMem[localIndex + 1] = input[globalIndex + 1];
		sharedMem[localIndex + 2] = input[globalIndex + 2];
	}

	__syncthreads();

	if (x < imagesizex && y < imagesizey)
	{
		float sumx = 0.0f, sumy = 0.0f, sumz = 0.0f;

		for (int dx = -kernelsize; dx <= kernelsize; dx++)
		{
			int localIndex = (localY * (blockDim.x + 2 * kernelsize) + (localX + dx)) * 3;
			float weight = weights[dx + kernelsize];

			sumx += weight * sharedMem[localIndex + 0];
			sumy += weight * sharedMem[localIndex + 1];
			sumz += weight * sharedMem[localIndex + 2];
		}

		int outputIndex = (y * imagesizex + x) * 3;
		output[outputIndex + 0] = sumx;
		output[outputIndex + 1] = sumy;
		output[outputIndex + 2] = sumz;
	}
}

__global__ void filter_vertical_gaussian(unsigned char *input, unsigned char *output, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsize, const float *weights)
{
	__shared__ unsigned char sharedMem[(16 + 2 * maxKernelSizeY) * (16) * 3];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int localX = threadIdx.x;
	int localY = threadIdx.y + kernelsize;

	// Load data into shared memory
	for (int dy = -kernelsize; dy <= kernelsize; dy++)
	{
		int globalX = x;
		int globalY = min(max(y + dy, 0), imagesizey - 1);

		int globalIndex = (globalY * imagesizex + globalX) * 3;
		int localIndex = ((localY + dy) * blockDim.x + localX) * 3;

		sharedMem[localIndex + 0] = input[globalIndex + 0];
		sharedMem[localIndex + 1] = input[globalIndex + 1];
		sharedMem[localIndex + 2] = input[globalIndex + 2];
	}

	__syncthreads();

	if (x < imagesizex && y < imagesizey)
	{
		float sumx = 0.0f, sumy = 0.0f, sumz = 0.0f;

		for (int dy = -kernelsize; dy <= kernelsize; dy++)
		{
			int localIndex = ((localY + dy) * blockDim.x + localX) * 3;
			float weight = weights[dy + kernelsize];

			sumx += weight * sharedMem[localIndex + 0];
			sumy += weight * sharedMem[localIndex + 1];
			sumz += weight * sharedMem[localIndex + 2];
		}

		int outputIndex = (y * imagesizex + x) * 3;
		output[outputIndex + 0] = sumx;
		output[outputIndex + 1] = sumy;
		output[outputIndex + 2] = sumz;
	}
}

__global__ void median_filter(unsigned char *input, unsigned char *output, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Shared memory to store the neighborhood pixels
	__shared__ unsigned char sharedMem[(16 + 2 * maxKernelSizeY) * (16 + 2 * maxKernelSizeX) * 3];

	int localX = threadIdx.x + kernelsizex;
	int localY = threadIdx.y + kernelsizey;

	// Load data into shared memory with border clamping
	for (int dy = -kernelsizey; dy <= kernelsizey; dy++)
	{
		for (int dx = -kernelsizex; dx <= kernelsizex; dx++)
		{
			int globalX = min(max(x + dx, 0), imagesizex - 1);
			int globalY = min(max(y + dy, 0), imagesizey - 1);

			int globalIndex = (globalY * imagesizex + globalX) * 3;
			int localIndex = ((localY + dy) * (blockDim.x + 2 * kernelsizex) + (localX + dx)) * 3;

			sharedMem[localIndex + 0] = input[globalIndex + 0];
			sharedMem[localIndex + 1] = input[globalIndex + 1];
			sharedMem[localIndex + 2] = input[globalIndex + 2];
		}
	}

	__syncthreads();

	// Calculate the median for each channel
	if (x < imagesizex && y < imagesizey)
	{
		for (int channel = 0; channel < 3; channel++)
		{
			unsigned char window[(2 * maxKernelSizeX + 1) * (2 * maxKernelSizeY + 1)];
			int idx = 0;
			for (int dy = -kernelsizey; dy <= kernelsizey; dy++)
			{
				for (int dx = -kernelsizex; dx <= kernelsizex; dx++)
				{
					int localIndex = ((localY + dy) * (blockDim.x + 2 * kernelsizex) + (localX + dx)) * 3;
					window[idx++] = sharedMem[localIndex + channel];
				}
			}
			// Sort and find median
			for (int i = 0; i < idx - 1; i++)
			{
				for (int j = i + 1; j < idx; j++)
				{
					if (window[i] > window[j])
					{
						unsigned char temp = window[i];
						window[i] = window[j];
						window[j] = temp;
					}
				}
			}
			int medianIndex = idx / 2;
			int outputIndex = (y * imagesizex + x) * 3;
			output[outputIndex + channel] = window[medianIndex];
		}
	}
}

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

void computeImagesSeparable(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *)malloc(imagesizex * imagesizey * 3);
	unsigned char *temp_output;
	cudaMalloc((void **)&dev_input, imagesizex * imagesizey * 3);
	cudaMemcpy(dev_input, image, imagesizey * imagesizex * 3, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_bitmap, imagesizex * imagesizey * 3);
	cudaMalloc((void **)&temp_output, imagesizex * imagesizey * 3);

	dim3 block_dim(16, 16);
	dim3 grid((imagesizex + block_dim.x - 1) / block_dim.x, (imagesizey + block_dim.y - 1) / block_dim.y);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	// First pass: horizontal filter
	filter_horizontal<<<grid, block_dim>>>(dev_input, temp_output, imagesizex, imagesizey, kernelsizex);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	// Second pass: vertical filter
	filter_vertical<<<grid, block_dim>>>(temp_output, dev_bitmap, imagesizex, imagesizey, kernelsizey);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaEventRecord(stop);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
	cudaMemcpy(pixels, dev_bitmap, imagesizey * imagesizex * 3, cudaMemcpyDeviceToHost);

	cudaFree(dev_bitmap);
	cudaFree(temp_output);
	cudaFree(dev_input);
}

void computeImagesGaussian(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *)malloc(imagesizex * imagesizey * 3);
	unsigned char *temp_output;
	cudaMalloc((void **)&dev_input, imagesizex * imagesizey * 3);
	cudaMemcpy(dev_input, image, imagesizey * imagesizex * 3, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_bitmap, imagesizex * imagesizey * 3);
	cudaMalloc((void **)&temp_output, imagesizex * imagesizey * 3);

	// Gaussian weights
	float h_weights[5] = {1 / 16.0f, 4 / 16.0f, 6 / 16.0f, 4 / 16.0f, 1 / 16.0f};
	float *d_weights;
	cudaMalloc((void **)&d_weights, 5 * sizeof(float));
	cudaMemcpy(d_weights, h_weights, 5 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 block_dim(16, 16);
	dim3 grid((imagesizex + block_dim.x - 1) / block_dim.x, (imagesizey + block_dim.y - 1) / block_dim.y);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	// First pass: horizontal filter
	filter_horizontal_gaussian<<<grid, block_dim>>>(dev_input, temp_output, imagesizex, imagesizey, kernelsizex, d_weights);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	// Second pass: vertical filter
	filter_vertical_gaussian<<<grid, block_dim>>>(temp_output, dev_bitmap, imagesizex, imagesizey, kernelsizey, d_weights);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaEventRecord(stop);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
	cudaMemcpy(pixels, dev_bitmap, imagesizey * imagesizex * 3, cudaMemcpyDeviceToHost);

	cudaFree(dev_bitmap);
	cudaFree(temp_output);
	cudaFree(dev_input);
	cudaFree(d_weights);
}

void computeImagesMedian(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *)malloc(imagesizex * imagesizey * 3);
	unsigned char *dev_input = nullptr;
	unsigned char *dev_bitmap = nullptr;

	// Allocate device memory and check for errors
	cudaError_t err;
	err = cudaMalloc((void **)&dev_input, imagesizex * imagesizey * 3);
	if (err != cudaSuccess)
	{
		printf("cudaMalloc failed for dev_input: %s\n", cudaGetErrorString(err));
		return;
	}
	err = cudaMalloc((void **)&dev_bitmap, imagesizex * imagesizey * 3);
	if (err != cudaSuccess)
	{
		printf("cudaMalloc failed for dev_bitmap: %s\n", cudaGetErrorString(err));
		cudaFree(dev_input);
		return;
	}

	cudaMemcpy(dev_input, image, imagesizey * imagesizex * 3, cudaMemcpyHostToDevice);

	dim3 block_dim(16, 16);
	dim3 grid((imagesizex + block_dim.x - 1) / block_dim.x, (imagesizey + block_dim.y - 1) / block_dim.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	median_filter<<<grid, block_dim>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);
	cudaDeviceSynchronize();

	err = cudaGetLastError(); // Check for errors after kernel launch
	if (err != cudaSuccess)
	{
		printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
	cudaMemcpy(pixels, dev_bitmap, imagesizey * imagesizex * 3, cudaMemcpyDeviceToHost);

	// Use cudaFree instead of free
	cudaFree(dev_input);
	cudaFree(dev_bitmap);
}

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *)malloc(imagesizex * imagesizey * 3);
	cudaMalloc((void **)&dev_input, imagesizex * imagesizey * 3);
	cudaMemcpy(dev_input, image, imagesizey * imagesizex * 3, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_bitmap, imagesizex * imagesizey * 3);
	dim3 blockDim(16, 16);
	dim3 grid((imagesizex + blockDim.x - 1) / blockDim.x, (imagesizey + blockDim.y - 1) / blockDim.y);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	filter<<<grid, blockDim>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);

	cudaDeviceSynchronize();
	//	Check for errors!
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaEventRecord(stop);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
	cudaMemcpy(pixels, dev_bitmap, imagesizey * imagesizex * 3, cudaMemcpyDeviceToHost);
	cudaFree(dev_bitmap);
	cudaFree(dev_input);
}

// Display images
void Draw()
{
	// Dump the whole picture onto the screen.
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image);
		glRasterPos2i(0, -1);
		glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels);
		glRasterPos2i(-1, 0);
		glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image);
	}
	glFlush();
}

void computeImagesNaive(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *)malloc(imagesizex * imagesizey * 3);
	cudaMalloc((void **)&dev_input, imagesizex * imagesizey * 3);
	cudaMemcpy(dev_input, image, imagesizey * imagesizex * 3, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_bitmap, imagesizex * imagesizey * 3);
	dim3 grid(imagesizex, imagesizey);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	filter<<<grid, 1>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
	cudaDeviceSynchronize();
	//	Check for errors!
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaEventRecord(stop);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
	cudaMemcpy(pixels, dev_bitmap, imagesizey * imagesizex * 3, cudaMemcpyDeviceToHost);
	cudaFree(dev_bitmap);
	cudaFree(dev_input);
}

// Main program, inits
int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);

    // Load the image
    image = readppm((char *)"maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);

    // Set the window size based on the image dimensions
    if (imagesizey >= imagesizex)
        glutInitWindowSize(imagesizex * 2, imagesizey);
    else
        glutInitWindowSize(imagesizex, imagesizey * 2);

    glutCreateWindow("Lab 5");
    glutDisplayFunc(Draw);

    ResetMilli();

    if (argc > 1) {
        int choice = atoi(argv[1]); 

        switch (choice) {
            case 1:
                computeImagesNaive(2, 2);
                break;
            case 2:
                computeImages(2, 2);
                break;
            case 3:
                computeImagesSeparable(3, 3);
                break;
            case 4:
                computeImagesGaussian(2, 2);
                break;
            case 5:
                computeImagesMedian(2, 2);
                break;
            default:
                printf("Invalid choice! Please provide a number between 1 and 5.\n");
                return 1; // Exit the program with an error code
        }
    } else {
        printf("Please provide a command-line argument (1-5) to choose an image processing method.\n");
        return 1; // Exit the program with an error code
    }

    // Main loop
    glutMainLoop();
    return 0;
}

