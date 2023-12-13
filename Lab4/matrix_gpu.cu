#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixAddition(float* A, float* B, float* C, int N) {
    int iy = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;
    int index = ix+iy*N;
    if (ix < N && iy < N)
    {
        C[index] = A[index] + B[index];
    }
}

void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i + N * j] = i + j;  
        }
    }
}

void printMatrix(float* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << matrix[i * N + j] << "  ";
        }
        cout << endl;
    }
}

int main() {
    const int N = 1024;  

    
    float* h_A = new float[N*N];
    float* h_B = new float[N*N];
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    // printMatrix(h_A, N);
    cout << "---------------------------------------------------------" << endl;
    // printMatrix(h_B, N);
    cout << "---------------------------------------------------------" << endl;

    float *h_C = new float[N * N];

    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((N+blockSize.x-1)/blockSize.x, (N+blockSize.y-1)/blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixAddition<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        cout << "Error: " << cudaGetErrorString(err) << endl;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Kernel Execution Time: " << milliseconds << " ms" << endl;

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // printMatrix(h_C, N);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}