#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixAddition(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int index = row * N + col;
        C[index] = A[index] + B[index];
    }
}

void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i * N + j] = i + j;  // You can set any initialization logic here
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
    const int N = 16;  // Matrix columns

    // Allocate and initialize matrices A and B on the host
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    printMatrix(h_A, N);
    cout << "---------------------------------------------------------" << endl;
    printMatrix(h_B, N);
    cout << "---------------------------------------------------------" << endl;

    // Allocate matrix C to store the result on the host
    float* h_C = new float[N * N];

    // Allocate device (GPU) memory
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(N, N);
    dim3 gridSize(1, 1);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch the kernel on the GPU
    matrixAddition<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Kernel Execution Time: " << milliseconds << " ms" << endl;

    // Copy the result back from the GPU to the CPU
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // ... Handle the result stored in h_C ...
    printMatrix(h_C, N);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}