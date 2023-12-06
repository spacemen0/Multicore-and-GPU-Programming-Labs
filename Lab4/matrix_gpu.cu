#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixAddition(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    printf("row: %d, col: %d\n", row, col);
    if (row < N && col < N)
    {
        int index = row * N + col;
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

    
    float* h_A = (float *)malloc(N * N * sizeof(float));
    float* h_B = (float *)malloc(N * N * sizeof(float));
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    // printMatrix(h_A, N);
    cout << "---------------------------------------------------------" << endl;
    // printMatrix(h_B, N);
    cout << "---------------------------------------------------------" << endl;

    float* h_C = (float *)malloc(N * N * sizeof(float));


    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(N, N);
    dim3 gridSize(1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixAddition<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

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

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}