
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int *A, const int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(int);

    std::vector<int> h_A(N), h_B(N), h_C(N);

    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() % 101;  // integers from 0 to 100
        h_B[i] = rand() % 101;
    }

    int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCudaError(cudaMalloc(&d_A, size), "Allocating d_A");
    checkCudaError(cudaMalloc(&d_B, size), "Allocating d_B");
    checkCudaError(cudaMalloc(&d_C, size), "Allocating d_C");

    checkCudaError(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice), "Copying h_A");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice), "Copying h_B");

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    checkCudaError(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost), "Copying result");

    // Save to file in column format
    std::ofstream outFile("vector_sum_output.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }

    outFile << std::setw(10) << "A[i]"
            << std::setw(10) << "B[i]"
            << std::setw(15) << "C[i] = A + B" << "\n";
    outFile << std::string(35, '-') << "\n";

    for (int i = 0; i < N; ++i) {
        outFile << std::setw(10) << h_A[i]
                << std::setw(10) << h_B[i]
                << std::setw(15) << h_C[i] << "\n";
    }

    outFile.close();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

