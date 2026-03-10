#include <stdio.h>
#include <cuda_runtime.h>

#define N 4   // matrix size (4x4)

// =======================
// GPU KERNEL FUNCTION
// =======================
__global__ void matrixAdd(int *A, int *B, int *C)
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    int index = row * N + col;

    C[index] = A[index] + B[index];
}

// =======================
// HOST FUNCTION (CPU)
// =======================
int main()
{
    int size = N * N * sizeof(int);

    int h_A[N][N], h_B[N][N], h_C[N][N];

    // initialize matrices
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
        {
            h_A[i][j] = i + j;
            h_B[i][j] = i * j;
        }

    int *d_A, *d_B, *d_C;

    // Allocate GPU memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy CPU → GPU (PCIe transfer)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 threads(N, N);
    matrixAdd<<<1, threads>>>(d_A, d_B, d_C);

    // Copy GPU → CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    printf("Result Matrix:\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++)
            printf("%d ", h_C[i][j]);
        printf("\n");
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}