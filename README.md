# HPC
HPC-Vector-Addition-CUDA
High Performance Computing Vector Addition using CUDA (CPU vs GPU comparison)

HPC Vector Addition using CUDA
This project demonstrates vector addition using both CPU and GPU using CUDA.

The purpose is to compare execution time between CPU and GPU computations.

Vector Addition Formula:

C[i] = A[i] + B[i]

Where: A = Input vector 1 B = Input vector 2 C = Result vector

CUDA GPU performs operations using parallel threads which significantly improves performance for large datasets.

Technologies Used:

CUDA
C++
NVIDIA GPU
Compilation:

nvcc vector_addition.cu -o vectoradd

Execution:

./vectoradd

Output: Execution time comparison between CPU and GPU.
# HPC-Matrix-Multiplication-CUDA1
# Matrix Multiplication using CUDA

This project demonstrates matrix multiplication using GPU parallel computing with CUDA.

Matrix multiplication formula:

C[i][j] = Σ A[i][k] * B[k][j]

CUDA uses thousands of threads to compute matrix multiplication efficiently.

Advantages:
- Faster execution for large matrices
- Parallel processing

Compilation:

nvcc matrix_multiplication.cu -o matrix

Execution:

./matrix


