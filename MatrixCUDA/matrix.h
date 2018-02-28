
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
int width;
int height;
float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16
#define BLOCK_ROWS 8
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
