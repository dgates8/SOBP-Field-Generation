#include "matrix.h"
#include <iostream>

/*************************************************************************
Handle-Error code for timing runs 
*************************************************************************/
static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//check to be sure matrix operations are in range
	if(row > A.height || col > B.width) return;

	//loop over elements for dot product
	for(int i = 0; i < A.width; i++){
		Cvalue += (A.elements[row * A.width + i]) * (B.elements[i * B.width + col]);
		C.elements[row * C.width + col] = Cvalue;
	}
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE = 16
void MatMul(const Matrix A, const Matrix B, Matrix C) {

// Load A and B to device memory
Matrix d_A;
d_A.width = A.width;
d_A.height = A.height;
size_t size = A.width * A.height * sizeof(float);
cudaError_t err = cudaMalloc(&d_A.elements, size);
printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
printf("Copy A to device: %s\n",cudaGetErrorString(err));

Matrix d_B;
d_B.width = B.width;
d_B.height = B.height;
size = B.width * B.height * sizeof(float);
err = cudaMalloc(&d_B.elements, size);
printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
printf("Copy B to device: %s\n",cudaGetErrorString(err));

Matrix d_C;
d_C.width = C.width;
d_C.height = C.height;
size = C.width * C.height * sizeof(float);
err = cudaMalloc(&d_C.elements, size);
printf("CUDA malloc A: %s\n",cudaGetErrorString(err));

dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);

	float time;
	cudaEvent_t start, stop;

	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );

//launch dimGrid blocks with dimBlock threads each 
MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
err = cudaThreadSynchronize();
printf("Run kernel: %s\n", cudaGetErrorString(err));

err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
printf("Copy C off of device: %s\n", cudaGetErrorString(err));

	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
	printf("Time to generate:  %3.1f ms \n", time);

cudaFree(d_A.elements);
cudaFree(d_B.elements);
cudaFree(d_C.elements);
}
/**********************************************************************************
Main body Execution of Matrix operations. For demonstration purposes, the matrices
are filled with random entries and then the runtime is analyzed.
Author: Drake Gates
**********************************************************************************/
int main(int argc, char* argv[]){
	Matrix A,B,C;
	int a, b, c, d;

	a = atoi(argv[1]); //height
	b = atoi(argv[2]); //width
	c = b;			   //height
	d = atoi(argv[3]); //width

	A.height = a;
	A.width = b;
	A.elements = (float*)malloc(A.width*A.height*sizeof(float));



	B.height = c;
	B.width = d;
	B.elements = (float*)malloc(B.width*B.height*sizeof(float));

	C.height = A.height;
	C.width = B.width;
	C.elements = (float*)malloc(C.width *C.height *sizeof(float));

	//fill matrices
	srand (time(NULL));
	for(int i = 0; i < A.height; i++){
		for(int j = 0; j < A.width; j++){
			A.elements[i*A.width + j] = (float)(rand() % 4);
		}
	}

	for(int i = 0; i < B.height; i++){
		for(int j = 0; j < B.width; j++){
			B.elements[i*B.width + j] = (float)(rand() % 2);
		}
	}

	MatMul(A,B,C);
/*
	for(int i = 0; i < C.height; i++){
		for(int j = 0; j < C.width; j++){
			printf("%f ", C.elements[i*C.width + j]);
			printf("\n");
		}
		printf("\n");
	}
*/
	return 0;

}


