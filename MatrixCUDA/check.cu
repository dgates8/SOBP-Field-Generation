#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void test(double *a, double *b, double *x, double *y, double *energy, int N, int size){

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < N){
		a[i] += 1;

		b[i] += 0;

	}
	__syncthreads();

	for(int j = 0; j< N; j++){
		if(a[i] == x[j] && b[i] == y[j]){
			for(int k = 0; k+j <= 100; k+=10){
				energy[i+k] += energy[j];
			}
		}
	}	
}


int main(){
	int size = 10;
	int size2 = 100;
	int GridDim= 1;
	int BlockSize = 10;

	double *e;
	e = (double*)malloc(size2 * sizeof(double));

	for(int k = 0; k < size2; k++){
		e[k] = k;
	}

	double *x, *y;
	x = (double*)malloc(size*sizeof(double));
	y = (double*)malloc(size*sizeof(double));


	for(int j= 0; j < 10; j++){
		x[j] = j;
		y[j] = 0;
	}


	for(int w = 0; w < 10; w++){
		std::cout << x[w] << "  "<< y[w] << std::endl;
	}

	//memcpy(x1,x,size*sizeof(int));
	//memcpy(y1,y,size*sizeof(int));

	double *a, *b, *c, *d, *nrg;
	cudaMalloc((void**)&a, size*sizeof(double));
	cudaMalloc((void**)&b, size*sizeof(double));
	cudaMalloc((void**)&nrg, size2*sizeof(double));
	cudaMalloc((void**)&c, size*sizeof(double));
	cudaMalloc((void**)&d, size*sizeof(double));

	cudaError_t err = cudaMemcpy(a, x, size*sizeof(double), cudaMemcpyHostToDevice);
	printf("CUDA malloc a: %s\n",cudaGetErrorString(err));

	err= cudaMemcpy(b, y, size*sizeof(double), cudaMemcpyHostToDevice);
	printf("CUDA malloc y: %s\n",cudaGetErrorString(err));
	
	err= cudaMemcpy(c, x, size*sizeof(double), cudaMemcpyHostToDevice);
	printf("CUDA malloc y: %s\n",cudaGetErrorString(err));
	
	err= cudaMemcpy(d, y, size*sizeof(double), cudaMemcpyHostToDevice);
	printf("CUDA malloc y: %s\n",cudaGetErrorString(err));

	err = cudaMemcpy(nrg, e, size2*sizeof(double), cudaMemcpyHostToDevice);
	printf("CUDA malloc energy: %s\n",cudaGetErrorString(err));


	test<<<GridDim, BlockSize>>>(a,b,c,d,nrg,size,size2);

	double *final;
	final = (double*)malloc(size2*sizeof(double));
	err = cudaMemcpy(final,nrg, size2*sizeof(double),cudaMemcpyDeviceToHost);
	printf("CUDA malloc final: %s\n",cudaGetErrorString(err));

	std::cout << "done" << std::endl;
	for(int i = 0; i< 100; i++){
		std::cout << final[i] << "  " << e[i] << std::endl;
	}
}
