#include <iostream>

__global__ void test(double *a, double *b, double *x, double *y, double *energy, int N, int size){
	//__shared__ double X[40000];
	//__shared__ double Y[40000];
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	//if(tid < N){
	//	X[tid] = x[tid];
	//	Y[tid] = y[tid];
	//}
	//if(tid < size){
	//	E[tid] = energy[tid];
	//}
	if(tid < N){
		a[tid] += 4;
		b[tid] += 3;
	}
	__syncthreads();

	if(tid < N){
		for(int i = 0; i < N; i++){
			if(a[tid] == x[i] && b[tid] == y[i]){
				for(int k = 0; k < size; k+=400){
					energy[tid] += energy[i+k];
				}
			}
		}
	}
}


int main(){
	int size = 40000;
	int size2 = 16000000;
	int GridDim= (size + 127)/(128);
	int BlockSize = 128;

	double *e;
	e = (double*)malloc(size2 * sizeof(double));

	for(int k = 0; k < size2; k++){
		e[k] = k;
	}

	double *x, *y;
	x = (double*)malloc(size*sizeof(double));
	y = (double*)malloc(size*sizeof(double));

	for(int i = 0; i < 200; i++){
		for(int j= 0; j < 200; j++){
			x[j] = j;
			y[j] = j;
		}
	}

	double *a, *b, *nrg;
	cudaMalloc((void**)&a, size*sizeof(double));
	cudaMalloc((void**)&b, size*sizeof(double));
	cudaMalloc((void**)&nrg, size2*sizeof(double));

	cudaMemcpy(a, x, size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(b, y, size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(nrg, e, size2*sizeof(double), cudaMemcpyHostToDevice);

	test<<<GridDim, BlockSize>>>(a,b,a,b,nrg,size,size2);

	double *final;
	final = (double*)malloc(size2*sizeof(double));
	cudaMemcpy(final,nrg, size2*sizeof(double),cudaMemcpyDeviceToHost);
	std::cout << "done" << std::endl;
	int count = 0;
	for(int y = 0; y< size2; y++){
			if(final[y] != e[y]){
				count++;
			}
	}
		std::cout << count << std::endl;
}