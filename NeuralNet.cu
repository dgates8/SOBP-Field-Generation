#include <iostream>
#include <math.h>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>   
#include <vector>
#include <fstream>
#include <iomanip>

#define mag 10000
#define BLOCKSIZE 128

//tune to get higher pass rate, currently tuned to ~ 90% which is good enough for my application
#define eta .00031

//macro for error checking
#define cudaCheckError(){	   												  \
	cudaError_t err = cudaGetLastError();											  \
	if(err != cudaSuccess){              											  \
		std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << " : " <<  cudaGetErrorString(err) << std::endl; \
		exit(EXIT_FAILURE);                 										  \
	}                                     											  \
}

__global__ void sum(float *data, float* out, int size){
	__shared__ float temp[128];

	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	int gridSize = blockDim.x*2*gridDim.x;

	//read into shared memory per block
	if(tid < size){
		temp[tid] = 0;
		__syncthreads();
	}

	while(i < size){
		temp[tid] += data[tid] + data[tid+blockDim.x];
		i += gridSize; 
	}
	__syncthreads();

	if(blockDim.x >= 128){
		if(tid < 64){
			temp[tid] += temp[tid + 64];
		}
		__syncthreads();
	}
	if(tid < 32){
		if(blockDim.x >= 64){
			temp[tid] += temp[tid+32];
		}
		if(blockDim.x >= 32){
			temp[tid] += temp[tid+16];
		}
		if(blockDim.x >= 16){
			temp[tid] += temp[tid+8];
			}
		if(blockDim.x >= 8){
			temp[tid] += temp[tid+4];
			}
		if(blockDim.x >= 4){
			temp[tid] += temp[tid+2];
			}
		if(blockDim.x >= 2){
			temp[tid] += temp[tid+1];
		}
	}

	//read back each minimum per block to global array
	if(tid == 0){
		out[blockIdx.x] = temp[0];
	}
}
__global__ void dot(float *a, float *b, float *c, int size){
	int i = blockIdx.x*blockDim.x*8 + threadIdx.x;
	if(i < size){
		#pragma unroll
		for(int k = 0; k < 8; k++){
			c[i + k*BLOCKSIZE] = a[i+k*BLOCKSIZE] * b[i+k*BLOCKSIZE];
		}
	}
}

__inline__ __device__ __host__ float absolute(float a){
	return a < 0 ? -1*a : a;
}

__global__ void error(float *sigma, float *z, float * x, float *b, float* w, float* out , int size){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < size){
		out[i] = ((w[i]*x[i] + b[i]) - z[i]) * sigma[i];	
	}
}

__global__ void sigmoid(float* sigma, float *w, float* b, float *a, float* temp, int size){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int GRIDSIZE = (mag+BLOCKSIZE-1)/(BLOCKSIZE);
	float product = 0.0;
	if(i == 0){
		dot<<<GRIDSIZE/8,BLOCKSIZE>>>(w,a,sigma,size);
		sum<<<GRIDSIZE,BLOCKSIZE>>>(sigma,temp,size);
		for(int i = 0; i < gridDim.x; i++){
			product += temp[i];
		} 
	}
	__syncthreads();
	if(i < size){
		sigma[i] = 1/(1+expf(-1*(product+b[i])));
	}		
}



int main(){
	
	cudaSetDevice(5);
	int GRIDSIZE = (mag+BLOCKSIZE-1)/(BLOCKSIZE);
	
	std::vector<float> vecx;
	
	float *x, *z, *w, *b;
	x = (float*)malloc(mag*sizeof(float));
	z = (float*)malloc(mag*sizeof(float));
	w = (float*)malloc(mag*sizeof(float));
	b = (float*)malloc(mag*sizeof(float));

	for(float idx = 0; idx < 10; idx+=.001){
		vecx.push_back(idx);
	}

	std::copy(vecx.begin(), vecx.end(), x);
	
	srand(time(NULL));
	//initialize training data
	for(int i = 0; i < mag; i++){
		w[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/4);	
		b[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);	
		for(int j = 0; j < mag; j++){
			z[i] = 4*x[i] + exp(-x[i]) + sin(x[i])-cos(x[i])*cos(x[i]);
		}
	}
 
	bool FLAG = true;
	int loop = 0;

	while(FLAG){
		float *d_x, *d_w, *d_b, *d_z, *d_min, *d_temp, *d_err, *d_cost;
		float  *o_z;
		float *h_z, *err, *h_cost;

		h_z = (float*)malloc(mag*sizeof(float));
		err = (float*)malloc(mag*sizeof(float));
		h_cost = (float*)malloc(mag*sizeof(float));
		
		cudaMalloc((void**)&d_x, mag*sizeof(float));
		cudaCheckError();
		cudaMalloc((void**)&d_w, mag*sizeof(float));
		cudaCheckError();
		cudaMalloc((void**)&d_b, mag*sizeof(float));
		cudaCheckError();
		cudaMalloc((void**)&d_z, mag*sizeof(float));
		cudaCheckError();
		cudaMalloc((void**)&d_min, mag*sizeof(float));
		cudaCheckError();
		cudaMalloc((void**)&d_temp, mag*sizeof(float));
		cudaCheckError();
		cudaMalloc((void**)&o_z, mag*sizeof(float));
		cudaCheckError();
		cudaMalloc((void**)&d_err, mag*sizeof(float));
		cudaCheckError();
		cudaMalloc((void**)&d_cost, mag*sizeof(float));
		cudaCheckError();


		cudaMemcpy(d_x, x, mag*sizeof(float), cudaMemcpyHostToDevice);	
		cudaCheckError();
		cudaMemcpy(d_w, w, mag*sizeof(float), cudaMemcpyHostToDevice);	
		cudaCheckError();
		cudaMemcpy(d_b, b, mag*sizeof(float), cudaMemcpyHostToDevice);	
		cudaCheckError();
		cudaMemcpy(o_z, z, mag*sizeof(float), cudaMemcpyHostToDevice);	
		cudaCheckError();
		
		sigmoid<<<GRIDSIZE,BLOCKSIZE>>>(d_z, d_w, d_b, d_x, d_temp, mag);
		error<<<GRIDSIZE,BLOCKSIZE>>>(d_z, o_z, d_x, d_b, d_w, d_err, mag);	
		sum<<<GRIDSIZE,BLOCKSIZE>>>(d_err, d_cost, mag);
 		
		cudaMemcpy(h_z, d_z, mag*sizeof(float), cudaMemcpyDeviceToHost);	
		cudaCheckError();
		cudaMemcpy(err, d_err, mag*sizeof(float), cudaMemcpyDeviceToHost);
		cudaCheckError();
		cudaMemcpy(h_cost, d_cost, mag*sizeof(float), cudaMemcpyDeviceToHost);
		cudaCheckError();
			
		float cost = 0;
		int count = 0;
		for(int i = 0; i < GRIDSIZE; i++){
			cost += h_cost[i];
		}
		
		for(int i = 0; i < mag; i++){
			if(w[i] > 5 || b[i] > 5 || w[i] < -5 || b[i] < -5){
				w[i] = 0;
				b[i] = 0;
			}
			if(err[i] < 0){
				w[i] -= eta*x[i]*err[i];
				b[i] -= eta*err[i];
			}else{
				w[i] += eta*x[i]*err[i];
				b[i] += eta*err[i];
			}
			if(absolute((w[i]*x[i] + b[i])-z[i]) < .0001){
				count++;			
			}		
		}
		
		if(loop % 1000 == 0){
			std::cout << count << std::endl;
		}

		if(count > 9200){
			FLAG = false;
		}

		loop++;
		cudaFree(d_x);
		cudaFree(d_w);
		cudaFree(d_b);
		cudaFree(d_z);
		cudaFree(d_temp);
		cudaFree(o_z);
		cudaFree(d_err);
		cudaFree(d_cost);
		free(h_z);
		free(h_cost);
				
	}
	
	float *out;
	out = (float*)malloc(mag*sizeof(float));
		
	std::cout << "Found weights and baises that generate the function" << std::endl;
	std::ofstream data("data.txt", std::ios::out);
	for(int i = 0; i < mag; i++){	
		out[i] = w[i]*x[i] + b[i];
		data << std::fixed << std::setprecision(3) << x[i] << "     "  << out[i] << "     " << z[i] << std::endl;
	}
	
}

