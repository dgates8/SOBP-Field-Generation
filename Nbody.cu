#include <fstream>
#include <iterator>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define blockSize 64
#define Ke 8.99e9
#define Qe -1.602e-19
#define epsilon 1e-8
#define mass 9.11e-31
 
//macro for error checking
#define cudaCheckError(){	   												  \
	cudaError_t err = cudaGetLastError();											  \
	if(err != cudaSuccess){              											  \
		std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << " : " <<  cudaGetErrorString(err) << std::endl; \
		exit(EXIT_FAILURE);                 										  \
	}                                     											  \
}

//calculate bodybody coulomb interactions
__device__ float3 bodyBodyCoulomb(float3 bi, float3 bj, float3 ai){
	float3 rij;
	
	//components of rij
	rij.x = bj.x - bi.x;
	rij.y = bj.y - bi.y;
	rij.z = bj.z - bi.z;
	
	//distance squared for solving force equation
	float distSquared = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z + epsilon;
	if(distSquared > 10){
		ai.x = -1;
		ai.y = -1;
		ai.z = -1;
		return ai;
	}
	//inverse cubed with softening factor
	float inverseDist = 1.0f*Qe/sqrtf(distSquared*distSquared*distSquared);

	//finish the equation by multiplying by charge
	float kernel = Ke*Qe*inverseDist;
	
	//get acceleration for each component
	ai.x += rij.x*kernel;
	ai.y += rij.y*kernel;
	ai.z += rij.z*kernel;
	return ai;
}

__device__ float3 tileFunction(float3 position, float3 acceleration, float3* shared){
	#pragma unroll
	for(int i = 0; i < blockDim.x; i++){
		acceleration = bodyBodyCoulomb(position, shared[i], acceleration);
	}
	return acceleration;
}

__global__ void find_forces(float3* X, float3* A, int numberOfBodies){
	float3 position;
	float3 acc = {0.0f, 0.0f, 0.0f};
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(tid < numberOfBodies){
		//read into shared memory and calculate tile
		position = X[tid];

		for(int i = 0, tile = 0; i < gridDim.x; i += blockSize, tile++){
			//declare shared memory bank
			__shared__ float3 sharedPosition[blockSize];

			int idx = tile*blockDim.x + threadIdx.x;
			sharedPosition[threadIdx.x] = X[idx];
			__syncthreads();
			acc = tileFunction(position, acc, sharedPosition);
			__syncthreads();
		}	

		//read back to global memory for integration step
		A[tid] = acc;	
	}
}

//main   
int main(const int argc, const char** argv){ 

	cudaSetDevice(10);
	
	//declare dt, numberofSteps from the command line	
	float dt = atof(argv[1]);
	int numberOfSteps = atoi(argv[2]);
	int numberOfBodies = atoi(argv[3]);
	
	//allocate random data array
	float3* x;
	x = (float3*)malloc(numberOfBodies*sizeof(float3));
	
	float3* v;
	v = (float3*)malloc(numberOfBodies*sizeof(float3));
	
	float3* a;
	a = (float3*)malloc(numberOfBodies*sizeof(float3));

	
	srand (time(NULL));
	//fill random starting position and acceleration
	for(int i = 0; i < numberOfBodies; i++){
		x[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		x[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		x[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		
		v[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		v[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		v[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		a[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		a[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		a[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

	}
	
	//allocate cuda memory
	float3 *d_x;
	cudaMalloc((void**)&d_x, numberOfBodies*sizeof(float3));
	
	float3 *d_a;
	cudaMalloc((void**)&d_a, numberOfBodies*sizeof(float3));
	
	//declare gridSize
	int gridSize = (numberOfBodies+blockSize-1)/(blockSize);

	//start loop over time steps
	for(int k = 0; k < numberOfSteps; k++){
		
		//copy position, acceleration to device
		cudaMemcpy(d_x, x, numberOfBodies*sizeof(float3), cudaMemcpyHostToDevice);
		cudaCheckError();
		cudaMemcpy(d_a, a, numberOfBodies*sizeof(float3), cudaMemcpyHostToDevice);
		cudaCheckError();


		//call kernel
		find_forces<<<gridSize, blockSize>>>(d_x, d_a, numberOfBodies);
		
		//copy position, acceleration off device
		cudaMemcpy(x, d_x, numberOfBodies*sizeof(float3), cudaMemcpyDeviceToHost);
		cudaCheckError();
		cudaMemcpy(a, d_a, numberOfBodies*sizeof(float3), cudaMemcpyDeviceToHost);
		cudaCheckError();

		for(int i = 0; i < numberOfBodies; i++){
			if(a[i].x == -1){
				v[i].x += 0;
				v[i].y += 0;
				v[i].z += 0;
			}
			else{
				v[i].x += 0.5*a[i].x*dt*dt/mass;
				v[i].y += 0.5*a[i].y*dt*dt/mass;
				v[i].z += 0.5*a[i].z*dt*dt/mass;

				x[i].x += v[i].x*dt;
				x[i].y += v[i].y*dt;
				x[i].z += v[i].z*dt;
			}
		}
	}
	
	//read out some results just for fun
	for(int i = 0; i < 10; i++){
		std::cout << x[i].x << "  " << a[i].x << std::endl;
	}
	
	free(x);
	free(a);		
	cudaFree(d_x);
	cudaFree(d_a);
}
