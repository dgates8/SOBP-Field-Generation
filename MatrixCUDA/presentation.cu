#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream> 
#include <iomanip>
#define speed 3.0e8
#define mass  0.511
#define hbar  1.68e-10
#define pi    3.1415
#define S     0.5 //Symmetry factor for two body event 
#define g     2.002319 //coupling constant for theory


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

/*************************************************************************
Device function to find energy using the standard formula 
*************************************************************************/
__device__ double Energy(int a){
	return sqrt(powf(a,2)*powf(speed,2) + powf(mass,2)*powf(speed,4));
}

/***************************************************************************
Scattering solves for the cross section of scattering for electron pair annhilation
e+ + e- -> y + y. No radial dependence so multiply by 4*pi. 

--Drake Gates example CUDA code
***************************************************************************/
__global__ void Scattering(int n, int *a, int *b, double *c){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n){
		double Etotal = 8*pi*(Energy(a[i])+Energy(b[i]));
		c[i] = powf((speed*hbar)/Etotal,2) * double(a[i])/double(b[i]) * 16*powf(g,4) * 4*pi;
	}
}	

int main(int argc, char* argv[]){
	int N = atoi(argv[1]);//Number of elements to be generated 
	int *v1, *v2, *d_v1, *d_v2;
	double *energy, *d_energy; //Declare vectors or matrices in row-max format

	//Allocate pointers of type int and length N 
	v1 = (int*)malloc(N*sizeof(int));
	v2 = (int*)malloc(N*sizeof(int));
	energy = (double*)malloc(N*sizeof(double));

	//random fill
	srand(time(NULL));
    for (int i = 0; i < N; i++){
    	v1[i] = mass*(rand() % 10) + 1;
    	v2[i] = mass*(rand() % 10) + 1;
    }

	//Allocate GPU pointers by reference and check errors 
	cudaError_t err = cudaMalloc(&d_v1, N*sizeof(int));
	printf("CUDA malloc v1: %s\n",cudaGetErrorString(err));

	err = cudaMalloc(&d_v2, N*sizeof(int));
	printf("CUDA malloc v2: %s\n",cudaGetErrorString(err));

	err = cudaMalloc(&d_energy, N*sizeof(double));
	printf("CUDA malloc energy: %s\n",cudaGetErrorString(err));

	//copy vectors to GPU and check for errors
	err = cudaMemcpy(d_v1, v1, N*sizeof(int), cudaMemcpyHostToDevice);
	printf("Copy v1 to device: %s\n",cudaGetErrorString(err));

	err= cudaMemcpy(d_v2, v2, N*sizeof(int), cudaMemcpyHostToDevice);
	printf("Copy v2 to device: %s\n",cudaGetErrorString(err));

	//time the computation
	float time;
	cudaEvent_t start, stop;

	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );

	//call kernel of 1 block with 10 threads  
	Scattering<<<(N+255)/256,256>>>(N, d_v1,d_v2,d_energy);
	//sync threads
	err = cudaThreadSynchronize();
	printf("Kernel Call: %s\n",cudaGetErrorString(err));

	//get resultant energy off of device 
	err = cudaMemcpy(energy, d_energy, N*sizeof(double), cudaMemcpyDeviceToHost);
	printf("Copy Energy off device: %s\n",cudaGetErrorString(err));

	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
	printf("Time to generate:  %3.1f ms \n", time);

	//std::fixed << std::setprecision(6)
	/*
	for(int k = 0; k < N; k++){
		std::cout <<  energy[k] << "    " << v1[k] << "   " << v2[k] << std::endl;
	}
	*/
	//free GPU cache
	cudaFree(d_v1);
	cudaFree(d_v2);
	cudaFree(d_energy);
	free(v1);
	free(v2);
	free(energy);

	return 0;
}
