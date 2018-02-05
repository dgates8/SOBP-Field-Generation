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

//Define the Z direction size as a global variable
#define z 400
#define blockSize 128
#define energySize 64000000


//define macro for error checking
#define cudaCheckError(){	   												   \
	cudaError_t err = cudaGetLastError();											   \
	if(err != cudaSuccess){              											   \
		std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << " : " <<  cudaGetErrorString(err) << std::endl; \
		exit(EXIT_FAILURE);                 										   \
	}                                     											   \
}

__inline__ __device__ int absolute(int value){
	return value < 0 ? value*-1 : value;
}

/***********************************************************************************
BeamShift takes in energy array from the host loop and moves the coordinates 
as specified to move the beam grid, given by the starting value 200*y + x, which 
corresponds to the beginning of the offset to write the energy value into 
corresponding to the "move". Alternatively, think of the "move" in x-y direction as
having to be mapped back into a location in the 16M element energy array, since the 
energy array contains all information about how the field is distributed and paraview
does the coordinate assignation.  
************************************************************************************/
__global__ void beamShift(double* energy, double *temp, int move_x, int move_y, const int size){
	int tid = blockIdx.x*blockDim.x*8 + threadIdx.x;
	int id = absolute(400*move_y + move_x + tid);
	if(id < size){
		#pragma unroll
		for(int i = 0; i < 8; i++){
			temp[tid+blockSize*i] = energy[id+blockSize*i]+energy[tid+blockSize*i];
		}
	}
}
/***********************************************************************************
Sum takes in each array after the move and sums them back together to get the correct 
energy values for the field. I used a tiling approach whereby the energy data is 
loaded into shared memory tiles for faster read/write access and then added together.
************************************************************************************/
__global__ void sum(double* temp, double* energy, const int size){
	int tid = blockIdx.x*blockDim.x*32 + threadIdx.x;
	if(tid < size){
		#pragma unroll
		for(int i = 0; i < 32; i++){
			temp[tid + i*blockSize] += energy[tid + i*blockSize];
		}
	}

}

//Read in energy source file data 
void getSourceFile(std::vector<double>& eNomVec, std::vector<double>& rangeVec, 
			std::vector<double>& sigmaXVec,std::vector<double>& sigmaYVec, 
			std::vector<double>& eMeanVec, std::vector<double>& sigmaEVec, 
			std::vector<double>& xVec, std::vector<double>& yVec, 
			std::vector<double>& nxVec,std::vector<double>& nyVec,
			std::vector<double>& weightVec, int& numGroups) 
{
	int dateOfMeasurement;
	long int numberOfGroups;
	double  eNom, range, sigmaX, sigmaY, eMean, sigmaE, xcoord, ycoord, weight, nx, ny;
	
	std::string line;
	//declare and open file
	std::ifstream ifile("IMPT_source.dat", std::ios::in);
	if(!ifile){
		std::cout << "Error, IMPT_source not found" << std::endl;
	}else{
		//read in date of measurement
		ifile >> dateOfMeasurement;
	
		//read in number of groups
		ifile >> numberOfGroups;
		numGroups = numberOfGroups;
		
		//skip over header line
		std::string e, r, x, y, m, s, nx1, ny1, x1, y1, w;
		ifile >> e;
		ifile >> r;
		ifile >> x;
		ifile >> y;
		ifile >> m;
		ifile >> s;
		ifile >> x1;
		ifile >> y1;
		ifile >> nx1;
		ifile >> ny1;
		ifile >> w;
	
		//intialize memory for faster read in 
		xVec.reserve(numberOfGroups);
		yVec.reserve(numberOfGroups);
		nxVec.reserve(numberOfGroups);
		nyVec.reserve(numberOfGroups);
		weightVec.reserve(numberOfGroups);
		eNomVec.reserve(numberOfGroups);					
		
		//read in data to vectorsG
		for(int i = 0; i < numberOfGroups; i++){
			ifile >> eNom;
			ifile >> range;
			ifile >> sigmaX;
			ifile >> sigmaY;
			ifile >> eMean;
			ifile >> sigmaE;
			ifile >> xcoord;
			ifile >> ycoord;
			ifile >> nx;
			ifile >> ny;
			ifile >> weight;

			eNomVec.push_back(eNom);
//			rangeVec.push_back(range);
//			sigmaXVec.push_back(sigmaX);
//			sigmaYVec.push_back(sigmaY);
//			eMeanVec.push_back(eMean);
			xVec.push_back(xcoord);
			yVec.push_back(ycoord);
			nxVec.push_back(nx);
			nyVec.push_back(ny);
			weightVec.push_back(weight);
		}
	}
}

int main(int argc, char** argv){
	
	//get command line arguments for the elements to loop over and error check	
	if(argc < 2){
		std::cout << "Too few arguments, need two for range of beam values" << std::endl;
		exit(EXIT_FAILURE);
	}else if(argc > 5 ){
		std::cout << "Too many arguments, need two for range of beam values" << std::endl;
		exit(EXIT_FAILURE);
	}else if(atoi(argv[1]) <= 0 || atoi(argv[1]) > 94 || atoi(argv[2]) <= 0 || atoi(argv[2]) > 94){
		std::cout << "Arguments out of range, must be in range [1,94]" << std::endl;
		exit(EXIT_FAILURE);
	}

	//declare stuff for source file read
	int numberOfGroups;
	std::vector<double> eNom, range, sigmaX, sigmaY, eMean, sigmaE, xCoord, yCoord, nx, ny, weight;
	getSourceFile(eNom, range, sigmaX, sigmaY, eMean, sigmaE, xCoord, yCoord, nx, ny, weight, numberOfGroups);

	//intialize device for faster update of values using all 16 GPUs as defined by the run.sh script	
	cudaSetDevice(atoi(argv[3]));

	for(int master = atoi(argv[1])-1; master < atoi(argv[2]); master++){
			
		//declare stream size variables and open file/check for errors
		std::streampos bufferSize;

		//create fileName to read in data
		std::ostringstream fName;
		if(master < 9){
			fName << std::fixed << "GyPerMU3D_beamlet_0" << master+1 << "_" << std::setprecision(1) << eNom[master] << "MeV.bin";
		}else{
			fName << std::fixed << "GyPerMU3D_beamlet_" << master+1 << "_"  << std::setprecision(1) << eNom[master] << "MeV.bin";

		}
		std::string fileName = fName.str();
		std::ifstream ifile(fileName.c_str(), std::ios::in | std::ios::binary);
		if(!ifile){
			std::cout << "Error, no file found" << std::endl;
			exit(1);
		}
		
		//get file size
		ifile.seekg(0, std::ios::end);
		bufferSize = ifile.tellg();
		ifile.seekg(0, std::ios::beg);

		//declare buffer
		std::vector<double> buffer(bufferSize/sizeof(double));
		
		//read in data
		ifile.read(reinterpret_cast<char*>(buffer.data()), bufferSize); 

		//declare size of data for later malloc's
		int size = bufferSize/(sizeof(double)*z);
		
		//fill new array of 400x400 field correctly with zeroes in each layer for extra points out of range
		std::vector<double> key;
		for(int k = 0; k < 400; k++){
			for(int i = 0; i < 400; i++){
				for(int j = 0; j < 400; j++){
					if(j < 100 && i < 100){
						key.push_back(0);
					}else if(j >= 100 && j < 300 && i >= 100 && i < 300){
						key.push_back(buffer[(200*(i-100) + (j-100))+(k*40000)]);
					}else{
						key.push_back(0);
					}
				}
			}
		}

		//copy memory from buffer to energy
		double *energy;
		energy = (double*)malloc(energySize*sizeof(double));
		std::copy(key.begin(), key.end(), energy);
		
		//free memory from buffer
		std::vector<double>().swap(buffer);
			
		///create #spots variable;
		int spots = nx[master]*ny[master];	

		//declare move arrays for grid distribution
		std::vector<float> move1, move2;
		move1.reserve(spots);
		move2.reserve(spots);

		int moveX[spots], moveY[spots];
		double fieldSize = -1*(xCoord[master]+yCoord[master])/(nx[master]-1);
		double field = -1*(xCoord[master]+yCoord[master]);
		//declare spacings
		int spaceX = ceil(field*10/(nx[master]));
		int spaceY = ceil(field*10/(ny[master]));		
		for(int i = 0, x = xCoord[master]*10; i < nx[master]; i++, x += spaceX){
			for(int j = 0, y = yCoord[master]*10; j < ny[master]; j++, y+= spaceY){
				move1.push_back(x);
				move2.push_back(y);
			}
		}

		//copy to arrays for cuda since vectors are more annoying to work with as they are heap objects 
		std::copy(move1.begin(), move1.end(), moveX);
		std::copy(move2.begin(), move2.end(), moveY);
	

	/***********************************************************************************************************/
		
		//Declare gridSize for cuda kernels
		int gridSize = (energySize+blockSize-1)/blockSize;

		//declare host arrays
		double *temp_energy, *h_energy; 
		
		//allocate an array to hold the sum of each movement on the device
		cudaMallocHost((void**)&temp_energy, energySize*sizeof(double));
		cudaCheckError();
	
		double *d_energy;
		cudaMalloc((void**)&d_energy, energySize*sizeof(double));
		cudaCheckError();
	
		cudaMemcpy(d_energy, energy, energySize*sizeof(double), cudaMemcpyHostToDevice);
		cudaCheckError();
	
		//loop to perform all the moves.
		for(int i = 0; i < spots; i++){		
	
			double *d_temp;
									
			cudaMalloc((void**)&d_temp, energySize*sizeof(double));
			cudaCheckError();
	
			//kernel to perform all the moves for the grid
			beamShift<<<gridSize/8, blockSize>>>(d_energy, d_temp, moveX[i], moveY[i], energySize);
			sum<<<gridSize/32, blockSize>>>(temp_energy, d_temp, energySize);
			
			cudaFree(d_temp);
		}

		//read off the temp_energy vector from the device	
		cudaMallocHost((void**)&h_energy, energySize*sizeof(double));
		cudaCheckError();

		cudaMemcpyAsync(h_energy, temp_energy, energySize*sizeof(double), cudaMemcpyDeviceToHost);
		cudaCheckError();
			
		
		//final kernel to subtract off extra copies of energy
		for(int i = 0; i < energySize; i++){
			h_energy[i] -= energy[i]*(spots);
			//h_energy[i] *= weight[master]; TODO: Uncomment for weighted run
			if(h_energy[i] < 0){
				h_energy[i] = 0;
			}
		}					
		
		//read out to 94 files individually
		std::ostringstream OName;
		if(master < 9){
			OName << std::fixed << "GyPerMU3D_0" << master+1 << "_" << std::setprecision(1) << eNom[master] << "MeV_field_" << std::setprecision(0) << nx[master] << "by" 
			      << std::setprecision(0) << ny[master] << "spots_" << std::setprecision(2) << fieldSize << "by" << std::setprecision(2) << fieldSize << "cm2spacing.bin";
		}else{
			OName << std::fixed << "GyPerMU3D_" << master+1 << "_"  << std::setprecision(1) << eNom[master] << "MeV_field_" << std::setprecision(0) << nx[master] << "by" 
		              << std::setprecision(0) << ny[master] << "spots_" << std::setprecision(2) << fieldSize << "by" << std::setprecision(2) << fieldSize << "cm2spacing.bin";

		}

		std::string fileNameOut = OName.str();
		std::ofstream ofile(fileNameOut.c_str() , std::ios::out | std::ios::binary);
		ofile.write(reinterpret_cast<char*>(h_energy), energySize*sizeof(double));
		cudaDeviceReset();
	}//end of master loop	
}//end of main
