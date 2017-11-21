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

		eNomVec.reserve(numberOfGroups);
		
		//read in data to vectors
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
//			xVec.push_back(xcoord);
//			yVec.push_back(ycoord);
//			nxVec.push_back(nx);
//			nyVec.push_back(ny);
//			weightVec.push_back(weight);
		}
	}
}
int main(){
	int numberOfGroups;
	std::vector<double> eNom, range, sigmaX, sigmaY, eMean, sigmaE, xCoord, yCoord, nx, ny, weight;
	getSourceFile(eNom, range, sigmaX, sigmaY, eMean, sigmaE, xCoord, yCoord, nx, ny, weight, numberOfGroups);
	
	std::vector< std::vector<double> > zRange;
	
	for(int master = 0; master < 94; master++){
		
		std::vector<double> temp;

		//declare stream size variables and open file/check for errors
		std::streampos bufferSize;

		//create fileName to read in data
		std::ostringstream fName;
		if(master < 9){
			fName << std::fixed << "PercentEdep3D_field_0" << master+1 << "_" << std::setprecision(1) << eNom[master] << "MeV.bin";
		}else{
			fName << std::fixed << "PercentEdep3D_field_" << master+1 << "_"  << std::setprecision(1) << eNom[master] << "MeV.bin";

		}
		std::string fileName = fName.str();
		std::cout << fileName << std::endl;
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

		int size = bufferSize/(sizeof(double)*400);
		
		//copy memory from buffer to energy
		double *energy;
		energy = (double*)malloc(size*sizeof(double)*400);
		std::copy(buffer.begin(), buffer.end(), energy);

		for(int i = 20100; i < 16000000; i+= 40000){
			temp.push_back(energy[i]);
		}	 
		zRange.push_back(temp);		
	}
	
	std::cout << zRange.size() << std::endl;
	std::ofstream zfile("depthDose.txt", std::ios::out);
	zfile << "Depth Dose Curve" << std::endl;
	zfile << std::endl;
	zfile << "Z(cm)";
	for(int i = 0; i < 94; i++){
		zfile << std::fixed << std::setprecision(1) << std::setw(11) << eNom[i] << "(MeV)";
	}	
	zfile << std::endl;
	for(int i = 0; i < 400; i++){
		zfile << i << "        ";
		for(int j = 0; j < 94; j++){
			zfile << std::fixed << std::setprecision(3) << std::setw(8) << zRange[j][i] << "        ";
		}
		zfile << std::endl;
	}
	zfile << std::endl;
}
