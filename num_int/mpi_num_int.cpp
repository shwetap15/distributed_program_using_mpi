#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <mpi.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

float f1(float x, int intensity);
float f2(float x, int intensity);
float f3(float x, int intensity);
float f4(float x, int intensity);

#ifdef __cplusplus
}
#endif

  
int main (int argc, char* argv[]) {
    
	if (argc < 6) {
		std::cerr<<"usage: "<<argv[0]<<" <functionid> <a> <b> <n> <intensity>"<<std::endl;
		return -1;
	}
	
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	MPI_Init(&argc,&argv);
    
	int function_id = atoi(argv[1]);
	float a = atof(argv[2]);
	float b = atof(argv[3]);
	int n = atoi(argv[4]);
	int intensity = atoi(argv[5]);
	float ba_by_n = ((b-a)/n);
	double output = 0;
	float local_ba_by_n = 0;
	float local_a = 0;
	
	int P;
	
	MPI_Comm_size(MPI_COMM_WORLD, &P);
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	double partial_output = 0;
	
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&intensity, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	for(long i = rank;i<n;i=i+P)
	{
		float x = 0;
		float j = 0;
		
		j = (i+0.5)*ba_by_n;
		int local_ba_by_n = 0;
		
		x = a+j;
		
		switch(function_id)
		{
			case 1 : {
					partial_output += f1(x,intensity);
					break;
				}
			case 2 : {
					partial_output += f2(x,intensity);
					break;
				}
			case 3 : {
					partial_output += f3(x,intensity);
					break;
				}
			case 4 : {
					partial_output += f4(x,intensity);
					break;
				}
			default:break;		
		}	
		
		//std::cout<<"partial_output is "<<partial_output<<std::endl;
	}
	
	MPI_Reduce(&partial_output, &output, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	output = output*ba_by_n;
	
	if(rank == 0)
	{
		std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed_seconds = end-start;

		std::cout<<output<<std::endl;
		std::cerr<<elapsed_seconds.count();
		
	}
	
	MPI_Finalize();
	
	
	return 0;
}
