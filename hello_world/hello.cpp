#include <mpi.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <limits.h>

int main(int argc, char*argv[]) {
	MPI_Init(&argc,&argv);
	//MPI_Init(NULL,NULL);
	
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char hostname[HOST_NAME_MAX];
	gethostname(hostname, HOST_NAME_MAX);

	std::cout<<"I am process "<< rank <<" out of "<<size<<". I am running on machine - "<<hostname<<"."<<std::endl;
	
	MPI_Finalize();
	return 0;
}
