#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <mpi.h>

float genA (int row, int col) {
	if (row > col)
		return 1.;
	else
		return 0.;
}

float genx0 (int i) {
	return 1.;
}


void checkx (int iter, long i, float xval) {
	if (iter == 1) {
	float shouldbe = i;
	if (fabs(xval/shouldbe) > 1.01 || fabs(xval/shouldbe) < .99 )
		std::cout<<"incorrect : x["<<i<<"] at iteration "<<iter<<" should be "<<shouldbe<<" not "<<xval<<std::endl;
	}

	if (iter == 2) {
	float shouldbe =(i-1)*i/2;
	if (fabs(xval/shouldbe) > 1.01 || fabs(xval/shouldbe) < .99)
		std::cout<<"incorrect : x["<<i<<"] at iteration "<<iter<<" should be "<<shouldbe<<" not "<<xval<<std::endl;
	}
}

//perform dense y=Ax on an n \times n matrix
void matmul(float*A, float*x, float*y, long n) {
	for (long row = 0; row<n; ++row) {
		float sum = 0;

		for (long col = 0; col<n; ++col) {
			sum += x[col] * A[row*n+col];
		}

		y[row] = sum;
	}
}

int main (int argc, char*argv[]) {

	if (argc < 3) {
		std::cout<<"usage: "<<argv[0]<<" <n> <iteration>"<<std::endl;
	}

	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	MPI_Init(&argc,&argv);
	
	//bool check = true;
	bool check = false;

	long n = atol(argv[1]);

	long iter = atol(argv[2]);


	//initialize data
	MPI_Comm row_comm, col_comm;
	
	int p,rank;
	int row_size,row_rank,col_size,col_rank,myrow,mycol;
	
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int root_p = sqrt(p);
	long block_size = n/root_p;
	
	float* local_A , *local_x, *x, *y;
	x = new float[n];
	y = new float[n];
	
	local_A = new float[block_size*block_size];
	local_x = new float[block_size];

	
	long local_col,local_row,index;
	
		
	if(rank%block_size == 0)
		local_row = rank;
	
	local_col = (rank%block_size)*block_size;
	
	index = 0;
	for(long row=local_row;row<(local_row+block_size);row++)
	{
		for(long col=local_col;col<(local_col+block_size);col++)
		{
			local_A[index] = genA(row, col);
			index++;
		}
	}
	
	/* Creating groups of procesors row wise */

	int row_color = rank/root_p;
	MPI_Comm_split(MPI_COMM_WORLD,row_color,rank,&row_comm);
	MPI_Comm_size(row_comm,&row_size);
	MPI_Comm_rank(row_comm,&row_rank);
	
	/* Creating groups of procesors column wise */
	int col_color = rank%root_p;
	MPI_Comm_split(MPI_COMM_WORLD,col_color,rank,&col_comm);
	MPI_Comm_size(col_comm,&col_size);
	MPI_Comm_rank(col_comm,&col_rank);
		
	float* local_y = new float[block_size];
	
	for (int it = 0; it<iter; ++it) {
		if(it==0)
		{
			for (long i=0; i<block_size; i++)	
			{
				local_x[i] = genx0(i);
			}
		}	

		matmul(local_A, local_x, local_y, block_size);

		{
			float* t = local_x;
			local_x=local_y;
			local_y=t;
		}
		
		//MPI_Reduce(&local_y, &y, 1, MPI_DOUBLE, MPI_SUM, 0, row_comm);
		MPI_Reduce(local_x, x, 1, MPI_DOUBLE, MPI_SUM, 0, row_comm);
		MPI_Bcast(x, 1, MPI_INT, 0, col_comm);
		

		if (check)
			for (long i = 0; i<n; ++i)
				checkx (it+1, i, local_x[i]);
	}	
	
	if(rank == 0)
	{
		std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed_seconds = end-start;

		std::cerr<<elapsed_seconds.count()<<std::endl;

	}

	MPI_Finalize();
	
	delete[] local_A;
	delete[] x;
	delete[] y;
	delete[] local_x;
	delete[] local_y;

	return 0;
}
