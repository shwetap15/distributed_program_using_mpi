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
			//sum += x[col] *A[row][col]
			sum += x[col] * A[row*n+col];
		}

		y[row] = sum;
	}
}

int main (int argc, char*argv[]) {
	
	MPI_Status status;     
	MPI_Comm row_comm, col_comm;
	MPI_Group MPI_GROUP_WORLD;

	if (argc < 3) {
		std::cout<<"usage: "<<argv[0]<<" <n> <iteration>"<<std::endl;
	}

	bool check = true;

	long n = atol(argv[1]);

	long iter = atol(argv[2]);


	//initialize data
	int block_size , rowsize, rowrank, colsize, colrank;

	int Numprocs, MyRank, Root = 0;
	int irow, icol, iproc, jproc, index, Proc_Id;
	int NoofRows, NoofCols, NoofRows_Bloc, NoofCols_Bloc;
	int Bloc_MatrixSize, Bloc_VectorSize, VectorSize;
	int Local_Index, Global_Row_Index, Global_Col_Index;

	float **Matrix, *Matrix_Array, *Bloc_Matrix, *Vector, *Bloc_Vector;
	float *FinalResult, *MyResult, *FinalVector;

	int *ranks;
	
	long block_rows,block_cols,block_x_size, myrow; 
	float* local_A ,*x, *A;
	
	
	float* y = new float[n];

	//std::cout<<"========================================"<<std::endl;
	
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

	
	
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	
	int p, root_p;
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	p = 4;
	
	root_p  = sqrt(p);
	//float* local_A = new float[(n*n)/p];
	
	
	if(rank == 0)
	{
		block_rows = block_cols = n/root_p;
		
		//matrix A creation
		float* A = new float[n*n];

		for (long row = 0; row<n; row++) {
			for (long col=0; col<n; col++) {
				A[row*n+col] = genA(row, col);
			}
		}
	
		//vector x creation
		float* x = new float[n];
	
		for (long i=0; i<n; ++i)
			x[i] = genx0(i);
	}	
	
	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Bcast (A, n*n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	/* Memory allocating for Bloc Matrix */
	block_x_size = n / root_p;
	block_size = block_rows*block_cols;
	local_A = (float *) malloc (block_size * sizeof(float));
	MPI_Scatter(local_A, block_size, MPI_FLOAT, local_A,  block_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD); 
	
	/* Creating groups of procesors row wise */
	myrow=rank/root_p;
	MPI_Comm_split(MPI_COMM_WORLD,myrow,rank,&row_comm);
	MPI_Comm_size(row_comm,&rowsize);
	MPI_Comm_rank(row_comm,&rowrank);

	/* Creating groups of procesors column wise */
	myrow=rank%root_p;
	MPI_Comm_split(MPI_COMM_WORLD,myrow,rank,&col_comm);
	MPI_Comm_size(col_comm,&colsize);
	MPI_Comm_rank(col_comm,&colrank);
	
	/* Scatter part of vector to all row master processors */
	Bloc_Vector = (float*) malloc(block_x_size * sizeof(float));
	if(rank == 0){
		MPI_Scatter(x, block_x_size, MPI_FLOAT, Bloc_Vector, block_x_size, MPI_FLOAT, 0,row_comm);
	}

	/* Row master broadcasts its vector part to processors in its column */
	MPI_Bcast(Bloc_Vector, block_x_size, MPI_FLOAT, 0, col_comm);

	/* Multiplication done by all procs */

	MyResult   = (float *) malloc(block_rows * sizeof(float));
	index = 0;
	for(irow=0; irow < block_rows; irow++){
		MyResult[irow]=0;
		for(icol=0;icol< block_cols; icol++){
			MyResult[irow] += local_A[index++] * Bloc_Vector[icol];
		}
	}

	/* collect partial product from all procs on to master processor 
		and add it to get final answer */
	if(rank == 0) 
		FinalResult = (float *)malloc(block_rows*p*sizeof(float));

	MPI_Gather (MyResult, block_rows, MPI_FLOAT, FinalResult, block_rows, MPI_FLOAT, 0, MPI_COMM_WORLD); 
	
	if(MyRank == 0){
		FinalVector = (float *) malloc(NoofRows * sizeof(float));
		index = 0;
		for(iproc=0; iproc < root_p; iproc++){
			for(irow=0; irow < block_rows; irow++){
				FinalVector[index]  = 0;
				for(jproc=0; jproc < root_p; jproc++){
					FinalVector[index] += FinalResult[iproc*root_p*block_rows + jproc*block_rows +irow];
				}
				index++;
			}
		}
		
		//std::cout<<"FinalVector["<<it+1<<"]: ";
		for (long i=0; i<n; ++i)
			std::cout<<FinalVector[i]<<" ";
		std::cout<<std::endl;
	}
	
	/*if(rank == 0)
	{
		for (int it = 0; it<iter; ++it) {

			matmul(A, x, y, n);

			{
				float*t = x;
				x=y;
				y=t;
			}

			if (check)
				for (long i = 0; i<n; ++i)
					checkx (it+1, i, x[i]);
		}
	}	*/
	

	/*std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cerr<<elapsed_seconds.count()<<std::endl;*/

	if(rank == 0)
	{
		std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed_seconds = end-start;

		//std::cout<<output<<std::endl;
		std::cerr<<elapsed_seconds.count();
		
	}
	
	if(rank == 0)
	delete[] A;
	delete[] x;
	delete[] y;

	MPI_Comm_free(&row_comm);
	MPI_Comm_free(&col_comm);

	MPI_Finalize();
	
	return 0;
}
