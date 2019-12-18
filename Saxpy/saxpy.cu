#include<stdio.h>
#include<iostream>

//using the struct of cudaDeviceProp getting the information of gpu
void printDeviceProp(cudaDeviceProp devProp){
	
		printf("Name: %s\n", devProp.name);
		printf("Maximum thread per block: %d\n", devProp.maxThreadsPerBlock);
		for(int i = 0; i < 3; i++)
		printf("Maximum dimension of block: %d\t %d\n",i, devProp.maxThreadsDim[i]);

	return;
}

//A single thread for each of the n elements, and each thread computes its array index using blockIdx.x*blockDim.x + threadIdx.x.

__global__ void saxpy(int n, float a, float *x, float *y){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n){
		
		y[i] = a*x[i] + y[i];
		__syncthreads();//syncronizing threads in kernel calls
	}

}


int main(int argc, char* argv[]){

	int devCount;
	int N;
	float A;

	float* d_x;
	float* d_y;

	//N = atoi(argv[1]);

	//A = atoi(argv[2]);

	cudaGetDeviceCount(&devCount);
	printf("%d Cuda devices\n", devCount);
	
	for(int k = 0; k < devCount; ++k){
		printf("\nCuda Device %d\n",k);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, k);
		printDeviceProp(devProp);
	}
		

	printf("Size of array N: ");
	scanf("%d", &N);

	printf("Size of scalar value A: ");
	scanf("%f", &A);

	//Allocates space from host memory
	float* h_x = (float*)malloc(N*sizeof(float));
	float* h_y = (float*)malloc(N*sizeof(float));

	cudaMalloc((void**)&d_x, N*sizeof(float));
	cudaMalloc((void**)&d_y, N*sizeof(float));

	for(int i=0; i < N; i++){
		
		//Generates random values between 0 and 256 and assigns to allocated space
		h_x[i] = (float) (rand() % 256);
		h_y[i] = (float) (rand() % 256);
		printf("x = %f\n",h_x[i]);
		printf("y = %f\n",h_y[i]);
	}

	//Sends from host to device
	cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);

	int blocks = (N + 255)/256;
	printf("Block number: %d\n",blocks);

	//calls the kernel function
	//The first argument in the execution specifies the number of thread blocks in the grid, and the second specifies the number of threads in a thread block.
	saxpy<<<blocks,256>>>(N, A, d_x, d_y);

	//returns the value from device to host 
	cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	//Printing the host value after returned from device to host
	for(int i=0; i < N; i++){
		
		printf("y = %f\n",h_y[i]);
	}
cudaFree(d_x);
cudaFree(d_y);
free(h_x);
free(h_y);

	return 0;
}
