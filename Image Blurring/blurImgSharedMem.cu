#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<time.h>
#include<iostream>
#include<stdio.h>
#include "device_launch_parameters.h"
#include "wb.h"

//https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf

#define BLOCK_SIZE 16
#define blurCol 4
#define blurRow 4
#define BLOCK_WIDTH (BLOCK_SIZE + blurCol -1)

__global__ void imgBlurShared(float* in, const float* filterData, float* out, int channels, int width, int height) {

	int ty, tx;

	int linearization, linearizationOfRowAndCol;

	int row_i, col_i, row_o, col_o;

	int Row, Col;
	
	__shared__ float sharedArray[BLOCK_WIDTH][BLOCK_WIDTH];  //block of image in shared memory


	for (int c = 0; c < channels; c++) {
		
		float sumOfPixels = 0;

		//flattening the 2D coordinates of the thread then takes the inverse operation and calculating the 2D coordinates of the thread with respect to the shared memory area.
		//Step 1:

		linearization = threadIdx.y * BLOCK_SIZE + threadIdx.x;

		ty = linearization / BLOCK_WIDTH;     //assigning row of shared memory 

		tx = linearization % BLOCK_WIDTH;		//assigning col of shared memory  
		
		row_o = blockIdx.y * BLOCK_SIZE + ty;

		col_o = blockIdx.x * BLOCK_SIZE + tx;

		row_i = row_o - blurRow / 2; // index to fetch data from input image because of getting values from neighbor tiles then you have to shift  

		col_i = col_o - blurRow / 2; // index to fetch data from input image because of getting values from neighbor tiles then you have to shift

		if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {

			sharedArray[ty][tx] = in[(row_i * width + col_i) * channels + c];  // copy element of image in shared memory
		}
		else {

			sharedArray[ty][tx] = 0;
		}

		__syncthreads();

		//Step 2:


		linearizationOfRowAndCol = threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE;

		ty = linearizationOfRowAndCol / BLOCK_WIDTH;	//assigning row of shared memory 

		tx = linearizationOfRowAndCol % BLOCK_WIDTH;	//assigning col of shared memory
		
		row_o = blockIdx.y * BLOCK_SIZE + ty;

		col_o = blockIdx.x * BLOCK_SIZE + tx;

		row_i = row_o - blurRow / 2;// index to fetch data from input image because of getting values from neighbor tiles then you have to shift

		col_i = col_o - blurRow / 2;// index to fetch data from input image because of getting values from neighbor tiles then you have to shift

		if (ty < BLOCK_WIDTH) {

			if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)

				sharedArray[ty][tx] = in[(row_i * width + col_i) * channels + c];// copy element of image in shared memory

			else

				sharedArray[ty][tx] = 0;

		}

		__syncthreads();

		//Blurring image by tracing coloums and rows and multiplies the filter array with the shared array values that comes from input image assigned to shared array 

		for (Row = 0; Row < blurCol ; Row++) {

			for (Col = 0; Col < blurRow ; Col++) {

				sumOfPixels = sumOfPixels + sharedArray[threadIdx.y + Row][threadIdx.x + Col] * filterData[Row * blurCol + Col];

			}
		}

		// assigning values to output variable 
		Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

		Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

		if (Row < height && Col < width) {

			out[(Row * width + Col) * channels + c] = sumOfPixels;
		}

	}
}


void wbImage_save(const wbImage_t& image, const char* fName) {
	std::ostringstream oss;
	oss << "P6\n" << "# Created for blurring output" << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";
	//oss << "P6\n" << "# Created by GIMP version 2.10.8 PNM plug-in" << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";

	std::string headerStr(oss.str());

	std::ofstream outFile(fName, std::ios::binary);
	outFile.write(headerStr.c_str(), headerStr.size());

	const int numElements = image.width * image.height * image.channels;

	unsigned char* rawData = new unsigned char[numElements];

	for (int i = 0; i < numElements; ++i)
	{
		rawData[i] = static_cast<unsigned char>(image.data[i] * wbInternal::kImageColorLimit + 0.5f);
	}

	outFile.write(reinterpret_cast<char*>(rawData), numElements);
	outFile.close();

	delete[] rawData;
}


int main(int argc, char** argv) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	char* inputImageFile;
	char* outputImageFile;

	wbImage_t inputImage;
	wbImage_t outputImage;

	float* hostInputImageData;
	float* hostOutputImageData;

	float* deviceInputImageData;
	float* deviceOutputImageData;
	float* deviceBlurData;

	float hostBlurData[blurRow * blurCol] = {  // I gave the filter values directly and in order to calculate 3x3 I gave the 4x4 values to the array and the size is related to that

			0.0625, 0.0625, 0.0625, 0.0625,

			0.0625, 0.0625, 0.0625, 0.0625,

			0.0625, 0.0625, 0.0625, 0.0625,

			0.0625, 0.0625, 0.0625, 0.0625

	};

	inputImageFile = argv[1];
	outputImageFile = argv[2];
	//argv[1] = "1.ppm";
	//argv[2] = "1out.ppm";
	//inputImageFile = "6.ppm";
	//outputImageFile = "out.ppm";
	printf("Loading %s\n", inputImageFile);
	inputImage = wbImport(inputImageFile);
	hostInputImageData = wbImage_getData(inputImage);

	int imageWidth = wbImage_getWidth(inputImage);
	int imageHeight = wbImage_getHeight(inputImage);
	int imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	hostOutputImageData = wbImage_getData(outputImage);

	hostInputImageData = wbImage_getData(inputImage);
	printf("%d %d %d\n", imageWidth, imageHeight, imageChannels);
	printf("%f %f %f\n", hostInputImageData[0], hostInputImageData[1], hostInputImageData[2]);
	/*for (int i = 0; i < 500; i++) {

		printf("Image: %f", hostInputImageData[i]);
	}*/

	/*YOUR CODE HERE*/
	size_t imageSize = sizeof(float) * imageWidth * imageHeight * imageChannels;
	printf("Image size: %zd\n", imageSize);
	

	//hostInputImageData = (float*)malloc(imageSize);
	/*for (int i = 0; i < 500; i++) {

		printf("Image commend malloc: %f\n", hostInputImageData[i]);
	}*/
	//hostOutputImageData = (float*)malloc(imageSize);

	/*printf("Host Input Malloc: %f",*hostInputImageData);
	printf("Host Output Malloc: %f", *hostOutputImageData);

	printf("Host Space Allocated\n");*/

	cudaMalloc((void**)&deviceInputImageData, imageSize);
	cudaMalloc((void**)&deviceOutputImageData, imageSize);
	cudaMalloc((void**)&deviceBlurData, sizeof(float) * blurRow * blurCol );

	//printf("Device Space Allocated\n");

	/*for (int i = 0; i < 100; i++) {
		printf("hostDeviceInput: %lf", hostInputImageData[i]);
	}*/

	cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice);
	//cudaMemcpy(deviceOutputImageData, hostOutputImageData, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBlurData, hostBlurData, sizeof(float) * blurRow * blurCol,cudaMemcpyHostToDevice);



	//printf("Mem copy from host to device\n");

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE, 1);

	dim3 dimGrid((float)ceil(imageWidth / BLOCK_SIZE), (float)ceil(imageHeight / BLOCK_SIZE), 1);

	//printf("Kernel Executing\n");

	cudaEventRecord(start);
	imgBlurShared << < dimGrid, dimBlock >> > (deviceInputImageData,deviceBlurData,deviceOutputImageData, imageChannels, imageWidth, imageHeight);
	cudaEventRecord(stop);

	//printf("Kernel terminate\n");

	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize, cudaMemcpyDeviceToHost);

	//printf("Copy from device to host\n");
	/*for (int i = 0; i < 100; i++) {
		printf("hostOut: %lf", hostOutputImageData[i]);
	}*/

	cudaEventSynchronize(stop);
	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Timer : %lf\n", milliseconds);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/*Just save the original image for now.*/
	//hostOutputImageData = hostInputImageData;
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	outputImage.data = hostOutputImageData;
	wbImage_save(outputImage, outputImageFile);

	return 0;
}
