#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<time.h>
#include<iostream>
#include<stdio.h>
#include "device_launch_parameters.h"
#include "wb.h"

/*https://www.slideshare.net/DarshanParsana/gaussian-image-blurring-in-cuda-c*/

#define BLUR_SIZE 3
#define BLOCK_SIZE 16

__global__ void imageBlur(float* in, float* out, int width, int height, int channelNumber) {
	
	float numPixels;
	int Row = blockIdx.y * blockDim.y + threadIdx.y; // assigning threads to row
	int Col = blockIdx.x * blockDim.x + threadIdx.x; // assigning threads to col

	if ((Row < height) && (Col < width))
	{
		
		for (int channel = 0; channel < channelNumber; channel++) { //Channels (r,g,b)

			numPixels = 0.0; 
			int counter = 0;

			for (int blurRow = 0; blurRow < BLUR_SIZE + 1; ++blurRow) {
				for (int blurCol = 0; blurCol < BLUR_SIZE + 1; ++blurCol) {


					int currentRow = Row + blurRow; //getting the current rows
					int currentCol = Col + blurCol; //getting the current cols

					if ((currentRow > -1) && (currentRow < height) && (currentCol > -1) && (currentCol < width)) {
						numPixels += in[(currentRow * width + currentCol) * channelNumber + channel]; //Linearization of an input array
						counter++;//keeping each pixel numbers up to 9 for this implementation because of blur size
						
					}
				}
			}

			out[channelNumber * (Row * width + Col) + channel] = (numPixels/counter); // assigning values to output variable
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

	inputImageFile = argv[1];
	outputImageFile = argv[2];
	//argv[1] = "1.ppm";
	//argv[2] = "1out.ppm";
	//inputImageFile = "1.ppm";
	//outputImageFile = "out.ppm";
	printf("Loading %s\n", inputImageFile);
	inputImage = wbImport(inputImageFile);
	hostInputImageData = wbImage_getData(inputImage);

	int imageWidth = wbImage_getWidth(inputImage);
	int imageHeight = wbImage_getHeight(inputImage);
	int imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth,imageHeight,imageChannels);
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

	//printf("Device Space Allocated\n");

	/*for (int i = 0; i < 100; i++) {
		printf("hostDeviceInput: %lf", hostInputImageData[i]);
	}*/

	cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutputImageData, hostOutputImageData, imageSize, cudaMemcpyHostToDevice);

	//printf("Mem copy from host to device\n");

	dim3 dimBlock(16, 16, 1);

	dim3 dimGrid((float)ceil(imageWidth /BLOCK_SIZE), (float)ceil(imageHeight /BLOCK_SIZE), 1);

	//printf("Kernel Executing\n");

	cudaEventRecord(start);
	imageBlur << < dimGrid, dimBlock >> > ( deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight,imageChannels);
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
