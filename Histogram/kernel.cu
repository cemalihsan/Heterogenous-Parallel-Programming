#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<time.h>
#include <stdlib.h>
#include<iostream>
#include<stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256
#define SCAN_SIZE 512

#ifndef IO_PGM_H_
#define IO_PGM_H_


#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

float* read_pgm_float(const char* fname, int* ncol, int* nrow);

void write_pgm_float(const char* fname, const float* data, int ncol, int nrow);

#endif

static void errorf(const char* msg, const char* name)
{
	fprintf(stderr, "%s %s.\n", msg, name);
	exit(EXIT_FAILURE);
}

static void error(const char* msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(EXIT_FAILURE);
}

static void skip_whites_and_comments(FILE* f)
{
	int c;
	do
	{
		while (isspace(c = getc(f))); /* skip spaces */
		if (c == '#') /* skip comments */
			while (c != '\n' && c != '\r' && c != EOF)
				c = getc(f);
	} while (c == '#' || isspace(c));
	if (c != EOF && ungetc(c, f) == EOF)
		error("Error: unable to 'ungetc' while reading PGM file.");
}

static unsigned int get_num(FILE* f)
{
	unsigned int num;
	int c;

	while (isspace(c = getc(f)));
	if (!isdigit(c)) error("Error: corrupted PGM file.");
	num = (unsigned int)(c - '0');
	while (isdigit(c = getc(f))) num = 10 * num + c - '0';
	if (c != EOF && ungetc(c, f) == EOF)
		error("Error: unable to 'ungetc' while reading PGM file.");

	return num;
}

float* read_pgm_float(const char* fname, int* ncol, int* nrow)
{
	FILE* f;
	int c, bin;
	int depth, i, j;
	float* data;

	/* open file */
	f = fopen(fname, "rb");
	if (f == NULL) errorf("Error: unable to open input image file ", fname);

	/* read header */
	if (getc(f) != 'P') errorf("Error: not a PGM file ", fname);
	if ((c = getc(f)) == '2') bin = FALSE;
	else if (c == '5') bin = TRUE;
	else errorf("Error: not a PGM file ", fname);
	skip_whites_and_comments(f);
	*ncol = get_num(f);            /* X size */
	skip_whites_and_comments(f);
	*nrow = get_num(f);            /* Y size */
	skip_whites_and_comments(f);
	depth = get_num(f);            /* depth */
	if (depth == 0) fprintf(stderr,
		"Warning: depth=0, probably invalid PGM file\n");
	/* white before data */
	if (!isspace(c = getc(f))) errorf("Error: corrupted PGM file ", fname);

	/* get memory */
	data = (float*)calloc((*ncol) * (*nrow), sizeof(float));
	if (data == NULL)
		error("not enough memory.");

	/* read data */

	/*If the depth is less than 256, it is 1 byte. Otherwise, it is 2 bytes*/
	if (depth < 256)
	{
		for (i = 0; i < *nrow; i++)
			for (j = 0; j < *ncol; j++)
				data[j + i * (*ncol)] = bin ?
				(float)getc(f) : (float)get_num(f);
	}
	/*16 bits PGM Most significant byte first
	 * see http://netpbm.sourceforge.net/doc/pgm.html
	 */
	else {
		for (i = 0; i < *nrow; i++)
			for (j = 0; j < *ncol; j++)
				/*most significant byte first*/
				data[j + i * (*ncol)] = bin ? ((float)getc(f) * 256) +
				(float)getc(f) : (float)get_num(f);

	}

	/* close file if needed */
	if (f != stdin && fclose(f) == EOF)
		errorf("Error: unable to close file while reading PGM file ", fname);

	return data;
}

void write_pgm_float(const char* fname, const float* data, int ncol, int nrow)
{
	FILE* f;
	int i, j;
	int v, max, min;

	/* check min and max values */
	max = min = 0;
	for (i = 0; i < nrow; i++)
		for (j = 0; j < ncol; j++)
		{
			v = (int)data[j + i * ncol];
			if (v > max) max = v;
			if (v < min) min = v;
		}

	if (min < 0) fprintf(stderr,
		"Warning: negative values in '%s'.\n",
		fname);
	if (max > 255) fprintf(stderr,
		"Warning: values exceeding 255 in '%s'.\n",
		fname);

	/* open file */
	if (strcmp(fname, "-") == 0) f = stdout;
	else f = fopen(fname, "w");
	if (f == NULL) errorf("Error: unable to open output image file ", fname);

	/* write header */
	fprintf(f, "P5\n");
	fprintf(f, "%d %d\n", ncol, nrow);
	fprintf(f, "%d\n", 255);

	/* write data */
	for (i = 0; i < nrow; i++) {
		for (j = 0; j < ncol; j++) {
			fputc((unsigned char)data[j + i * ncol], f);
		}


	}


	/* close file if needed */
	if (f != stdout && fclose(f) == EOF)
		errorf("Error: unable to close file while writing PGM file ", fname);
}


__global__ void createHistogram(unsigned int* histogram, float* img, int size) {
	
	__shared__ unsigned int private_histo[HISTOGRAM_LENGTH];
	int threadId = threadIdx.x;

	if (threadIdx.x < HISTOGRAM_LENGTH) {
		private_histo[threadId] = 0;
	}
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < size) {

		atomicAdd(&(private_histo[(int)img[i]]), 1);
		i += stride;

	}
	__syncthreads();

	if (threadIdx.x < HISTOGRAM_LENGTH) {
		atomicAdd(&(histogram[threadId]), private_histo[threadId]);
	}

}

__global__ void histogramEqualize(unsigned int* charImage, unsigned int* cdf, int size) {
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < HISTOGRAM_LENGTH) {
		float temp = 255.0F * (cdf[i] - (float)cdf[0]) / (size - (float)cdf[0]);
		float start = 0.0F, end = 255.0F;

		if (start > temp) {
			temp = start;
		}
		if (end < temp) {
			temp = end;
		}
		
		charImage[i] = (unsigned int)temp;
	}
	if (i == 0) {
		printf("Equalize[%d]: %u\n", i, charImage[i]);
	}
}

__global__ void cumulativeDistribution(unsigned int* cdf, unsigned int* histogram, int sizeOfHisto) {

	__shared__ unsigned int XY[SCAN_SIZE];
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < SCAN_SIZE && i < sizeOfHisto) {
		XY[i] = histogram[i];
	}

	__syncthreads();

	//work efficient reduction phase
	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		
		unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
		
		if (index < SCAN_SIZE && index < sizeOfHisto) {

			XY[index] = XY[index] + XY[index - stride];
		}
		__syncthreads();
	}

	//work efficient post reduction phase
	for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		
		__syncthreads();
		
		unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
		
		if (index + stride < SCAN_SIZE && index + stride < sizeOfHisto) {
			
			XY[index + stride] = XY[index + stride] + XY[index];
		}
	}

	__syncthreads();

	if (i < SCAN_SIZE && i < sizeOfHisto) {

		cdf[i] = cdf[i] + XY[threadIdx.x];

	}
	if (i == 0) {
		printf("cdf[%d]: %u\n", i, cdf[i]);
	}
}


int main(int argc, char** argv) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	char* inputImageFile;
	char* outputImageFile;
	
	unsigned int imageSize;
	unsigned int histogramSize;
	unsigned int cdfSize;
	int sizeOfImage;

	float* hostInputImageData;
	float* hostOutputImageData;

	unsigned int* hostEqualized;
	unsigned int* hostCdf;
	unsigned int* hostHistogram;

	float* deviceInputImageData;
	float* deviceOutputImageData;

	unsigned int* deviceHistogram;
	unsigned int* deviceCdf;
	unsigned int* deviceEqualize;

	int iterator = 0;
	int pgmRow;//width
	int pgmCol;//heigth
	
	inputImageFile = argv[1];
	outputImageFile = argv[2];

	hostInputImageData = read_pgm_float(inputImageFile, &pgmRow, &pgmCol);

	imageSize = sizeof(float) * pgmRow * pgmCol;
	printf("imageSize:%u\n",imageSize);
	histogramSize = sizeof(unsigned int) * HISTOGRAM_LENGTH;
	printf("histogramSize:%u\n", histogramSize);
	cdfSize = sizeof(unsigned int) * HISTOGRAM_LENGTH;
	printf("cdfSize:%u\n", cdfSize);
	sizeOfImage = pgmRow * pgmCol;

	printf("Width: %d\n", pgmRow);
	printf("Heigth: %d\n", pgmCol);

	//In oreder to check values using cuda mem copy
	hostOutputImageData = (float*)malloc(imageSize);
	hostEqulizeData = (float*)malloc(sizeOfImage);
	hostHistogram = (unsigned int*)malloc(histogramSize);
	hostCdf = (unsigned int*)malloc(cdfSize);
	hostEqualized = (unsigned int*)malloc(cdfSize);

	cudaMalloc((void**)&deviceInputImageData,imageSize);
	cudaMalloc((void**)&deviceOutputImageData,imageSize);
	cudaMalloc((void**)&deviceHistogram,histogramSize);
	cudaMalloc((void**)&deviceCdf,cdfSize);
	cudaMalloc((void**)&deviceEqualize, cdfSize);

	cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((pgmRow * pgmCol - 1 ) / SCAN_SIZE + 1, 1, 1);
	dim3 dimGridHistogram((pgmRow * pgmCol - 1) / HISTOGRAM_LENGTH + 1, 1, 1);

	cudaEventRecord(start);

	printf("Kernel Executing\n");
	
	dim3 dimBlockH(HISTOGRAM_LENGTH, 1);
	dim3 dimGridH((pgmRow * pgmCol - 1) / SCAN_SIZE + 1, 1, 1);
	createHistogram << <dimGridH,dimBlock >> > (deviceHistogram,deviceInputImageData, pgmRow * pgmCol);

	//IN ORDER TO CHECK VALUES
	
	/*cudaMemcpy(hostHistogram, deviceHistogram, histogramSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 256; i++) {
		printf("histo[%d]: %d\n", i, hostHistogram[i]);
	}*/
	
	cumulativeDistribution << <dimGridHistogram, dimBlock>> > (deviceCdf, deviceHistogram, HISTOGRAM_LENGTH);

	//IN ORDER TO CHECK VALUES
	
	/*cudaMemcpy(hostCdf, deviceCdf, histogramSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 256; i++) {
		printf("cdf[%d]: %d\n", i, hostCdf[i]);
	}*/
	
	dim3 eqoGrid((pgmRow * pgmCol - 1) / SCAN_SIZE + 1, 1, 1);
	histogramEqualize << <eqoGrid, dimBlock >> > (deviceEqualize,deviceCdf, pgmRow * pgmCol);

	//IN ORDER TO CHECK VALUES
	
	cudaMemcpy(hostEqualized, deviceEqualize,cdfSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 256; i++) {
		printf("equalize[%d]: %d\n", i, hostEqualized[i]);
	}

	//In order to put equlized data into a host input array by searching each equlized data in inputData and write to the output data
	while (iterator < sizeOfImage) {

		hostOutputImageData[iterator] = (float)hostEqualized[(int)hostInputImageData[iterator]];
		iterator++;
	 }

	printf("Kernel terminate\n");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Timer : %lf\n", milliseconds);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	write_pgm_float(outputImageFile, hostOutputImageData, pgmRow, pgmCol);

	return 0;
}
