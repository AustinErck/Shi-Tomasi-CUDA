/*
	File: gpu.cu
    Author(s): 
		Austin Erck - University of the Pacific, ECPE 251, Spring 2021
	Description:
    	This program implements Shi Tomasi Feature Detection using NVIDIA's CUDA framework. 
*/

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <argp.h>
#include <cuda.h>
#include "gpu.h"
#include "image_template.h"

#define RADIUS_OF_FEATURE_MARKER 8

int main(int argc, char **argv){

	// Handle arguments
	// TODO: Read from argv
	char *filepath = NULL;
	uint8_t verbosity = 0; // Determines how much information should be shown
	float sigma = 1.1; // Sigma of the gaussian distribution
	uint64_t blockSize = 16; // CUDA block size
	uint64_t windowSize = 4; // Size of a pixel 'neighborhood'
	float sensitivity = 0.1; // Number of features = sensitivity*image_width

	// Setup timers
    struct timeval computationStart, computationEnd;

	// Setup CUDA pointers
	float *h_data1; //host pointers
	float *d_data1; //device pointers

	// Read image into first data array
	int width = 0, height = 0;
	//const float &initialImage = h1_data;
	read_image_template(filepath, &h_data1, &width, &height); // h_data1 = initialImage

	// Generate kernels
	int GWidth, dGWidth;
	//gaussian(sigma, &h_G, &GWidth);
	//gaussianDerivative(sigma, &h_dG, &dGWidth);

	// Malloc data on devices
	cudaMalloc((void **)&d_data1, sizeof(float) * width * height);

	// Begin computation timer
    gettimeofday(&computationStart, NULL);

	// Populate data on devices
	cudaMemcpy(d_data1, h_data1, sizeof(float) * width * height, cudaMemcpyHostToDevice);

	// Setup CUDA grid and blocks
    dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(width/blockSize, height/blockSize); // ASSUMPTION: Image is divisible by 16

	// TODO: Image processing

	// Copy data from device to host
	cudaMemcpy(h_data1, d_data1, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	// Sync CUDA threads and measure computation time
    cudaDeviceSynchronize();
    gettimeofday(&computationEnd, NULL);

    // Save output image to disk
    const char outputFilename[] = "corners.pgm";
    write_image_template(outputFilename, h_data1, width, height);

    // Free data from host and devices
    free(h_data1);
    cudaFree(d_data1);

    // Print benchmarching information
	printf("%d, %f, %x, %x, %f, %Lf\n", width, sigma, blockSize, windowSize, sensitivity, calculateTime(computationStart, computationEnd));

	return 0;
}

long double calculateTime(struct timeval start, struct timeval end) {
	return (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);	
}