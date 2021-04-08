/*
	File: main.cu
	Author(s): 
		Austin Erck - University of the Pacific, ECPE 251, Spring 2021
	Description:
		This program implements Shi Tomasi Feature Detection using NVIDIA's CUDA framework. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "shi_tomasi.h"

int main(int argc, char **argv) {

	// Check that image path and sigma are provided
	if( argc != 5 ) {
		printf("Please provide an image path, sigma value in the following format:\n./goodfeatures <full path to image> [sigma] [window size] [sensitivity] [block size]\n");
		return 1;
	}

	// Get image path
	char* filepath = argv[1];

	// Get sigma value
	char* sigmaConversionError;
	char* sigmaString = argv[2];
	float sigma = (float)strtod(sigmaString, &sigmaConversionError);
	if (*sigmaConversionError != 0) {
		printf("Fatal Error: Invalid sigma value provided. Must be numerial value\n");
		return 1;
	}

	// Get window size value
	int windowSize = atoi(argv[3]);

	// Get sigma value
	char* sensitivityConversionError;
	char* sensitivityString = argv[4];
	float sensitivity = (float)strtod(sensitivityString, &sensitivityConversionError);
	if (*sensitivityConversionError != 0) {
		printf("Fatal Error: Invalid sensitivity value provided. Must be numerial value\n");
		return 1;
	}

	// Get block size size value
	int blockSize = atoi(argv[2]);

	// Setup timers
	struct timeval computationStart, computationEnd;

	// Run algorihm
	shiTomasi(filepath, sigma, sensitivity, windowSize, blockSize, &computationStart, &computationEnd);

	// Print benchmarching information
	printf("%d, %f, %d, %d, %f, %Lf\n", 0, sigma, blockSize, windowSize, sensitivity, calculateTime(computationStart, computationEnd));

	return 0;
};
