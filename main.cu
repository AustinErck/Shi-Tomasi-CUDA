/*
	File: main.cu
	Author(s): 
		Austin Erck - University of the Pacific, ECPE 251, Spring 2021
	Description:
		This program implements Shi Tomasi Feature Detection using NVIDIA's CUDA framework. 
*/

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <argp.h>
#include "shi_tomasi.h"

int main(int argc, char **argv) {

	// Handle arguments
	// TODO: Read from argv
	char *filepath = NULL;
	uint8_t verbosity = 0; // Determines how much information should be shown
	float sigma = 1.1; // Sigma of the gaussian distribution
	uint8_t blockSize = 16; // CUDA block size
	uint8_t windowSize = 4; // Size of a pixel 'neighborhood'
	float sensitivity = 0.1; // Number of features = sensitivity*image_width

    // Setup timers
    struct timeval computationStart, computationEnd;

    // Run algorihm
	shiTomasi(filepath, sigma, sensitivity, windowSize, blockSize, verbosity, computationStart, computationEnd);

	// Print benchmarching information
	//printf("%d, %f, %x, %x, %f, %Lf\n", 0, sigma, blockSize, windowSize, sensitivity, calculateTime(computationStart, computationEnd));

	return 0;
};
