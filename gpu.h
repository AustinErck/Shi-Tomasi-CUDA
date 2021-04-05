/*
	File: serialprogram.h
    Author(s): 
		Yang Liu - University of the Pacific, ECPE 293, Spring 2017
		Cody Balos - University of the Pacific, ECPE 293, Spring 2017
	Description:
    	Declares the functions used for the serial Shi Tomasi feature detection program.
*/

#ifndef SHI_TOMASI_GPU_H
#define SHI_TOMASI_GPU_H

/**
*   Calclates the number of microseconds between two timeval structures
*   
*   \param start Beginning of the measured time period
*	\param end End of the measured time period
*   \return long double
*
**/
long double calculateTime(struct timeval start, struct timeval end);

/**
*   Generates guassian and derivative guassian kernels needed to perform convolution in Shi-Tomasi
*   
*   \param start Beginning of the measured time period
*	\param end End of the measured time period
*   \return long double
*
**/
void generateKernels(float* G, float* DG, int* width, const float sigma);

/**
*   Performs a convolve operation using CUDA and optimized with a shared memory implementation.
*	*Assumes that the provided kernel has odd dimensions
*   
*   \param image Input image
*	\param outputImage Output image
*   \param imageWidth Input & output image width
*   \param imageHeight Input & output image height
*   \param kernel Pointer to kernel
*   \param kernelWidth Kernel width
*   \param kernelHeight Kernel height
*
**/
__global__
void convolve(const float* image, float* outputImage, const int imageWidth, const int imageHeight, const float* kernel, const int kernelWidth, const int kernelHeight);

/**
*   Calculates eigenvalues for horizontal and vertical images and returns an image with the lowest value for each pixel. Uses CUDA and optimized with a shared memory implementation.
*	*Assumes that the provided windowSize has odd dimensions
*   
*   \param eigenValues Output image with eigen values
*	\param horizontalImage Horizontal input image
*	\param verticalImage Vertical input image
*   \param imageWidth Input & output image width
*   \param imageHeight Input & output image height
*   \param windowSize Dimension of square window used to calculate eigenvalues
*
**/
__global__
void computeEigenValues(float* eigenValues, const float* horizontalImage, const float* verticalImage, const int imageWidth, const int imageHeight, const int windowSize);

#endif