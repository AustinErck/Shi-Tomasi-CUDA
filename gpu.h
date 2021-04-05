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

typedef struct FloatWrap {
	float data;
	int x;
	int y;
} FloatWrap;

/**
*   Sort function to properly sort FloatWrap arrays
*   
*   \param A First FloatWrap
*	\param B Second FloatWrap
*   \return If A should be sorted after B
*
**/
bool FloatWrap_sort(FloatWrap A, FloatWrap B);

/**
*   Calculates the number of microseconds between two timeval structures
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
*	\param horizontalImage Horizontal input image
*	\param verticalImage Vertical input image
*   \param eigenvalues Output image with eigen values
*   \param imageWidth Input & output image width
*   \param imageHeight Input & output image height
*   \param windowSize Dimension of square window used to calculate eigenvalues
*
**/
__global__
void computeEigenvalues(const float* horizontalImage, const float* verticalImage, float* eigenvalues, const int imageWidth, const int imageHeight, const int windowSize);

/**
*   Combines float array data with local positions into wrappedArray using CUDA
*   
*	\param array Array data to wrap
*	\param wrappedArray Output array
*   \param imageWidth Input & output image width
*   \param imageHeight Input & output image height
*
**/
__global__
void wrapFloatArray(const float* array, FloatWrap* wrappedArray, const int imageWidth, const int imageHeight);

/**
*   TODO
*   
*	\param inputImage Input image that results will be added to
*	\param wrappedEigenvalues Array data to wrap
*	\param outputImage Output image
*   \param imageWidth Input & output image width
*   \param imageHeight Input & output image height
*	\param sensitivity Percentage used to limit the amount of features that will be considered (should default to 0.1)
*
**/
__global__
void findFeatures(const FloatWrap* wrappedEigenvalues, float* outputImage, const int imageWidth, const int imageHeight, const float sensitivity);

#endif