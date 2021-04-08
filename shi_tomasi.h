/*
	File: shi_tomasi.cu
	Author(s): 
		Austin Erck - University of the Pacific, ECPE 251, Spring 2021
	Description:
		This program implements Shi Tomasi Feature Detection using NVIDIA's CUDA framework. 
*/

#ifndef SHI_TOMASI_GPU_H
#define SHI_TOMASI_GPU_H

#include <stdint.h>

template <typename T> 
struct LocationData {
	T data;
	int x;
	int y;

	__host__ __device__
	bool operator < (const LocationData<T>& B) const {
		return data > B.data;
	}

	/*__host__ __device__
	bool operator()(const LocationData<T>& A, const LocationData<T>& B) {
		return (A.data > B.data);
	}*/
};

/**
*   Performs feature finding using the Shi-Tomasi algorihm. Output is saved to the current directory
*   
*   \param filepath Full file path to image
*	\param sigma Sigma value used to generate gaussian kernels
*   \param sensitivity Percent of total image size that will determine many features will be considered
*	\param windowSize How large the window for eigenvalue calculation will be
*   \param blockSize Size of CUDA blocks
*   \param verbosity Should additional debugging output be provided
*   \param computationStart Pointer to timeval start
*   \param computationEnd Pointer to timeval end
*
**/
void shiTomasi(char* filepath, const float sigma, const float sensitivity, const unsigned int windowSize, const unsigned int blockSize, bool verbosity, struct timeval* computationStart, struct timeval* computationEnd);

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
*   \param G Pointer to save gaussian kernel results to
*   \param DG Pointer to save derivative gaussian kernel results to
*   \param width Pointer to save kernel width to
*   \param sigma Sigma value provided by user
*
**/
void generateKernels(float** G, float** DG, int* width, const float sigma);

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
void generateLocationData(const float* array, LocationData<float>* wrappedArray, const int imageWidth);

/**
*   Boxes around a given pixel on top of the provided image
*   
*	\param image Input image that results will be added to
*	\param x Feature x coordinate to be outlined
*	\param y Feature y coordinate to be outlined
*   \param imageWidth Input & output image width
*   \param imageHeight Input & output image height
*
**/
void drawBox(float* image, const int x, const int y, const int imageWidth, const int imageHeight);

/**
*   Uses pre-sorted descending ordered wrapped eigenvalues and draws a box around the most prominent features that are at least 8 pixels apart
*   
*	\param image Input image that results will be added to
*	\param wrappedEigenvalues Array data to wrap
*   \param imageWidth Input & output image width
*   \param imageHeight Input & output image height
*	\param sensitivity Percentage used to limit the amount of features that will be considered (should default to 0.1)
*
**/
void findFeatures(float* image, const LocationData<float>* wrappedEigenvalues, const int imageWidth, const int imageHeight, const float sensitivity);

#endif