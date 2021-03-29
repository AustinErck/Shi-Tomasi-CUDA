/*
	File: serialprogram.h
    Author(s): 
		Yang Liu - University of the Pacific, ECPE 293, Spring 2017
		Cody Balos - University of the Pacific, ECPE 293, Spring 2017
	Description:
    	Declares the functions used for the serial Shi Tomasi feature detection program.
*/

#ifndef SHI_TOMASI_H
#define SHI_TOMASI_H

#define BUFFER 512

typedef struct data_wrapper_t {
	float data;
	size_t x;
	size_t y;
} data_wrapper_t;

/// Draws a box at specfied location in the image. Used for markgin features.
void drawbox(float *image, int i, int j, int image_width, int image_height);

/// Find features in an image.
void find_features(float *eigenvalues, float sensitivity, int image_width, int image_height, float *output_image);

/// Defines comparison for data_wrapper_t. When used with qsort, it will result in a descending order array.
int data_wrapper_compare(const void *a, const void *b);

/// Compute the eigenvalues of a pixel's Z matrix.
void compute_eigenvalues(float *hgrad, float *vgrad, int image_height, int image_width, int windowsize, float *eigenvalues);

/// Calculate the minimum eigenvalue.
float min_eigenvalue(float a, float b, float c, float d);

/// Produce the images horizontal and vertical gradients.
void convolve(float *kernel, float *image, float *resultimage, int image_width, int image_height, int kernel_width, int kernel_height, int half);

/// Creates Gaussian kernel and Gaussian derivative kernel for image gradient/convolution procedure.
void gen_kernel(float *gkernel, float *dkernel, float sigma, int a, int w);

/// Prints out the program help menu.
void help(const char *err);

/// Returns the elapsed time between two timeval in milliseconds.
long get_elapsed_time(struct timeval start, struct timeval end);

#endif