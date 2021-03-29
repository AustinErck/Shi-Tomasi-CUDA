/*
	File: gpu.cu
    Author(s): 
		Austin Erck - University of the Pacific, ECPE 251, Spring 2021
	Description:
    	This program implements Shi Tomasi Feature Detection using NVIDIA's CUDA framework. 
*/

#include <argp.h>
#include "image_template.h"

#define RADIUS_OF_FEATURE_MARKER 8

int main(int argc, char **argv){

	// Handle arguments
	char *filepath = NULL;
    uint8_t verbosity = 0; // Determines how much information should be shown
    float sigma = 1.1; // Sigma of the gaussian distribution
	uint64_t windowsize = 4; // Size of a pixel 'neighborhood'
	float sensitivity = 0.1; // Number of features = sensitivity*image_width

	// Setup CUDA pointers
    float *h_data1; //host pointers
	float *d_data1; //device pointers

	// Read image into first data array
    int width = 0, height = 0;
    read_image_template(filepath, &h_data1, &width, &height); // h_data1 = initialImage

    // Generate kernels
	float *gkernel = (float *)malloc(sizeof(float) * kernel_width);
	float *dkernel = (float *)malloc(sizeof(float) * kernel_width);
	gen_kernel(gkernel, dkernel, sigma, a, kernel_width);

}

void gaussian(const float sigma, float** kernel, int* width) {
    
    float a = roundf(2.5 * sigma - 0.5);
    float w = 2 * a + 1;
    float* G = (float*)malloc(w * sizeof(float));
    float sum = 0;

    //printf("G_w: %f\n", w);
    int i;
    for(i = 0; i < w; i++) {
        G[i] = expf(-1.0 * powf((float)(i) - a, 2.0) / (2.0 * powf(sigma, 2.0)));
        sum += G[i];
    }

    // Normalize gaussian kernel peaks
    for(i = 0; i < w; i++) {
        G[i] = G[i] / sum;
        //printf("G[%d]: %f\n", i, G[i]);
    }

    // Set results to pointers
    *kernel = G;
    *width = (int)w;
}

void gen_kernel(float *gkernel, float *dkernel, float sigma, int a, int w){
	a = (int)round(2.5 * sigma -.5);
	kernel_width = 2 * a + 1;
	
	float sum_gkern = 0;
	float sum_dkern = 0;

	int i;
	for(i = 0; i < w; i++){
		gkernel[i] = (float)exp( (float)(-1.0 * (i-a) * (i-a)) / (2 * sigma * sigma));

		dkernel[i] = (-1 * (i - a)) * );
		 G[i] = expf(-1.0 * );
		sum_gkern = sum_gkern + gkernel[i];
		sum_dkern = sum_dkern - (float)i * dkernel[i];
	}

	//reverse the kernel by creating a new kernel, yes not ideal
	float *newkernel = (float *)malloc(sizeof(float) * w);
	for (i = 0; i < w; i++){
		dkernel[i] = dkernel[i] / sum_dkern;
		gkernel[i] = gkernel[i] / sum_gkern;
		newkernel[w-i] = dkernel[i];
	}

	//copy new kernel back in
	for (i = 0; i < w; i++){
		dkernel[i] = newkernel[i+1];
	}
	free(newkernel);
}