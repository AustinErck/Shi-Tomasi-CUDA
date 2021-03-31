/*
	File: serialprogram.c
    Author(s): 
		Yang Liu - University of the Pacific, ECPE 293, Spring 2017
		Cody Balos - University of the Pacific, ECPE 293, Spring 2017
	Description:
    	This program implements Shi Tomasi Feature Detection.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "image_template_serial.h"
#include "serialprogram.h"

#define RADIUS_OF_FEATURE_MARKER 8

int main(int argc, char **argv){
	// Start the total execution timer which times entire program execution.
	struct timeval start, end;
	gettimeofday(&start, NULL);

	// User provided arguments

	// path to the image to process
	char *filepath = NULL;
	// how much information should the program print to the console
    int verbose_lvl = 0;
	// sigma of the gaussian distribution
    float sigma = 1.1;
	// size of a pixel 'neighborhood'
	int windowsize = 4;
	// # of features = sensitivity*image_width
	float sensitivity = 0.1;

	// argument parsing logic
    if (argc > 1) {
        if (!strcmp(argv[1], "-h")) {
            help(NULL);
        }
        else if (!strcmp(argv[1], "-v")) {
            verbose_lvl = 1;
            filepath = argv[2];
            if (argc >= 4)
                sigma = atof(argv[3]);
			if (argc >= 5)
				windowsize = atof(argv[4]);
			if (argc >= 6)
				sensitivity = atof(argv[5]);
        }
        else if (!strcmp(argv[1], "-vv")) {
            verbose_lvl = 2;
            filepath = argv[2];
            if (argc >= 4)
                sigma = atof(argv[3]);
			if (argc >= 5)
				windowsize = atof(argv[4]);
			if (argc >= 6)
				sensitivity = atof(argv[5]);
        }
        else {
            filepath = argv[1];
            if (argc >= 3)
                sigma = atof(argv[2]);
			if (argc >= 4)
				windowsize = atof(argv[3]);
			if (argc >= 5)
				sensitivity = atof(argv[4]);
        }
    } else {
        help("You must provide the path to the image to process.");
    }

	if (verbose_lvl > 0) {
		printf("detecting features for %s\n", filepath);
		printf("sigma = %0.3f, windowsize = %d, sensitivity = %0.3f\n", sigma, windowsize, sensitivity);
	}

	int width;
	int height;
	int kernel_width;
	int a;

	// calculate kernel width based on sigma
	a = (int)round(2.5 * sigma -.5);
	kernel_width = 2 * a + 1;

	// malloc and generate the kernels
	float *gkernel = (float *)malloc(sizeof(float) * kernel_width);
	float *dkernel = (float *)malloc(sizeof(float) * kernel_width);
	gen_kernel(gkernel, dkernel, sigma, a, kernel_width);

	// malloc and read the image to be processed
	float *original_image;
	read_imagef(filepath, &original_image, &width, &height);

	// create hgrad and vgrad and temp
	float *hgrad = (float *)malloc(sizeof(float) * width * height);
	float *vgrad =  (float *)malloc(sizeof(float) * width * height);
	float *tmp_image = (float *)malloc(sizeof(float) * width * height);
	float *eigenvalues = (float *)malloc(sizeof(float) * width * height);

	// Start the core tasks timers immediately prior to performing convolution.
	struct timeval core_start, core_end;
	gettimeofday(&core_start, NULL);

	// convolve to get the vgrad and hgrad
	convolve(gkernel, original_image, tmp_image, width, height, kernel_width, 1, a);
	convolve(dkernel, tmp_image, vgrad, width, height, 1, kernel_width, a);
	convolve(gkernel, original_image,tmp_image, width, height, 1, kernel_width, a);
	convolve(dkernel, tmp_image, hgrad, width, height, kernel_width, 1, a);
	free(tmp_image);
	free(gkernel);
	free(dkernel);

	// do the eigenvalues and detection here
	compute_eigenvalues(hgrad, vgrad, height, width, windowsize, eigenvalues);
	free(hgrad);
	free(vgrad);
	find_features(eigenvalues, sensitivity, width, height, original_image);

	// Set the end time as time to complete the core tasks.
	gettimeofday(&core_end, NULL);

	// Now we write the output.
	char corner_image[30];
	sprintf(corner_image, "corners.pgm");
	write_imagef(corner_image, original_image, width, height);

	// Free stuff leftover.
	free(original_image);
	free(eigenvalues);

	// End the total execution timer.
	gettimeofday(&end, NULL);

	// Print out final program stats.
	printf("core execution time: %ldms\n", get_elapsed_time(core_start, core_end));
	printf("total execution time: %ldms\n", get_elapsed_time(start, end));

	return 0;
}

void drawbox(float *image, int i, int j, int image_width, int image_height){
	int radius = image_width*0.0025;
	for (int k = -1 * radius; k <= radius; k++ ){
		for (int m = -1 * radius; m <= radius; m++){
			if ((i + k) >= 0 && (i + k) < image_height && (j + m) >= 0 && (j+m) <image_width){
				image[(i + k) * image_width + (j+m) ] = 0;
			}
		}
	}
}

void find_features(float *eigenvalues, float sensitivity, int image_width, int image_height, float *output_image){
	size_t image_size = image_height*image_width;

	// Use data_wrapper to pair each eigenvalue with its index so that we can use
	// the index after it the eigenvalues are sorted.
	data_wrapper_t *wrapped_eigs = malloc(sizeof(data_wrapper_t)*image_size);
	for (int i = 0; i < image_height; ++i) {
		for (int j = 0; j < image_width; ++j) {
			data_wrapper_t wrapper;
			wrapper.data = eigenvalues[i * image_width + j];
			wrapper.x = i;
			wrapper.y = j;
			wrapped_eigs[i * image_width + j] = wrapper;
		}
	}

	// Sort eigenvalues in descending order while keeping their corresponding pixel index in the image.
	qsort(wrapped_eigs, image_height*image_width, sizeof *wrapped_eigs, data_wrapper_compare);
	
	// Create the features buffer based on the sensitivity value (acts as a percentage of the image size).
	int max_features = ceil(image_width*sensitivity);
	data_wrapper_t *features = malloc(sizeof(data_wrapper_t)*max_features);

	// Set the first feature so we have a starting point.
	features[0] = wrapped_eigs[0];
	int features_count = 1;

	// Fill the features buffer!
	for (int i = 0; i < image_size && features_count < max_features; ++i) {
		// Check if prospective feature is more than 8 manhattan distance away from any existing feature.
		for (int j = 0; j < features_count; ++j) {
			int manhattan = abs(features[j].x - wrapped_eigs[i].x) + abs(features[j].y - wrapped_eigs[i].y);
			if (manhattan > 8) { 
				features[features_count] = wrapped_eigs[i];
				features_count++;
				break;
			}
		}
	}

	for (int i = 0; i < features_count; i++)
		drawbox(output_image, features[i].x, features[i].y, image_width, image_height);
	
	free(wrapped_eigs);
	free(features);
}

int data_wrapper_compare(const void *a, const void *b){
	const data_wrapper_t *aa = (const data_wrapper_t *) a;
	const data_wrapper_t *bb = (const data_wrapper_t *) b;
	return (aa->data < bb->data) - (aa->data > bb->data);
}

void compute_eigenvalues(float *hgrad, float *vgrad, int image_height, int image_width, int windowsize, float *eigenvalues){
	int w = floor(windowsize/2);

 	int i, j, k, m, offseti, offsetj;
	float ixx_sum, iyy_sum, ixiy_sum;

	for (i = 0; i < image_height; i++){
		for (j = 0; j < image_width; j++){
			ixx_sum = 0;
			iyy_sum = 0;
			ixiy_sum = 0;

			for (k = 0; k < windowsize; k++){
				for (m = 0; m < windowsize; m++){
					offseti = -1 * w + k;
					offsetj = -1 * w + m;
					if (i+offseti >= 0 && i+offseti < image_height && j + offsetj >= 0 && j+offsetj < image_width){
						ixx_sum += hgrad[(i +offseti) * image_width  + (j + offsetj)] * hgrad[(i +offseti) * image_width  + (j + offsetj)];
						iyy_sum += vgrad[(i +offseti) * image_width  + (j + offsetj)] * vgrad[(i +offseti) * image_width  + (j + offsetj)];
						ixiy_sum += hgrad[(i +offseti) * image_width  + (j + offsetj)] * vgrad[(i +offseti) * image_width  + (j + offsetj)];
					}
				}
			}

			eigenvalues[i * image_width + j] = min_eigenvalue(ixx_sum, ixiy_sum, ixiy_sum, iyy_sum);
		}
	}
}

float min_eigenvalue(float a, float b, float c, float d){
	float ev_one = (a + d)/2 + pow(((a + d) * (a + d))/4 - (a * d - b * c), 0.5);
	float ev_two = (a + d)/2 - pow(((a + d) * (a + d))/4 - (a * d - b * c), 0.5);
	if (ev_one >= ev_two){
		return ev_two;
	}
	else{
		return ev_one;
	}
}

void convolve(float *kernel, float *image, float *resultimage, int image_width, int image_height, int kernel_width, int kernel_height, int half){
	float sum;
	int i, j, k, m, offsetj, offseti;
	//assign the kernel to the new array
	for (i = 0; i < image_height; i++){
		for (j = 0; j < image_width; j++){

			//reset tracker
			sum= 0.0;
			//for each item in the kernel
			for (k = 0; k < kernel_height; k++){
				for (m = 0; m < kernel_width; m++){
					offseti = -1 * (kernel_height/2) + k;
					offsetj = -1 * (kernel_width/2) + m;
					if (i+offseti >= 0 && i+offseti < image_height && j + offsetj >= 0 && j+offsetj < image_width){
						sum+=(float)(image[(i+offseti) * image_width + (j+offsetj)])*kernel[k*kernel_width +m];
					}
				}
			}
			//copy it back
			resultimage[i * image_width + j] = sum;
		}
	}
}

void gen_kernel(float *gkernel, float *dkernel, float sigma, int a, int w){
	int i;
	float sum_gkern;
	float sum_dkern;
	sum_gkern= 0;
	sum_dkern= 0;
	for(i = 0; i < w; i++){
		gkernel[i] = (float)exp( (float)(-1.0 * (i-a) * (i-a)) / (2 * sigma * sigma));
		dkernel[i] = (float)(-1 * (i - a)) * (float)exp( (float)(-1.0 * (i-a) * (i-a)) / (2 * sigma * sigma));
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

void help(const char *err) 
{
    if (err != NULL)
        printf("%s\n", err);
    printf("usage: ./goodfeatures [-v,-vv] <full path to the image> [sigma] [windowsize] [sensitivity] \n");
    printf("flags:\n");
    printf("\t-h: show this help menu\n");
    printf("\t-v: output basic execution information\n");
    printf("\t-vv: output all information... good for debugging\n");
	printf("arguments:\n");
	printf("\tsigma: the sigma value for the Gaussian distribution used to form the convolution mask.\n");
	printf("\twindowsize: the size of a pixel 'neighborhood' in an image\n");
	printf("\tsensitivity: determines the amount of features to detect... can be 0.0 to 1.0\n");
    exit(0);
}

long get_elapsed_time(struct timeval start, struct timeval end)
{
	return ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))/1000.0;
}
