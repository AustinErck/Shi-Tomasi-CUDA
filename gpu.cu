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
#include "image_template.h"
#include "gpu.h"

#define RADIUS_OF_FEATURE_MARKER 8

int main(int argc, char **argv){

	// Handle arguments
	// TODO: Read from argv
	char *filepath = NULL;
	int verbosity = 0; // Determines how much information should be shown
	float sigma = 1.1; // Sigma of the gaussian distribution
	int blockSize = 16; // CUDA block size
	int windowSize = 4; // Size of a pixel 'neighborhood'
	float sensitivity = 0.1; // Number of features = sensitivity*image_width

	// Setup timers
    struct timeval computationStart, computationEnd;

	// Setup CUDA pointers
	float *h_data1, *h_G, *h_DG; //host pointers
	float *d_data1, *d_data2, *d_data3, *d_G, *d_DG; //device pointers

	// Read image into first data array
	int width = 0, height = 0;
	//const float &initialImage = h1_data;
	read_image_template(filepath, &h_data1, &width, &height); // h_data1 = initialImage

	// Calculate constants
	const int bytesPerImage = sizeof(float) * width * height;
	const int bytesPerBlock = sizeof(float) * blockSize * blockSize;

	// Setup CUDA grid and blocks based on image size
	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(width/blockSize, height/blockSize); // ASSUMPTION: Image is divisible by 16

	// Generate kernels
	int kernelWidth;
	generateKernels(h_G, h_DG, &kernelWidth, sigma);

	// Malloc data on devices
	cudaMalloc((void **)&d_data1, bytesPerImage);
	cudaMalloc((void **)&d_data2, bytesPerImage);
	cudaMalloc((void **)&d_data3, bytesPerImage);
	cudaMalloc((void **)&d_G, sizeof(float) * kernelWidth);
	cudaMalloc((void **)&d_DG, sizeof(float) * kernelWidth);

	// Begin computation timer
    gettimeofday(&computationStart, NULL);

	// Populate data on devices from host
	cudaMemcpy(d_data1, h_data1, bytesPerImage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_G, h_G, sizeof(float) * kernelWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_DG, h_DG, sizeof(float) * kernelWidth, cudaMemcpyHostToDevice);

	// Temp Horizontal/Vertical convolutions
	convolve<<<dimGrid,dimBlock, bytesPerBlock>>>(d_data1, d_data2, width, height, d_G, 1, kernelWidth); // data1 = temp_horizontal
    convolve<<<dimGrid,dimBlock, bytesPerBlock>>>(d_data1, d_data3, width, height, d_G, kernelWidth, 1); // data2 = temp_vertical
   
    // Horizontal/Vertical convolutions
    convolve<<<dimGrid,dimBlock, bytesPerBlock>>>(d_data2, d_data1, width, height, d_DG, kernelWidth, 1); // data1 = horizontal
    convolve<<<dimGrid,dimBlock, bytesPerBlock>>>(d_data3, d_data2, width, height, d_DG, 1, kernelWidth); // data2 = vertical

	// Compute eigen values
	computeEigenValues<<<dimGrid,dimBlock, bytesPerBlock * 2>>>(d_data1, d_data2, d_data3, width, height, windowSize); // d_data3 = eigenValues

	// TODO: Find features

	// Copy data from device to host
	cudaMemcpy(h_data1, d_data1, bytesPerImage, cudaMemcpyDeviceToHost);

	// Sync CUDA threads and measure computation time
    cudaDeviceSynchronize();
    gettimeofday(&computationEnd, NULL);

    // Save output image to disk
    char outputFilename[] = "corners.pgm";
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

void generateKernels(float* G, float* DG, int* width, const float sigma){

	// Calculate a and w(idth) variables used in guassian and derivative guassian kernel calculations
	float a = roundf(2.5 * sigma - 0.5);
    int w = 2 * a + 1;

	// Track total of all values used in each kernel
    float sumG = 0, sumDG = 0;

	// Loop through the width of the kernel and populate kernels while calculating the sum of each
	int i;
	for(i = 0; i < w; i++) {
        G[i] = expf(-1.0 * powf((float)(i) - a, 2.0) / (2.0 * powf(sigma, 2.0)));
		DG[i] = -1.0 * ( (float)(i + 1) - 1.0 - a) * expf(-1.0 * powf((float)(i + 1) - 1.0 - a, 2.0) / (2.0 * powf(sigma, 2.0)));
        sumG += G[i];
		sumDG -= i * DG[i];
    }

	// Divide each value in the kernel by the total sum of the kernel
	for (i = 0; i < w; i++){
		G[i] = G[i] / sumG;
		DG[i] = DG[i] / sumDG;
	}

	// Flip derivative kernel
	for (i = 0; i < w/2; i++) {
        const float temp = DG[w - i - 1];
        DG[w - i - 1] = DG[i];
        DG[i] = temp;
    }
}

__global__
void convolve(const float* image, float* outputImage, const int imageWidth, const int imageHeight, const float* kernel, const int kernelWidth, const int kernelHeight) {

    // Calculate kernel center constants
    const int kernelCenterX = kernelWidth / 2;
    const int kernelCenterY = kernelHeight / 2;
    
    // Set initial pixel value to zero
    float sum = 0;

    // Get x and y based on thread and block index
    const int xBlockOffset = blockIdx.x * blockDim.x;
    const int yBlockOffset = blockIdx.y * blockDim.y;
    const int xLocal = threadIdx.x;
    const int yLocal = threadIdx.y;
    const int xGlobal = xLocal + xBlockOffset;
	const int yGlobal = yLocal + yBlockOffset;

    // Setup shared data array
    extern __shared__ float blockData[];
    blockData[yLocal * blockDim.x + xLocal] = image[yGlobal * imageWidth + xGlobal];
    __syncthreads();

    // Loop through each pixel of the   kernel
    int i, j;
    for(j = 0; j < kernelHeight; j++) {
        for(i = 0; i < kernelWidth; i++) {
        
            // Calculate offset based on current pixel in kernel
            int xCalculated = xGlobal + (i - kernelCenterX);
            int yCalculated = yGlobal + (j - kernelCenterY);
            
            // Check that pixel is not out of bounds
            if(xCalculated < 0 || xCalculated >= imageWidth || yCalculated < 0 || yCalculated >= imageHeight) {
                continue;
            }

            // Add image value multipled by kernel value to sum
			// Both cases perform the same action, however if possible local data(blockData) is used instead of global data(image)
            if(xCalculated >= xBlockOffset && xCalculated < xBlockOffset + blockDim.x && yCalculated >= yBlockOffset && yCalculated < yBlockOffset + blockDim.y) {
                // Go to block data

                // Calculate part of the convolve value based on image and kernel pixel.
                sum += kernel[j * kernelWidth + i] * blockData[(yCalculated - yBlockOffset) * blockDim.x + (xCalculated - xBlockOffset)];
            } else {
                // Go to global data

                // Calculate part of the convolve value based on image and kernel pixel.
                sum += kernel[j * kernelWidth + i] * image[yCalculated * imageWidth + xCalculated];
            }
        }
    }

    // Write sum to memory
    outputImage[yGlobal * imageWidth + xGlobal] = sum;
}

__global__
void computeEigenValues(const float* horizontalImage, const float* verticalImage, float* eigenValues, const int imageWidth, const int imageHeight, const int windowSize) {

	// Calculate window center constant
    const int windowCenter = windowSize / 2;
    
    // Set initial pixel value to zero
    float sumIXX = 0, sumIYY = 0, sumIXIY = 0;

    // Get x and y based on thread and block index
    const int xBlockOffset = blockIdx.x * blockDim.x;
    const int yBlockOffset = blockIdx.y * blockDim.y;
    const int xLocal = threadIdx.x;
    const int yLocal = threadIdx.y;
    const int xGlobal = xLocal + xBlockOffset;
	const int yGlobal = yLocal + yBlockOffset;

    // Setup shared data array
    extern __shared__ float sharedData[];
	float *horizontalImageLocal = (float*)&sharedData; // Use first half of shared memory for horizontal image
	float *verticalImageLocal = (float*)&sharedData + (sizeof(float) * imageWidth * imageHeight); // Use second half of shared memory for vertical image
    horizontalImageLocal[yLocal * blockDim.x + xLocal] = horizontalImage[yGlobal * imageWidth + xGlobal];
	verticalImageLocal[yLocal * blockDim.x + xLocal] = verticalImage[yGlobal * imageWidth + xGlobal];
    __syncthreads();

    // Loop through each pixel of the   kernel
    int i, j;
    for(j = 0; j < windowSize; j++) {
        for(i = 0; i < windowSize; i++) {
        
            // Calculate offset based on current pixel in kernel
            const int xCalculated = xGlobal + (i - windowCenter);
            const int yCalculated = yGlobal + (j - windowCenter);
            
            // Check that pixel is not out of bounds. Skip if it is
            if(xCalculated < 0 || xCalculated >= imageWidth || yCalculated < 0 || yCalculated >= imageHeight) {
                continue;
            }

            // Determine sum values for ixx, iyy, and ixiy
			// Both cases perform the same action, however if possible local data(horizontalImageLocal & verticalImageLocal) is used instead of global data(horizontalImage & verticalImage)
            if(xCalculated >= xBlockOffset && xCalculated < xBlockOffset + blockDim.x && yCalculated >= yBlockOffset && yCalculated < yBlockOffset + blockDim.y) {
                // Go to local data

				// Calculate array offset
				const int arrayOffset = (yCalculated - yBlockOffset) * blockDim.x + (xCalculated - xBlockOffset);

                // Calculate part of the convolve value based on image and kernel pixel.
                sumIXX += powf(horizontalImageLocal[arrayOffset], 2.0); // horizontalImage^2
				sumIYY += powf(verticalImageLocal[arrayOffset], 2.0); // verticalImage^2
				sumIXIY += horizontalImageLocal[arrayOffset] * verticalImageLocal[arrayOffset]; // horizontalImage * verticalImage
            } else {
                // Go to global data

				// Calculate part of the convolve value based on image and kernel pixel.
                sumIXX += powf(horizontalImage[yCalculated * imageWidth + xCalculated], 2.0); // horizontalImage^2
				sumIYY += powf(verticalImage[yCalculated * imageWidth + xCalculated], 2.0); // verticalImage^2
				sumIXIY += horizontalImage[yCalculated * imageWidth + xCalculated] * verticalImageLocal[yCalculated * imageWidth + xCalculated]; // horizontalImage * verticalImage
            }
        }
    }

	// Calculate eigen values
	const float temp1 = (sumIXX + sumIYY)/2;
	const float temp2 = powf( powf(sumIXX + sumIYY, 2.0)/4.0 - (sumIXX * sumIYY - powf(sumIXIY, 2.0)), 0.5);
	float eigenValue1 = temp1 + temp2;
	float eigenValue2 = temp1 - temp2;

	// Save smaller of the two eigen values
	eigenValues[j * imageWidth + i] = (eigenValue1 >= eigenValue2) ? eigenValue2 : eigenValue1;
}
