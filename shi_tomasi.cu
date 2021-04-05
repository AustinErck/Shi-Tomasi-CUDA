/*
	File: shi_tomasi.cu
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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include "shi_tomasi.h"
#include "image_template.h"

void shiTomasi(char* filepath, const float sigma, const float sensitivity, const unsigned int windowSize, const unsigned int blockSize, bool verbosity, struct timeval* computationStart, struct timeval* computationEnd) {

	// Setup CUDA pointers
	float *h_data1, *h_G, *h_DG; //host pointers
	float *d_data1, *d_data2, *d_data3, *d_G, *d_DG; //device pointers
	LocationData<float> *h_ld; // Additional host pointer
	LocationData<float> *d_ld; // Additional device pointer

	// Read image into first data array
	int width = 0, height = 0;
	read_image_template(filepath, &h_data1, &width, &height); // h_data1 = initialImage

	// Calculate constants
	const int imageSize = width * height;
	const int bytesPerImage = sizeof(float) * imageSize;
	const int bytesPerBlock = sizeof(float) * blockSize * blockSize;

	// Generate kernels
	int kernelWidth;
	generateKernels(&h_G, &h_DG, &kernelWidth, sigma);

	// Setup CUDA grid and blocks based on image size
	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(width/blockSize, height/blockSize); // ASSUMPTION: Image is divisible by 16
	printf("%d %d %d %d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

	// Malloc data on host
	h_ld = (LocationData<float>*)malloc(sizeof(LocationData<float>) * imageSize);

	// Malloc data on devices
	cudaMalloc((void **)&d_data1, bytesPerImage);
	cudaMalloc((void **)&d_data2, bytesPerImage);
	cudaMalloc((void **)&d_data3, bytesPerImage);
	cudaMalloc((void **)&d_G, sizeof(float) * kernelWidth);
	cudaMalloc((void **)&d_DG, sizeof(float) * kernelWidth);
	cudaMalloc((void **)&d_ld, sizeof(LocationData<float>) * imageSize);

	// Begin computation timer
	gettimeofday(computationStart, NULL);

	// Populate data on devices from host
	cudaMemcpy(d_data1, h_data1, bytesPerImage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_G, h_G, sizeof(float) * kernelWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_DG, h_DG, sizeof(float) * kernelWidth, cudaMemcpyHostToDevice);

	// Temp Horizontal/Vertical convolutions
	convolve<<<dimGrid, dimBlock, bytesPerBlock>>>(d_data1, d_data2, width, height, d_G, 1, kernelWidth); // data1(input) => data2(temp_horizontal)
	convolve<<<dimGrid, dimBlock, bytesPerBlock>>>(d_data1, d_data3, width, height, d_G, kernelWidth, 1); // data1(input) => data3(temp_vertical)

	// Horizontal/Vertical convolutions
	convolve<<<dimGrid, dimBlock, bytesPerBlock>>>(d_data2, d_data1, width, height, d_DG, kernelWidth, 1); // data2(temp_horizontal) => data1(horizontal)
	convolve<<<dimGrid, dimBlock, bytesPerBlock>>>(d_data3, d_data2, width, height, d_DG, 1, kernelWidth); // data3(temp_vertical) => data2(vertical)

	// Compute eigenvalues
	//computeEigenvalues<<<dimGrid, dimBlock, bytesPerBlock * 2>>>(d_data1, d_data2, d_data3, width, height, windowSize); // data1(horizontal), data2(vertical) => data3(eigenvalues)

	// Wrap eigenvalues with LocationData struct
	//generateLocationData<<<dimGrid, dimBlock>>>(d_data3, d_ld, width);

	// Sort array of wrapped eigenvalues
	//thrust::device_ptr< LocationData<float> > thr_d(d_ld);
	//thrust::device_vector< LocationData<float> >d_sortedLocationData(thr_d, thr_d + (height * width));
	//thrust::sort(d_sortedLocationData.begin(), d_sortedLocationData.end(), LocationData<float>());

	// Copy sorted LocationData array back to the host
	//cudaMemcpy(h_ld, d_ld, bytesPerImage, cudaMemcpyDeviceToHost);

	// Find features
	//findFeatures(h_data1, h_ld, width, height, sensitivity);

	// Measure computation time
	gettimeofday(computationEnd, NULL);

	// Save output image to disk
	char outputFilename[] = "shiTomasi_cuda.pgm";
	write_image_template(outputFilename, h_data1, width, height);

	// Free data from host and devices
	free(h_data1);
	free(h_G);
	//free(h_DG);
	//free(h_ld);
	cudaFree(d_data1);
	cudaFree(d_data2);
	//cudaFree(d_data3);
	cudaFree(d_G);
	//cudaFree(d_DG);
	//cudaFree(d_ld);
}

long double calculateTime(struct timeval start, struct timeval end) {
	return (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);	
}

void generateKernels(float** G, float** DG, int* width, const float sigma){

	// Calculate a and w(idth) variables used in guassian and derivative guassian kernel calculations
	const float a = roundf(2.5 * sigma - 0.5);
	const int w = (int)(2.0 * a + 1.0);

	// Malloc data on host for kernels
	float* g = (float*)malloc(w * sizeof(float));
	float* dg = (float*)malloc(w * sizeof(float));

	// Track total of all values used in each kernel
	float sumG = 0, sumDG = 0;

	// Loop through the width of the kernel and populate kernels while calculating the sum of each
	int i;
	for(i = 0; i < w; i++) {
		g[i] = expf(-1.0 * powf((float)(i) - a, 2.0) / (2.0 * powf(sigma, 2.0)));
		dg[i] = -1.0 * ( (float)(i + 1) - 1.0 - a) * expf(-1.0 * powf((float)(i + 1) - 1.0 - a, 2.0) / (2.0 * powf(sigma, 2.0)));
		sumG += g[i];
		sumDG -= i * dg[i];
	}

	// Divide each value in the kernel by the total sum of the kernel
	for (i = 0; i < w; i++){
		g[i] = g[i] / sumG;
		dg[i] = dg[i] / sumDG;
	}

	// Flip derivative kernel
	for (i = 0; i < w/2; i++) {
		const float temp = dg[w - i - 1];
		dg[w - i - 1] = dg[i];
		dg[i] = temp;
	}

	// Save kernels to provided pointers
	*width = w;
	*G = g;
	*DG = dg;
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
				//sum += kernel[j * kernelWidth + i] * blockData[(yCalculated - yBlockOffset) * blockDim.x + (xCalculated - xBlockOffset)];
			} else {
				// Go to global data

				// Calculate part of the convolve value based on image and kernel pixel.
				//sum += kernel[j * kernelWidth + i] * image[yCalculated * imageWidth + xCalculated];
			}
		}
	}

	// Write sum to memory
	outputImage[yGlobal * imageWidth + xGlobal] = sum;
}

__global__
void computeEigenvalues(const float* horizontalImage, const float* verticalImage, float* eigenvalues, const int imageWidth, const int imageHeight, const int windowSize) {

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

	// Calculate eigenvalues
	const float temp1 = (sumIXX + sumIYY)/2;
	const float temp2 = powf( powf(sumIXX + sumIYY, 2.0)/4.0 - (sumIXX * sumIYY - powf(sumIXIY, 2.0)), 0.5);
	float eigenvalue1 = temp1 + temp2;
	float eigenvalue2 = temp1 - temp2;

	// Save smaller of the two eigenvalues
	eigenvalues[yGlobal * imageWidth + xGlobal] = (eigenvalue1 >= eigenvalue2) ? eigenvalue2 : eigenvalue1;
}

__global__
void generateLocationData(const float* array, LocationData<float>* wrappedArray, const int imageWidth) {
	
	// Get x and y based on thread and block index
	const int xBlockOffset = blockIdx.x * blockDim.x;
	const int yBlockOffset = blockIdx.y * blockDim.y;
	const int xLocal = threadIdx.x;
	const int yLocal = threadIdx.y;
	const int xGlobal = xLocal + xBlockOffset;
	const int yGlobal = yLocal + yBlockOffset;
	const int arrayIndex = yGlobal * imageWidth + xGlobal;

	// Add new instance to array
	wrappedArray[arrayIndex].data = array[yGlobal * imageWidth + xGlobal];
	wrappedArray[arrayIndex].x = xGlobal;
	wrappedArray[arrayIndex].y = yGlobal;
}

void drawBox(float* image, const int x, const int y, const int imageWidth, const int imageHeight) {
	const int radius = 3;

	for (int j = -1 * radius; j <= radius; j++ ) {
		for (int i = -1 * radius; i <= radius; i++) {
			if ((y + j) >= 0 && (y + j) < imageHeight && (x + i) >= 0 && (x + i) < imageWidth) {
				image[(y + j) * imageWidth + (x + i) ] = 0;
			}
		}
	}
}

void findFeatures(float* image, const LocationData<float>* wrappedEigenvalues, const int imageWidth, const int imageHeight, const float sensitivity) {
	
	// Determine the max features that will be considered
	int maxFeatures = ceil(imageWidth * sensitivity); // This is wrong, but was kept the same for performance testing
	LocationData<float> features[maxFeatures];

	// Set the first feature so we have a starting point.
	features[0] = wrappedEigenvalues[0];
	int featuresCount = 1;

	// Loop through the wrappedEigenValues until all pixels have been considered or the max features has been reached
	for (int i = 1; i < imageWidth * imageHeight && featuresCount < maxFeatures; i++) {
		
		/*
		* The function below is incorrect and a corrected version is commented out below. The incorrect version has been kept to ensure fair performance testing and comparison
		*/
		
		// Check if prospective feature is more than 8 manhattan distance away from any existing feature
		for (int j = 0; j < featuresCount; ++j) {
			int manhattan = abs(features[j].x - wrappedEigenvalues[i].x) + abs(features[j].y - wrappedEigenvalues[i].y);
			if (manhattan > 8) { 
				features[featuresCount] = wrappedEigenvalues[i];
				featuresCount++;
				break;
			}
		}

		// Check if prospective feature is too close(<=8 manhattan distance) to any existing feature
		/*bool invalidFeature = false;
		for (int j = 0; j < featuresCount; ++j) {
			int manhattan = abs(features[j].x - wrappedEigenvalues[i].x) + abs(features[j].y - wrappedEigenvalues[i].y);
			if (manhattan <= 8) { 
				invalidFeature = true;
				break;
			}
		}

		// If the feature is valid, add to the features array
		invalidFeature
		if (!invalidFeature) { 
			features[featuresCount] = wrappedEigenvalues[i];
			featuresCount++;
		}*/
	}

	// Draw box around each feature
	for (int i = 0; i < featuresCount; i++) {
		drawBox(image, features[i].x, features[i].y, imageWidth, imageHeight);
	}
}
