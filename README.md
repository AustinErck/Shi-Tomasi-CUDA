# CUDA Corner Detection - Shi Tomasi

## Usage

    ./goodfeatures <full path to image> [sigma] [window size] [sensitivity] [cuda block size]

**arguments:**

 - sigma: the sigma value for the Gaussian distribution used to form the convolution mask
 - window size: the size of a pixel 'neighborhood' in an image
 - sensitivity: determines the amount of features to detect... can be 0.0 to 1.0
 - cude block size: determines how large of a CUDA kernel to use. Default to 16 for a 16x16 kernel

##  Contributors
Thank you to these wonderful people from University of the Pacific for providing various part of the code and debugging support!

* Vivek Pallipuram
* Yang Liu
* Cody Balos
