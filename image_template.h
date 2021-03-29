#ifndef IMAGE_TEMPLATE_H
#define IMAGE_TEMPLATE_H

/* 
This program was originally written by
Sumedh Naik (now at Intel) at Clemson University
as a part of his thesis titled, "Connecting Architectures,
fitness, Optimizations and Performance using an Anisotropic
Diffusion Filter. This header was also used
in Dr. Pallipuram's dissertation work. 

Modified by Cody Balos in 2017 to comply with the C99 standard.
*/

#include <stdio.h> 
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER 512

#define READ_IMAGE_TEMPLATE(T) { \
	unsigned char *temp_img; \
	read_imagec(name, &temp_img, im_width, im_height); \
	*image = malloc(sizeof(T)*(*im_width)*(*im_height)); \
	for(int i = 0; i < (*im_width) * (*im_height); i++) \
		(*image)[i] =(T)temp_img[i]; \
	free(temp_img); \
}

#define WRITE_IMAGE_TEMPLATE(T) { \
	unsigned char *temp_img = malloc(sizeof(unsigned char)*im_width*im_height); \
	for(int i = 0;i < (im_width*im_height); i++) \
		temp_img[i] = image[i]; \
	write_imagec(name, temp_img, im_width, im_height); \
	free(temp_img); \
}

// Function declarations
void read_image(char *name, double **image, int *im_width, int *im_height);
void read_imagef(char *name, float **image, int *im_width, int *im_height);
void read_imagel(char *name, long **image, int *im_width, int *im_height);
void read_imagec(char *name, unsigned char **image, int *im_width, int *im_height);
void write_image(char *name, double *image, int im_width, int im_height);
void write_imagef(char *name, float *image, int im_width, int im_height);
void write_imagel(char *name, long *image, int im_width, int im_height);
void write_imagec(char *name, unsigned char *image, int im_width, int im_height);

// Function definitions
void read_image(char *name, double **image, int *im_width, int *im_height)
{
	READ_IMAGE_TEMPLATE(double)
}

void read_imagef(char *name, float **image, int *im_width, int *im_height)
{
	READ_IMAGE_TEMPLATE(float)
}

void read_imagel(char *name, long **image, int *im_width, int *im_height)
{
	READ_IMAGE_TEMPLATE(long)
}

void read_imagec(char *name, unsigned char **image, int *im_width, int *im_height)
{
	FILE *fip;
	char buf[BUFFER];
	char *parse;
	int im_size;
	
	fip=fopen(name,"rb");
	if(fip==NULL)
	{
		fprintf(stderr,"ERROR:Cannot open %s\n",name);
		exit(0);
	}
	fgets(buf,BUFFER,fip);
	do
	{
		fgets(buf,BUFFER,fip);
	}
	while(buf[0]=='#');
	parse=strtok(buf," ");
	(*im_width)=atoi(parse);

	parse=strtok(NULL,"\n");
	(*im_height)=atoi(parse);

	fgets(buf,BUFFER,fip);
	parse=strtok(buf," ");
	
	im_size=(*im_width)*(*im_height);
	(*image)=malloc(sizeof(unsigned char)*im_size);
	fread(*image,1,im_size,fip);
	
	fclose(fip);
}

void write_image(char *name, double *image, int im_width, int im_height)
{
	WRITE_IMAGE_TEMPLATE(double)
}

void write_imagef(char *name, float *image, int im_width, int im_height)
{
	WRITE_IMAGE_TEMPLATE(float)
}

void write_imagel(char *name, long *image, int im_width, int im_height)
{
	WRITE_IMAGE_TEMPLATE(long)
}

void write_imagec(char *name, unsigned char *image, int im_width, int im_height)
{
	FILE *fop; 
	int im_size=im_width*im_height;
	
	fop=fopen(name,"w+");
	fprintf(fop,"P5\n%d %d\n255\n",im_width,im_height);
	fwrite(image,sizeof(unsigned char),im_size,fop);
	
	fclose(fop);
}

#endif

