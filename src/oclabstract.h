#ifndef OCL_ABS_H
#define OCL_ABS_H

#include <CL/opencl.h>

//define some additional constants
#define INPUT_SIZE 128*6
#define OUTPUT_SIZE 128*6
#define CONCAT_SIZE INPUT_SIZE*2

bool setupOclEnv(char *kernel_file);
void matrixMultiplyCl(cl_float *a, cl_float *b, cl_float *output);
void matrixAddCl(cl_float *a, cl_float *b, cl_float *output);
void sigmoidCl(cl_float *in, cl_float *out);
void tanhCl(cl_float *in, cl_float *out);
void matrixConcatCl(cl_float *a, cl_float *b, cl_float *output);

#endif

