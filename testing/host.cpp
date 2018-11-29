/*

Filename: host.cpp
Author: Zach Sherer
Purpose: Host application for the BFS OpenCL project. This application handles the scheduling of the kernels as well as the
summation and management of frontiers. When finished, the application will be able to handle both bottom-up and top-down
traversal.

Acknowledgements:

AOCLUtils and associated functions written by Altera Corporation.
Stratix V is a trademark of the Altera Corporation.
DE5NET is a trademark of Terasic Inc.

Date		Change
----------------------------------------------------------------------
8/2/17		File created.
		Changed names of buffers and added the graph structure to the code.
8/3/17		Debugging infinite looping error.
8/4/17		Kernel changes facilitate additional changes to code:
			- added status_prev and status_next
			- added new kernel functionality for update_status and switchable bottom/top
			- adding multiple kernels may facilitate adding events back in for synchronization. Research ongoing.
8/7/17		Corrected allocation size of the csr and beg_pos buffers to be the correct size for each array
		First working implementation realized.
8/8/17		Started change to hybrid implementation.
11/11/18	Hello from the future.
		This file is being repurposed for a new project: the Data fusion RNN activity recognition project.
			- Buffers now represent weight memory or intermediate data for the LSTM cell.
			- Weight memory is loaded in from a file at the beginning of the forward pass

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <CL/opencl.h>
#include "wtime.h"

#define WINDOW_SIZE 128
#define MATRIX_SIZE WINDOW_SIZE*6
#define STATE_SIZE 6*6
#define INDEX(ROW, COLUMN, WIDTH) ((ROW)*WIDTH + (COLUMN))

//    D A T A   S T R U C T U R E S    //

cl_platform_id 		platform = NULL;
cl_device_id 		device = NULL;
cl_context		context = NULL;
cl_program		program = NULL;
cl_command_queue	queue = NULL;

cl_kernel		add_kernel = NULL;
cl_kernel		sig_kernel = NULL;
cl_kernel		tanh_kernel = NULL;

//Weight memory
cl_mem			weight_buf = NULL;
cl_mem			bias_buf = NULL;
cl_mem			input_concat_buf = NULL;
cl_mem			final_output_buf = NULL;

//Host-side buffers
cl_float		*weight;
cl_float		*bias;
cl_float		*input_concat;
cl_float		*final_output;

cl_int status;

//function prototypes
bool init_env();
void init_data();
void run_kernel();
void cleanup();
void sigmoidtest(float*, float*);
void tanhtest(float*, float*);
void addtest(float*, float*, float*);
void matmul(float*, float*, float*);
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

void sigmoidtest(float *input, float *output)
{
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		float intr_val = exp((double) -input[i]);
		output[i] = (1/(1+intr_val));
	}
}
void tanhtest(float *input, float *output)
{
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		output[i] = tanh(input[i]);
	}
}
void addtest(float *a, float *b, float *output)
{
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		output[i] = a[i] + b[i];
	}
}
void matmul(float *a, float *b, float *output)
{
	/*
	   this attempts an auto-transpose operation by addressing the matrix differently,
	   possible because the matrix shape is known and constant
	   
	   input_a shape: 6 rows, WINDOW_SIZE cols
	   input_b shape: WINDOW_SIZE rows, 6 cols
	*/
	float sum;
	for(unsigned i = 0; i < 6; i++)
	{
		for(unsigned j = 0; j < 6; j++)
		{
			sum = 0;
			for(unsigned k = 0; k < WINDOW_SIZE; k++)
			{
				register float first = a[INDEX(i, k, WINDOW_SIZE)];
				register float second = b[INDEX(j, k, WINDOW_SIZE)];

				sum += first * second;
			}
			output[INDEX(i, j, 6)] = sum;
		}
	}
}
//TODO make this a macro so that __LINE__ actually does what we want
void checkError(int err, int lineno)
{
	if(err != CL_SUCCESS)
	{
		printf("OpenCL error %d, line %d\n", err, lineno);
		exit(0);
	}
}

int main(int argc, char** argv)
{
	//TODO: set up buffers for test data
	//these will be of fixed size
	//see if the reqd_wg_size attribute works outside of altera
	srand(time(NULL));
	weight = (cl_float*)malloc(sizeof(cl_float) * MATRIX_SIZE);
	bias = (cl_float*)malloc(sizeof(cl_float) * STATE_SIZE);
	input_concat = (cl_float*)malloc(sizeof(cl_float) * MATRIX_SIZE);
        final_output = (cl_float*)malloc(sizeof(cl_float) * STATE_SIZE);

	if(!init_env())
	{
		return -1;
	}
	run_kernel()
	cleanup();
	return 0;
}

bool init_env()
{
	//Load in kernel file and extract source and length
	ifstream kernel_file("kernels.cl");
	std::stringstream kernel_source_reader;
	size_t filesize;
	
	kernel_source_reader << kernel_file.rdbuf();
	string kernel_source_str(kernel_source_reader.str());
	const char* kernel_source = kernel_source_str.c_str();
	kernel_file.seekg(0, ios::beg);
	filesize = kernel_file.tellg();
	kernel_file.seekg(0, ios::end);
	filesize = kernel_file.tellg() - filesize;

	cl_int status; //holds status of each operation for error checking

	//Get platform
	status = clGetPlatformIDs(1, &platform, NULL);
	checkError(status, __LINE__);

	//Get device ID. Since this is only running on the DE5NET for now, we only need to get one device.
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	checkError(status, __LINE__);
	size_t paramsize;
	char *param;
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &paramsize);
	param = (char*)malloc(paramsize);
	clGetDeviceInfo(device, CL_DEVICE_NAME, paramsize, param, NULL);
	printf("Device param: %s\n", param);
	free(param);

	//Create context
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	checkError(status, __LINE__);

	//Create program
	program = clCreateProgramWithSource(context, 1, &kernel_source, &filesize, NULL);
	checkError(status, __LINE__);

	//Build program
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	if(status != CL_SUCCESS)
	{
		printf("Program build failed.\n");
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &paramsize);
		param = (char*)malloc(paramsize);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, paramsize, param, NULL);
		printf("%s\n", param);
		checkError(status, __LINE__);
	}

	//Create cmd queue
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, __LINE__);

	//Create kernels
	printf("Attempting to build kernel for %s... ", kernel_name);
	add_kernel = clCreateKernel(program, "matrix_add", &status);
	checkError(status, __LINE__);
	sig_kernel = clCreateKernel(program, "sigmoid_activation", &status);
	checkError(status, __LINE__);
	tanh_kernel = clCreateKernel(program, "tanh_activation", &status);
	checkError(status, __LINE__);
	
	//Create buffers
	weight_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					sizeof(cl_float)*MATRIX_SIZE,
					NULL, 
					&status);
	checkError(status, __LINE__);
	bias_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					sizeof(cl_float)*STATE_SIZE,
					NULL, 
					&status);
	checkError(status, __LINE__);
	input_concat_buf = clCreateBuffer(context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					sizeof(cl_float)*MATRIX_SIZE,
					NULL, 
					&status);
	checkError(status, __LINE__);
	final_output_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					sizeof(cl_float)*STATE_SIZE,
					NULL, 
					&status);
	checkError(status, __LINE__);
	//initialize input buffers with some data
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		if(i < STATE_SIZE)
		{
			bias[i] = rand_float();
			final_output[i] = 0;
		}
		weight[i] = rand_float();
		input_concat[i] = rand_float();
	}
	return true;
}

void run_kernel(int kernel_args, std::string kernel_name)
{
	double starttime, endtime;
	const size_t global_work_size = MATRIX_SIZE;
	const size_t local_work_size = WINDOW_SIZE;

	//requires knowledge of number of kernel args from the function parameter
	clEnqueueWriteBuffer(	queue, 
				input_a_buf,
				CL_FALSE, 0, 
				sizeof(cl_float)*MATRIX_SIZE,
				input_a, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				input_b_buf,
				CL_FALSE, 0, 
				sizeof(cl_float)*MATRIX_SIZE,
				input_b, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(cl_float)*MATRIX_SIZE,
				output, 
				0, NULL, NULL);

	if(kernel_args == 2)
	{
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_a_buf);
		checkError(status, __LINE__);
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
		checkError(status, __LINE__);
	}
	else if(kernel_args == 3)
	{
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_a_buf);
		checkError(status, __LINE__);
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_b_buf);
		checkError(status, __LINE__);
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buf);
		checkError(status, __LINE__);
	}
	//TODO: every time you want a new test, change this function call
	printf("Running CPU test for %s:\n", kernel_name.c_str());
	if(kernel_name == "matrix_add")
	{
		starttime = wtime();
		addtest();
		endtime = wtime();
	}
	else if(kernel_name == "matrix_mul")
	{
		starttime = wtime();
		multest();
		endtime = wtime();
	}
	else if(kernel_name == "sigmoid_activation")
	{
		starttime = wtime();
		sigmoidtest();
		endtime = wtime();
	}
	else if(kernel_name == "tanh_activation")
	{
		starttime = wtime();
		tanhtest();
		endtime = wtime();
	}
	else
	{
		printf("invalid kernel name, should have been caught earlier!\n");
	}

	printf("Time for CPU test: %g\n", endtime-starttime);

	starttime = wtime();
	clEnqueueNDRangeKernel(	queue, 
				kernel,
				1, NULL, 
				&global_work_size, &local_work_size,
				0, NULL, NULL);
	clFinish(queue);
	endtime = wtime();
	printf("Time for accelerated test: %g\n", endtime-starttime);

	clEnqueueReadBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(cl_float)*MATRIX_SIZE,
				output, 
				0, NULL, NULL);
	clFinish(queue);

	checkOutput();

	printf("Kernel run successfully\n\n"); 
}

//Required function for AOCL_utils
void cleanup()
{
	clReleaseContext(context);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseKernel(kernel);
	clReleaseMemObject(input_a_buf);
	clReleaseMemObject(input_b_buf);
	clReleaseMemObject(output_buf);
}
