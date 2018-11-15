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
#include <cstring>
#include <math.h>
#include <assert.h>
#include <CL/opencl.h>
#include "AOCLUtils/aocl_utils.h"
#include "graph.h"

#define WINDOW_SIZE 128
#define MATRIX_SIZE WINDOW_SIZE*6

//    D A T A   S T R U C T U R E S    //

cl_platform_id 		platform = NULL;
cl_device_id 		device = NULL;
cl_context		context = NULL;
cl_program		program = NULL;
cl_command_queue	queue = NULL;
cl_kernel 		k_matrix_add = NULL;
cl_kernel		k_matrix_mul = NULL;
cl_kernel		k_sigmoid = NULL;
cl_kernel		k_tanh = NULL;
cl_kernel		k_concat = NULL;

//Weight memory
cl_mem			input_a_buf = NULL;
cl_mem			input_b_buf = NULL;
cl_mem			output_buf = NULL;

//Host-side buffers
cl_float		*input_a;
cl_float		*input_b;
cl_float		*output;

//function prototypes
bool init_env(char*);
void init_data();
void run_kernel(int);
void cleanup();
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

int main(int argc, char** argv)
{
	//TODO: set up buffers for test data
	//these will be of fixed size
	//see if the reqd_wg_size attribute works outside of altera
	input_a = (cl_float*)malloc(sizeof(cl_float) * WINDOW_SIZE);
	input_b = (cl_float*)malloc(sizeof(cl_float) * WINDOW_SIZE);
	output 	= (cl_float*)malloc(sizeof(cl_float) * WINDOW_SIZE);

	if(!init_env(argv[1]))
	{
		return -1;
	}
	run_kernel(atoi(argv[2]);
	cleanup();
	return 0;
}

bool init_env(char* kernel_name)
{
	cl_int status; //holds status of each operation for error checking

	//Get platform
	platform = findPlatform("Intel(R) FPGA");
	if (platform == NULL)
	{
		printf("Unable to find FPGA OpenCL platform. Exiting.");
		return false;
	}

	//Get device ID. Since this is only running on the DE5NET for now, we only need to get one device.
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	checkError(status, "Failed to get devices");

	printf("Platform: %s\n", getPlatformName(platform).c_str());
	printf("Using %s for calculation.\n", getDeviceName(device).c_str());

	//Create context
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	checkError(status, "Unable to create OpenCL context.");

	//Create program
	program = clCreateProgramWithSource(context, "kernels.cl", &device, 1);

	//Build program
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	//Create cmd queue
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create queue");

	//Create kernels
	k_matrix_add = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel");
	
	//Create buffers
	input_a_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					CONCAT_SIZE,
					&status);
	checkError(status, "Failed to create buffer for first input");
	input_b_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					CONCAT_SIZE,
					&status);
	checkError(status, "Failed to create buffer for second input");
	output_buf = clCreateBuffer(	context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					CONCAT_SIZE,
					&status);
	checkError(status, "Failed to create buffer for output");
	return true;

}

void run_kernel(int kernel_args)
{
	//TODO: set up kernel test here
	//requires knowledge of number of kernel args from the function parameter
	clEnqueueWriteBuffer(	queue, 
				input_a_buf,
				CL_FALSE, 0, 
				sizeof(cl_float)*CONCAT_SIZE,
				input_a, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				input_b_buf,
				CL_FALSE, 0, 
				sizeof(cl_float)*CONCAT_SIZE,
				input_b, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(cl_float)*OUTPUT_SIZE,
				output, 
				0, NULL, NULL);

	if(kernel_args == 2)
	{
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_a_buf);
		checkError(status, "Failed to set kernel arg 0");
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
		checkError(status, "Failed to set kernel arg 1");
	}
	else if(kernel_args == 3)
	{
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_a_buf);
		checkError(status, "Failed to set kernel arg 0");
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_b_buf);
		checkError(status, "Failed to set kernel arg 1");
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buf);
		checkError(status, "Failed to set kernel arg 2");
	}
	clEnqueueNDRangeKernel(	queue, 
				k_matrix_mul, 
				1, NULL, 
				&global_work_size, &local_work_size, 
				0, NULL, NULL);
	checkError(status, "Failed to launch matrix mul kernel");

	clEnqueueReadBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(cl_float)*OUTPUT_SIZE,
				output, 
				0, NULL, NULL);
	clFinish(queue);
}

//Required function for AOCL_utils
void cleanup()
{
}
