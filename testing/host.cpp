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
11/25/18	Updating for compatibility with target system.
			- This means using binaries for the kernel code and using aocl_utils
11/28/18	Updated to be more modular from the command line.
			- Specifying the test no longer requires a recompile
			- specifying window size still does though

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <CL/opencl.h>
#include "AOCLUtils/aocl_utils.h"
#include "wtime.h"

#define WINDOW_SIZE 2048
#define MATRIX_SIZE WINDOW_SIZE*6
#define INDEX(ROW, COLUMN, WIDTH) ((ROW)*WIDTH + (COLUMN))

//    D A T A   S T R U C T U R E S    //

cl_platform_id 		platform = NULL;
cl_device_id 		device = NULL;
cl_context		context = NULL;
cl_program		program = NULL;
cl_command_queue	queue = NULL;
cl_kernel		kernel = NULL;

//Weight memory
cl_mem			input_a_buf = NULL;
cl_mem			input_b_buf = NULL;
cl_mem			output_buf = NULL;

//Host-side buffers
cl_float		*input_a;
cl_float		*input_b;
cl_float		*output;
cl_float		*test_data;

cl_int			status;

//function prototypes
bool init_env(char*);
void init_data();
void run_kernel(int, std::string);
void cleanup();
void checkOutput();
void sigmoidtest();
void tanhtest();
void addtest();
void multest();
void concattest();
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

using namespace std; 
using namespace aocl_utils;

void sigmoidtest()
{
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		float intr_val = exp((double) -input_a[i]);
		test_data[i] = (1/(1+intr_val));
	}
}
void tanhtest()
{
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		test_data[i] = tanh(input_a[i]);
	}
}
void addtest()
{
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		test_data[i] = input_a[i] + input_b[i];
	}
}
void multest()
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
				sum += input_a[INDEX(i, k, WINDOW_SIZE)] * input_b[INDEX(j, k, WINDOW_SIZE)];
			}
			test_data[INDEX(i, j, 6)] = sum;
		}
	}
}
void concattest()
{
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		test_data[i] = input_a[i] + input_b[i];
	}
}

void checkOutput()
{
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		if(output[i] != test_data[i])
		{
			printf("output differs from test data:\n");
			printf("output: %f\ttest data: %f\n", output[i], test_data[i]);
			printf("index: %d\n", i);
			//exit(0);
		}
	}
}

////TODO make this a macro so that __LINE__ actually does what we want
//void checkError(int err, int lineno)
//{
//	if(err != CL_SUCCESS)
//	{
//		printf("OpenCL error %d, line %d\n", err, lineno);
//		exit(0);
//	}
//}

int main(int argc, char** argv)
{
	//TODO: set up buffers for test data
	//these will be of fixed size
	//see if the reqd_wg_size attribute works outside of altera
	input_a = (cl_float*)alignedMalloc(sizeof(cl_float) * MATRIX_SIZE);
	input_b = (cl_float*)alignedMalloc(sizeof(cl_float) * MATRIX_SIZE);
	output 	= (cl_float*)alignedMalloc(sizeof(cl_float) * MATRIX_SIZE);

	test_data = (cl_float*)malloc(sizeof(cl_float) * MATRIX_SIZE);
	srand(time(NULL));
	//initialize input buffers with some data
	for(unsigned i = 0; i < MATRIX_SIZE; i++)
	{
		input_a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		input_b[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	std::string kernel_name = argv[1];
	if(!init_env(argv[1]))
	{
		return -1;
	}
	run_kernel(atoi(argv[2]), kernel_name);
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
	std::string binary_file = getBoardBinaryFile("kernels", device);
	printf("Using binary %s to program FPGA\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	//Build program
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	//Create cmd queue
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create queue");

	//Create kernels
	printf("Attempting to build kernel for %s... ", kernel_name);
	kernel = clCreateKernel(program, kernel_name, &status);
	switch(status)
	{
		case CL_INVALID_KERNEL_NAME: 
			printf("Invalid kernel name. Please use a valid kernel in kernels.cl.\n");
			exit(0);
			break;
		case CL_SUCCESS:
			printf("Build successful.\n");
		default: checkError(status, "building kernel");
	}
	
	//Create buffers
	input_a_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					sizeof(float)*MATRIX_SIZE,
					NULL, 
					&status);
	checkError(status, "input_a");
	input_b_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					sizeof(float)*MATRIX_SIZE,
					NULL, 
					&status);
	checkError(status, "input_b");
	output_buf = clCreateBuffer(	context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					sizeof(float)*MATRIX_SIZE,
					NULL, 
					&status);
	checkError(status, "output");
	return true;
}

void run_kernel(int kernel_args, std::string kernel_name)
{
	double starttime, endtime;
	const size_t global_work_size = WINDOW_SIZE;
	const size_t local_work_size = WINDOW_SIZE;

	printf("global work size: %u\n", global_work_size);

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
		checkError(status, "kernel arg 0");
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
		checkError(status, "kernel arg 1");
	}
	else if(kernel_args == 3)
	{
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_a_buf);
		checkError(status, "kernel arg 0");
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_b_buf);
		checkError(status, "kernel arg 1");
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buf);
		checkError(status, "kernel arg 2");
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
	status = clEnqueueNDRangeKernel(queue, 
				kernel,
				1, NULL, 
				&global_work_size, &local_work_size,
				0, NULL, NULL);
	checkError(status, "Failed to launch kernel");
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
