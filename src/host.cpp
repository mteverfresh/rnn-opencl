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

#define ROOTNODE 0
using namespace aocl_utils;

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
cl_mem			w_forget_buf = NULL;
cl_mem			w_input_buf = NULL;
cl_mem			w_internal_buf = NULL;
cl_mem			w_output_buf = NULL;

//LSTM I/O
cl_mem			curr_input_buf;
cl_mem			curr_output_buf;
cl_mem			prev_output_buf;
cl_mem			curr_state_buf;
cl_mem			prev_state_buf;

//function prototypes
bool init_env();
void init_data();
void run_kernel(const unsigned int alpha, const unsigned int beta);
void cleanup();
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

int main(int argc, char** argv)
{
	if(!init_env())
	{
		return -1;
	}
	run_kernel(alpha, beta);
	cleanup();
	return 0;
}

bool init_env()
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
	std::string binary_file = getBoardBinaryFile("bfs", device);
	printf("Using binary %s to program FPGA\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	//Build program
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	//Create cmd queue
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create queue");

	//Create kernels
	k_matrix_add = clCreateKernel(program, "matrix_add", &status);
	checkError(status, "Failed to create kernel \"matrix_add\"");
	k_matrix_mul = clCreateKernel(program, "matrix_mul", &status);
	checkError(status, "Failed to create kernel \"matrix_mul\"");
	k_sigmoid = clCreateKernel(program, "sigmoid_activation", &status);
	checkError(status, "Failed to create kernel \"sigmoid_activation\"");
	k_tanh = clCreateKernel(program, "tanh_activation", &status);
	checkError(status, "Failed to create kernel \"tanh_activation\"");
	k_concat = clCreateKernel(program, "matrix_concat", &status);
	checkError(status, "Failed to create kernel \"matrix_concat\"");

	//Create buffers
	w_forget_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of buffer
					&status);
	checkError(status, "Failed to create buffer for forget weight");
	w_input_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of buffer
					&status);
	checkError(status, "Failed to create buffer for input weight");
	w_internal_buf = clCreateBuffer(context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of buffer
					&status);
	checkError(status, "Failed to create buffer for internal weight");
	w_output_buf = clCreateBuffer(context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of buffer
					&status);
	checkError(status, "Failed to create buffer for output weight");
	curr_input_buf = clCreateBuffer(context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of buffer
					&status);
	checkError(status, "Failed to create buffer for current input");
	curr_output_buf = clCreateBuffer(context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of buffer
					&status);
	checkError(status, "Failed to create buffer for current output");
	prev_output_buf = clCreateBuffer(context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of buffer
					&status);
	checkError(status, "Failed to create buffer for previous output");
	curr_state_buf = clCreateBuffer(context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of buffer
					&status);
	checkError(status, "Failed to create buffer for current state");
	prev_state_buf = clCreateBuffer(context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of buffer
					&status);
	checkError(status, "Failed to create buffer for previous state");

	return true;
}

void run_kernel(const unsigned int alpha, const unsigned int beta)
{
	//Calls to LSTM organizing functions will be here
	//TODO: create a class for the LSTM whose members can be called here
}

//Required function for AOCL_utils
void cleanup()
{
}
