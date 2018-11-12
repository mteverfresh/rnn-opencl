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

alpha and beta variable names are borrowed from the paper "Direction Optimizing Breadth-First Search" by Scott Beamer, Krste
Asanovic, and David Patterson.

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

//OpenCL runtime structures
cl_platform_id 		platform = NULL;
cl_device_id 		device = NULL;
cl_context		context = NULL;
cl_program		program = NULL;
cl_command_queue	queue = NULL;
cl_kernel 		bfs_top_kernel = NULL;
cl_kernel		bfs_bottom_kernel = NULL;
cl_kernel		update_status_kernel = NULL;

//Memory buffers
cl_mem			csr_buf = NULL;
cl_mem			beg_pos_buf = NULL;
cl_mem			front_comm_buf = NULL;
cl_mem			status_prev_buf = NULL; //new for 8/4/17
cl_mem			status_next_buf = NULL; //new for 8/4/17
cl_mem			level_buf = NULL;

//graph structures
graph<cl_long, cl_long, int, long, cl_long, char> *ginst;
cl_short *front_comm;
//Currently, the program uses a previous status array and a next status array. This may be concatenated into a single status array
//in future version.
cl_short *status_prev;
cl_short *status_next;
cl_short level = 0;

//function prototypes
bool init_env();
void init_data();
void run_kernel(const unsigned int alpha, const unsigned int beta);
void cleanup();
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

int main(int argc, char** argv)
{
	if(argc != 6) 
	{
		std::cout << "Invalid arguments.\n";
		return -1;
	}
	//assign the arguments to constants
	const char *beg_file = argv[1];
	const char *csr_file = argv[2];
	const char *weight_file = argv[3];
	//const char *direction = argv[4];
	const unsigned int alpha = atoi(argv[4]);
	const unsigned int beta = atoi(argv[5]);
	
	//allocate the graph object. This will not change, and it must be accessible globally
	ginst = new graph<cl_long, cl_long, int, long, cl_long, char>(beg_file, csr_file, weight_file);

	//allocate the communication arrays for the kernel
	front_comm = new cl_short[ginst->vert_count];
	status_prev = new cl_short[ginst->vert_count];
	status_next = new cl_short[ginst->vert_count];

	for(unsigned i = 0; i < ginst->vert_count; i++)
	{
		front_comm[i] = 0;
		status_prev[i] = -1;
		status_next[i] = -1;
	}
	status_prev[ROOTNODE] = 0; //hard-code 0 as root node. Could be replaced with cmd arg later
	status_next[ROOTNODE] = 0;
	
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
	bfs_top_kernel = clCreateKernel(program, "bfs_top", &status);
	checkError(status, "Failed to create kernel \"bfs_top\"");
	bfs_bottom_kernel = clCreateKernel(program, "bfs_bottom", &status);
	checkError(status, "Failed to create kernel \"bfs_bottom\"");
	update_status_kernel = clCreateKernel(program, "update_status", &status);
	checkError(status, "Failed to create kernel \"update_status\"");

	//Create buffers
	csr_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, ginst->beg_pos[ginst->vert_count]*sizeof(cl_long), NULL, &status);
	checkError(status, "Failed to create buffer for csr");
	beg_pos_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, ((ginst->vert_count) + 1)*sizeof(cl_long), NULL, &status);
	checkError(status, "Failed to create buffer for beg_pos");
	front_comm_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ginst->vert_count*sizeof(cl_short), NULL, &status);
	checkError(status, "Failed to create buffer for fontiers");
	status_prev_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ginst->vert_count*sizeof(cl_short), NULL, &status);
	checkError(status, "Failed to create buffer for previous status"); 
	status_next_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ginst->vert_count*sizeof(cl_short), NULL, &status);
	checkError(status, "Failed to create buffer for next status");
	level_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_short), NULL, &status);
	checkError(status, "Failed to create buffer for level");

	return true;
}

void run_kernel(const unsigned int alpha, const unsigned int beta)
{
	cl_int status;
	unsigned frontiers = 0; 	//total frontier count
	level = 0;			//runnning level (depth) counter
	bool has_frontiers = true;	//outer loop condition

	std::cout << "Constants " << alpha << " " << beta <<"\n";
	//cl_events used for precise management of when kernels and buffers are manipulated. 
	cl_event initial_event[2];
	cl_event write_event[3];
	cl_event read_event[2];
	cl_event bfs_event;
	
	//Enqueue the input buffers
	clEnqueueWriteBuffer(queue, csr_buf, CL_FALSE, 0, ginst->beg_pos[ginst->vert_count]*sizeof(cl_long), ginst->csr, 0, NULL, &initial_event[0]);
	clEnqueueWriteBuffer(queue, beg_pos_buf, CL_FALSE, 0, ((ginst->vert_count) + 1)*sizeof(cl_long), ginst->beg_pos, 0, NULL, &initial_event[1]);

	//Release the events. This implementation ensures that the buffers are written in the correct order, but it may not 
	//be necessary to be this detailed in the final version.
	clReleaseEvent(initial_event[0]);
	clReleaseEvent(initial_event[1]);

	while(has_frontiers)
	{
		frontiers = 0;

		clEnqueueWriteBuffer(queue, front_comm_buf, CL_FALSE, 0, ginst->vert_count*sizeof(cl_short), front_comm, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, status_prev_buf, CL_FALSE, 0, ginst->vert_count*sizeof(cl_short), status_prev, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, status_next_buf, CL_FALSE, 0, ginst->vert_count*sizeof(cl_short), status_next, 0, NULL, NULL);

		const size_t global_work_size = ginst->vert_count;

		//now, use level parameters to switch between implementations.
		//Update for 8/8/17: moved kernel enqueueing into these loops
		if(level < alpha || level >= beta)
		{
			//Set the kernel args for the top-down kernel
			std::cout << "Using top-down" << "\n";
			status = clSetKernelArg(bfs_top_kernel, 0, sizeof(cl_mem), &csr_buf);
			checkError(status, "Failed to set bfs_top_kernel arg 0");
			status = clSetKernelArg(bfs_top_kernel, 1, sizeof(cl_mem), &beg_pos_buf);
			checkError(status, "Failed to set bfs_top_kernel arg 1");
			status = clSetKernelArg(bfs_top_kernel, 2, sizeof(cl_mem), &front_comm_buf);
			checkError(status, "Failed to set bfs_top_kernel arg 2");
			status = clSetKernelArg(bfs_top_kernel, 3, sizeof(cl_mem), &status_prev_buf);
			checkError(status, "Failed to set bfs_top_kernel arg 3");
			status = clSetKernelArg(bfs_top_kernel, 4, sizeof(cl_mem), &status_next_buf);
			checkError(status, "Failed to set bfs_top_kernel arg 4");
			status = clSetKernelArg(bfs_top_kernel, 5, sizeof(cl_short), &level);
			checkError(status, "Failed to set bfs_top_kernel arg 5");
			status = clSetKernelArg(bfs_top_kernel, 6, sizeof(cl_long), &ginst->vert_count);
			checkError(status, "Failed to set bfs_top_kernel arg 6");

			status = clEnqueueNDRangeKernel(queue, bfs_top_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		else if(level >= alpha)
		{
			//Set the kernel args for the bottom-up kernel
			std::cout << "Using bottom-up" << "\n";
			status = clSetKernelArg(bfs_bottom_kernel, 0, sizeof(cl_mem), &csr_buf);
			checkError(status, "Failed to set bfs_bottom_kernel arg 0");
			status = clSetKernelArg(bfs_bottom_kernel, 1, sizeof(cl_mem), &beg_pos_buf);
			checkError(status, "Failed to set bfs_bottom_kernel arg 1");
			status = clSetKernelArg(bfs_bottom_kernel, 2, sizeof(cl_mem), &front_comm_buf);
			checkError(status, "Failed to set bfs_bottom_kernel arg 2");
			status = clSetKernelArg(bfs_bottom_kernel, 3, sizeof(cl_mem), &status_prev_buf);
			checkError(status, "Failed to set bfs_bottom_kernel arg 3");
			status = clSetKernelArg(bfs_bottom_kernel, 4, sizeof(cl_mem), &status_next_buf);
			checkError(status, "Failed to set bfs_bottom_kernel arg 4");
			status = clSetKernelArg(bfs_bottom_kernel, 5, sizeof(cl_short), &level);
			checkError(status, "Failed to set bfs_bottom_kernel arg 5");
			status = clSetKernelArg(bfs_bottom_kernel, 6, sizeof(cl_long), &ginst->vert_count);
			checkError(status, "Failed to set bfs_bottom_kernel arg 6");

			status = clEnqueueNDRangeKernel(queue, bfs_bottom_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		//Enqueue the kernel
		/*if(!strcmp(direction, "top"))
		{
			status = clEnqueueNDRangeKernel(queue, bfs_top_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		else if(!strcmp(direction, "bottom"))
		{
			status = clEnqueueNDRangeKernel(queue, bfs_bottom_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}*/

		//Enqueue a read action to read from the status buffer and front_comm buffer
		clEnqueueReadBuffer(queue, front_comm_buf, CL_FALSE, 0, ginst->vert_count*sizeof(cl_short), front_comm, 0, NULL, NULL);
		clEnqueueReadBuffer(queue, status_next_buf, CL_FALSE, 0, ginst->vert_count*sizeof(cl_short), status_next, 0, NULL, NULL);

		clFinish(queue);

		//At this point, the "threads" from the kernel have joined and have reported their frontier counts to the front_comm
		//array. The total number of frontiers is calculated here in a single-threaded process.

		//@TODO change to pipelined adder implementation (opencl library?)
		//actually won't have to do this coming up because of prefix scan
		for(unsigned i = 0; i < ginst->vert_count; i++)
		{	
			frontiers += front_comm[i];
			front_comm[i] = 0; //zero out front_comm for next iteration
		}
		std::cout << "Level " << level << ": found " << frontiers << " frontiers\n";
		if(frontiers == 0)
		{
			has_frontiers = false;
		}
		level++;

		//Now, copy the old status_next into status_prev

		//First, set the args for the update_status_kernel kernel
		status = clSetKernelArg(update_status_kernel, 0, sizeof(cl_mem), &status_prev_buf);
		checkError(status, "Failed to set update_status_kernel arg 0");
		status = clSetKernelArg(update_status_kernel, 1, sizeof(cl_mem), &status_next_buf);
		checkError(status, "Failed to set update_status_kernel arg 1");

		//Second, enqueue the kernel
		status = clEnqueueNDRangeKernel(queue, update_status_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

		//Last, read from the buffers and finish the queue.
		clEnqueueReadBuffer(queue, status_prev_buf, CL_FALSE, 0, ginst->vert_count*sizeof(cl_short), status_prev, 0, NULL, NULL);
		clFinish(queue);

		//end loop
	}	
}

//Required function for AOCL_utils
void cleanup()
{
	if(bfs_top_kernel)
		clReleaseKernel(bfs_top_kernel);
	if(bfs_bottom_kernel)
		clReleaseKernel(bfs_bottom_kernel);
	if(update_status_kernel)
		clReleaseKernel(update_status_kernel);
	if(queue)
		clReleaseCommandQueue(queue);
	if(csr_buf)
		clReleaseMemObject(csr_buf);
	if(beg_pos_buf)
		clReleaseMemObject(beg_pos_buf);
	if(front_comm_buf)
		clReleaseMemObject(front_comm_buf);
	if(status_prev_buf)
		clReleaseMemObject(status_prev_buf);
	if(status_next_buf)
		clReleaseMemObject(status_next_buf);
	if(program)
		clReleaseProgram(program);
	if(context)
		clReleaseContext(context);
}
