#include "lstm.hpp"

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

bool setupOclEnv(char *kernel_file)
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
	std::string binary_file = getBoardBinaryFile(kernel_file, device);
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
	input_a_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of concatenated matrix
					&status);
	checkError(status, "Failed to create buffer for first input");
	input_b_buf = clCreateBuffer(	context, 
					CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//size of concatenated matrix
					&status);
	checkError(status, "Failed to create buffer for second input");
	output_buf = clCreateBuffer(	context, 
					CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
					NULL, 
					//longest dim of concatenated matrix square
					&status);
	checkError(status, "Failed to create buffer for output");
	return true;
	
}

//    O P E N C L   A B S T R A C T I O N   F U N C T I O N S    //

//These functions make the acceleration functions into simple function calls

void matrixMultiplyCl(float *a, float *b, float *output)
{
	//TODO these may be wrong or subject to change, keep an eye on this section
	const size_t global_work_size = OUTPUT_SIZE;
	const size_t local_work_size = WINDOW_COLS;

	clEnqueueWriteBuffer(	queue, 
				input_a_buf,
				CL_FALSE, 0, 
				sizeof(float)*CONCAT_SIZE,
				a, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				input_b_buf,
				CL_FALSE, 0, 
				sizeof(float)*CONCAT_SIZE,
				b, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*OUTPUT_SIZE,
				output, 
				0, NULL, NULL);

	status = clSetKernelArg(k_matrix_mul, 0, sizeof(cl_mem), &input_a_buf);
	checkError(status, "Failed to set matrix_mul arg 0");
	status = clSetKernelArg(k_matrix_mul, 1, sizeof(cl_mem), &input_b_buf);
	checkError(status, "Failed to set matrix_mul arg 1");
	status = clSetKernelArg(k_matrix_mul, 2, sizeof(cl_mem), &output_buf);
	checkError(status, "Failed to set matrix_mul arg 2");

	clEnqueueNDRangeKernel(	queue, 
				k_matrix_mul, 
				1, NULL, 
				&global_work_size, &local_work_size, 
				0, NULL, NULL);
	checkError(status, "Failed to launch matrix mul kernel");

	clEnqueueReadBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*OUTPUT_SIZE,
				output, 
				0, NULL, NULL);
	clFinish(queue);
}
void matrixAddCl(float *a, float *b, float *output)
{
	//TODO these may be wrong or subject to change, keep an eye on this section
	const size_t global_work_size = OUTPUT_SIZE;
	const size_t local_work_size = OUTPUT_SIZE;

	clEnqueueWriteBuffer(	queue, 
				input_a_buf,
				CL_FALSE, 0, 
				sizeof(float)*CONCAT_SIZE,
				a, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				input_b_buf,
				CL_FALSE, 0, 
				sizeof(float)*CONCAT_SIZE,
				b, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*CONCAT_SIZE,
				output, 
				0, NULL, NULL);

	status = clSetKernelArg(k_matrix_add, 0, sizeof(cl_mem), &input_a_buf);
	checkError(status, "Failed to set matrix_add arg 0");
	status = clSetKernelArg(k_matrix_add, 1, sizeof(cl_mem), &input_b_buf);
	checkError(status, "Failed to set matrix_add arg 1");
	status = clSetKernelArg(k_matrix_add, 2, sizeof(cl_mem), &output_buf);
	checkError(status, "Failed to set matrix_add arg 2");

	clEnqueueNDRangeKernel(	queue, 
				k_matrix_add, 
				1, NULL, 
				&global_work_size, &local_work_size, 
				0, NULL, NULL);
	checkError(status, "Failed to launch matrix add kernel");

	clEnqueueReadBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*INPUT_SIZE,
				output, 
				0, NULL, NULL);
	clFinish(queue);
}
void sigmoidCl(float *in, float *out)
{
	//TODO these may be wrong or subject to change, keep an eye on this section
	const size_t global_work_size = OUTPUT_SIZE;
	const size_t local_work_size = OUTPUT_SIZE;

	clEnqueueWriteBuffer(	queue, 
				input_a_buf,
				CL_FALSE, 0, 
				sizeof(float)*OUTPUT_SIZE,
				a, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*OUTPUT_SIZE,
				output, 
				0, NULL, NULL);

	status = clSetKernelArg(k_sigmoid, 0, sizeof(cl_mem), &input_a_buf);
	checkError(status, "Failed to set concat arg 0");
	status = clSetKernelArg(k_sigmoid, 1, sizeof(cl_mem), &output_buf);
	checkError(status, "Failed to set concat arg 1");

	clEnqueueNDRangeKernel(	queue, 
				k_sigmoid, 
				1, NULL, 
				&global_work_size, &local_work_size, 
				0, NULL, NULL);
	checkError(status, "Failed to launch matrix add kernel");

	clEnqueueReadBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*OUTPUT_SIZE;
				output, 
				0, NULL, NULL);
	clFinish(queue);
}
void tanhCl(float *in, float *out)
{
	//TODO these may be wrong or subject to change, keep an eye on this section
	const size_t global_work_size = OUTPUT_SIZE;
	const size_t local_work_size = OUTPUT_SIZE;

	clEnqueueWriteBuffer(	queue, 
				input_a_buf,
				CL_FALSE, 0, 
				sizeof(float)*OUTPUT_SIZE,
				a, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*OUTPUT_SIZE,
				output, 
				0, NULL, NULL);

	status = clSetKernelArg(k_tanh, 0, sizeof(cl_mem), &input_a_buf);
	checkError(status, "Failed to set concat arg 0");
	status = clSetKernelArg(k_tanh, 1, sizeof(cl_mem), &output_buf);
	checkError(status, "Failed to set concat arg 1");

	clEnqueueNDRangeKernel(	queue, 
				k_tanh, 
				1, NULL, 
				&global_work_size, &local_work_size, 
				0, NULL, NULL);
	checkError(status, "Failed to launch matrix add kernel");

	clEnqueueReadBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*OUTPUT_SIZE;
				output, 
				0, NULL, NULL);
	clFinish(queue);
}
void matrixConcatCl(float *a, float *b, float *output)
{
	//TODO these may be wrong or subject to change, keep an eye on this section
	const size_t global_work_size = CONCAT_SIZE;
	const size_t local_work_size = INPUT_SIZE;

	clEnqueueWriteBuffer(	queue, 
				input_a_buf,
				CL_FALSE, 0, 
				sizeof(float)*INPUT_SIZE
				a, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				input_b_buf,
				CL_FALSE, 0, 
				sizeof(float)*OUTPUT_SIZE
				b, 
				0, NULL, NULL);
	clEnqueueWriteBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*CONCAT_SIZE,
				output, 
				0, NULL, NULL);

	status = clSetKernelArg(k_concat, 0, sizeof(cl_mem), &input_a_buf);
	checkError(status, "Failed to set concat arg 0");
	status = clSetKernelArg(k_concat, 1, sizeof(cl_mem), &input_b_buf);
	checkError(status, "Failed to set concat arg 1");
	status = clSetKernelArg(k_concat, 2, sizeof(cl_mem), &output_buf);
	checkError(status, "Failed to set concat arg 2");

	clEnqueueNDRangeKernel(	queue, 
				k_concat, 
				1, NULL, 
				&global_work_size, &local_work_size, 
				0, NULL, NULL);
	checkError(status, "Failed to launch matrix add kernel");

	clEnqueueReadBuffer(	queue, 
				output_buf,
				CL_FALSE, 0, 
				sizeof(float)*CONCAT_SIZE,
				output, 
				0, NULL, NULL);
	clFinish(queue);
}
