/* 
Filename: kernels.cl
Author: Zach Sherer
Purpose: Kernels for the LSTM.
Notes:
Acknowledgements:

Date		|	Change
--------------------------------------------------------------------------------------
11/04/18	|	File created to provide the acceleration for the LSTM.
		|
11/21/18	|	All kernels finished. Proceeding with testing for 2-input kernels.

*/

#define X 0
#define Y 1
#define SOURCES 6
#define MAX_WINDOW_SIZE 1024
#define PREV_GROUP 0
#define CURR_GROUP 1

#define INDEX(ROW, COLUMN, WIDTH) ((ROW) * (WIDTH) + (COLUMN))

__kernel void matrix_add(
	__global const	float *restrict a,
	__global const	float *restrict b,
	__global	float *restrict out
)
{
	const int tid = get_global_id(X);
	out[tid] = a[tid] + b[tid];
}

//requries 2d kernel of 128*6
__attribute__((max_work_group_size(MAX_WINDOW_SIZE)))
__kernel void matrix_mul(
	__global const	float *restrict a,
	__global const	float *restrict b,
	__global	float *restrict out
)
{
	const int tid = get_global_id(0);
	const int lid = get_local_id(0);
	const int gid = get_group_id(0);
	const int xsize = get_local_size(0);

	__local float products[MAX_WINDOW_SIZE]; //related to local size

	for(unsigned i = 0; i < SOURCES; i++)
	{
		//Multiply without transpose, something you can do when the sizes are set
		for(unsigned j = 0; j < SOURCES; j++)
		{
			products[tid] = a[INDEX(i, tid, xsize)] * b[INDEX(j, tid, xsize)];
			barrier(CLK_LOCAL_MEM_FENCE);

			//Now do the reduction
			for(unsigned k = xsize/2; k > 0; k /= 2)
			{
				if(tid < k)
				{
					float sum = products[tid] + products[tid + k];
					products[tid] = sum;
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
			if(tid == 0)
			{
				out[INDEX(i, j, SOURCES)] = products[0];
			}
		}
	}
}

__kernel void sigmoid_activation(
	__global const	float *restrict input,
	__global 	float *restrict output
)
{
	int tid = get_global_id(0);
	output[tid] = 1.0 / (1 + exp(-input[tid]));
}
__kernel void tanh_activation(
	__global const	float *restrict input,
	__global 	float *restrict output
)
{
	int tid = get_global_id(0);
	output[tid] = tanh(input[tid]);
}
//dimensions must be hardened for this kernel
//exactly two workgroups, one to write the vertices from each matrix
__kernel void matrix_concat(
	__global const 	float *restrict prev_input,
	__global const 	float *restrict curr_input,
	__global 	float *restrict output
)
{
	int tid = get_local_id(0);
	int gid = get_group_id(0);
	int abs_id = tid + gid*get_local_size(0);

	switch(gid)
	{
	case PREV_GROUP:	output[abs_id] = prev_input[tid]; break;
	case CURR_GROUP:	output[abs_id] = curr_input[tid]; break;
	}
}
	
