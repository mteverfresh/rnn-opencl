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
__kernel void matrix_mul(
	__global const	float *restrict a,
	__global const	float *restrict b,
	__global	float *restrict out
)
{
	const int row = get_global_id(X);
	const int col = get_global_id(Y);
	const int xsize = get_local_size(X);
	const int ysize = get_local_size(Y);

	float sum = 0;

	for(unsigned i = 0; i < ysize; i++)
	{
		sum += a[INDEX(row, i, xsize)] * b[INDEX(i, col, ysize)];
	}
	out[INDEX(row, col, xsize)] = sum;
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
	
