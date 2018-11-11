/* 
Filename: kernels.cl
Author: Zach Sherer
Purpose: Kernels for the LSTM.
Notes:
Acknowledgements:

Date		|	Change
--------------------------------------------------------------------------------------
11/04/18	|	File created to provide the acceleration for the LSTM.

*/

#define X 0
#define Y 1
#define SOURCES 6
#define PREV_GROUP 0
#define CURR_GROUP 1

__kernel void matrix_add(
	__global const	double restrict *a,
	__global const	double restrict *b,
	__global	double restrict *out
)
{
	const int tid = get_global_id(X);
	out[tid] = a[tid] + b[tid];
}

__kernel void matrix_mul(
	__global const	double restrict *a,
	__global const	double restrict *b,
	__global	double restrict *out
)
{
	const int row = get_global_id(X);
	const int col = get_global_id(Y);

	double sum = 0;
	for(unsigned i = 0; i < SOURCES; i++)
	{
		sum += a[i*
	
}

__kernel void sigmoid_activation(
	__global const	double restrict *input,
	__global 	double restrict *output
)
{
	int tid = get_global_id(0);
	output[tid] = 1.0 / (1 + exp(-input[tid]));
}
__kernel void tanh_activation(
	__global const	double restrict *input,
	__global 	double restrict *output
)
{
	int tid = get_global_id(0);
	output[tid] = tanh(input[tid]);
}
//dimensions must be hardened for this kernel
//exactly two workgroups, one to write the vertices from each matrix
__kernel void matrix_concat(
	__global const 	double restrict *prev_input,
	__global const 	double restrict *curr_input,
	__global 	double restrict *output
)
{
	int tid = get_local_id(0);
	int gid = get_group_id(0);
	int abs_id = tid + gid*get_local_size(0);

	switch(gid)
	{
	PREV_GROUP:	output[abs_id] = prev_input[tid];
	CURR_GROUP:	output[abs_id] = curr_input[tid];
	}
}
	
