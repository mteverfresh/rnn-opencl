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
		|	
11/28/18	|	Out of time! Adapting for single pass kernels.

*/

#define X 0
#define Y 1
#define SOURCES 6
#define MAX_WINDOW_SIZE 1024
#define PREV_GROUP 0
#define CURR_GROUP 1

#define INDEX(ROW, COLUMN, WIDTH) ((ROW) * (WIDTH) + (COLUMN))

__attribute__((max_work_group_size(MAX_WINDOW_SIZE)))
__kernel void sigpass(
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
				out[INDEX(i, j, SOURCES)] = 1.0 / (1 + exp(-(products[0] + a[j])));
			}
		}
	}
}
