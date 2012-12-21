#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <ctime>

#define BLOCKS_NUM 4
#define BLOCK_SIZE 256
#define DATA_TYPE int


__global__ void reduce( DATA_TYPE* in, DATA_TYPE* out ){
	__shared__ int data[BLOCK_SIZE];

	int tid = threadIdx.x;
	int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

	data[ tid ] = in [ i ] + in[ i + blockDim.x ];
	__syncthreads();

	for ( int s = blockDim.x / 2; s > 0; s >>= 1 ){
		if ( tid < s) data[tid] += data[tid + s];
		__syncthreads();
	}
	
	__syncthreads();

	if ( tid == 0 ) out[blockIdx.x] = data[0];
}

using namespace std;

int main(){
	DATA_TYPE nums[ BLOCKS_NUM * BLOCK_SIZE ];

	int res = 0;
	srand(time(0));
	for( int i = 0; i < BLOCKS_NUM * BLOCK_SIZE; i++ ){
		if ( i < 1000 ) nums[ i ] = rand()%100 - 50;
		else nums[ i ] = 0;
		res += nums[ i ];
	}

	cout << "For summ: " << res << endl;

	cudaSetDevice( 0 );
	DATA_TYPE* in;
	DATA_TYPE* out;
	
	unsigned int in_memory_size = sizeof( DATA_TYPE ) * BLOCKS_NUM * BLOCK_SIZE;
	unsigned int out_memory_size = sizeof( DATA_TYPE ) * BLOCKS_NUM;

	cudaMalloc( ( void** ) &in, in_memory_size );
	cudaMalloc( ( void** ) &out, out_memory_size );
	
	cudaMemcpy( in, nums, in_memory_size, cudaMemcpyHostToDevice );
	
	
	dim3 block( BLOCK_SIZE );
	dim3 grid( BLOCKS_NUM );
	
	reduce<<< grid, block >>>( in, out );
	cudaDeviceSynchronize();
	cudaMemcpy( nums, out, out_memory_size, cudaMemcpyDeviceToHost );
		
	res = 0;
	for (int i = 0; i < 2; i++) res += nums[i];
	cout << "CUDA summ: " << res << endl;

	cin.get();

	cudaFree( in );
	cudaFree( out );
	return 0;	
}

