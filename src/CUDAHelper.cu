#include "cudaUtil.h"
#include <stdio.h>

__global__ void testAngleThread()
{
	printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void initAngleThread(float3* ptr_angles, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
	if (idx < size) {
		ptr_angles[idx] = make_float3(0.0f, 0.0f, 0.0f);	
	}
}


extern "C" void initAngle(void* ptr_angles, unsigned int m_nNodes)
{
	const unsigned int threadsPerBlock = 1024;
	const unsigned int blocks = ceil(m_nNodes/float(threadsPerBlock));
	printf("block %d, thread %d\n\n", blocks, threadsPerBlock);
	initAngleThread<<<blocks, threadsPerBlock>>> ( (float3*)ptr_angles, m_nNodes );
	//testAngleThread<<<blocks, threadsPerBlock>>> ();
	cudaSafeCall(cudaDeviceSynchronize());	
}

