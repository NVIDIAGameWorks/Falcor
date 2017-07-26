
#include <helper_math_cufalcor.h>


__constant__ uchar *d_vertPosBuff;

extern "C"
__global__ void kernelProcessVertices(float scale, uint numVertices, uint stride, size_t attribSize, uint attribOffset)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if (x < numVertices){

		float *vertComponent = (float*)( d_vertPosBuff+(x*(stride)+attribOffset + 1 * attribSize) );

		(*vertComponent) *= scale;

	}

}


