/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***************************************************************************/
#include <helper_math_cufalcor.h>


__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
	return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}


// Simple copy kernel
extern "C"
__global__ void kernelProcessTexture(	cudaTextureObject_t inputTexObj,
										cudaSurfaceObject_t outputSurfObj,
										int width, int height, uchar3 constColorAdd)
{

	float px = 1.0 / float(width);
	float py = 1.0 / float(height);


	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < (width-1) && y < (height-1) ) {
		//uchar4 data;
		float4 data=make_float4(0.0f);

		// Read from input texture
		uint i = 0; uint j = 0;
		for (j = 0; j < 4; j++)
		for (i = 0; i < 4; i++)
		{
			data += tex2D<float4>(inputTexObj, (x + i) * px, (y + j) * py);
		}

		data /= 16.0f;
		data.x *= 255.0;
		data.y *= 255.0;
		data.z *= 255.0;
		data.w *= 255.0;
	
        data.x += float(constColorAdd.x);
		data.y += float(constColorAdd.y);
		data.z += float(constColorAdd.z);
		//data.w += float(constColorAdd.w);
        
        data = clamp(data, make_float4(0,0,0,0), make_float4(255,255,255,255));
		// Write to output surface
		surf2Dwrite(to_uchar4(data), outputSurfObj, x * sizeof(uchar4), y);

		//surf2Dwrite(data, outputSurfObj, x, y);
	}
}


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


