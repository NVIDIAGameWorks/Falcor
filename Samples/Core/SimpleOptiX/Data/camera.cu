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
#include "declarations.cuh"

rtDeclareVariable(int,           gSupersample, , );
rtDeclareVariable(float,         gExposure, , );

_fn int reverseBits(const int i,const int p) {if (i==0) return i; else return p-i;}
template<int b>
_fn float halton(int j)
{
	const int primes[61]=
    {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,
	83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,
	191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283
    };
	const int p = primes[b]; 
	double h = 0.0, f = 1.0 / (double)p;
    double factor = f;

	while (j > 0) 
    {
		h += reverseBits(j % p, p) * factor; 
        j /= p; 
        factor *= f;
	}
	return float(h);
}

RT_PROGRAM void pinhole()
{
    vec2 offset = v2(0.5f);
    if(gSupersample != 0)
    {
        // 0th and 1st prime number corresponds to the 2,3 Halton sequence
        offset.x = halton<0>(int(gIterationCount));
		offset.y = halton<1>(int(gIterationCount));
    }

    float2 d = (v2(launch_index) + offset) / v2(launch_dim) * 2.f - 1.f;
	float3 ray_origin = gCams[0].position;
	float3 ray_direction = normalize(d.x*gCams[0].cameraU + d.y*gCams[0].cameraV + gCams[0].cameraW);
  
    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

    PerRayData prd;
    prd.result = v3(0.38f, 0.52f, 0.10f);
    prd.depth = 0;

    rtTrace(top_object, ray, prd);

    accumulate(prd.result * gExposure);
}
