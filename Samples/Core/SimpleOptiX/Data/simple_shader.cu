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

rtDeclareVariable(int,          gBounces, , );
rtDeclareVariable(PerRayData, prd, rtPayload, );

#define RAY_PRIMARY     0
#define RAY_SHADOW      1
#define RAY_INDIRECT    2

RT_PROGRAM void closest_hit_radiance()
{
    if(prd.depth == RAY_SHADOW)
    {
        prd.t = t_hit;
        return;
    }
    else if(prd.depth == RAY_PRIMARY)
        prd.result = v3(0.f);
    
    // Fetch, interpolate, and prepare colors and fetch textures
    ShadingAttribs shAttr;
    interpolateAndPrepareShadingAttribs(shAttr);

    ShadingOutput result;

    // Evaluate all lights
    for(size_t i=0;i<gLights.size();++i)
    {
	    LightAttribs LAttr;
	    prepareLightAttribs(gLights[i], shAttr, LAttr);
        optix::Ray ray2 = optix::make_Ray(shAttr.P, LAttr.L, 0, scene_epsilon, RT_DEFAULT_MAX);
        // Do shadowing if enabled
        PerRayData prd2;
        prd2.t = RT_DEFAULT_MAX;
        prd2.depth = RAY_SHADOW;
        if(gBounces > 0)
            rtTrace(top_object, ray2, prd2);
        // Evaluate material
        float lDist = length(gLights[i].worldPos - shAttr.P);
        if(prd2.t >= lDist * 0.9999f)
            evalMaterial(shAttr, LAttr, result);
    }

    // Diffuse indirect bounce
    if(prd.depth == RAY_PRIMARY && gBounces > 1)
    {
        // Sample diffuse
        uint seed = launch_index.x * 67 + 53 * launch_index.y + 71 * gIterationCount;
        vec3 wi = cosine_sample_hemisphere(rand_next(seed), rand_next(seed));
        vec3 outDir = toLocal(wi, shAttr.T, shAttr.B, shAttr.N);
        optix::Ray ray2 = optix::make_Ray(shAttr.P, outDir, 0, scene_epsilon, RT_DEFAULT_MAX);
        PerRayData prd2;
        prd2.depth = RAY_INDIRECT;
        prd2.result = v3(0.f);
        rtTrace(top_object, ray2, prd2);
        prd.result += v3(getDiffuseColor(shAttr)) * prd2.result;
    }

    prd.result += result.finalValue;
}
