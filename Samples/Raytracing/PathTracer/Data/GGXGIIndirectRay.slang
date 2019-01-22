/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
import ShaderCommon;
import Raytracing;
import Helpers;
import GGXGICommon;
#include "HostDeviceSharedMacros.h"

[shader("miss")]
void IndirectMiss(inout IndirectRayPayload rayData)
{
    // Sample the environment map
    float2 uv = dirToSphericalCrd(WorldRayDirection());
    float2 dims;
    gEnvMap.GetDimensions(dims.x, dims.y);
    rayData.color = float3(gEnvMap[uint2(uv * dims)].rgb);
}

[shader("anyhit")]
void IndirectAnyHit(inout IndirectRayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    if (evalRtAlphaTest(attribs))
        IgnoreHit();
}

[shader("closesthit")]
void IndirectClosestHit(inout IndirectRayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    VertexOut vOut = getVertexAttributes(PrimitiveIndex(), attribs);
    ShadingData sd = prepareShadingData(vOut, gMaterial, WorldRayOrigin(), 0);

    // Add emissive color
    rayData.color = gEmitMult * sd.emissive.rgb;

    // Do direct illumination at this hit location
    if (gDoDirectGI) rayData.color += ggxDirect(rayData.rndSeed, sd);

    // Do indirect illumination at this hit location (if we haven't traversed too far)
    // Indirect bounces after the first will not check against geometric normals (and therefore possibly leak light) because
    //  prepareShadingData does not provide that information. This will still produce a good enough result.
    if (rayData.rayDepth < gMaxDepth) rayData.color += ggxIndirect(rayData.rndSeed, sd, sd.N, rayData.rayDepth);
}
