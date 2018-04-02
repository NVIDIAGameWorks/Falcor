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
#ifndef __RAYTRACING_H__
#define __RAYTRACING_H__

__exported __import Shading;
__exported __import DefaultVS;

ByteAddressBuffer gIndices      : register(t50);
ByteAddressBuffer gTexCrds      : register(t51);
ByteAddressBuffer gNormals      : register(t52);
ByteAddressBuffer gBitangents   : register(t53);
ByteAddressBuffer gPositions    : register(t54);
ByteAddressBuffer gLightMapUVs  : register(t55);
shared RaytracingAccelerationStructure gRtScene : register(t56);

struct BuiltinIntersectionAttribs
{
    float2 baryCrd;
};

shared cbuffer DxrPerFrame : register(b13)
{
    uint hitProgramCount;
};

uint3 getIndices(uint triangleIndex)
{
    uint baseIndex = triangleIndex * 3;
    int address = baseIndex * 4;
    return gIndices.Load3(address);
}

VertexOut getVertexAttributes(uint triangleIndex, float3 barycentrics)
{
    uint3 indices = getIndices(triangleIndex);
    VertexOut v;
    v.texC = 0;
    v.normalW = 0;
    v.bitangentW = 0;
    v.posW = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    v.colorV = 0;
    v.prevPosH = 0;
    v.lightmapC = 0;

    [unroll]
    for (int i = 0; i < 3; i++)
    {
        int address = (indices[i] * 3) * 4;
        v.texC       += asfloat(gTexCrds.Load2(address))     * barycentrics[i];
        v.normalW    += asfloat(gNormals.Load3(address))     * barycentrics[i];
        v.bitangentW += asfloat(gBitangents.Load3(address))  * barycentrics[i];
        v.lightmapC  += asfloat(gLightMapUVs.Load2(address)) * barycentrics[i];
    }
    v.normalW = normalize(v.normalW);
    v.bitangentW = normalize(v.bitangentW);
    return v;
}

VertexOut getVertexAttributes(uint triangleIndex, BuiltinIntersectionAttribs attribs)
{
    float3 barycentrics = float3(1.0 - attribs.baryCrd.x - attribs.baryCrd.y, attribs.baryCrd.x, attribs.baryCrd.y);
    return getVertexAttributes(triangleIndex, barycentrics);
}

float3 getGeoNormal(const float3 e[2])
{
    return normalize(cross(e[0], e[1]));
}

void getTriNormalsAndEdges(uint triangleIndex, out float3 n[3], out float3 e[2])
{
    uint3 indices = getIndices(triangleIndex);

    float3 p[3];
    p[0] = asfloat(gPositions.Load3((indices[0] * 3) * 4));
    p[1] = asfloat(gPositions.Load3((indices[1] * 3) * 4));
    p[2] = asfloat(gPositions.Load3((indices[2] * 3) * 4));

    e[0] = p[1] - p[0];
    e[1] = p[2] - p[0];

    n[0] = asfloat(gNormals.Load3((indices[0] * 3) * 4));
    n[1] = asfloat(gNormals.Load3((indices[1] * 3) * 4));
    n[2] = asfloat(gNormals.Load3((indices[2] * 3) * 4));
}
#endif // __RAYTRACING_H__