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

ByteAddressBuffer gIndices       : register(t50);
ByteAddressBuffer gTexCrds       : register(t51);
ByteAddressBuffer gNormals       : register(t52);
ByteAddressBuffer gBitangents    : register(t53);
ByteAddressBuffer gPositions     : register(t54);
ByteAddressBuffer gPrevPositions : register(t55);
ByteAddressBuffer gLightMapUVs   : register(t56);
shared RaytracingAccelerationStructure gRtScene : register(t57);

// If defined, hit position is computed by barycentric interpolation of the vertex positions. 
// Otherwise it is computed based on the ray equation in world space: p=o+t*d, which is numerically unstable.
// Unfortunately, interpolating the position incurs the extra cost of fetching 3x12B positions and one matrix multiply.
#define USE_INTERPOLATED_POSITION

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
#ifdef USE_INTERPOLATED_POSITION
    v.posW = 0;
#else
    v.posW = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
#endif
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
#ifdef USE_INTERPOLATED_POSITION
        v.posW       += asfloat(gPositions.Load3(address))   * barycentrics[i];
#endif
    }
#ifdef USE_INTERPOLATED_POSITION
    v.posW = mul(float4(v.posW, 1.f), gWorldMat[0]).xyz;
#endif
#ifndef _MS_DISABLE_INSTANCE_TRANSFORM
    // Transform normal/bitangent to world space
    // PETRIK TODO: LightTransport relies on getVertexAttributes() returning normalW and bitangentW in world space (as their names imply). If this is made the default, the Falcor RT sample(s) need to be updated.
    v.normalW = mul(v.normalW, (float3x3)gWorldInvTransposeMat[0]).xyz;
    v.bitangentW = mul(v.bitangentW, (float3x3)gWorldMat[0]).xyz;
#endif
    v.normalW = normalize(v.normalW);
    v.bitangentW = normalize(v.bitangentW);
    return v;
}

VertexOut getVertexAttributes(uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
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

/** Returns geometric normal of the specified triangle.
    \param[in] triangleIndex Index of the triangle.
    \return Geometric normal. Front facing for counter-clockwise winding.
*/
float3 getGeometricNormal(uint triangleIndex)
{
    uint3 indices = getIndices(triangleIndex);

    float3 p[3];
    p[0] = asfloat(gPositions.Load3((indices[0] * 3) * 4));
    p[1] = asfloat(gPositions.Load3((indices[1] * 3) * 4));
    p[2] = asfloat(gPositions.Load3((indices[2] * 3) * 4));

    float3 e[2];
    e[0] = p[1] - p[0];
    e[1] = p[2] - p[0];

    return getGeoNormal(e);
}

/** Returns position on triangle in the previous frame in world space.
*/
float3 getPrevPosW(uint triangleIndex, float3 barycentrics)
{
    uint3 indices = getIndices(triangleIndex);
    float3 prevPos = float3(0, 0, 0);

    [unroll]
    for (int i = 0; i < 3; i++)
    {
        // Load vertex in object space from vertex buffer for previous frame if it exists, otherwise from the current frame.
        int address = (indices[i] * 3) * 4;
        prevPos += asfloat(gPrevPositions.Load3(address)) * barycentrics[i];
    }

    return mul(float4(prevPos, 1.f), gPrevWorldMat[0]).xyz;
}

float3 getPrevPosW(uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    return getPrevPosW(triangleIndex, barycentrics);
}

#endif // __RAYTRACING_H__