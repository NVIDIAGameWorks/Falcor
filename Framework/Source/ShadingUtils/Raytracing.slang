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

#include "HostDeviceData.h"

// If defined, hit position is computed by barycentric interpolation of the vertex positions. 
// Otherwise it is computed based on the ray equation in world space: p=o+t*d, which is numerically unstable.
// Unfortunately, interpolating the position incurs the extra cost of fetching 3x12B positions and one matrix multiply.
#define USE_INTERPOLATED_POSITION

shared cbuffer DxrPerFrame : register(b13)
{
    uint hitProgramCount;
};

#ifndef FALCOR_VK

ByteAddressBuffer gIndices       : register(t50);
ByteAddressBuffer gTexCrds       : register(t51);
ByteAddressBuffer gNormals       : register(t52);
ByteAddressBuffer gBitangents    : register(t53);
ByteAddressBuffer gPositions     : register(t54);
ByteAddressBuffer gPrevPositions : register(t55);
ByteAddressBuffer gLightMapUVs   : register(t56);
shared RaytracingAccelerationStructure gRtScene : register(t57);

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
    v.normalW = mul(v.normalW, (float3x3)gWorldInvTransposeMat[0]).xyz;
    v.bitangentW = mul(v.bitangentW, (float3x3)gWorldMat[0]).xyz;
#endif
    v.normalW = normalize(v.normalW);
    v.bitangentW = normalize(v.bitangentW);
    return v;
}

/** Returns a triangle's vertex normals and edges in object space.
    \param[in] triangleIndex Index of the triangle.
    \param[out] n Vertex normals in object space.
    \param[out] e Triangle edges in object space.
*/
void getTriNormalsAndEdgesInObjectSpace(uint triangleIndex, out float3 n[3], out float3 e[2])
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

/** Returns a triangle's geometric normal in world space.
    \param[in] triangleIndex Index of the triangle.
    \return Geometric normal in world space. Front facing for counter-clockwise winding.
*/
float3 getGeometricNormalW(uint triangleIndex)
{
    uint3 indices = getIndices(triangleIndex);

    float3 p[3];
    p[0] = asfloat(gPositions.Load3((indices[0] * 3) * 4));
    p[1] = asfloat(gPositions.Load3((indices[1] * 3) * 4));
    p[2] = asfloat(gPositions.Load3((indices[2] * 3) * 4));

    float3 e[2];
    e[0] = p[1] - p[0];
    e[1] = p[2] - p[0];

    float3 N = getGeoNormal(e);
    return mul(N, (float3x3)gWorldInvTransposeMat[0]).xyz;
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

/** Prepare the hit-point data
*/
ShadingData rtPrepareShadingData(VertexOut v, MaterialData m, float3 camPosW)
{
    return prepareShadingData(v, m, camPosW);
}

ShadingData rtPrepareShadingData(VertexOut v, MaterialData m, float3 camPosW, float lod)
{
    return prepareShadingData(v, m, camPosW, lod);
}

#else // if defined(FALCOR_VK)

#ifndef RT_GEOMETRY_COUNT
#define RT_GEOMETRY_COUNT 1
#endif

StructuredBuffer<uint> gIndices[RT_GEOMETRY_COUNT];
StructuredBuffer<float> gTexCrds[RT_GEOMETRY_COUNT];
StructuredBuffer<float> gNormals[RT_GEOMETRY_COUNT];
StructuredBuffer<float> gBitangents[RT_GEOMETRY_COUNT];
StructuredBuffer<float> gPositions[RT_GEOMETRY_COUNT];
StructuredBuffer<float> gPrevPositions[RT_GEOMETRY_COUNT];
StructuredBuffer<float> gLightMapUVs[RT_GEOMETRY_COUNT];
RaytracingAccelerationStructure gRtScene;

// Replaces InternalPerMeshCB
[vk::shader_record]
cbuffer InternalShaderRecord
{
    float4x4 gWorldMatLocal;              // Per-instance world transforms
    float4x4 gPrevWorldMatLocal;          // Previous frame world transforms
    float3x4 gWorldInvTransposeMatLocal;  // Per-instance matrices for transforming normals

    uint gGeometryID;
};

struct MaterialConstants
{
    float4 baseColor;
    float4 specular;
    float3 emissive;
    float padf;
    float alphaThreshold;
    float IoR;
    uint32_t id;
    uint32_t flags;
    float2 heightScaleOffset;
    float2 pad;
};

struct MaterialArray
{
    Texture2D baseColor[RT_GEOMETRY_COUNT];
    Texture2D specular[RT_GEOMETRY_COUNT];
    Texture2D emissive[RT_GEOMETRY_COUNT];
    Texture2D normalMap[RT_GEOMETRY_COUNT];

    Texture2D occlusionMap[RT_GEOMETRY_COUNT];
    Texture2D lightMap[RT_GEOMETRY_COUNT];
    Texture2D heightMap[RT_GEOMETRY_COUNT];
    SamplerState samplerState[RT_GEOMETRY_COUNT];

    RWStructuredBuffer<MaterialConstants> constants;
};

// Replaces gMaterial
ParameterBlock<MaterialArray> gRtMaterials;

#define bindless(descArray) (descArray[NonUniformResourceIndex(gGeometryID)])

float2 loadFloat2(StructuredBuffer<float> buffer, uint address)
{
    float2 v = 0.0;
    v[0] = buffer[address];
    v[1] = buffer[address + 1];

    return v;
}

float3 loadFloat3(StructuredBuffer<float> buffer, uint address)
{
    float3 v = 0.0;
    v[0] = buffer[address];
    v[1] = buffer[address + 1];
    v[2] = buffer[address + 2];

    return v;
}

uint3 getIndices(uint triangleIndex)
{
    uint baseIndex = triangleIndex * 3;

    uint3 indices;
    indices[0] = bindless(gIndices)[baseIndex];
    indices[1] = bindless(gIndices)[baseIndex + 1];
    indices[2] = bindless(gIndices)[baseIndex + 2];

    return indices;
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
        int address = indices[i] * 3; // Stride is 3 for all vertex attributes
        v.texC       += loadFloat2(bindless(gTexCrds), address)     * barycentrics[i];
        v.normalW    += loadFloat3(bindless(gNormals), address)     * barycentrics[i];
        v.bitangentW += loadFloat3(bindless(gBitangents), address)  * barycentrics[i];
        v.lightmapC  += loadFloat2(bindless(gLightMapUVs), address) * barycentrics[i];
#ifdef USE_INTERPOLATED_POSITION
        v.posW       += loadFloat3(bindless(gPositions), address)   * barycentrics[i];
#endif
    }
#ifdef USE_INTERPOLATED_POSITION
    v.posW = mul(float4(v.posW, 1.f), gWorldMatLocal).xyz;
#endif
#ifndef _MS_DISABLE_INSTANCE_TRANSFORM
    // Transform normal/bitangent to world space
    v.normalW = mul(v.normalW, (float3x3)gWorldInvTransposeMatLocal).xyz;
    v.bitangentW = mul(v.bitangentW, (float3x3)gWorldMatLocal).xyz;
#endif
    v.normalW = normalize(v.normalW);
    v.bitangentW = normalize(v.bitangentW);
    return v;
}

/** Returns a triangle's vertex normals and edges in object space.
    \param[in] triangleIndex Index of the triangle.
    \param[out] n Vertex normals in object space.
    \param[out] e Triangle edges in object space.
*/
void getTriNormalsAndEdgesInObjectSpace(uint triangleIndex, out float3 n[3], out float3 e[2])
{
    uint3 indices = getIndices(triangleIndex);

    float3 p[3];
    p[0] = loadFloat3(bindless(gPositions), indices[0]);
    p[1] = loadFloat3(bindless(gPositions), indices[1]);
    p[2] = loadFloat3(bindless(gPositions), indices[2]);

    e[0] = p[1] - p[0];
    e[1] = p[2] - p[0];

    n[0] = loadFloat3(bindless(gNormals), indices[0]);
    n[1] = loadFloat3(bindless(gNormals), indices[1]);
    n[2] = loadFloat3(bindless(gNormals), indices[2]);
}

/** Returns a triangle's geometric normal in world space.
    \param[in] triangleIndex Index of the triangle.
    \return Geometric normal in world space. Front facing for counter-clockwise winding.
*/
float3 getGeometricNormalW(uint triangleIndex)
{
    uint3 indices = getIndices(triangleIndex);

    float3 p[3];
    p[0] = loadFloat3(bindless(gPositions), indices[0]);
    p[1] = loadFloat3(bindless(gPositions), indices[1]);
    p[2] = loadFloat3(bindless(gPositions), indices[2]);

    float3 e[2];
    e[0] = p[1] - p[0];
    e[1] = p[2] - p[0];

    float3 N = getGeoNormal(e);
    return mul(N, (float3x3)gWorldInvTransposeMatLocal).xyz;
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
        prevPos += loadFloat3(bindless(gPositions), indices[i]) * barycentrics[i];
    }

    return mul(float4(prevPos, 1.f), gPrevWorldMatLocal).xyz;
}

/** Prepare the hit-point data
*/
ShadingData rtPrepareShadingData(VertexOut v, MaterialData _, float3 camPosW)
{
    ExplicitLodTextureSampler lod = { 0 };

    MaterialConstants m = bindless(gRtMaterials.constants);

    ShadingData sd = initShadingData();

#ifdef _MS_STATIC_MATERIAL_FLAGS
    m.flags = _MS_STATIC_MATERIAL_FLAGS;
#endif

    // Sample the diffuse texture and apply the alpha test
    float4 baseColor = sampleTexture(bindless(gRtMaterials.baseColor), bindless(gRtMaterials.samplerState), v.texC, m.baseColor, EXTRACT_DIFFUSE_TYPE(m.flags), lod);
    sd.opacity = m.baseColor.a;
    applyAlphaTest(m.flags, baseColor.a, m.alphaThreshold, v.posW);

    sd.posW = v.posW;
    sd.uv = v.texC;
    sd.V = normalize(camPosW - v.posW);
    sd.N = normalize(v.normalW);

    sd.B = normalize(v.bitangentW - sd.N * (dot(v.bitangentW, sd.N)));
    sd.T = normalize(cross(sd.B, sd.N));

    // Sample the spec texture
    bool sampleOcclusion = EXTRACT_OCCLUSION_MAP(m.flags) > 0;
    float4 spec = sampleTexture(bindless(gRtMaterials.specular), bindless(gRtMaterials.samplerState), v.texC, m.specular, EXTRACT_SPECULAR_TYPE(m.flags), lod);
    if (EXTRACT_SHADING_MODEL(m.flags) == ShadingModelMetalRough)
    {
        // R - Occlusion; G - Roughness; B - Metalness
        sd.diffuse = lerp(baseColor.rgb, float3(0), spec.b);

        // UE4 uses 0.08 multiplied by a default specular value of 0.5 as a base, hence the 0.04
        sd.specular = lerp(float3(0.04f), baseColor.rgb, spec.b);
        sd.linearRoughness = spec.g;

        if (sampleOcclusion) sd.occlusion = spec.r;
    }
    else // if (EXTRACT_SHADING_MODEL(m.flags) == ShadingModelSpecGloss)
    {
        sd.diffuse = baseColor.rgb;
        sd.specular = spec.rgb;
        sd.linearRoughness = 1 - spec.a;

        if(sampleOcclusion)
        {
            sd.occlusion = sampleTexture(bindless(gRtMaterials.occlusionMap), bindless(gRtMaterials.samplerState), v.texC, 1, ChannelTypeTexture, lod);
        }
    }

    sd.linearRoughness = max(0.08, sd.linearRoughness); // Clamp the roughness so that the BRDF won't explode
    sd.roughness = sd.linearRoughness * sd.linearRoughness;
    sd.emissive = sampleTexture(bindless(gRtMaterials.emissive), bindless(gRtMaterials.samplerState), v.texC, float4(m.emissive, 1), EXTRACT_EMISSIVE_TYPE(m.flags), lod).rgb;
    sd.IoR = m.IoR;
    sd.doubleSidedMaterial = EXTRACT_DOUBLE_SIDED(m.flags);

#define channel_type(extract) (extract(m.flags) ? ChannelTypeTexture : ChannelTypeUnused)
    sd.lightMap = sampleTexture(bindless(gRtMaterials.lightMap), bindless(gRtMaterials.samplerState), v.lightmapC, 1, channel_type(EXTRACT_LIGHT_MAP), lod).rgb;
    sd.height = sampleTexture(bindless(gRtMaterials.heightMap), bindless(gRtMaterials.samplerState), v.texC, 1, channel_type(EXTRACT_HEIGHT_MAP), lod).xy;
    sd.height = sd.height * m.heightScaleOffset.x + m.heightScaleOffset.y;
#undef channel_type

    // Apply normal map
    uint mapType = EXTRACT_NORMAL_MAP_TYPE(m.flags);
    if(mapType != NormalMapUnused)
    {
        float3 mapN = lod.sampleTexture(bindless(gRtMaterials.normalMap), bindless(gRtMaterials.samplerState), sd.uv).xyz;
        switch(mapType)
        {
        case NormalMapRGB:
            mapN = RgbToNormal(mapN);
            break;
        case NormalMapRG:
            mapN = RgToNormal(mapN.rg);
            break;
        }

        // Apply the transformation. Everything is normalized already
        sd.N = sd.T * mapN.x + sd.B * mapN.y + sd.N * mapN.z;
        sd.B = normalize(sd.B - sd.N * dot(sd.B, sd.N));
        sd.T = normalize(cross(sd.B, sd.N));
    }

    sd.NdotV = dot(sd.N, sd.V);

    // Flip the normal if it's backfacing
    if(sd.NdotV <= 0 && sd.doubleSidedMaterial)
    {
        sd.N = -sd.N;
        sd.NdotV = -sd.NdotV;
    }

    return sd;
}

ShadingData rtPrepareShadingData(VertexOut v, MaterialData m, float3 camPosW, float lod)
{
    return rtPrepareShadingData(v, m, camPosW);
}

#endif

// Common

float3 getGeoNormal(const float3 e[2])
{
    return normalize(cross(e[0], e[1]));
}

float3 getPrevPosW(uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    return getPrevPosW(triangleIndex, barycentrics);
}

VertexOut getVertexAttributes(uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    return getVertexAttributes(triangleIndex, barycentrics);
}

#endif // __RAYTRACING_H__
