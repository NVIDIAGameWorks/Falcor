/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "Data/HostDeviceData.h"    // For MAX_INSTANCES_BONES define

cbuffer PerModelCB
{
    float4x4 gBoneMat[MAX_BONES];               // Per-model bone matrices
    float4x4 gInvTransposeBoneMat[MAX_BONES];   // Per-model bone inverse transpose matrices
};

cbuffer PerMeshCB
{
    uint gNumVertices;
};

// Input vertex buffers
ByteAddressBuffer gPositions;
ByteAddressBuffer gNormals;
ByteAddressBuffer gBitangents;
ByteAddressBuffer gBoneWeights;
ByteAddressBuffer gBoneIds;

// Output vertex buffers
RWByteAddressBuffer gSkinnedPositions;
RWByteAddressBuffer gSkinnedPrevPositions;
RWByteAddressBuffer gSkinnedNormals;
RWByteAddressBuffer gSkinnedBitangents;

struct Vertex
{
    float3 pos;
    float3 prevPos;
#ifdef HAS_NORMAL
    float3 normal;
#endif
#ifdef HAS_BITANGENT
    float3 bitangent;
#endif
    float4 boneWeights;
    uint4 boneIds;
};

uint4 unpackUint4x8(uint packedInput)
{
    return uint4(packedInput, packedInput >> 8, packedInput >> 16, packedInput >> 24) & 0xff;
}

Vertex loadVertexAttributes(uint vertexIndex)
{
    // Load vertex data. The buffers are in RGB32Float format, so no addition conversion needed.
    Vertex v;
    v.pos = asfloat(gPositions.Load3((vertexIndex * 3) * 4));
#ifdef FIRST_FRAME
    v.prevPos = float3(0, 0, 0);
#else
    v.prevPos = asfloat(gSkinnedPositions.Load3((vertexIndex * 3) * 4));    // Load previously skinned position
#endif
#ifdef HAS_NORMAL
    v.normal = asfloat(gNormals.Load3((vertexIndex * 3) * 4));
#endif
#ifdef HAS_BITANGENT
    v.bitangent = asfloat(gBitangents.Load3((vertexIndex * 3) * 4));
#endif

    // Load bone weights and IDs
    v.boneWeights = asfloat(gBoneWeights.Load4((vertexIndex * 4) * 4));     // RGBA32Float
    v.boneIds = unpackUint4x8(gBoneIds.Load(vertexIndex * 4));              // RGBA8Uint

    return v;
}

void storeVertexAttributes(uint vertexIndex, Vertex v)
{
    gSkinnedPositions.Store3((vertexIndex * 3) * 4, asuint(v.pos));
    gSkinnedPrevPositions.Store3((vertexIndex * 3) * 4, asuint(v.prevPos));
#ifdef HAS_NORMAL
    gSkinnedNormals.Store3((vertexIndex * 3) * 4, asuint(v.normal));
#endif
#ifdef HAS_BITANGENT
    gSkinnedBitangents.Store3((vertexIndex * 3) * 4, asuint(v.bitangent));
#endif
}

float4x4 getBlendedBoneMat(float4 weights, uint4 ids)
{
    float4x4 boneMat = gBoneMat[ids.x] * weights.x;
    boneMat += gBoneMat[ids.y] * weights.y;
    boneMat += gBoneMat[ids.z] * weights.z;
    boneMat += gBoneMat[ids.w] * weights.w;

    return boneMat;
}

float3x3 getBlendedInvTransposeBoneMat(float4 weights, uint4 ids)
{
    float3x3 mat = (float3x3)gInvTransposeBoneMat[ids.x] * weights.x;
    mat += (float3x3)gInvTransposeBoneMat[ids.y] * weights.y;
    mat += (float3x3)gInvTransposeBoneMat[ids.z] * weights.z;
    mat += (float3x3)gInvTransposeBoneMat[ids.w] * weights.w;

    return mat;
}


[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint vertexId = dispatchThreadID.x;
    if (vertexId >= gNumVertices) return;

    Vertex vIn = loadVertexAttributes(vertexId);

    float4x4 boneMat = getBlendedBoneMat(vIn.boneWeights, vIn.boneIds);
    float3x3 invTransposeBoneMat = getBlendedInvTransposeBoneMat(vIn.boneWeights, vIn.boneIds);

    Vertex vOut;
    vOut.pos = mul(float4(vIn.pos, 1.f), boneMat).xyz;
#ifdef FIRST_FRAME
    vOut.prevPos = vOut.pos;    // First frame only, copy position to prevPos to avoid undefined values in shaders using it
#else
    vOut.prevPos = vIn.prevPos;
#endif
#ifdef HAS_NORMAL
    vOut.normal = mul(vIn.normal, invTransposeBoneMat).xyz;
#endif
#ifdef HAS_BITANGENT
    vOut.bitangent = mul(vIn.bitangent, (float3x3)boneMat).xyz;
#endif

    storeVertexAttributes(vertexId, vOut);
}
