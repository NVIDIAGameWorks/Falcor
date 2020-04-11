/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"
#ifdef HOST_CODE
#include "glm/packing.hpp"
#endif

BEGIN_NAMESPACE_FALCOR

struct MeshDesc
{
    uint vbOffset;
    uint ibOffset;
    uint vertexCount; // #SCENE This is probably only needed on the CPU
    uint indexCount; // #SCENE This is probably only needed on the CPU
    uint materialID;
};

enum MeshInstanceFlags
{
    None = 0x0,
    Flipped = 0x1
};

// TODO: Pack into 16B by encoding IDs and flags with fewer bits. See #795.
struct MeshInstanceData
{
    uint globalMatrixID;
    uint materialID;
    uint meshID;
    uint flags; ///< MeshInstanceFlags
    uint vbOffset;
    uint ibOffset;
    uint pad[2];
};

struct StaticVertexData
{
    float3 position;
    float3 normal;
    float3 bitangent;
    float2 texCrd;
};

/** Vertex data packed into 32B for aligned access.
*/
struct PackedStaticVertexData
{
    float3 position;
    float3 packedNormalBitangent; // 3xfp16 each
    float2 texCrd;

#ifdef HOST_CODE
    PackedStaticVertexData(const StaticVertexData& v) { pack(v); }
    void pack(const StaticVertexData& v)
    {
        position = v.position;
        texCrd = v.texCrd;

        auto asfloat = [](uint32_t v) { return *reinterpret_cast<float*>(&v); };
        //auto asfloat = [](uint32_t v) { return std::bit_cast<float>(v); }; // TODO: Not yet available in VS2019

        packedNormalBitangent.x = asfloat(glm::packHalf2x16({ v.normal.x, v.normal.y }));
        packedNormalBitangent.y = asfloat(glm::packHalf2x16({ v.normal.z, v.bitangent.x }));
        packedNormalBitangent.z = asfloat(glm::packHalf2x16({ v.bitangent.y, v.bitangent.z }));
    }

#else // !HOST_CODE
    [mutating] void pack(const StaticVertexData v)
    {
        position = v.position;
        texCrd = v.texCrd;

        uint3 n = f32tof16(v.normal);
        uint3 b = f32tof16(v.bitangent);

        packedNormalBitangent.x = asfloat((n.y << 16) | n.x);
        packedNormalBitangent.y = asfloat((b.x << 16) | n.z);
        packedNormalBitangent.z = asfloat((b.z << 16) | b.y);
    }

    StaticVertexData unpack()
    {
        StaticVertexData v;
        v.position = position;
        v.texCrd = texCrd;

        float3 n;
        n.x = f16tof32(asuint(packedNormalBitangent.x) & 0xffff);
        n.y = f16tof32(asuint(packedNormalBitangent.x) >> 16);
        n.z = f16tof32(asuint(packedNormalBitangent.y) & 0xffff);
        v.normal = normalize(n);

        float3 b;
        b.x = f16tof32(asuint(packedNormalBitangent.y) >> 16);
        b.y = f16tof32(asuint(packedNormalBitangent.z) & 0xffff);
        b.z = f16tof32(asuint(packedNormalBitangent.z) >> 16);
        v.bitangent = normalize(b);

        return v;
    }
#endif
};

struct PrevVertexData
{
    float3 position;
};

struct DynamicVertexData
{
    uint4 boneID;
    float4 boneWeight;
    uint staticIndex;   // The index in the static vertex buffer
    uint globalMatrixID;
};

struct VertexData
{
    float3 posW;            ///< Position in world space.
    float3 normalW;         ///< Shading normal in world space.
    float3 bitangentW;      ///< Shading bitangent in world space.
    float2 texC;            ///< Texture coordinate.
    float3 faceNormalW;     ///< Face normal in world space.
    float  coneTexLODValue; ///< Texture LOD data for cone tracing. This is zero, unless getVertexDataRayCones() is used
};

END_NAMESPACE_FALCOR
