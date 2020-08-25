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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
#include "Utils/Math/PackedFormats.h"
#else
import Utils.Math.PackedFormats;
#endif

BEGIN_NAMESPACE_FALCOR

struct MeshDesc
{
    uint vbOffset;      ///< Offset into global vertex buffer.
    uint ibOffset;      ///< Offset into global index buffer, or zero if non-indexed.
    uint vertexCount;   ///< Vertex count.
    uint indexCount;    ///< Index count, or zero if non-indexed.
    uint materialID;

    uint getTriangleCount() CONST_FUNCTION
    {
        return (indexCount > 0 ? indexCount : vertexCount) / 3;
    }
};

enum class MeshInstanceFlags
// TODO: Remove the ifdefs and the include when Slang supports enum type specifiers.
#ifdef HOST_CODE
    : uint32_t
#endif
{
    None = 0x0,
    Flipped = 0x1
};

struct MeshInstanceData
{
    uint globalMatrixID;
    uint materialID;
    uint meshID;
    uint flags;         ///< MeshInstanceFlags.
    uint vbOffset;      ///< Offset into global vertex buffer.
    uint ibOffset;      ///< Offset into global index buffer, or zero if non-indexed.
};

/** Mesh instance data packed into 16B.
*/
struct PackedMeshInstanceData
{
    uint materialID;
    uint packedIDs;     ///< Packed meshID, globalMatrixID, and flags packed into 32 bits.
    uint vbOffset;      ///< Offset into global vertex buffer.
    uint ibOffset;      ///< Offset into global index buffer, or zero if non-indexed.

    // Packed representation

    static const uint kMatrixBits = 16;
    static const uint kMeshBits = 15;
    static const uint kFlagsBits = 1;

    static const uint kMatrixOffset = 0;
    static const uint kMeshOffset = kMatrixOffset + kMatrixBits;
    static const uint kFlagsOffset = kMeshOffset + kMeshBits;

#ifdef HOST_CODE
    void pack(const MeshInstanceData& d)
    {
        materialID = d.materialID;
        vbOffset = d.vbOffset;
        ibOffset = d.ibOffset;

        assert(d.flags < (1 << kFlagsBits));
        assert(d.meshID < (1 << kMeshBits));
        assert(d.globalMatrixID < (1 << kMatrixBits));
        packedIDs = (d.flags << kFlagsOffset) | (d.meshID << kMeshOffset) | (d.globalMatrixID << kMatrixOffset);
    }
#endif

    MeshInstanceData unpack()
    {
        MeshInstanceData d;
        d.materialID = materialID;
        d.vbOffset = vbOffset;
        d.ibOffset = ibOffset;

        d.globalMatrixID = (packedIDs >> kMatrixOffset) & ((1 << kMatrixBits) - 1);
        d.meshID = (packedIDs >> kMeshOffset) & ((1 << kMeshBits) - 1);
        d.flags = (packedIDs >> kFlagsOffset) & ((1 << kFlagsBits) - 1);

        return d;
    }
};

struct StaticVertexData
{
    float3 position;    ///< Position.
    float3 normal;      ///< Shading normal.
    float4 tangent;     ///< Shading tangent. The bitangent is computed: cross(normal, tangent.xyz) * tangent.w.
    float2 texCrd;      ///< Texture coordinates.
};

/** Vertex data packed into 32B for aligned access.
*/
struct PackedStaticVertexData
{
    float3 position;
    float3 packedNormalTangent;
    float2 texCrd;

#ifdef HOST_CODE
    PackedStaticVertexData(const StaticVertexData& v) { pack(v); }
    void pack(const StaticVertexData& v)
    {
        position = v.position;
        texCrd = v.texCrd;

        packedNormalTangent.x = asfloat(glm::packHalf2x16({ v.normal.x, v.normal.y }));
        packedNormalTangent.y = asfloat(glm::packHalf2x16({ v.normal.z, v.tangent.w }));
        packedNormalTangent.z = asfloat(encodeNormal2x16(v.tangent.xyz));
    }

#else // !HOST_CODE
    [mutating] void pack(const StaticVertexData v)
    {
        position = v.position;
        texCrd = v.texCrd;

        uint3 n = f32tof16(v.normal);
        uint t_w = f32tof16(v.tangent.w);

        packedNormalTangent.x = asfloat((n.y << 16) | n.x);
        packedNormalTangent.y = asfloat((t_w << 16) | n.z);
        packedNormalTangent.z = asfloat(encodeNormal2x16(v.tangent.xyz));
    }

    StaticVertexData unpack()
    {
        StaticVertexData v;
        v.position = position;
        v.texCrd = texCrd;

        v.normal.x = f16tof32(asuint(packedNormalTangent.x) & 0xffff);
        v.normal.y = f16tof32(asuint(packedNormalTangent.x) >> 16);
        v.normal.z = f16tof32(asuint(packedNormalTangent.y) & 0xffff);
        v.normal = normalize(v.normal);
       
        v.tangent.xyz = decodeNormal2x16(asuint(packedNormalTangent.z));
        v.tangent.w = f16tof32(asuint(packedNormalTangent.y) >> 16);

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
    uint staticIndex;       ///< The index in the static vertex buffer.
    uint globalMatrixID;
};

/** Struct representing interpolated vertex attributes in world space.
    Note the tangent is not guaranteed to be orthogonal to the normal.
    The bitangent should be computed: cross(normal, tangent.xyz) * tangent.w.
    The tangent space is orthogonalized in prepareShadingData().
*/
struct VertexData
{
    float3 posW;            ///< Position in world space.
    float3 normalW;         ///< Shading normal in world space (normalized).
    float4 tangentW;        ///< Shading tangent in world space (normalized). The last component is guaranteed to be +-1.0 or zero if tangents are missing.
    float2 texC;            ///< Texture coordinate.
    float3 faceNormalW;     ///< Face normal in world space (normalized).
    float  coneTexLODValue; ///< Texture LOD data for cone tracing. This is zero, unless getVertexDataRayCones() is used.
};

END_NAMESPACE_FALCOR
