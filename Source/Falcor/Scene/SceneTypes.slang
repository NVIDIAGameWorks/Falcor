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

enum class MeshFlags : uint32_t
{
    None = 0x0,
    Use16BitIndices = 0x1,  ///< Indices are in 16-bit format. The default is 32-bit.
    HasDynamicData = 0x2,   ///< Mesh has dynamic vertex data.
    IsFrontFaceCW = 0x4,    ///< Front-facing side has clockwise winding in object space. Note that the winding in world space may be flipped due to the instance transform.
};

/** Mesh data stored in 64B.
*/
struct MeshDesc
{
    uint vbOffset;          ///< Offset into global vertex buffer.
    uint ibOffset;          ///< Offset into global index buffer, or zero if non-indexed.
    uint vertexCount;       ///< Vertex count.
    uint indexCount;        ///< Index count, or zero if non-indexed.
    uint dynamicVbOffset;   ///< Offset into dynamic vertex buffer, or zero if no dynamic data.
    uint materialID;        ///< Material ID.
    uint flags;             ///< See MeshFlags.
    uint _pad;

    uint getTriangleCount() CONST_FUNCTION
    {
        return (indexCount > 0 ? indexCount : vertexCount) / 3;
    }

    bool use16BitIndices() CONST_FUNCTION
    {
        return (flags & (uint)MeshFlags::Use16BitIndices) != 0;
    }

    bool isFrontFaceCW() CONST_FUNCTION
    {
        return (flags & (uint)MeshFlags::IsFrontFaceCW) != 0;
    }
};

enum class MeshInstanceFlags
// TODO: Remove the ifdefs and the include when Slang supports enum type specifiers.
#ifdef HOST_CODE
    : uint32_t
#endif
{
    None = 0x0,
    Use16BitIndices = 0x1,      ///< Indices are in 16-bit format. The default is 32-bit.
    HasDynamicData = 0x2,       ///< Mesh has dynamic vertex data.
    TransformFlipped = 0x4,     ///< Instance transform flips the coordinate system handedness. TODO: Deprecate this flag if we need an extra bit.
    IsObjectFrontFaceCW = 0x8,  ///< Front-facing side has clockwise winding in object space. Note that the winding in world space may be flipped due to the instance transform.
    IsWorldFrontFaceCW = 0x10,  ///< Front-facing side has clockwise winding in world space. This is the combination of the mesh winding and instance transform handedness.
};

struct MeshInstanceData
{
    uint globalMatrixID;
    uint materialID;
    uint meshID;
    uint flags;         ///< See MeshInstanceFlags.
    uint vbOffset;      ///< Offset into global vertex buffer.
    uint ibOffset;      ///< Offset into global index buffer, or zero if non-indexed.

    bool hasDynamicData() CONST_FUNCTION
    {
        return (flags & (uint)MeshInstanceFlags::HasDynamicData) != 0;
    }
};

/** Mesh instance data packed into 16B.
*/
struct PackedMeshInstanceData
{
    uint packedIDs[2];  ///< Packed materialID, meshID, globalMatrixID, and flags packed into 64 bits.
    uint vbOffset;      ///< Offset into global vertex buffer.
    uint ibOffset;      ///< Offset into global index buffer, or zero if non-indexed.

    // Packed representation
    // The pack/unpack logic assumes that meshID representation is split between packedIds[0] and packedIds[1]
    // (i.e., kFlagBigs + kMatrixBits < 32, kFlagsBits + kMatrixBits + kMeshBits > 32)
    // bit position  31.......................................0
    // packedIds[1]   |    kMaterialBits   |    kMeshBitsHi   |
    // packedIDs[0]   | kMeshBitsLo | kMatrixBits | kFlagBits |

    static const uint kMaterialBits = 17;
    static const uint kMatrixBits = 21;
    static const uint kMeshBits = 21;
    static const uint kFlagsBits = 5;

    static const uint kFlagsOffset = 0;
    static const uint kMatrixOffset = kFlagsOffset + kFlagsBits;
    static const uint kMeshOffsetLo = kMatrixOffset + kMatrixBits;
    static const uint kMeshBitsLo = 32 - kMeshOffsetLo;
    static const uint kMeshOffsetHi = 0;
    static const uint kMeshBitsHi = kMeshBits - kMeshBitsLo;
    static const uint kMaterialOffset = kMeshOffsetHi + kMeshBitsHi;

#ifdef HOST_CODE
    void pack(const MeshInstanceData& d)
    {
        vbOffset = d.vbOffset;
        ibOffset = d.ibOffset;

        assert(d.flags < (1 << kFlagsBits));
        assert(d.meshID < (1 << kMeshBits));
        assert(d.globalMatrixID < (1 << kMatrixBits));
        assert(d.materialID < (1 << kMaterialBits));
        packedIDs[0] = (d.flags << kFlagsOffset) | (d.globalMatrixID << kMatrixOffset) | ((d.meshID & ((1 << kMeshBitsLo) -1)) << kMeshOffsetLo);
        packedIDs[1] = (d.meshID >> kMeshBitsLo) | (d.materialID << kMaterialOffset);
    }
#endif

    MeshInstanceData unpack()
    {
        MeshInstanceData d;
        d.vbOffset = vbOffset;
        d.ibOffset = ibOffset;

        d.materialID = (packedIDs[1] >> kMaterialOffset) & ((1 << kMaterialBits) - 1);
        d.meshID = ((packedIDs[0] >> kMeshOffsetLo) & ((1 << kMeshBitsLo) - 1)) | (((packedIDs[1] >> kMeshOffsetHi) & ((1 << kMeshBitsHi) -1)) << kMeshBitsLo);
        d.globalMatrixID = (packedIDs[0] >> kMatrixOffset) & ((1 << kMatrixBits) - 1);
        d.flags = (packedIDs[0] >> kFlagsOffset) & ((1 << kFlagsBits) - 1);

        return d;
    }
};

struct StaticVertexData
{
    float3 position;    ///< Position.
    float3 normal;      ///< Shading normal.
    float4 tangent;     ///< Shading tangent. The bitangent is computed: cross(normal, tangent.xyz) * tangent.w. NOTE: The tangent is *only* valid when tangent.w != 0.
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

struct ProceduralPrimitiveData
{
    uint typeID;        ///< Type ID corresponding to how it was defined during Scene creation.
    uint instanceIdx;   ///< Of custom primitives of this type, what instance index it is, in the order of addition to the scene.

    uint AABBOffset;    ///< Offset into global AABB buffer.
    uint AABBCount;     ///< Number of AABBs within a custom primitive.
};

struct CurveDesc
{
    uint vbOffset;      ///< Offset into global curve vertex buffer.
    uint ibOffset;      ///< Offset into global curve index buffer.
    uint vertexCount;   ///< Vertex count.
    uint indexCount;    ///< Index count.
    uint degree;        ///< Polynomial degree of curve; linear (1) by default.
    uint materialID;    ///< Material ID.

    uint getSegmentCount() CONST_FUNCTION
    {
        return indexCount;
    }
};

struct CurveInstanceData
{
    uint globalMatrixID;
    uint materialID;
    uint curveID;
    uint vbOffset;      ///< Offset into global curve vertex buffer.
    uint ibOffset;      ///< Offset into global curve index buffer.
};

struct StaticCurveVertexData
{
    float3 position;    ///< Position.
    float radius;       ///< Radius of the sphere at curve ends.
    float3 tangent;     ///< Shading tangent.
    float3 normal;      ///< Shading normal.
    float2 texCrd;      ///< Texture coordinates.
};

END_NAMESPACE_FALCOR
