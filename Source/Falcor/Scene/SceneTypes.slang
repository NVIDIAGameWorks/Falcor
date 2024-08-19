/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "SceneDefines.slangh"

#ifdef HOST_CODE
#include "Utils/Math/PackedFormats.h"
#include "VertexData.slang"
#else
import Utils.Math.PackedFormats;
import Utils.SlangUtils;
import Utils.Attributes;
__exported import Scene.VertexData;
#endif

BEGIN_NAMESPACE_FALCOR

/** Geometry instance ID.
    This uniquely identifies a geometry instance in the scene.
    All instances are sequentially indexed, with mesh instances first.
    This may change in the future, but a lot of existing code relies on it.
*/
struct GeometryInstanceID
{
    uint index;             ///< Global instance index. This is computed as InstanceID() + GeometryIndex().

#ifndef HOST_CODE
    /** Construct a geometry instance ID.
        \param[in] instanceID The DXR InstanceID() system value.
        \param[in] geometryIndex The DXR GeometryIndex() system value.
    */
    __init(uint instanceID, uint geometryIndex)
    {
        index = instanceID + geometryIndex;
    }
#endif
};

/** Geometry types in the scene.
*/
enum class GeometryType : uint32_t
{
    None                    = GEOMETRY_TYPE_NONE,
    TriangleMesh            = GEOMETRY_TYPE_TRIANGLE_MESH,
    DisplacedTriangleMesh   = GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH,
    Curve                   = GEOMETRY_TYPE_CURVE,
    SDFGrid                 = GEOMETRY_TYPE_SDF_GRID,
    Custom                  = GEOMETRY_TYPE_CUSTOM,

    Count
};

/** Flags indicating what geometry types exist in the scene.
*/
enum class GeometryTypeFlags : uint32_t
{
    TriangleMesh            = (1u << GEOMETRY_TYPE_TRIANGLE_MESH),
    DisplacedTriangleMesh   = (1u << GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH),
    Curve                   = (1u << GEOMETRY_TYPE_CURVE),
    SDFGrid                 = (1u << GEOMETRY_TYPE_SDF_GRID),
    Custom                  = (1u << GEOMETRY_TYPE_CUSTOM),
};

#ifdef HOST_CODE
FALCOR_ENUM_CLASS_OPERATORS(GeometryTypeFlags);
#endif

enum class GeometryInstanceFlags : uint32_t
{
    None = 0x0,

    // Mesh flags.
    Use16BitIndices = 0x1,      ///< Indices are in 16-bit format. The default is 32-bit.
    IsDynamic = 0x2,            ///< Mesh is dynamic, either through skinning or vertex animations.
    TransformFlipped = 0x4,     ///< Instance transform flips the coordinate system handedness. TODO: Deprecate this flag if we need an extra bit.
    IsObjectFrontFaceCW = 0x8,  ///< Front-facing side has clockwise winding in object space. Note that the winding in world space may be flipped due to the instance transform.
    IsWorldFrontFaceCW = 0x10,  ///< Front-facing side has clockwise winding in world space. This is the combination of the mesh winding and instance transform handedness.
};

struct GeometryInstanceData
{
    static const uint kTypeBits = 3;
    static const uint kTypeOffset = 32 - kTypeBits;

    uint flags;             ///< Upper kTypeBits bits are reserved for storing the type.
    uint globalMatrixID;    ///< This is actually CPU-side NodeID, with a confusing name.
    uint materialID;
    uint geometryID;

    uint vbOffset; ///< Offset into vertex buffer.
    uint ibOffset; ///< Offset into index buffer, or zero if non-indexed.

    uint instanceIndex; ///< InstanceIndex in TLAS.
    uint geometryIndex; ///< GeometryIndex in BLAS.

#ifdef HOST_CODE
    GeometryInstanceData() = default;

    GeometryInstanceData(GeometryType type)
        : flags((uint32_t)type << kTypeOffset)
    {}
#endif

    GeometryType getType() CONST_FUNCTION
    {
        return GeometryType(flags >> kTypeOffset);
    }

    bool isDynamic() CONST_FUNCTION
    {
        return (flags & (uint)GeometryInstanceFlags::IsDynamic) != 0;
    }

    bool isWorldFrontFaceCW() CONST_FUNCTION
    {
        return (flags & (uint)GeometryInstanceFlags::IsWorldFrontFaceCW) != 0;
    }
};

enum class MeshFlags : uint32_t
{
    None = 0x0,
    Use16BitIndices = 0x1,  ///< Indices are in 16-bit format. The default is 32-bit.
    IsSkinned = 0x2,        ///< Mesh is skinned and has corresponding vertex data.
    IsFrontFaceCW = 0x4,    ///< Front-facing side has clockwise winding in object space. Note that the winding in world space may be flipped due to the instance transform.
    IsDisplaced = 0x8,      ///< Mesh has displacement map.
    IsAnimated = 0x10,      ///< Mesh is affected by vertex-animations.
};

/** Mesh data stored in 32B.
*/
struct MeshDesc
{
    uint vbOffset;          ///< Offset into global vertex buffer.
    uint ibOffset;          ///< Offset into global index buffer, or zero if non-indexed.
    uint vertexCount;       ///< Vertex count.
    uint indexCount;        ///< Index count, or zero if non-indexed.
    uint skinningVbOffset;  ///< Offset into skinning data buffer, or zero if no skinning data.
    uint prevVbOffset;      ///< Offset into previous vertex data buffer, or zero if neither skinned or animated.
    uint materialID;        ///< Material ID.
    uint flags;             ///< See MeshFlags.

    uint getVertexCount() CONST_FUNCTION
    {
        return vertexCount;
    }

    uint getTriangleCount() CONST_FUNCTION
    {
        return (indexCount > 0 ? indexCount : vertexCount) / 3;
    }

    bool use16BitIndices() CONST_FUNCTION
    {
        return (flags & (uint)MeshFlags::Use16BitIndices) != 0;
    }

    bool useVertexIndices() CONST_FUNCTION
    {
        return indexCount > 0;
    }

    bool isSkinned() CONST_FUNCTION
    {
        return (flags & (uint)MeshFlags::IsSkinned) != 0;
    }

    bool isAnimated() CONST_FUNCTION
    {
        return (flags & (uint)MeshFlags::IsAnimated) != 0;
    }

    bool isDynamic() CONST_FUNCTION
    {
        return isSkinned() || isAnimated();
    }

    bool isFrontFaceCW() CONST_FUNCTION
    {
        return (flags & (uint)MeshFlags::IsFrontFaceCW) != 0;
    }

    bool isDisplaced() CONST_FUNCTION
    {
        return (flags & (uint)MeshFlags::IsDisplaced) != 0;
    }
};

struct StaticVertexData
{
    float3 position;    ///< Position.
    float3 normal;      ///< Shading normal.
    float4 tangent;     ///< Shading tangent. The bitangent is computed: cross(normal, tangent.xyz) * tangent.w. NOTE: The tangent is *only* valid when tangent.w != 0.
    float2 texCrd;      ///< Texture coordinates.
    float curveRadius;  ///< Curve cross-sectional radius. Valid only for geometry generated from curves.
};

/** Vertex data packed into 32B for aligned access.
*/
struct PackedStaticVertexData
{
    float3 position;
    float3 packedNormalTangentCurveRadius;
    float2 texCrd;

#ifdef HOST_CODE
    PackedStaticVertexData() = default;
    PackedStaticVertexData(const StaticVertexData& v) { pack(v); }
    void pack(const StaticVertexData& v)
    {
        position = v.position;
        texCrd = v.texCrd;

        uint3 n = f32tof16(v.normal);

        float packedTangentSignCurveRadius = v.tangent.w;

        if (v.curveRadius > 0.f)
        {
            // This is safe because if v.curveRadius > 0 then v.tangent.w != 0 (curves always have valid tangents).
            FALCOR_ASSERT(v.tangent.w != 0.f);
            packedTangentSignCurveRadius *= v.curveRadius;
        }
        uint t_w = f32tof16(packedTangentSignCurveRadius);

        packedNormalTangentCurveRadius.x = asfloat((n.y << 16) | n.x);
        packedNormalTangentCurveRadius.y = asfloat((t_w << 16) | n.z);
        packedNormalTangentCurveRadius.z = asfloat(encodeNormal2x16(v.tangent.xyz()));
    }

#else // !HOST_CODE
    [mutating] void pack(const StaticVertexData v)
    {
        position = v.position;
        texCrd = v.texCrd;

        uint3 n = f32tof16(v.normal);

        float packedTangentSignCurveRadius = v.tangent.w;
        // This is safe because if v.curveRadius > 0 then v.tangent.w != 0 (curves always have valid tangents).
        if (v.curveRadius > 0.f) packedTangentSignCurveRadius *= v.curveRadius;
        uint t_w = f32tof16(packedTangentSignCurveRadius);

        packedNormalTangentCurveRadius.x = asfloat((n.y << 16) | n.x);
        packedNormalTangentCurveRadius.y = asfloat((t_w << 16) | n.z);
        packedNormalTangentCurveRadius.z = asfloat(encodeNormal2x16(v.tangent.xyz));
    }
#endif

    StaticVertexData unpack() CONST_FUNCTION
    {
        StaticVertexData v;
        v.position = position;
        v.texCrd = texCrd;

        v.normal.x = f16tof32(asuint(packedNormalTangentCurveRadius.x) & 0xffff);
        v.normal.y = f16tof32(asuint(packedNormalTangentCurveRadius.x) >> 16);
        v.normal.z = f16tof32(asuint(packedNormalTangentCurveRadius.y) & 0xffff);
        v.normal = normalize(v.normal);

        float3 tangent = decodeNormal2x16(asuint(packedNormalTangentCurveRadius.z));
        float packedTangentSignCurveRadius = f16tof32(asuint(packedNormalTangentCurveRadius.y) >> 16);
        v.tangent = float4(tangent, sign(packedTangentSignCurveRadius));

        v.curveRadius = STD_NAMESPACE abs(packedTangentSignCurveRadius);

        return v;
    }
};

struct PrevVertexData
{
    float3 position;
};

struct SkinningVertexData
{
    uint4 boneID;
    float4 boneWeight;
    uint staticIndex; ///< The index in the static vertex buffer.
    uint bindMatrixID;
    uint skeletonMatrixID;
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

struct StaticCurveVertexData
{
    float3 position;    ///< Position.
    float radius;       ///< Radius of the sphere at curve ends.
    float2 texCrd;      ///< Texture coordinates.
};

struct DynamicCurveVertexData
{
    float3 position;    ///< Position.
};

/** Custom primitive data.
    The custom primitives are currently mapped 1:1 to the list of custom primitive AABBs.
*/
struct CustomPrimitiveDesc
{
    uint userID;        ///< User-defined ID that is specified during scene creation. This can be used to identify different sub-types of custom primitives.
    uint aabbOffset;    ///< Offset into list of procedural primitive AABBs.
};

#ifndef HOST_CODE

#ifndef SCENE_INDEX_BUFFER_COUNT
// #error "Define SCENE_INDEX_BUFFER_COUNT, SCENE_INDEX_BUFFER_INDEX_BITS"
#define SCENE_INDEX_BUFFER_COUNT 1 // here for the benefit of the IntelliSense
#define SCENE_INDEX_BUFFER_INDEX_BITS 1
#endif // SCENE_INDEX_BUFFER_COUNT

/**
 * GPU representation for SplitBuffer<uint32_T>.
 * Unlike the SplitVertexBuffer, this wraps a ByteAddressBuffer access rather than
 * StructuredBuffer, but since we do not have templates, Load<uint16_t3> has to be
 * explicitly instantiated.
 *
 * Functions as an adaptor when we need larger-than-4GB buffers.
 * In such case, the top bits of the index are used to select which buffers
 * is the item in, while the lower bits address the item inside the buffer.
 * Host is only used for static_asserts.
 */
struct SplitIndexBuffer
{
    typedef uint ElementType;
    static constexpr uint kBufferIndexBits = SCENE_INDEX_BUFFER_INDEX_BITS; // log2(sizeof(ElementType))
    static constexpr uint kBufferIndexOffset = 32 - kBufferIndexBits;
    static constexpr uint kElementIndexMask = (1u << kBufferIndexOffset) - 1;
    static constexpr uint kBufferCount = SCENE_INDEX_BUFFER_COUNT;

#if SCENE_INDEX_BUFFER_COUNT > 0
#if SCENE_INDEX_BUFFER_COUNT > 1
    /// TODO: Once the [root] signature issue has been solved, this should be the only version
    ByteAddressBuffer data[kBufferCount];
#else
    [root] ByteAddressBuffer data[kBufferCount];
#endif

    /// baseOffset - offset to the start of the mesh
    /// byteIndex - bytes to read from offset
    /// Returns bytes starting baseOffset*4 + byteIndex, handling the uint overflow
    /// for large buffers.
    /// expected usage is LoadUint16_t3(baseOffset, baseOffset*4 + index)
    uint16_t3 Load16b(uint baseOffset, uint triangleIndex)
    {
        if (kBufferCount == 1)
            return data[0].Load<uint16_t3>(baseOffset * 4 + triangleIndex * 6);
        uint bufferIndex = baseOffset >> kBufferIndexOffset;
        uint byteOffset = (baseOffset & kElementIndexMask) * 4;
        return data[bufferIndex].Load<uint16_t3>(byteOffset + triangleIndex * 6);
    }

    uint3 Load32b(uint baseOffset, uint triangleIndex)
    {
        if (kBufferCount == 1)
            return data[0].Load3(baseOffset * 4 + triangleIndex * 12);
        uint bufferIndex = baseOffset >> kBufferIndexOffset;
        uint byteOffset = (baseOffset & kElementIndexMask) * 4;
        return data[bufferIndex].Load3(byteOffset + triangleIndex * 12);
    }
#endif // SCENE_INDEX_BUFFER_COUNT > 0
};

#ifndef SCENE_VERTEX_BUFFER_COUNT
// #error "Define SCENE_VERTEX_BUFFER_COUNT, SCENE_VERTEX_BUFFER_INDEX_BITS"
#define SCENE_VERTEX_BUFFER_COUNT 1 // here for the benefit of the IntelliSense
#define SCENE_VERTEX_BUFFER_INDEX_BITS 1
#endif // SCENE_VERTEX_BUFFER_COUNT

/**
 * GPU representation for SplitBuffer<PackedStaticVertexData>.
 * All comments apply to RWSplitVertexBuffer below as well.
 *
 * Functions as an adaptor when we need larger-than-4GB buffers.
 * In such case, the top bits of the index are used to select which buffers
 * is the item in, while the lower bits address the item inside the buffer.
 * Host is only used for static_asserts.
 */
struct SplitVertexBuffer
{
    typedef PackedStaticVertexData ElementType;
    static constexpr uint kBufferIndexBits = SCENE_VERTEX_BUFFER_INDEX_BITS;
    static constexpr uint kBufferIndexOffset = 32 - kBufferIndexBits;
    static constexpr uint kElementIndexMask = (1u << kBufferIndexOffset) - 1;
    static constexpr uint kBufferCount = SCENE_VERTEX_BUFFER_COUNT;

#if SCENE_VERTEX_BUFFER_COUNT > 1
    /// TODO: Once the [root] signature issue has been solved, this should be the only version
    StructuredBuffer<ElementType> data[ArrayMax<1, kBufferCount>.value];
#else
    [root] StructuredBuffer<ElementType> data[1];
#endif

    __subscript(uint index)->ElementType
    {
        get {
            if (kBufferCount == 1)
                return data[0][index];
            uint bufferIndex = index >> kBufferIndexOffset;
            uint elementIndex = index & kElementIndexMask;
            return data[bufferIndex][elementIndex];
        }
    }
};

struct RWSplitVertexBuffer
{
    typedef PackedStaticVertexData ElementType;
    static constexpr uint kBufferIndexBits = SCENE_VERTEX_BUFFER_INDEX_BITS;
    static constexpr uint kBufferIndexOffset = 32 - kBufferIndexBits;
    static constexpr uint kElementIndexMask = (1u << kBufferIndexOffset) - 1;
    static constexpr uint kBufferCount = SCENE_VERTEX_BUFFER_COUNT;

#if SCENE_VERTEX_BUFFER_COUNT > 1
    /// TODO: Once the [root] signature issue has been solved, this should be the only version
    RWStructuredBuffer<ElementType> data[ArrayMax<1, kBufferCount>.value];
#else
    [root] RWStructuredBuffer<ElementType> data[1];
#endif

    __subscript(uint index)->ElementType
    {
        get {
            if (kBufferCount == 1)
                return data[0][index];
            uint bufferIndex = index >> kBufferIndexOffset;
            uint elementIndex = index & kElementIndexMask;
            return data[bufferIndex][elementIndex];
        }

        set {
            if (kBufferCount == 1)
            {
                data[0][index] = newValue;
            }
            else
            {
                uint bufferIndex = index >> kBufferIndexOffset;
                uint elementIndex = index & kElementIndexMask;
                data[bufferIndex][elementIndex] = newValue;
            }
        }
    }
};

#endif /// HOST_CODE

END_NAMESPACE_FALCOR
