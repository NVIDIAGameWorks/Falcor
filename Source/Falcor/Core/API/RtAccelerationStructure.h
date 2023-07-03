/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "fwd.h"
#include "RtAccelerationStructurePostBuildInfoPool.h"
#include "Handles.h"
#include "Formats.h"
#include "Buffer.h"
#include "ResourceViews.h"
#include "Utils/Math/Matrix.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include <slang-gfx.h>
#include <vector>
#include <cstdint>

namespace Falcor
{
const uint64_t kAccelerationStructureByteAlignment = 256;

class RtAccelerationStructure;

using DeviceAddress = uint64_t;

enum class RtGeometryInstanceFlags
{
    // The enum values are kept consistent with D3D12_RAYTRACING_INSTANCE_FLAGS
    // and VkGeometryInstanceFlagBitsKHR.
    None = 0,
    TriangleFacingCullDisable = 0x00000001,
    TriangleFrontCounterClockwise = 0x00000002,
    ForceOpaque = 0x00000004,
    NoOpaque = 0x00000008,
};
FALCOR_ENUM_CLASS_OPERATORS(RtGeometryInstanceFlags);

// The layout of this struct is intentionally consistent with D3D12_RAYTRACING_INSTANCE_DESC
// and VkAccelerationStructureInstanceKHR.
struct RtInstanceDesc
{
    float transform[3][4];
    uint32_t instanceID : 24;
    uint32_t instanceMask : 8;
    uint32_t instanceContributionToHitGroupIndex : 24;
    RtGeometryInstanceFlags flags : 8;
    DeviceAddress accelerationStructure;

    /**
     * Sets the transform matrix using a float4x4 value.
     * @param[in] matrix A 4x4 matrix to set into transform.
     */
    RtInstanceDesc& setTransform(const float4x4& matrix);
};

enum class RtAccelerationStructureKind
{
    TopLevel,
    BottomLevel
};

enum class RtAccelerationStructureBuildFlags
{
    // The enum values are intentionally consistent with
    // D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.
    None,
    AllowUpdate = 1,
    AllowCompaction = 2,
    PreferFastTrace = 4,
    PreferFastBuild = 8,
    MinimizeMemory = 16,
    PerformUpdate = 32
};
FALCOR_ENUM_CLASS_OPERATORS(RtAccelerationStructureBuildFlags);

enum class RtGeometryType
{
    Triangles,
    ProcedurePrimitives
};

enum class RtGeometryFlags
{
    // The enum values are intentionally consistent with
    // D3D12_RAYTRACING_GEOMETRY_FLAGS.
    None,
    Opaque = 1,
    NoDuplicateAnyHitInvocation = 2
};
FALCOR_ENUM_CLASS_OPERATORS(RtGeometryFlags);

struct RtTriangleDesc
{
    DeviceAddress transform3x4;
    ResourceFormat indexFormat;
    ResourceFormat vertexFormat;
    uint32_t indexCount;
    uint32_t vertexCount;
    DeviceAddress indexData;
    DeviceAddress vertexData;
    uint64_t vertexStride;
};

struct RtAABBDesc
{
    /// Number of AABBs.
    uint64_t count;

    /// Pointer to an array of `ProceduralAABB` values in device memory.
    DeviceAddress data;

    /// Stride in bytes of the AABB values array.
    uint64_t stride;
};

struct RtGeometryDesc
{
    RtGeometryType type;
    RtGeometryFlags flags;
    union
    {
        RtTriangleDesc triangles;
        RtAABBDesc proceduralAABBs;
    } content;
};

struct RtAccelerationStructurePrebuildInfo
{
    uint64_t resultDataMaxSize;
    uint64_t scratchDataSize;
    uint64_t updateScratchDataSize;
};

struct RtAccelerationStructureBuildInputs
{
    RtAccelerationStructureKind kind;

    RtAccelerationStructureBuildFlags flags;

    uint32_t descCount;

    /// Array of `InstanceDesc` values in device memory.
    /// Used when `kind` is `TopLevel`.
    DeviceAddress instanceDescs;

    /// Array of `GeometryDesc` values.
    /// Used when `kind` is `BottomLevel`.
    const RtGeometryDesc* geometryDescs;
};

/**
 * Abstract the API acceleration structure object.
 * An acceleration structure object is a wrapper around a buffer resource that stores the contents
 * of an acceleration structure. It does not own the backing buffer resource, which is similar to
 * a resource view.
 */
class FALCOR_API RtAccelerationStructure : public Object
{
    FALCOR_OBJECT(RtAccelerationStructure)
public:
    class FALCOR_API Desc
    {
    public:
        friend class RtAccelerationStructure;

        /**
         * Set acceleration structure kind.
         * @param[in] kind Kind of the acceleration structure.
         */
        Desc& setKind(RtAccelerationStructureKind kind);

        /**
         * Set backing buffer of the acceleration structure.
         * @param[in] buffer The buffer to store the acceleration structure contents.
         * @param[in] offset The offset within the buffer for the acceleration structure contents.
         * @param[in] offset The size in bytes to use for the acceleration structure.
         */
        Desc& setBuffer(ref<Buffer> buffer, uint64_t offset, uint64_t size);

        ref<Buffer> getBuffer() const { return mBuffer; }

        uint64_t getOffset() const { return mOffset; }

        uint64_t getSize() const { return mSize; }

        RtAccelerationStructureKind getKind() const { return mKind; }

    protected:
        RtAccelerationStructureKind mKind = RtAccelerationStructureKind::BottomLevel;
        ref<Buffer> mBuffer = nullptr;
        uint64_t mOffset = 0;
        uint64_t mSize = 0;
    };

    struct BuildDesc
    {
        RtAccelerationStructureBuildInputs inputs;
        RtAccelerationStructure* source;
        RtAccelerationStructure* dest;
        DeviceAddress scratchData;
    };

    /**
     * Create a new acceleration structure.
     * @param[in] desc Describes acceleration structure settings.
     * @return A new object, or throws an exception if creation failed.
     */
    static ref<RtAccelerationStructure> create(ref<Device> pDevice, const Desc& desc);

    static RtAccelerationStructurePrebuildInfo getPrebuildInfo(Device* pDevice, const RtAccelerationStructureBuildInputs& inputs);

    ~RtAccelerationStructure();

    uint64_t getGpuAddress();

    const Desc& getDesc() const { return mDesc; }

    gfx::IAccelerationStructure* getGfxAccelerationStructure() const { return mGfxAccelerationStructure; }

protected:
    RtAccelerationStructure(ref<Device> pDevice, const Desc& desc);

    ref<Device> mpDevice;
    Desc mDesc;

    Slang::ComPtr<gfx::IAccelerationStructure> mGfxAccelerationStructure;
};

/**
 * A helper class to translate `RtAccelerationStructureBuildInputs` into `gfx::IAccelerationStructure::BuildInputs`.
 */
struct GFXAccelerationStructureBuildInputsTranslator
{
public:
    gfx::IAccelerationStructure::BuildInputs& translate(const RtAccelerationStructureBuildInputs& buildInputs);

private:
    gfx::IAccelerationStructure::BuildInputs mDesc = {};
    gfx::IAccelerationStructure::PrebuildInfo mPrebuildInfo = {};
    std::vector<gfx::IAccelerationStructure::GeometryDesc> mGeomDescs;

    gfx::IAccelerationStructure::GeometryFlags::Enum translateGeometryFlags(RtGeometryFlags flags)
    {
        return (gfx::IAccelerationStructure::GeometryFlags::Enum)flags;
    }
};

gfx::QueryType getGFXAccelerationStructurePostBuildQueryType(RtAccelerationStructurePostBuildInfoQueryType type);
} // namespace Falcor
