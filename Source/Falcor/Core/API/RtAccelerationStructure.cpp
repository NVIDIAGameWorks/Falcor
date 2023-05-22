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
#include "RtAccelerationStructure.h"
#include "Device.h"
#include "GFXHelpers.h"
#include "GFXAPI.h"

namespace Falcor
{
gfx::IAccelerationStructure::Kind getGFXAccelerationStructureKind(RtAccelerationStructureKind kind)
{
    switch (kind)
    {
    case RtAccelerationStructureKind::TopLevel:
        return gfx::IAccelerationStructure::Kind::TopLevel;
    case RtAccelerationStructureKind::BottomLevel:
        return gfx::IAccelerationStructure::Kind::BottomLevel;
    default:
        FALCOR_UNREACHABLE();
        return gfx::IAccelerationStructure::Kind::BottomLevel;
    }
}

RtAccelerationStructure::Desc& RtAccelerationStructure::Desc::setKind(RtAccelerationStructureKind kind)
{
    mKind = kind;
    return *this;
}

RtAccelerationStructure::Desc& RtAccelerationStructure::Desc::setBuffer(ref<Buffer> buffer, uint64_t offset, uint64_t size)
{
    mBuffer = buffer;
    mOffset = offset;
    mSize = size;
    return *this;
}

ref<RtAccelerationStructure> RtAccelerationStructure::create(ref<Device> pDevice, const Desc& desc)
{
    return ref<RtAccelerationStructure>(new RtAccelerationStructure(pDevice, desc));
}

uint64_t RtAccelerationStructure::getGpuAddress()
{
    return mDesc.mBuffer->getGpuAddress() + mDesc.mOffset;
}

RtInstanceDesc& RtInstanceDesc::setTransform(const float4x4& matrix)
{
    std::memcpy(transform, &matrix, sizeof(transform));
    return *this;
}

RtAccelerationStructure::RtAccelerationStructure(ref<Device> pDevice, const RtAccelerationStructure::Desc& desc)
    : mpDevice(pDevice), mDesc(desc)
{
    gfx::IAccelerationStructure::CreateDesc createDesc = {};
    createDesc.buffer = mDesc.getBuffer()->getGfxBufferResource();
    createDesc.kind = getGFXAccelerationStructureKind(mDesc.mKind);
    createDesc.offset = mDesc.getOffset();
    createDesc.size = mDesc.getSize();
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createAccelerationStructure(createDesc, mGfxAccelerationStructure.writeRef()));
}

RtAccelerationStructure::~RtAccelerationStructure()
{
    mpDevice->releaseResource(mGfxAccelerationStructure);
}

RtAccelerationStructurePrebuildInfo RtAccelerationStructure::getPrebuildInfo(
    Device* pDevice,
    const RtAccelerationStructureBuildInputs& inputs
)
{
    static_assert(sizeof(RtAccelerationStructurePrebuildInfo) == sizeof(gfx::IAccelerationStructure::PrebuildInfo));

    gfx::IAccelerationStructure::BuildInputs gfxBuildInputs;

    GFXAccelerationStructureBuildInputsTranslator translator;
    gfxBuildInputs = translator.translate(inputs);

    gfx::IAccelerationStructure::PrebuildInfo gfxPrebuildInfo;
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->getAccelerationStructurePrebuildInfo(gfxBuildInputs, &gfxPrebuildInfo));

    RtAccelerationStructurePrebuildInfo result = {};
    result.resultDataMaxSize = gfxPrebuildInfo.resultDataMaxSize;
    result.scratchDataSize = gfxPrebuildInfo.scratchDataSize;
    result.updateScratchDataSize = gfxPrebuildInfo.updateScratchDataSize;
    return result;
}

gfx::IAccelerationStructure::BuildInputs& GFXAccelerationStructureBuildInputsTranslator::translate(
    const RtAccelerationStructureBuildInputs& buildInputs
)
{
    if (buildInputs.geometryDescs)
    {
        mGeomDescs.resize(buildInputs.descCount);
        for (size_t i = 0; i < mGeomDescs.size(); i++)
        {
            auto& inputGeomDesc = buildInputs.geometryDescs[i];
            mGeomDescs[i].flags = translateGeometryFlags(inputGeomDesc.flags);

            switch (inputGeomDesc.type)
            {
            case RtGeometryType::Triangles:
                mGeomDescs[i].type = gfx::IAccelerationStructure::GeometryType::Triangles;
                mGeomDescs[i].content.triangles.indexData = inputGeomDesc.content.triangles.indexData;
                mGeomDescs[i].content.triangles.indexCount = inputGeomDesc.content.triangles.indexCount;
                mGeomDescs[i].content.triangles.indexFormat = getGFXFormat(inputGeomDesc.content.triangles.indexFormat);
                mGeomDescs[i].content.triangles.transform3x4 = inputGeomDesc.content.triangles.transform3x4;
                mGeomDescs[i].content.triangles.vertexData = inputGeomDesc.content.triangles.vertexData;
                mGeomDescs[i].content.triangles.vertexStride = inputGeomDesc.content.triangles.vertexStride;
                mGeomDescs[i].content.triangles.vertexCount = inputGeomDesc.content.triangles.vertexCount;
                mGeomDescs[i].content.triangles.vertexFormat = getGFXFormat(inputGeomDesc.content.triangles.vertexFormat);
                break;

            case RtGeometryType::ProcedurePrimitives:
                mGeomDescs[i].type = gfx::IAccelerationStructure::GeometryType::ProcedurePrimitives;
                mGeomDescs[i].content.proceduralAABBs.count = static_cast<gfx::GfxCount>(inputGeomDesc.content.proceduralAABBs.count);
                mGeomDescs[i].content.proceduralAABBs.data = inputGeomDesc.content.proceduralAABBs.data;
                mGeomDescs[i].content.proceduralAABBs.stride = inputGeomDesc.content.proceduralAABBs.stride;
                break;

            default:
                FALCOR_UNREACHABLE();
            }
        }
    }

    mDesc.descCount = buildInputs.descCount;

    switch (buildInputs.kind)
    {
    case RtAccelerationStructureKind::TopLevel:
        mDesc.kind = gfx::IAccelerationStructure::Kind::TopLevel;
        mDesc.instanceDescs = buildInputs.instanceDescs;
        break;

    case RtAccelerationStructureKind::BottomLevel:
        mDesc.kind = gfx::IAccelerationStructure::Kind::BottomLevel;
        mDesc.geometryDescs = &mGeomDescs[0];
        break;
    }

    mDesc.flags = (gfx::IAccelerationStructure::BuildFlags::Enum)buildInputs.flags;
    return mDesc;
}

gfx::QueryType getGFXAccelerationStructurePostBuildQueryType(RtAccelerationStructurePostBuildInfoQueryType type)
{
    switch (type)
    {
    case RtAccelerationStructurePostBuildInfoQueryType::CompactedSize:
        return gfx::QueryType::AccelerationStructureCompactedSize;
    case RtAccelerationStructurePostBuildInfoQueryType::SerializationSize:
        return gfx::QueryType::AccelerationStructureSerializedSize;
    case RtAccelerationStructurePostBuildInfoQueryType::CurrentSize:
        return gfx::QueryType::AccelerationStructureCurrentSize;
    default:
        FALCOR_UNREACHABLE();
        return gfx::QueryType::AccelerationStructureCompactedSize;
    }
}
} // namespace Falcor
