/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "GFXRtAccelerationStructure.h"
#include "GFXFormats.h"
#include "Core/API/RtAccelerationStructure.h"
#include "Core/API/Device.h"
#include "Core/API/GFX/GFXAPI.h"

namespace Falcor
{

    RtAccelerationStructure::RtAccelerationStructure(const RtAccelerationStructure::Desc& desc)
        : mDesc(desc)
    {
    }

    RtAccelerationStructure::~RtAccelerationStructure()
    {
        gpDevice->releaseResource(mApiHandle);
    }

    RtAccelerationStructurePrebuildInfo RtAccelerationStructure::getPrebuildInfo(const RtAccelerationStructureBuildInputs& inputs)
    {
        static_assert(sizeof(RtAccelerationStructurePrebuildInfo) == sizeof(gfx::IAccelerationStructure::PrebuildInfo));

        gfx::IAccelerationStructure::BuildInputs gfxBuildInputs;

        GFXAccelerationStructureBuildInputsTranslator translator;
        gfxBuildInputs = translator.translate(inputs);

        FALCOR_ASSERT(gpDevice);
        Slang::ComPtr<gfx::IDevice> pDevice = gpDevice->getApiHandle();

        gfx::IAccelerationStructure::PrebuildInfo gfxPrebuildInfo;
        pDevice->getAccelerationStructurePrebuildInfo(gfxBuildInputs, &gfxPrebuildInfo);

        RtAccelerationStructurePrebuildInfo result = {};
        result.resultDataMaxSize = gfxPrebuildInfo.resultDataMaxSize;
        result.scratchDataSize = gfxPrebuildInfo.scratchDataSize;
        result.updateScratchDataSize = gfxPrebuildInfo.updateScratchDataSize;
        return result;
    }

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

    bool RtAccelerationStructure::apiInit()
    {
        gfx::IAccelerationStructure::CreateDesc createDesc = {};
        createDesc.buffer = static_cast<gfx::IBufferResource*>(mDesc.getBuffer()->getApiHandle().get());
        createDesc.kind = getGFXAccelerationStructureKind(mDesc.mKind);
        createDesc.offset = mDesc.getOffset();
        createDesc.size = mDesc.getSize();
        SLANG_RETURN_FALSE_ON_FAIL(gpDevice->getApiHandle()->createAccelerationStructure(createDesc, mApiHandle.writeRef()));
        return true;
    }

    AccelerationStructureHandle RtAccelerationStructure::getApiHandle() const
    {
        return mApiHandle;
    }

    gfx::IAccelerationStructure::BuildInputs& GFXAccelerationStructureBuildInputsTranslator::translate(const RtAccelerationStructureBuildInputs& buildInputs)
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
}
