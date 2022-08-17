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
#include "Core/API/RtAccelerationStructurePostBuildInfoPool.h"
#include "Core/API/CopyContext.h"
#include "Core/API/D3D12/D3D12API.h"

namespace Falcor
{
    RtAccelerationStructurePostBuildInfoPool::RtAccelerationStructurePostBuildInfoPool(const Desc& desc)
        : mDesc(desc)
    {
        switch (mDesc.queryType)
        {
        case RtAccelerationStructurePostBuildInfoQueryType::CompactedSize:
        {
            mElementSize = sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC);
            break;
        }
        case RtAccelerationStructurePostBuildInfoQueryType::SerializationSize:
        {
            mElementSize = sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC);
            break;
        }
        case RtAccelerationStructurePostBuildInfoQueryType::CurrentSize:
        {
            mElementSize = sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE_DESC);
            break;
        }
        default:
            FALCOR_UNREACHABLE();
            break;
        }
        mpPostbuildInfoBuffer = Buffer::create(desc.elementCount * mElementSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
        mpPostbuildInfoStagingBuffer = Buffer::create(desc.elementCount * mElementSize, Buffer::BindFlags::None, Buffer::CpuAccess::Read);
    }

    RtAccelerationStructurePostBuildInfoPool::~RtAccelerationStructurePostBuildInfoPool()
    {
        if (mMappedPostBuildInfo)
        {
            mpPostbuildInfoStagingBuffer->unmap();
        }
    }

    uint64_t RtAccelerationStructurePostBuildInfoPool::getElement(CopyContext* pContext, uint32_t index)
    {
        if (!mStagingBufferUpToDate)
        {
            pContext->copyResource(mpPostbuildInfoStagingBuffer.get(), mpPostbuildInfoBuffer.get());
            pContext->flush(true);
            mStagingBufferUpToDate = true;

            mMappedPostBuildInfo = mpPostbuildInfoStagingBuffer->map(Buffer::MapType::Read);
        }

        FALCOR_ASSERT(index < mDesc.elementCount);

        switch (mDesc.queryType)
        {
        case RtAccelerationStructurePostBuildInfoQueryType::CompactedSize:
        {
            const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC* mappedPostBuildInfo =
                (const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC*)mMappedPostBuildInfo;
            return mappedPostBuildInfo[index].CompactedSizeInBytes;
        }
        case RtAccelerationStructurePostBuildInfoQueryType::SerializationSize:
        {
            const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC* mappedPostBuildInfo =
                (const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC*)mMappedPostBuildInfo;
            return mappedPostBuildInfo[index].SerializedSizeInBytes;
        }
        case RtAccelerationStructurePostBuildInfoQueryType::CurrentSize:
        {
            const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE_DESC* mappedPostBuildInfo =
                (const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE_DESC*)mMappedPostBuildInfo;
            return mappedPostBuildInfo[index].CurrentSizeInBytes;
        }
        default:
            FALCOR_UNREACHABLE();
            return 0;
        }

    }

    uint64_t RtAccelerationStructurePostBuildInfoPool::getBufferAddress(uint32_t index)
    {
        return mpPostbuildInfoBuffer->getGpuAddress() + mElementSize * index;
    }

    void RtAccelerationStructurePostBuildInfoPool::reset(CopyContext* pContext)
    {
        mStagingBufferUpToDate = false;
        if (mMappedPostBuildInfo)
        {
            mpPostbuildInfoStagingBuffer->unmap();
            mMappedPostBuildInfo = nullptr;
        }
        pContext->resourceBarrier(mpPostbuildInfoBuffer.get(), Resource::State::UnorderedAccess);
    }

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE translatePostBuildInfoType(RtAccelerationStructurePostBuildInfoQueryType type)
    {
        switch (type)
        {
        case RtAccelerationStructurePostBuildInfoQueryType::CompactedSize:
            return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
        case RtAccelerationStructurePostBuildInfoQueryType::SerializationSize:
            return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION;
        case RtAccelerationStructurePostBuildInfoQueryType::CurrentSize:
            return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE;
        default:
            FALCOR_UNREACHABLE();
            return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
        }
    }
}
