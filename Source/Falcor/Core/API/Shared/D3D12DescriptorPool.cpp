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
#include "stdafx.h"

#if FALCOR_D3D12_AVAILABLE

#include "D3D12DescriptorPool.h"
#include "D3D12DescriptorData.h"

namespace Falcor
{
    D3D12_DESCRIPTOR_HEAP_TYPE falcorToDxDescType(D3D12DescriptorPool::Type t)
    {
        switch (t)
        {
        case D3D12DescriptorPool::Type::TextureSrv:
        case D3D12DescriptorPool::Type::TextureUav:
        case D3D12DescriptorPool::Type::RawBufferSrv:
        case D3D12DescriptorPool::Type::RawBufferUav:
        case D3D12DescriptorPool::Type::TypedBufferSrv:
        case D3D12DescriptorPool::Type::TypedBufferUav:
        case D3D12DescriptorPool::Type::StructuredBufferSrv:
        case D3D12DescriptorPool::Type::StructuredBufferUav:
        case D3D12DescriptorPool::Type::AccelerationStructureSrv:
        case D3D12DescriptorPool::Type::Cbv:
            return D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        case D3D12DescriptorPool::Type::Dsv:
            return D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        case D3D12DescriptorPool::Type::Rtv:
            return D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        case D3D12DescriptorPool::Type::Sampler:
            return D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        default:
            FALCOR_UNREACHABLE();
            return D3D12_DESCRIPTOR_HEAP_TYPE(-1);
        }
    }

    uint32_t D3D12DescriptorPool::getMaxShaderVisibleSamplerHeapSize()
    {
        return D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
    }

    const D3D12DescriptorPool::ApiHandle& D3D12DescriptorPool::getApiHandle(uint32_t heapIndex) const
    {
        FALCOR_ASSERT(heapIndex < arraysize(mpApiData->pHeaps));
        return mpApiData->pHeaps[heapIndex]->getApiHandle();
    }

    D3D12DescriptorPool::SharedPtr D3D12DescriptorPool::create(const Desc& desc, const GpuFence::SharedPtr& pFence)
    {
        return SharedPtr(new D3D12DescriptorPool(desc, pFence));
    }

    D3D12DescriptorPool::D3D12DescriptorPool(const Desc& desc, const GpuFence::SharedPtr& pFence)
        : mDesc(desc)
        , mpFence(pFence)
    {
        // Find out how many heaps we need
        static_assert(D3D12DescriptorPool::kTypeCount == 13, "Unexpected desc count, make sure all desc types are supported");
        uint32_t descCount[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES] = { 0 };

        descCount[D3D12_DESCRIPTOR_HEAP_TYPE_RTV] = mDesc.mDescCount[(uint32_t)Type::Rtv];
        descCount[D3D12_DESCRIPTOR_HEAP_TYPE_DSV] = mDesc.mDescCount[(uint32_t)Type::Dsv];
        descCount[D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER] = mDesc.mDescCount[(uint32_t)Type::Sampler];
        descCount[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV] = mDesc.mDescCount[(uint32_t)Type::Cbv];
        descCount[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV] += mDesc.mDescCount[(uint32_t)Type::TextureSrv] + mDesc.mDescCount[(uint32_t)Type::RawBufferSrv] + mDesc.mDescCount[(uint32_t)Type::TypedBufferSrv] + mDesc.mDescCount[(uint32_t)Type::StructuredBufferSrv];
        descCount[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV] += mDesc.mDescCount[(uint32_t)Type::TextureUav] + mDesc.mDescCount[(uint32_t)Type::RawBufferUav] + mDesc.mDescCount[(uint32_t)Type::TypedBufferUav] + mDesc.mDescCount[(uint32_t)Type::StructuredBufferUav];
        descCount[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV] += mDesc.mDescCount[(uint32_t)Type::AccelerationStructureSrv];

        mpApiData = std::make_shared<DescriptorPoolApiData>();
        for (uint32_t i = 0; i < arraysize(mpApiData->pHeaps); i++)
        {
            if (descCount[i] > 0)
            {
                mpApiData->pHeaps[i] = D3D12DescriptorHeap::create(D3D12_DESCRIPTOR_HEAP_TYPE(i), descCount[i], mDesc.mShaderVisible);
            }
        }
    }

    D3D12DescriptorPool::~D3D12DescriptorPool() = default;

    void D3D12DescriptorPool::executeDeferredReleases()
    {
        uint64_t gpuVal = mpFence->getGpuValue();
        while (mpDeferredReleases.size() && mpDeferredReleases.top().fenceValue <= gpuVal)
        {
            mpDeferredReleases.pop();
        }
    }

    void D3D12DescriptorPool::releaseAllocation(std::shared_ptr<DescriptorSetApiData> pData)
    {
        DeferredRelease d;
        d.pData = pData;
        d.fenceValue = mpFence->getCpuValue();
        mpDeferredReleases.push(d);
    }
}

#endif // FALCOR_D3D12_AVAILABLE
