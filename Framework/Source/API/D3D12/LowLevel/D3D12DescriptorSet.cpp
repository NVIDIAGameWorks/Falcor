/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "API/DescriptorSet.h"
#include "D3D12DescriptorHeap.h"
#include "D3D12DescriptorData.h"
#include "API/Device.h"

namespace Falcor
{
    D3D12_DESCRIPTOR_HEAP_TYPE falcorToDxDescType(DescriptorPool::Type t);

    static D3D12DescriptorHeap* getHeap(const DescriptorPool* pPool, DescriptorSet::Type type)
    {
        auto dxType = falcorToDxDescType(type);
        D3D12DescriptorHeap* pHeap = pPool->getApiData()->pHeaps[dxType].get();
        assert(pHeap->getType() == dxType);
        return pHeap;
    }

    bool DescriptorSet::apiInit()
    {
        mpApiData = std::make_shared<DescriptorSetApiData>();

        // For each range we need to allocate a table from a heap
        mpApiData->allocations.resize(mLayout.getRangeCount());
        for (size_t i = 0; i < mpApiData->allocations.size(); i++)
        {
            const auto& range = mLayout.getRange(i);
            D3D12DescriptorHeap* pHeap = getHeap(mpPool.get(), range.type);
            mpApiData->allocations[i] = pHeap->allocateDescriptors(range.descCount);
            if (mpApiData->allocations[i] == false)
            {
                // Execute deferred releases and try again
                mpPool->executeDeferredReleases();
                mpApiData->allocations[i] = pHeap->allocateDescriptors(range.descCount);
                if (!mpApiData->allocations[i])
                {
                    return false;
                }
            }
        }
        return true;
    }

    DescriptorSet::CpuHandle DescriptorSet::getCpuHandle(uint32_t rangeIndex, uint32_t descInRange) const
    {
        return mpApiData->allocations[rangeIndex]->getCpuHandle(descInRange);
    }

    DescriptorSet::GpuHandle DescriptorSet::getGpuHandle(uint32_t rangeIndex, uint32_t descInRange) const
    {
        return mpApiData->allocations[rangeIndex]->getGpuHandle(descInRange);
    }

    void setCpuHandle(DescriptorSet* pSet, uint32_t rangeIndex, uint32_t descIndex, const DescriptorSet::CpuHandle& handle)
    {
        auto dstHandle = pSet->getCpuHandle(rangeIndex, descIndex);
        gpDevice->getApiHandle()->CopyDescriptorsSimple(1, dstHandle, handle, falcorToDxDescType(pSet->getRange(rangeIndex).type));
    }

    void DescriptorSet::setSrv(uint32_t rangeIndex, uint32_t descIndex, const ShaderResourceView* pSrv)
    {
        setCpuHandle(this, rangeIndex, descIndex, pSrv->getApiHandle()->getCpuHandle(0));
    }

    void DescriptorSet::setUav(uint32_t rangeIndex, uint32_t descIndex, const UnorderedAccessView* pUav)
    {
        setCpuHandle(this, rangeIndex, descIndex, pUav->getApiHandle()->getCpuHandle(0));
    }

    void DescriptorSet::setSampler(uint32_t rangeIndex, uint32_t descIndex, const Sampler* pSampler)
    {
        setCpuHandle(this, rangeIndex, descIndex, pSampler->getApiHandle()->getCpuHandle(0));
    }

    void DescriptorSet::bindForGraphics(CopyContext* pCtx, const RootSignature* pRootSig, uint32_t rootIndex)
    {
        pCtx->getLowLevelData()->getCommandList()->SetGraphicsRootDescriptorTable(rootIndex, getGpuHandle(0));
    }

    void DescriptorSet::bindForCompute(CopyContext* pCtx, const RootSignature* pRootSig, uint32_t rootIndex)
    {
        pCtx->getLowLevelData()->getCommandList()->SetComputeRootDescriptorTable(rootIndex, getGpuHandle(0));
    }

    void DescriptorSet::setCb(uint32_t rangeIndex, uint32_t descIndex, const Buffer* pBuffer)
    {
        UNSUPPORTED_IN_D3D12("DescriptorSet::setCb");
    }
}