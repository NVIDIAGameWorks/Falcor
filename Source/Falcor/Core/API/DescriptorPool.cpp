/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "DescriptorPool.h"

namespace Falcor
{
    DescriptorPool::SharedPtr DescriptorPool::create(const Desc& desc, const GpuFence::SharedPtr& pFence)
    {
        return SharedPtr(new DescriptorPool(desc, pFence));
    }

    DescriptorPool::DescriptorPool(const Desc& desc, const GpuFence::SharedPtr& pFence)
        : mDesc(desc)
        , mpFence(pFence)
    {
        apiInit();
    }

    DescriptorPool::~DescriptorPool() = default;

    void DescriptorPool::executeDeferredReleases()
    {
        uint64_t gpuVal = mpFence->getGpuValue();
        while (mpDeferredReleases.size() && mpDeferredReleases.top().fenceValue <= gpuVal)
        {
            mpDeferredReleases.pop();
        }
    }

    void DescriptorPool::releaseAllocation(std::shared_ptr<DescriptorSetApiData> pData)
    {
        DeferredRelease d;
        d.pData = pData;
        d.fenceValue = mpFence->getCpuValue();
        mpDeferredReleases.push(d);
    }
}
