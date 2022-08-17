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
#include "GpuMemoryHeap.h"
#include "GpuFence.h"
#include "Core/Assert.h"
#include "Utils/Math/Common.h"

namespace Falcor
{
    GpuMemoryHeap::~GpuMemoryHeap()
    {
        mDeferredReleases = decltype(mDeferredReleases)();
    }

    GpuMemoryHeap::GpuMemoryHeap(Type type, size_t pageSize, const GpuFence::SharedPtr& pFence)
        : mType(type)
        , mPageSize(pageSize)
        , mpFence(pFence)
    {
        allocateNewPage();
    }

    GpuMemoryHeap::SharedPtr GpuMemoryHeap::create(Type type, size_t pageSize, const GpuFence::SharedPtr& pFence)
    {
        return SharedPtr(new GpuMemoryHeap(type, pageSize, pFence));
    }

    void GpuMemoryHeap::allocateNewPage()
    {
        if (mpActivePage)
        {
            mUsedPages[mCurrentPageId] = std::move(mpActivePage);
        }

        if (mAvailablePages.size())
        {
            mpActivePage = std::move(mAvailablePages.front());
            mAvailablePages.pop();
            mpActivePage->allocationsCount = 0;
            mpActivePage->currentOffset = 0;
        }
        else
        {
            mpActivePage = std::make_unique<PageData>();
            initBasePageData((*mpActivePage), mPageSize);
        }

        mpActivePage->currentOffset = 0;
        mCurrentPageId++;
    }

    GpuMemoryHeap::Allocation GpuMemoryHeap::allocate(size_t size, size_t alignment)
    {
        Allocation data;
        if (size > mPageSize)
        {
            data.pageID = GpuMemoryHeap::Allocation::kMegaPageId;
            initBasePageData(data, size);
        }
        else
        {
            // Calculate the start
            size_t currentOffset = align_to(alignment, mpActivePage->currentOffset);
            if (currentOffset + size > mPageSize)
            {
                currentOffset = 0;
                allocateNewPage();
            }

            data.pageID = mCurrentPageId;
            data.offset = currentOffset;
            data.pData = mpActivePage->pData + currentOffset;
            data.pResourceHandle = mpActivePage->pResourceHandle;
            mpActivePage->currentOffset = currentOffset + size;
            mpActivePage->allocationsCount++;
        }

        data.fenceValue = mpFence->getCpuValue();
        return data;
    }

    void GpuMemoryHeap::release(Allocation& data)
    {
        FALCOR_ASSERT(data.pResourceHandle);
        mDeferredReleases.push(data);
    }

    void GpuMemoryHeap::executeDeferredReleases()
    {
        uint64_t gpuVal = mpFence->getGpuValue();
        while (mDeferredReleases.size() && mDeferredReleases.top().fenceValue <= gpuVal)
        {
            const Allocation& data = mDeferredReleases.top();
            if (data.pageID == mCurrentPageId)
            {
                mpActivePage->allocationsCount--;
                if (mpActivePage->allocationsCount == 0)
                {
                    mpActivePage->currentOffset = 0;
                }
            }
            else
            {
                if (data.pageID != Allocation::kMegaPageId)
                {
                    auto& pData = mUsedPages[data.pageID];
                    pData->allocationsCount--;
                    if (pData->allocationsCount == 0)
                    {
                        mAvailablePages.push(std::move(pData));
                        mUsedPages.erase(data.pageID);
                    }
                }
                // else it's a mega-page. Popping it will release the resource
            }
            mDeferredReleases.pop();
        }
    }
}
