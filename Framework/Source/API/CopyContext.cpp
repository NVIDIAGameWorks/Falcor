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
#include "API/CopyContext.h"
#include "API/Device.h"
#include "API/Buffer.h"
#include <queue>

namespace Falcor
{
    CopyContext::~CopyContext() = default;

    CopyContext::SharedPtr CopyContext::create(CommandQueueHandle queue)
    {
        SharedPtr pCtx = SharedPtr(new CopyContext());
        pCtx->mpLowLevelData = LowLevelContextData::create(LowLevelContextData::CommandQueueType::Copy, queue);
        return pCtx->mpLowLevelData ? pCtx : nullptr;
    }

    void CopyContext::flush(bool wait)
    {
        if (mCommandsPending)
        {
            mpLowLevelData->flush();
            mCommandsPending = false;
        }

        bindDescriptorHeaps();

        if (wait)
        {
            mpLowLevelData->getFence()->syncCpu();
        }
        gEventCounter.numFlushes++;
    }
    
    void CopyContext::updateTexture(const Texture* pTexture, const void* pData)
    {
        mCommandsPending = true;
        uint32_t subresourceCount = pTexture->getArraySize() * pTexture->getMipCount();
        if (pTexture->getType() == Texture::Type::TextureCube)
        {
            subresourceCount *= 6;
        }
        updateTextureSubresources(pTexture, 0, subresourceCount, pData);
    }

    CopyContext::ReadTextureTask::SharedPtr CopyContext::asyncReadTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex)
    {
        return CopyContext::ReadTextureTask::create(shared_from_this(), pTexture, subresourceIndex);
    }

    std::vector<uint8> CopyContext::readTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex)
    {
        CopyContext::ReadTextureTask::SharedPtr pTask = asyncReadTextureSubresource(pTexture, subresourceIndex);
        return pTask->getData();
    }



    void CopyContext::resourceBarrier(const Resource* pResource, Resource::State newState, const ResourceViewInfo* pViewInfo)
    {
        const Texture* pTexture = dynamic_cast<const Texture*>(pResource);
        if (pTexture)
        {
            bool globalBarrier = pTexture->isStateGlobal();
            if (pViewInfo)
            {
                globalBarrier = globalBarrier && pViewInfo->firstArraySlice == 0;
                globalBarrier = globalBarrier && pViewInfo->mostDetailedMip == 0;
                globalBarrier = globalBarrier && pViewInfo->mipCount == pTexture->getMipCount();
                globalBarrier = globalBarrier && pViewInfo->arraySize == pTexture->getArraySize();
            }

            if (globalBarrier)
            {
                textureBarrier(pTexture, newState);
            }
            else
            {
                subresourceBarriers(pTexture, newState, pViewInfo);
            }
        }
        else
        {
            const Buffer* pBuffer = dynamic_cast<const Buffer*>(pResource);
            bufferBarrier(pBuffer, newState);
        }
    }

    void CopyContext::subresourceBarriers(const Texture* pTexture, Resource::State newState, const ResourceViewInfo* pViewInfo)
    {
        ResourceViewInfo fullResource;
        bool setGlobal = false;
        if (pViewInfo == nullptr)
        {
            fullResource.arraySize = pTexture->getArraySize();
            fullResource.firstArraySlice = 0;
            fullResource.mipCount = pTexture->getMipCount();
            fullResource.mostDetailedMip = 0;
            setGlobal = true;
            pViewInfo = &fullResource;
        }

        for (uint32_t a = pViewInfo->firstArraySlice; a < pViewInfo->firstArraySlice + pViewInfo->arraySize; a++)
        {
            for (uint32_t m = pViewInfo->mostDetailedMip; m < pViewInfo->mipCount + pViewInfo->mostDetailedMip; m++)
            {
                Resource::State oldState = pTexture->getSubresourceState(a, m);
                if (oldState != newState)
                {
                    apiSubresourceBarrier(pTexture, newState, oldState, a, m);
                    if (setGlobal == false) pTexture->setSubresourceState(a, m, newState);
                    mCommandsPending = true;
                }
            }
        }
        if (setGlobal) pTexture->setGlobalState(newState);
    }
}