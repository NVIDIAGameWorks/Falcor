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
}