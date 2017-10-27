/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "API/ComputeContext.h"
#include "API/Device.h"
#include "API/DescriptorSet.h"
#include <cstring>

namespace Falcor
{
    void ComputeContext::prepareForDispatch()
    {
        assert(mpComputeState);
        if(mpComputeVars) applyComputeVars();

        ComputeStateObject::SharedPtr pCso = mpComputeState->getCSO(mpComputeVars.get());
        vkCmdBindPipeline(mpLowLevelData->getCommandList(), VK_PIPELINE_BIND_POINT_COMPUTE, pCso->getApiHandle());
        mBindComputeRootSig = false;
        mCommandsPending = true;
    }

    template<typename ViewType, typename ClearType>
    void clearColorImageCommon(CopyContext* pCtx, const ViewType* pView, const ClearType& clearVal)
    {
        if(pView->getApiHandle().getType() != VkResourceType::Image)
        {
            logWarning("Looks like you are trying to clear a buffer. Vulkan only supports clearing Buffers with a single uint value. Please use the uint version of clearUav(). Call is ignored");
            should_not_get_here();
            return;
        }
        pCtx->resourceBarrier(pView->getResource(), Resource::State::CopyDest);
        VkClearColorValue colVal;
        assert(sizeof(ClearType) <= sizeof(colVal.float32));
        std::memcpy(colVal.float32, &clearVal, sizeof(clearVal)); // VkClearColorValue is a union, so should work regardless of the ClearType
        VkImageSubresourceRange range;
        const auto& viewInfo = pView->getViewInfo();
        range.baseArrayLayer = viewInfo.firstArraySlice;
        range.baseMipLevel = viewInfo.mostDetailedMip;
        range.layerCount = viewInfo.arraySize;
        range.levelCount = viewInfo.mipCount;
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        vkCmdClearColorImage(pCtx->getLowLevelData()->getCommandList(), pView->getResource()->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &colVal, 1, &range);
    }

    template void clearColorImageCommon(CopyContext* pCtx, const RenderTargetView* pView, const vec4& clearVal);

    void ComputeContext::clearUAV(const UnorderedAccessView* pUav, const vec4& value)
    {
        clearColorImageCommon(this, pUav, value);
        mCommandsPending = true;
    }

    void ComputeContext::clearUAV(const UnorderedAccessView* pUav, const uvec4& value)
    {
        if(pUav->getApiHandle().getType() == VkResourceType::Buffer)
        {
            if ((value.x != value.y) || ((value.x != value.z) && (value.x != value.w)))
            {
                logWarning("Vulkan buffer clears only support a single element. A vector was supplied which has different elements per channel. only `x` will be used'");
            }
            const Buffer* pBuffer = dynamic_cast<const Buffer*>(pUav->getResource());
            vkCmdFillBuffer(getLowLevelData()->getCommandList(), pBuffer->getApiHandle(), pBuffer->getGpuAddressOffset(), pBuffer->getSize(), value.x);
        }
        else
        {
            clearColorImageCommon(this, pUav, value);
        }
        mCommandsPending = true;
    }

    void ComputeContext::clearUAVCounter(const StructuredBuffer::SharedPtr& pBuffer, uint32_t value)
    {
        if (pBuffer->hasUAVCounter())
        {
            clearUAV(pBuffer->getUAVCounter()->getUAV().get(), uvec4(value));
        }
    }

    void ComputeContext::initDispatchCommandSignature()
    {

    }

    void ComputeContext::dispatch(uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ)
    {
        prepareForDispatch();
        vkCmdDispatch(mpLowLevelData->getCommandList(), groupSizeX, groupSizeY, groupSizeZ);
    }

    void ComputeContext::dispatchIndirect(const Buffer* pArgBuffer, uint64_t argBufferOffset)
    {
        prepareForDispatch();
        resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
        vkCmdDispatchIndirect(mpLowLevelData->getCommandList(), pArgBuffer->getApiHandle(), pArgBuffer->getGpuAddressOffset() + argBufferOffset);
    }
}
