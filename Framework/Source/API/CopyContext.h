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
#pragma once
#include "API/Resource.h"
#ifdef FALCOR_LOW_LEVEL_API
#include "API/LowLevel/LowLevelContextData.h"
#endif

namespace Falcor
{
    class Texture;
    class Buffer;

    class CopyContext : public std::enable_shared_from_this<CopyContext>
    {
    public:
        using SharedPtr = std::shared_ptr<CopyContext>;
        using SharedConstPtr = std::shared_ptr<const CopyContext>;
        virtual ~CopyContext();

        static SharedPtr create(CommandQueueHandle queue);
        void updateBuffer(const Buffer* pBuffer, const void* pData, size_t offset = 0, size_t numBytes = 0);
        void updateTexture(const Texture* pTexture, const void* pData);
        void updateTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex, const void* pData);
        void updateTextureSubresources(const Texture* pTexture, uint32_t firstSubresource, uint32_t subresourceCount, const void* pData);
        std::vector<uint8> readTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex);

        /** Reset
        */
        virtual void reset();

        /** Flush the command list. This doesn't reset the command allocator, just submits the commands
            \param[in] wait If true, will block execution until the GPU finished processing the commands
        */
        virtual void flush(bool wait = false);

        /** Check if we have pending commands
        */
        bool hasPendingCommands() const { return mCommandsPending; }

        /** Signal the context that we have pending commands. Useful in case you make raw API calls
        */
        void setPendingCommands(bool commandsPending) { mCommandsPending = commandsPending; }

        /** Insert a resource barrier
        */
        virtual void resourceBarrier(const Resource* pResource, Resource::State newState);

        /** Copy an entire resource
        */
        void copyResource(const Resource* pDst, const Resource* pSrc);

        /** Copy a subresource
        */
        void copySubresource(const Texture* pDst, uint32_t dstSubresourceIdx, const Texture* pSrc, uint32_t srcSubresourceIdx);

        /** Copy part of a buffer 
        */
        void copyBufferRegion(const Buffer* pDst, uint64_t dstOffset, const Buffer* pSrc, uint64_t srcOffset, uint64_t numBytes);

#ifdef FALCOR_LOW_LEVEL_API
        /** Get the low-level context data
        */
        virtual LowLevelContextData::SharedPtr getLowLevelData() const { return mpLowLevelData; }

        /** Override the low-level context data with a user provided object
        */
        void setLowLevelContextData(LowLevelContextData::SharedPtr pLowLevelData) { mpLowLevelData = pLowLevelData; }
#endif
    protected:
        void bindDescriptorHeaps();
        CopyContext() = default;
        bool mCommandsPending = false;
#ifdef FALCOR_LOW_LEVEL_API
        LowLevelContextData::SharedPtr mpLowLevelData;
#endif
    };
}
