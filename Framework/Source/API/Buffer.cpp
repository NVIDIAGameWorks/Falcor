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
#include "API/Buffer.h"
#include "API/Device.h"
#include <cstring>

namespace Falcor
{
    size_t getBufferDataAlignment(const Buffer* pBuffer);
    void* mapBufferApi(const Buffer::ApiHandle& apiHandle, size_t size);

    Buffer::SharedPtr Buffer::create(size_t size, BindFlags usage, CpuAccess cpuAccess, const void* pInitData)
    {
        Buffer::SharedPtr pBuffer = SharedPtr(new Buffer(size, usage, cpuAccess));
        if (pBuffer->apiInit(pInitData != nullptr))
        {
            if (pInitData) pBuffer->updateData(pInitData, 0, size);
            return pBuffer;
        }
        else return nullptr;
    }

    Buffer::~Buffer()
    {
        if (mDynamicData.pResourceHandle)
        {
            gpDevice->getResourceAllocator()->release(mDynamicData);
        }
        else
        {
            gpDevice->releaseResource(mApiHandle);
        }
    }

    void Buffer::updateData(const void* pData, size_t offset, size_t size)
    {
        if (mCpuAccess == CpuAccess::Write)
        {
            uint8_t* pDst = (uint8_t*)map(MapType::WriteDiscard) + offset;
            std::memcpy(pDst, pData, size);
        }
        else
        {
            gpDevice->getRenderContext()->updateBuffer(this, pData, offset, size);
        }
    }

    void* Buffer::map(MapType type)
    {
        if (type == MapType::WriteDiscard)
        {
            if (mCpuAccess != CpuAccess::Write)
            {
                logError("Trying to map a buffer for write, but it wasn't created with the write permissions");
                return nullptr;
            }

            // Allocate a new buffer
            if (mDynamicData.pResourceHandle)
            {
                gpDevice->getResourceAllocator()->release(mDynamicData);
            }
            mDynamicData = gpDevice->getResourceAllocator()->allocate(mSize, getBufferDataAlignment(this));
            mApiHandle = mDynamicData.pResourceHandle;
            invalidateViews();
            return mDynamicData.pData;
        }
        else
        {
            assert(type == MapType::Read);

            if (mBindFlags == BindFlags::None)
            {
                return mapBufferApi(mApiHandle, mSize);
            }
            else
            {
                logWarning("Buffer::map() performance warning - using staging resource which require us to flush the pipeline and wait for the GPU to finish its work");
                if (mpStagingResource == nullptr)
                {
                    mpStagingResource = Buffer::create(mSize, Buffer::BindFlags::None, Buffer::CpuAccess::Read, nullptr);
                }

                // Copy the buffer and flush the pipeline
                RenderContext* pContext = gpDevice->getRenderContext().get();
                pContext->copyResource(mpStagingResource.get(), this);
                pContext->flush(true);
                return mpStagingResource->map(MapType::Read);
            }
        }
    }

    void CopyContext::updateBuffer(const Buffer* pBuffer, const void* pData, size_t offset, size_t numBytes)
    {
        if (numBytes == 0)
        {
            numBytes = pBuffer->getSize() - offset;
        }

        if (pBuffer->adjustSizeOffsetParams(numBytes, offset) == false)
        {
            logWarning("CopyContext::updateBuffer() - size and offset are invalid. Nothing to update.");
            return;
        }

        mCommandsPending = true;
        // Allocate a buffer on the upload heap
        uint8_t* pInitData = (uint8_t*)pData + offset;
        Buffer::SharedPtr pUploadBuffer = Buffer::create(numBytes, Buffer::BindFlags::None, Buffer::CpuAccess::Write, pInitData);

        copyBufferRegion(pBuffer, offset, pUploadBuffer.get(), 0, numBytes);
    }
}
