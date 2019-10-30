/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "Buffer.h"
#include "Device.h"

namespace Falcor
{
    size_t getBufferDataAlignment(const Buffer* pBuffer);
    void* mapBufferApi(const Buffer::ApiHandle& apiHandle, size_t size);

    Buffer::SharedPtr Buffer::create(size_t size, BindFlags usage, CpuAccess cpuAccess, const void* pInitData)
    {
        Buffer::SharedPtr pBuffer = SharedPtr(new Buffer(size, usage, cpuAccess));
        if (pBuffer->apiInit(pInitData != nullptr))
        {
            if (pInitData) pBuffer->setBlob(pInitData, 0, size);
            return pBuffer;
        }
        else return nullptr;
    }

    Buffer::SharedPtr Buffer::aliasResource(Resource::SharedPtr pBaseResource, GpuAddress offset, size_t size, Resource::BindFlags bindFlags)
    {
        assert(pBaseResource->asBuffer()); // Only aliasing buffers for now
        CpuAccess cpuAccess = pBaseResource->asBuffer() ? pBaseResource->asBuffer()->getCpuAccess() : CpuAccess::None;
        if (cpuAccess != CpuAccess::None)
        {
            logError("Buffer::aliasResource() - trying to alias a buffer with CpuAccess::" + to_string(cpuAccess) + " which is illegal. Aliased resource must have CpuAccess::None");
            return nullptr;
        }

        if ((pBaseResource->getBindFlags() & bindFlags) != bindFlags)
        {
            logError("Buffer::aliasResource() - requested buffer bind-flags don't match the aliased resource bind flags.\nRequested = " + to_string(bindFlags) + "\nAliased = " + to_string(pBaseResource->getBindFlags()));
            return nullptr;
        }

        if (offset >= pBaseResource->getSize() || (offset + size) >= pBaseResource->getSize())
        {
            logError("Buffer::aliasResource() - requested offset and size don't fit inside the alias resource dimensions. Requesed size = " +
                to_string(size) + ", offset = " + to_string(offset) + ". Aliased resource size = " + to_string(pBaseResource->getSize()));
            return nullptr;
        }

        SharedPtr pBuffer = SharedPtr(new Buffer(size, bindFlags, CpuAccess::None));
        pBuffer->mpAliasedResource = pBaseResource;
        pBuffer->mApiHandle = pBaseResource->getApiHandle();
        pBuffer->mGpuVaOffset = offset;
        return pBuffer;
    }

    Buffer::SharedPtr Buffer::createFromApiHandle(ApiHandle handle, size_t size, Resource::BindFlags usage, CpuAccess cpuAccess)
    {
        Buffer::SharedPtr pBuffer = SharedPtr(new Buffer(size, usage, cpuAccess));
        pBuffer->mApiHandle = handle;
        return pBuffer->mApiHandle ? pBuffer : nullptr;
    }

    Buffer::~Buffer()
    {
        if (mpAliasedResource) return;

        if (mDynamicData.pResourceHandle)
        {
            gpDevice->getUploadHeap()->release(mDynamicData);
        }
        else
        {
            gpDevice->releaseResource(mApiHandle);
        }
    }

    template<typename ViewClass>
    using CreateFuncType = std::function<typename ViewClass::SharedPtr(Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)>;

    template<typename ViewClass, typename ViewMapType>
    typename ViewClass::SharedPtr findViewCommon(Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount, ViewMapType& viewMap, CreateFuncType<ViewClass> createFunc)
    {
        ResourceViewInfo view = ResourceViewInfo(firstElement, elementCount);

        if (viewMap.find(view) == viewMap.end())
        {
            viewMap[view] = createFunc(pBuffer, firstElement, elementCount);
        }

        return viewMap[view];
    }

    ShaderResourceView::SharedPtr Buffer::getSRV(uint32_t firstElement, uint32_t elementCount)
    {
        auto createFunc = [](Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
        {
            return ShaderResourceView::create(pBuffer->shared_from_this(), firstElement, elementCount);
        };

        return findViewCommon<ShaderResourceView>(this, firstElement, elementCount, mSrvs, createFunc);
    }

    UnorderedAccessView::SharedPtr Buffer::getUAV(uint32_t firstElement, uint32_t elementCount)
    {
        auto createFunc = [](Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
        {
            return UnorderedAccessView::create(pBuffer->shared_from_this(), firstElement, elementCount);
        };

        return findViewCommon<UnorderedAccessView>(this, firstElement, elementCount, mUavs, createFunc);
    }

    bool Buffer::setBlob(const void* pData, size_t offset, size_t size)
    {
        if (offset + size > mSize)
        {
            logError("Error when setting blob to buffer. Blob to large and will result in an overflow. Ignoring call");
            return false;
        }

        if (mCpuAccess == CpuAccess::Write)
        {
            uint8_t* pDst = (uint8_t*)map(MapType::WriteDiscard) + offset;
            std::memcpy(pDst, pData, size);
        }
        else
        {
            gpDevice->getRenderContext()->updateBuffer(this, pData, offset, size);
        }
        return true;
    }

    void Buffer::updateData(const void* pData, size_t offset, size_t size)
    {
        setBlob(pData, offset, size);
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
                gpDevice->getUploadHeap()->release(mDynamicData);
            }
            mDynamicData = gpDevice->getUploadHeap()->allocate(mSize, getBufferDataAlignment(this));
            mApiHandle = mDynamicData.pResourceHandle;
            mGpuVaOffset = mDynamicData.offset;
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
                RenderContext* pContext = gpDevice->getRenderContext();
                pContext->copyResource(mpStagingResource.get(), this);
                pContext->flush(true);
                return mpStagingResource->map(MapType::Read);
            }
        }
    }

    SCRIPT_BINDING(Buffer)
    {
        m.regClass(Buffer);
    }
}
