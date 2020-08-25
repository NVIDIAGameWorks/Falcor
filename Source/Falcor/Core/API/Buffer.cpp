/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Buffer.h"
#include "Device.h"

namespace Falcor
{
    namespace
    {
        Buffer::SharedPtr createStructuredFromType(
            const ReflectionType* pType,
            const std::string& varName,
            uint32_t elementCount,
            ResourceBindFlags bindFlags,
            Buffer::CpuAccess cpuAccess,
            const void* pInitData,
            bool createCounter)
        {
            const ReflectionResourceType* pResourceType = pType->unwrapArray()->asResourceType();
            if (!pResourceType || pResourceType->getType() != ReflectionResourceType::Type::StructuredBuffer)
            {
                throw std::exception(("Can't create a structured buffer from the variable '" + varName + "'. The variable is not a structured buffer.").c_str());
            }

            assert(pResourceType->getSize() <= std::numeric_limits<uint32_t>::max());
            return Buffer::createStructured((uint32_t)pResourceType->getSize(), elementCount, bindFlags, cpuAccess, pInitData, createCounter);
        }
    }

    size_t getBufferDataAlignment(const Buffer* pBuffer);
    void* mapBufferApi(const Buffer::ApiHandle& apiHandle, size_t size);

    Buffer::Buffer(size_t size, BindFlags bindFlags, CpuAccess cpuAccess)
        : Resource(Type::Buffer, bindFlags, size)
        , mCpuAccess(cpuAccess)
    {
    }

    Buffer::SharedPtr Buffer::create(size_t size, BindFlags bindFlags, CpuAccess cpuAccess, const void* pInitData)
    {
        Buffer::SharedPtr pBuffer = SharedPtr(new Buffer(size, bindFlags, cpuAccess));
        pBuffer->apiInit(pInitData != nullptr);
        if (pInitData) pBuffer->setBlob(pInitData, 0, size);
        return pBuffer;
    }

    Buffer::SharedPtr Buffer::createTyped(ResourceFormat format, uint32_t elementCount, BindFlags bindFlags, CpuAccess cpuAccess, const void* pInitData)
    {
        size_t size = elementCount * getFormatBytesPerBlock(format);
        SharedPtr pBuffer = create(size, bindFlags, cpuAccess, pInitData);
        assert(pBuffer);

        pBuffer->mFormat = format;
        pBuffer->mElementCount = elementCount;
        return pBuffer;
    }

    Buffer::SharedPtr Buffer::createStructured(
        uint32_t structSize,
        uint32_t elementCount,
        ResourceBindFlags bindFlags,
        CpuAccess cpuAccess,
        const void* pInitData,
        bool createCounter)
    {
        size_t size = structSize * elementCount;
        Buffer::SharedPtr pBuffer = create(size, bindFlags, cpuAccess, pInitData);
        assert(pBuffer);

        pBuffer->mElementCount = elementCount;
        pBuffer->mStructSize = structSize;
        static const uint32_t zero = 0;
        if (createCounter)
        {
            pBuffer->mpUAVCounter = Buffer::create(sizeof(uint32_t), Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
        }
        return pBuffer;
    }

    Buffer::SharedPtr Buffer::createStructured(
        const ShaderVar& shaderVar,
        uint32_t elementCount,
        ResourceBindFlags bindFlags,
        CpuAccess cpuAccess,
        const void* pInitData,
        bool createCounter)
    {
        return createStructuredFromType(shaderVar.getType().get(), "<Unknown ShaderVar>", elementCount, bindFlags, cpuAccess, pInitData, createCounter);
    }

    Buffer::SharedPtr Buffer::createStructured(
        const Program* pProgram,
        const std::string& name,
        uint32_t elementCount,
        ResourceBindFlags bindFlags,
        CpuAccess cpuAccess,
        const void* pInitData,
        bool createCounter)
    {
        const auto& pDefaultBlock = pProgram->getReflector()->getDefaultParameterBlock();
        const ReflectionVar* pVar = pDefaultBlock ? pDefaultBlock->getResource(name).get() : nullptr;
        if (pVar == nullptr)
        {
            throw std::exception(("Can't find a structured buffer named '" + name + "' in the program").c_str());
        }
        return createStructuredFromType(pVar->getType().get(), name, elementCount, bindFlags, cpuAccess, pInitData, createCounter);
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
            logError("Buffer::aliasResource() - requested offset and size don't fit inside the alias resource dimensions. Requested size = " +
                std::to_string(size) + ", offset = " + std::to_string(offset) + ". Aliased resource size = " + std::to_string(pBaseResource->getSize()));
            return nullptr;
        }

        SharedPtr pBuffer = SharedPtr(new Buffer(size, bindFlags, CpuAccess::None));
        pBuffer->mpAliasedResource = pBaseResource;
        pBuffer->mApiHandle = pBaseResource->getApiHandle();
        pBuffer->mGpuVaOffset = offset;
        return pBuffer;
    }

    Buffer::SharedPtr Buffer::createFromApiHandle(ApiHandle handle, size_t size, Resource::BindFlags bindFlags, CpuAccess cpuAccess)
    {
        assert(handle);
        Buffer::SharedPtr pBuffer = SharedPtr(new Buffer(size, bindFlags, cpuAccess));
        pBuffer->mApiHandle = handle;
        return pBuffer;
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
            return ShaderResourceView::create(std::static_pointer_cast<Buffer>(pBuffer->shared_from_this()), firstElement, elementCount);
        };

        return findViewCommon<ShaderResourceView>(this, firstElement, elementCount, mSrvs, createFunc);
    }

    ShaderResourceView::SharedPtr Buffer::getSRV()
    {
        return getSRV(0);
    }

    UnorderedAccessView::SharedPtr Buffer::getUAV(uint32_t firstElement, uint32_t elementCount)
    {
        auto createFunc = [](Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
        {
            return UnorderedAccessView::create(std::static_pointer_cast<Buffer>(pBuffer->shared_from_this()), firstElement, elementCount);
        };

        return findViewCommon<UnorderedAccessView>(this, firstElement, elementCount, mUavs, createFunc);
    }

    UnorderedAccessView::SharedPtr Buffer::getUAV()
    {
        return getUAV(0);
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

    void* Buffer::map(MapType type)
    {
        if (type == MapType::Write)
        {
            if (mCpuAccess != CpuAccess::Write)
            {
                logError("Trying to map a buffer for write, but it wasn't created with the write permissions");
                return nullptr;
            }
            return mDynamicData.pData;
        }
        else if (type == MapType::WriteDiscard)
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
            mpCBV = nullptr;
            mDynamicData = gpDevice->getUploadHeap()->allocate(mSize, getBufferDataAlignment(this));
            mApiHandle = mDynamicData.pResourceHandle;
            mGpuVaOffset = mDynamicData.offset;
            invalidateViews();
            return mDynamicData.pData;
        }
        else
        {
            assert(type == MapType::Read);

            if (mCpuAccess == CpuAccess::Write)
            {
                // Buffers on the upload heap are already mapped, just return the ptr.
                assert(mDynamicData.pResourceHandle);
                assert(mDynamicData.pData);
                return mDynamicData.pData;
            }
            else if (mCpuAccess == CpuAccess::Read)
            {
                assert(mBindFlags == BindFlags::None);
                return mapBufferApi(mApiHandle, mSize);
            }
            else
            {
                // For buffers without CPU access we must copy the contents to a staging buffer.
                logWarning("Buffer::map() performance warning - using staging resource which require us to flush the pipeline and wait for the GPU to finish its work");
                if (mpStagingResource == nullptr)
                {
                    mpStagingResource = Buffer::create(mSize, Buffer::BindFlags::None, Buffer::CpuAccess::Read, nullptr);
                }

                // Copy the buffer and flush the pipeline
                RenderContext* pContext = gpDevice->getRenderContext();
                assert(mGpuVaOffset == 0);
                pContext->copyResource(mpStagingResource.get(), this);
                pContext->flush(true);
                return mpStagingResource->map(MapType::Read);
            }
        }
    }

    ConstantBufferView::SharedPtr Buffer::getCBV()
    {
        if (!mpCBV) mpCBV = ConstantBufferView::create(std::static_pointer_cast<Buffer>(shared_from_this()));
        return mpCBV;
    }

    SCRIPT_BINDING(Buffer)
    {
        pybind11::class_<Buffer, Buffer::SharedPtr>(m, "Buffer");
    }
}
