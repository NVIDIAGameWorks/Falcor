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
#include "Buffer.h"
#include "Device.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/Program/Program.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"

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
                throw RuntimeError("Can't create a structured buffer from the variable '{}'. The variable is not a structured buffer.", varName);
            }

            FALCOR_ASSERT(pResourceType->getSize() <= std::numeric_limits<uint32_t>::max());
            return Buffer::createStructured((uint32_t)pResourceType->getSize(), elementCount, bindFlags, cpuAccess, pInitData, createCounter);
        }
    }

    size_t getBufferDataAlignment(const Buffer* pBuffer);
    void* mapBufferApi(const Buffer::ApiHandle& apiHandle, size_t size);

    Buffer::Buffer(size_t size, BindFlags bindFlags, CpuAccess cpuAccess)
        : Resource(Type::Buffer, bindFlags, size)
        , mCpuAccess(cpuAccess)
    {
        checkArgument(size > 0, "Can't create GPU buffer of size zero");

        // Check that buffer size is within 4GB limit. Larger buffers are currently not well supported in D3D12.
        // TODO: Revisit this check in the future.
        if (size > (1ull << 32))
        {
            logWarning("Creating GPU buffer of size {} bytes. Buffers above 4GB are not currently well supported.", size);
        }
    }

    Buffer::SharedPtr Buffer::create(size_t size, BindFlags bindFlags, CpuAccess cpuAccess, const void* pInitData)
    {
        Buffer::SharedPtr pBuffer = SharedPtr(new Buffer(size, bindFlags, cpuAccess));
        pBuffer->apiInit(pInitData != nullptr);
        if (pInitData) pBuffer->setBlob(pInitData, 0, size);
        pBuffer->mElementCount = uint32_t(size);
        return pBuffer;
    }

    Buffer::SharedPtr Buffer::createTyped(ResourceFormat format, uint32_t elementCount, BindFlags bindFlags, CpuAccess cpuAccess, const void* pInitData)
    {
        size_t size = (size_t)elementCount * getFormatBytesPerBlock(format);
        SharedPtr pBuffer = create(size, bindFlags, cpuAccess, pInitData);
        FALCOR_ASSERT(pBuffer);

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
        size_t size = (size_t)structSize * elementCount;
        Buffer::SharedPtr pBuffer = create(size, bindFlags, cpuAccess, pInitData);
        FALCOR_ASSERT(pBuffer);

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
            throw RuntimeError("Can't find a structured buffer named '{}' in the program", name);
        }
        return createStructuredFromType(pVar->getType().get(), name, elementCount, bindFlags, cpuAccess, pInitData, createCounter);
    }

    Buffer::SharedPtr Buffer::aliasResource(Resource::SharedPtr pBaseResource, GpuAddress offset, size_t size, Resource::BindFlags bindFlags)
    {
        FALCOR_ASSERT(pBaseResource && pBaseResource->asBuffer()); // Only aliasing buffers for now
        CpuAccess cpuAccess = pBaseResource->asBuffer() ? pBaseResource->asBuffer()->getCpuAccess() : CpuAccess::None;
        checkArgument(cpuAccess != CpuAccess::None, "'pBaseResource' has CpuAccess:{} which is illegal. Aliased resources must have CpuAccess::None.", to_string(cpuAccess));
        checkArgument((pBaseResource->getBindFlags() & bindFlags) != bindFlags, "'bindFlags' ({}) don't match aliased resource bind flags {}.", to_string(bindFlags), to_string(pBaseResource->getBindFlags()));
        if (offset >= pBaseResource->getSize() || (offset + size) >= pBaseResource->getSize())
        {
            throw ArgumentError("'offset' ({}) and 'size' ({}) don't fit inside the aliased resource size {}.", offset, size, pBaseResource->getSize());
        }

        SharedPtr pBuffer = SharedPtr(new Buffer(size, bindFlags, CpuAccess::None));
        pBuffer->mpAliasedResource = pBaseResource;
        pBuffer->mApiHandle = pBaseResource->getApiHandle();
        pBuffer->mGpuVaOffset = offset;
        return pBuffer;
    }

    Buffer::SharedPtr Buffer::createFromApiHandle(ApiHandle handle, size_t size, Resource::BindFlags bindFlags, CpuAccess cpuAccess)
    {
        FALCOR_ASSERT(handle);
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
#ifdef FALCOR_D3D12
            gpDevice->releaseResource(mApiHandle);
#elif defined(FALCOR_GFX)
            gpDevice->releaseResource(static_cast<ApiObjectHandle>(mApiHandle.get()));
#endif
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

    void Buffer::setBlob(const void* pData, size_t offset, size_t size)
    {
        if (offset + size > mSize)
        {
            throw ArgumentError("'offset' ({}) and 'size' ({}) don't fit the buffer size {}.", offset, size, mSize);
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
    }

    void* Buffer::map(MapType type)
    {
        if (type == MapType::Write)
        {
            checkArgument(mCpuAccess == CpuAccess::Write, "Trying to map a buffer for write, but it wasn't created with the write permissions.");
            return mDynamicData.pData;
        }
        else if (type == MapType::WriteDiscard)
        {
            checkArgument(mCpuAccess == CpuAccess::Write, "Trying to map a buffer for write, but it wasn't created with the write permissions.");

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
            FALCOR_ASSERT(type == MapType::Read);

            if (mCpuAccess == CpuAccess::Write)
            {
                // Buffers on the upload heap are already mapped, just return the ptr.
                FALCOR_ASSERT(mDynamicData.pResourceHandle);
                FALCOR_ASSERT(mDynamicData.pData);
                return mDynamicData.pData;
            }
            else if (mCpuAccess == CpuAccess::Read)
            {
                FALCOR_ASSERT(mBindFlags == BindFlags::None);
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
                FALCOR_ASSERT(mGpuVaOffset == 0);
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

    uint32_t Buffer::getElementSize() const
    {
        if (mStructSize != 0) return mStructSize;
        if (mFormat == ResourceFormat::Unknown) return 1;

        throw RuntimeError("Inferring element size from resource format is not implemented");
    }

    bool Buffer::adjustSizeOffsetParams(size_t& size, size_t& offset) const
    {
        if (offset >= mSize)
        {
            logWarning("Buffer::adjustSizeOffsetParams() - offset is larger than the buffer size.");
            return false;
        }

        if (offset + size > mSize)
        {
            logWarning("Buffer::adjustSizeOffsetParams() - offset + size will cause an OOB access. Clamping size");
            size = mSize - offset;
        }
        return true;
    }


    FALCOR_SCRIPT_BINDING(Buffer)
    {
        pybind11::class_<Buffer, Buffer::SharedPtr>(m, "Buffer");
    }
}
