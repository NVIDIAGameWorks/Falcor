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
#pragma once
#include "Handles.h"
#include "Shared/D3D12Handles.h"
#include "Core/Macros.h"
#include "Core/Program/ProgramReflection.h"
#include <vector>
#include <memory>

namespace Falcor
{
    class Resource;
    class Texture;
    class Buffer;
    using ResourceWeakPtr = std::weak_ptr<Resource>;
    using ResourceSharedPtr = std::shared_ptr<Resource>;
    using ConstTextureSharedPtrRef = const std::shared_ptr<Texture>&;
    using ConstBufferSharedPtrRef = const std::shared_ptr<Buffer>&;

    struct FALCOR_API ResourceViewInfo
    {
        ResourceViewInfo() = default;
        ResourceViewInfo(uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
            : mostDetailedMip(mostDetailedMip), mipCount(mipCount), firstArraySlice(firstArraySlice), arraySize(arraySize) {}

        ResourceViewInfo(uint32_t firstElement, uint32_t elementCount)
            : firstElement(firstElement), elementCount(elementCount) {}

        static const uint32_t kMaxPossible = -1;

        // Textures
        uint32_t mostDetailedMip = 0;
        uint32_t mipCount = kMaxPossible;
        uint32_t firstArraySlice = 0;
        uint32_t arraySize = kMaxPossible;

        // Buffers
        uint32_t firstElement = 0;
        uint32_t elementCount = kMaxPossible;

        bool operator==(const ResourceViewInfo& other) const
        {
            return (firstArraySlice == other.firstArraySlice)
                && (arraySize == other.arraySize)
                && (mipCount == other.mipCount)
                && (mostDetailedMip == other.mostDetailedMip)
                && (firstElement == other.firstElement)
                && (elementCount == other.elementCount);
        }
    };

    /** Abstracts API resource views.
    */
    template<typename ApiHandleType>
    class FALCOR_API ResourceView
    {
    public:
        using ApiHandle = ApiHandleType;
        using Dimension = ReflectionResourceType::Dimensions;
        static const uint32_t kMaxPossible = -1;
        virtual ~ResourceView();

        ResourceView(ResourceWeakPtr& pResource, ApiHandle handle, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
            : mApiHandle(handle), mpResource(pResource), mViewInfo(mostDetailedMip, mipCount, firstArraySlice, arraySize) {}

        ResourceView(ResourceWeakPtr& pResource, ApiHandle handle, uint32_t firstElement, uint32_t elementCount)
            : mApiHandle(handle), mpResource(pResource), mViewInfo(firstElement, elementCount) {}

        ResourceView(ResourceWeakPtr& pResource, ApiHandle handle)
            : mApiHandle(handle), mpResource(pResource) {}

        /** Get the raw API handle.
        */
        const ApiHandle& getApiHandle() const { return mApiHandle; }

        /** Get information about the view.
        */
        const ResourceViewInfo& getViewInfo() const { return mViewInfo; }

        /** Get the resource referenced by the view.
        */
        ResourceSharedPtr getResource() const { return mpResource.lock(); }

#if FALCOR_HAS_CUDA
        /** Get the CUDA device address for this view.
        */
        void* getCUDADeviceAddress() const
        {
            return mpResource.lock()->getCUDADeviceAddress(mViewInfo);
        }
#endif

    protected:
        ApiHandle mApiHandle;
        ResourceViewInfo mViewInfo;
        ResourceWeakPtr mpResource;
    };

    class FALCOR_API ShaderResourceView : public ResourceView<SrvHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<ShaderResourceView>;
        using SharedConstPtr = std::shared_ptr<const ShaderResourceView>;

        static SharedPtr create(ConstTextureSharedPtrRef pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize);
        static SharedPtr create(ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount);
        static SharedPtr create(Dimension dimension);
#ifdef FALCOR_D3D12
        static SharedPtr createViewForAccelerationStructure(ConstBufferSharedPtrRef pBuffer);
#endif
        static SharedPtr getNullView(Dimension dimension);

        /** Get the D3D12 CPU descriptor handle representing this resource view.
            Valid only when D3D12 is the underlying API.
        */
        D3D12DescriptorCpuHandle getD3D12CpuHeapHandle() const;

    private:
        ShaderResourceView(ResourceWeakPtr pResource, ApiHandle handle, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
            : ResourceView(pResource, handle, mostDetailedMip, mipCount, firstArraySlice, arraySize) {}
        ShaderResourceView(ResourceWeakPtr pResource, ApiHandle handle, uint32_t firstElement, uint32_t elementCount)
            : ResourceView(pResource, handle, firstElement, elementCount) {}
        ShaderResourceView(ResourceWeakPtr pResource, ApiHandle handle)
            : ResourceView(pResource, handle) {}
    };

    class FALCOR_API DepthStencilView : public ResourceView<DsvHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<DepthStencilView>;
        using SharedConstPtr = std::shared_ptr<const DepthStencilView>;

        static SharedPtr create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize);
        static SharedPtr create(Dimension dimension);

        static SharedPtr getNullView(Dimension dimension);

        /** Get the D3D12 CPU descriptor handle representing this resource view.
            Valid only when D3D12 is the underlying API.
        */
        D3D12DescriptorCpuHandle getD3D12CpuHeapHandle() const;
    private:
        DepthStencilView(ResourceWeakPtr pResource, ApiHandle handle, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize) :
            ResourceView(pResource, handle, mipLevel, 1, firstArraySlice, arraySize) {}
    };

    class FALCOR_API UnorderedAccessView : public ResourceView<UavHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<UnorderedAccessView>;
        using SharedConstPtr = std::shared_ptr<const UnorderedAccessView>;

        static SharedPtr create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize);
        static SharedPtr create(ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount);
        static SharedPtr create(Dimension dimension);

        static SharedPtr getNullView(Dimension dimension);

        /** Get the D3D12 CPU descriptor handle representing this resource view.
            Valid only when D3D12 is the underlying API.
        */
        D3D12DescriptorCpuHandle getD3D12CpuHeapHandle() const;

    private:
        UnorderedAccessView(ResourceWeakPtr pResource, ApiHandle handle, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize) :
            ResourceView(pResource, handle, mipLevel, 1, firstArraySlice, arraySize) {}

        UnorderedAccessView(ResourceWeakPtr pResource, ApiHandle handle, uint32_t firstElement, uint32_t elementCount)
            : ResourceView(pResource, handle, firstElement, elementCount) {}
    };

    class FALCOR_API RenderTargetView : public ResourceView<RtvHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<RenderTargetView>;
        using SharedConstPtr = std::shared_ptr<const RenderTargetView>;
        static SharedPtr create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize);
        static SharedPtr create(Dimension dimension);

        static SharedPtr getNullView(Dimension dimension);

        /** Get the D3D12 CPU descriptor handle representing this resource view.
            Valid only when D3D12 is the underlying API.
        */
        D3D12DescriptorCpuHandle getD3D12CpuHeapHandle() const;

        ~RenderTargetView();

    private:
        RenderTargetView(ResourceWeakPtr pResource, ApiHandle handle, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize) :
            ResourceView(pResource, handle, mipLevel, 1, firstArraySlice, arraySize) {}
    };

    class FALCOR_API ConstantBufferView : public ResourceView<CbvHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<ConstantBufferView>;
        using SharedConstPtr = std::shared_ptr<const ConstantBufferView>;
        static SharedPtr create(ConstBufferSharedPtrRef pBuffer);
        static SharedPtr create();

        static SharedPtr getNullView();

        /** Get the D3D12 CPU descriptor handle representing this resource view.
            Valid only when D3D12 is the underlying API.
        */
        D3D12DescriptorCpuHandle getD3D12CpuHeapHandle() const;

    private:
        ConstantBufferView(ResourceWeakPtr pResource, ApiHandle handle) :
            ResourceView(pResource, handle, 0, 1, 0, 1) {}
    };
}
