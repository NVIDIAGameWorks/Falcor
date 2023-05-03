/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "fwd.h"
#include "Handles.h"
#include "NativeHandle.h"
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
        : mostDetailedMip(mostDetailedMip), mipCount(mipCount), firstArraySlice(firstArraySlice), arraySize(arraySize)
    {}

    ResourceViewInfo(uint32_t firstElement, uint32_t elementCount) : firstElement(firstElement), elementCount(elementCount) {}

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
        return (firstArraySlice == other.firstArraySlice) && (arraySize == other.arraySize) && (mipCount == other.mipCount) &&
               (mostDetailedMip == other.mostDetailedMip) && (firstElement == other.firstElement) && (elementCount == other.elementCount);
    }
};

/**
 * Abstracts API resource views.
 */
class FALCOR_API ResourceView
{
public:
    using SharedPtr = std::shared_ptr<ResourceView>;

    using Dimension = ReflectionResourceType::Dimensions;
    static const uint32_t kMaxPossible = -1;
    virtual ~ResourceView();

    ResourceView(
        std::shared_ptr<Device> pDevice,
        ResourceWeakPtr& pResource,
        Slang::ComPtr<gfx::IResourceView> gfxResourceView,
        uint32_t mostDetailedMip,
        uint32_t mipCount,
        uint32_t firstArraySlice,
        uint32_t arraySize
    )
        : mpDevice(std::move(pDevice))
        , mGfxResourceView(gfxResourceView)
        , mViewInfo(mostDetailedMip, mipCount, firstArraySlice, arraySize)
        , mpResource(pResource)
    {}

    ResourceView(
        std::shared_ptr<Device> pDevice,
        ResourceWeakPtr& pResource,
        Slang::ComPtr<gfx::IResourceView> gfxResourceView,
        uint32_t firstElement,
        uint32_t elementCount
    )
        : mpDevice(std::move(pDevice)), mGfxResourceView(gfxResourceView), mViewInfo(firstElement, elementCount), mpResource(pResource)
    {}

    ResourceView(std::shared_ptr<Device> pDevice, ResourceWeakPtr& pResource, Slang::ComPtr<gfx::IResourceView> gfxResourceView)
        : mpDevice(std::move(pDevice)), mGfxResourceView(gfxResourceView), mpResource(pResource)
    {}

    gfx::IResourceView* getGfxResourceView() const { return mGfxResourceView; }

    /**
     * Returns the native API handle:
     * - D3D12: D3D12_CPU_DESCRIPTOR_HANDLE
     * - Vulkan: VkImageView for texture views, VkBufferView for typed buffer views, VkBuffer for untyped buffer views
     */
    NativeHandle getNativeHandle() const;

    /**
     * Get information about the view.
     */
    const ResourceViewInfo& getViewInfo() const { return mViewInfo; }

    /**
     * Get the resource referenced by the view.
     */
    ResourceSharedPtr getResource() const { return mpResource.lock(); }

protected:
    std::shared_ptr<Device> mpDevice;
    Slang::ComPtr<gfx::IResourceView> mGfxResourceView;
    ResourceViewInfo mViewInfo;
    ResourceWeakPtr mpResource;
};

class FALCOR_API ShaderResourceView : public ResourceView
{
public:
    using SharedPtr = std::shared_ptr<ShaderResourceView>;

    static SharedPtr create(
        Device* pDevice,
        ConstTextureSharedPtrRef pTexture,
        uint32_t mostDetailedMip,
        uint32_t mipCount,
        uint32_t firstArraySlice,
        uint32_t arraySize
    );
    static SharedPtr create(Device* pDevice, ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount);
    static SharedPtr create(Device* pDevice, Dimension dimension);

private:
    ShaderResourceView(
        std::shared_ptr<Device> pDevice,
        ResourceWeakPtr pResource,
        Slang::ComPtr<gfx::IResourceView> gfxResourceView,
        uint32_t mostDetailedMip,
        uint32_t mipCount,
        uint32_t firstArraySlice,
        uint32_t arraySize
    )
        : ResourceView(pDevice, pResource, gfxResourceView, mostDetailedMip, mipCount, firstArraySlice, arraySize)
    {}
    ShaderResourceView(
        std::shared_ptr<Device> pDevice,
        ResourceWeakPtr pResource,
        Slang::ComPtr<gfx::IResourceView> gfxResourceView,
        uint32_t firstElement,
        uint32_t elementCount
    )
        : ResourceView(pDevice, pResource, gfxResourceView, firstElement, elementCount)
    {}
    ShaderResourceView(std::shared_ptr<Device> pDevice, ResourceWeakPtr pResource, Slang::ComPtr<gfx::IResourceView> gfxResourceView)
        : ResourceView(pDevice, pResource, gfxResourceView)
    {}
};

class FALCOR_API DepthStencilView : public ResourceView
{
public:
    using SharedPtr = std::shared_ptr<DepthStencilView>;

    static SharedPtr create(
        Device* pDevice,
        ConstTextureSharedPtrRef pTexture,
        uint32_t mipLevel,
        uint32_t firstArraySlice,
        uint32_t arraySize
    );
    static SharedPtr create(Device* pDevice, Dimension dimension);

private:
    DepthStencilView(
        std::shared_ptr<Device> pDevice,
        ResourceWeakPtr pResource,
        Slang::ComPtr<gfx::IResourceView> gfxResourceView,
        uint32_t mipLevel,
        uint32_t firstArraySlice,
        uint32_t arraySize
    )
        : ResourceView(pDevice, pResource, gfxResourceView, mipLevel, 1, firstArraySlice, arraySize)
    {}
};

class FALCOR_API UnorderedAccessView : public ResourceView
{
public:
    using SharedPtr = std::shared_ptr<UnorderedAccessView>;

    static SharedPtr create(
        Device* pDevice,
        ConstTextureSharedPtrRef pTexture,
        uint32_t mipLevel,
        uint32_t firstArraySlice,
        uint32_t arraySize
    );
    static SharedPtr create(Device* pDevice, ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount);
    static SharedPtr create(Device* pDevice, Dimension dimension);

private:
    UnorderedAccessView(
        std::shared_ptr<Device> pDevice,
        ResourceWeakPtr pResource,
        Slang::ComPtr<gfx::IResourceView> gfxResourceView,
        uint32_t mipLevel,
        uint32_t firstArraySlice,
        uint32_t arraySize
    )
        : ResourceView(pDevice, pResource, gfxResourceView, mipLevel, 1, firstArraySlice, arraySize)
    {}

    UnorderedAccessView(
        std::shared_ptr<Device> pDevice,
        ResourceWeakPtr pResource,
        Slang::ComPtr<gfx::IResourceView> gfxResourceView,
        uint32_t firstElement,
        uint32_t elementCount
    )
        : ResourceView(pDevice, pResource, gfxResourceView, firstElement, elementCount)
    {}
};

class FALCOR_API RenderTargetView : public ResourceView
{
public:
    using SharedPtr = std::shared_ptr<RenderTargetView>;

    static SharedPtr create(
        Device* pDevice,
        ConstTextureSharedPtrRef pTexture,
        uint32_t mipLevel,
        uint32_t firstArraySlice,
        uint32_t arraySize
    );
    static SharedPtr create(Device* pDevice, Dimension dimension);

private:
    RenderTargetView(
        std::shared_ptr<Device> pDevice,
        ResourceWeakPtr pResource,
        Slang::ComPtr<gfx::IResourceView> gfxResourceView,
        uint32_t mipLevel,
        uint32_t firstArraySlice,
        uint32_t arraySize
    )
        : ResourceView(pDevice, pResource, gfxResourceView, mipLevel, 1, firstArraySlice, arraySize)
    {}
};
} // namespace Falcor
