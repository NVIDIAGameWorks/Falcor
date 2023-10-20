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
#include "ResourceViews.h"
#include "Texture.h"
#include "Buffer.h"
#include "Device.h"
#include "GFXHelpers.h"
#include "GFXAPI.h"
#include "NativeHandleTraits.h"
#include "Core/ObjectPython.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{

NativeHandle ResourceView::getNativeHandle() const
{
    FALCOR_ASSERT(mpDevice != nullptr && mpResource != nullptr);
    gfx::InteropHandle gfxNativeHandle = {};
    FALCOR_GFX_CALL(mGfxResourceView->getNativeHandle(&gfxNativeHandle));
#if FALCOR_HAS_D3D12
    if (mpDevice->getType() == Device::Type::D3D12)
        return NativeHandle(D3D12_CPU_DESCRIPTOR_HANDLE{gfxNativeHandle.handleValue});
#endif
#if FALCOR_HAS_VULKAN
    if (mpDevice->getType() == Device::Type::Vulkan)
    {
        if (mpResource)
        {
            if (mpResource->getType() == Resource::Type::Buffer)
            {
                if (mGfxResourceView->getViewDesc()->format == gfx::Format::Unknown)
                    return NativeHandle(reinterpret_cast<VkBuffer>(gfxNativeHandle.handleValue));
                else
                    return NativeHandle(reinterpret_cast<VkBufferView>(gfxNativeHandle.handleValue));
            }
            else
            {
                return NativeHandle(reinterpret_cast<VkImageView>(gfxNativeHandle.handleValue));
            }
        }
    }
#endif
    return {};
}

ResourceView::~ResourceView()
{
    if (mGfxResourceView)
        mpDevice->releaseResource(mGfxResourceView);
}

void ResourceView::invalidate()
{
    if (mpDevice)
    {
        mpDevice->releaseResource(mGfxResourceView);
        mGfxResourceView = nullptr;
        mpResource = nullptr;
        mpDevice = nullptr;
    }
}

ref<ShaderResourceView> ShaderResourceView::create(
    Device* pDevice,
    Texture* pTexture,
    uint32_t mostDetailedMip,
    uint32_t mipCount,
    uint32_t firstArraySlice,
    uint32_t arraySize
)
{
    FALCOR_CHECK(is_set(pTexture->getBindFlags(), ResourceBindFlags::ShaderResource), "Texture does not have SRV bind flag set.");
    Slang::ComPtr<gfx::IResourceView> handle;
    gfx::IResourceView::Desc desc = {};
    desc.format = getGFXFormat(depthToColorFormat(pTexture->getFormat()));
    desc.type = gfx::IResourceView::Type::ShaderResource;
    desc.subresourceRange.baseArrayLayer = firstArraySlice;
    desc.subresourceRange.layerCount = arraySize;
    desc.subresourceRange.mipLevel = mostDetailedMip;
    desc.subresourceRange.mipLevelCount = mipCount;
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createTextureView(pTexture->getGfxTextureResource(), desc, handle.writeRef()));
    return ref<ShaderResourceView>(new ShaderResourceView(pDevice, pTexture, handle, mostDetailedMip, mipCount, firstArraySlice, arraySize)
    );
}

static void fillBufferViewDesc(gfx::IResourceView::Desc& desc, Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
{
    auto format = depthToColorFormat(pBuffer->getFormat());
    desc.format = getGFXFormat(format);

    uint32_t bufferElementSize = 0;
    uint64_t bufferElementCount = 0;
    if (pBuffer->isTyped())
    {
        FALCOR_ASSERT(getFormatPixelsPerBlock(format) == 1);
        bufferElementSize = getFormatBytesPerBlock(format);
        bufferElementCount = pBuffer->getElementCount();
    }
    else if (pBuffer->isStructured())
    {
        bufferElementSize = pBuffer->getStructSize();
        bufferElementCount = pBuffer->getElementCount();
        desc.format = gfx::Format::Unknown;
        desc.bufferElementSize = bufferElementSize;
    }
    else
    {
        desc.format = gfx::Format::Unknown;
        bufferElementSize = 4;
        bufferElementCount = pBuffer->getSize();
    }

    bool useDefaultCount = (elementCount == ShaderResourceView::kMaxPossible);
    FALCOR_ASSERT(useDefaultCount || (firstElement + elementCount) <= bufferElementCount); // Check range
    desc.bufferRange.firstElement = firstElement;
    desc.bufferRange.elementCount = useDefaultCount ? (bufferElementCount - firstElement) : elementCount;
}

ref<ShaderResourceView> ShaderResourceView::create(Device* pDevice, Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
{
    Slang::ComPtr<gfx::IResourceView> handle;
    gfx::IResourceView::Desc desc = {};
    desc.type = gfx::IResourceView::Type::ShaderResource;
    fillBufferViewDesc(desc, pBuffer, firstElement, elementCount);

    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createBufferView(pBuffer->getGfxBufferResource(), nullptr, desc, handle.writeRef()));
    return ref<ShaderResourceView>(new ShaderResourceView(pDevice, pBuffer, handle, firstElement, elementCount));
}

ref<ShaderResourceView> ShaderResourceView::create(Device* pDevice, Dimension dimension)
{
    // Create a null view of the specified dimension.
    return ref<ShaderResourceView>(new ShaderResourceView(pDevice, nullptr, nullptr, 0, 0));
}

ref<DepthStencilView> DepthStencilView::create(
    Device* pDevice,
    Texture* pTexture,
    uint32_t mipLevel,
    uint32_t firstArraySlice,
    uint32_t arraySize
)
{
    FALCOR_CHECK(is_set(pTexture->getBindFlags(), ResourceBindFlags::DepthStencil), "Texture does not have DSV bind flag set.");
    Slang::ComPtr<gfx::IResourceView> handle;
    gfx::IResourceView::Desc desc = {};
    desc.format = getGFXFormat(pTexture->getFormat());
    desc.type = gfx::IResourceView::Type::DepthStencil;
    desc.subresourceRange.baseArrayLayer = firstArraySlice;
    desc.subresourceRange.layerCount = arraySize;
    desc.subresourceRange.mipLevel = mipLevel;
    desc.subresourceRange.mipLevelCount = 1;
    desc.subresourceRange.aspectMask = gfx::TextureAspect::Depth;
    desc.renderTarget.shape = pTexture->getGfxTextureResource()->getDesc()->type;
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createTextureView(pTexture->getGfxTextureResource(), desc, handle.writeRef()));
    return ref<DepthStencilView>(new DepthStencilView(pDevice, pTexture, handle, mipLevel, firstArraySlice, arraySize));
}

ref<DepthStencilView> DepthStencilView::create(Device* pDevice, Dimension dimension)
{
    return ref<DepthStencilView>(new DepthStencilView(pDevice, nullptr, nullptr, 0, 0, 0));
}

ref<UnorderedAccessView> UnorderedAccessView::create(
    Device* pDevice,
    Texture* pTexture,
    uint32_t mipLevel,
    uint32_t firstArraySlice,
    uint32_t arraySize
)
{
    FALCOR_CHECK(is_set(pTexture->getBindFlags(), ResourceBindFlags::UnorderedAccess), "Texture does not have UAV bind flag set.");
    Slang::ComPtr<gfx::IResourceView> handle;
    gfx::IResourceView::Desc desc = {};
    desc.format = getGFXFormat(pTexture->getFormat());
    desc.type = gfx::IResourceView::Type::UnorderedAccess;
    desc.subresourceRange.baseArrayLayer = firstArraySlice;
    desc.subresourceRange.layerCount = arraySize;
    desc.subresourceRange.mipLevel = mipLevel;
    desc.subresourceRange.mipLevelCount = 1;
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createTextureView(pTexture->getGfxTextureResource(), desc, handle.writeRef()));
    return ref<UnorderedAccessView>(new UnorderedAccessView(pDevice, pTexture, handle, mipLevel, firstArraySlice, arraySize));
}

ref<UnorderedAccessView> UnorderedAccessView::create(Device* pDevice, Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
{
    Slang::ComPtr<gfx::IResourceView> handle;
    gfx::IResourceView::Desc desc = {};
    desc.type = gfx::IResourceView::Type::UnorderedAccess;
    fillBufferViewDesc(desc, pBuffer, firstElement, elementCount);
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createBufferView(
        pBuffer->getGfxBufferResource(),
        pBuffer->getUAVCounter() ? pBuffer->getUAVCounter()->getGfxBufferResource() : nullptr,
        desc,
        handle.writeRef()
    ));
    return ref<UnorderedAccessView>(new UnorderedAccessView(pDevice, pBuffer, handle, firstElement, elementCount));
}

ref<UnorderedAccessView> UnorderedAccessView::create(Device* pDevice, Dimension dimension)
{
    return ref<UnorderedAccessView>(new UnorderedAccessView(pDevice, nullptr, nullptr, 0, 0));
}

ref<RenderTargetView> RenderTargetView::create(
    Device* pDevice,
    Texture* pTexture,
    uint32_t mipLevel,
    uint32_t firstArraySlice,
    uint32_t arraySize
)
{
    FALCOR_CHECK(is_set(pTexture->getBindFlags(), ResourceBindFlags::RenderTarget), "Texture does not have RTV bind flag set.");
    Slang::ComPtr<gfx::IResourceView> handle;
    gfx::IResourceView::Desc desc = {};
    desc.format = getGFXFormat(pTexture->getFormat());
    desc.type = gfx::IResourceView::Type::RenderTarget;
    desc.subresourceRange.baseArrayLayer = firstArraySlice;
    desc.subresourceRange.layerCount = arraySize;
    desc.subresourceRange.mipLevel = mipLevel;
    desc.subresourceRange.mipLevelCount = 1;
    desc.subresourceRange.aspectMask = gfx::TextureAspect::Color;
    desc.renderTarget.shape = pTexture->getGfxTextureResource()->getDesc()->type;
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createTextureView(pTexture->getGfxTextureResource(), desc, handle.writeRef()));
    return ref<RenderTargetView>(new RenderTargetView(pDevice, pTexture, handle, mipLevel, firstArraySlice, arraySize));
}

gfx::IResource::Type getGFXResourceType(RenderTargetView::Dimension dim)
{
    switch (dim)
    {
    case RenderTargetView::Dimension::Buffer:
        return gfx::IResource::Type::Buffer;
    case RenderTargetView::Dimension::Texture1D:
    case RenderTargetView::Dimension::Texture1DArray:
        return gfx::IResource::Type::Texture1D;
    case RenderTargetView::Dimension::Texture2D:
    case RenderTargetView::Dimension::Texture2DMS:
    case RenderTargetView::Dimension::Texture2DMSArray:
    case RenderTargetView::Dimension::Texture2DArray:
        return gfx::IResource::Type::Texture2D;
    case RenderTargetView::Dimension::Texture3D:
        return gfx::IResource::Type::Texture3D;
    case RenderTargetView::Dimension::TextureCube:
    case RenderTargetView::Dimension::TextureCubeArray:
        return gfx::IResource::Type::TextureCube;
    default:
        FALCOR_UNREACHABLE();
        return gfx::IResource::Type::Texture2D;
    }
}

ref<RenderTargetView> RenderTargetView::create(Device* pDevice, Dimension dimension)
{
    Slang::ComPtr<gfx::IResourceView> handle;
    gfx::IResourceView::Desc desc = {};
    desc.format = gfx::Format::R8G8B8A8_UNORM;
    desc.type = gfx::IResourceView::Type::RenderTarget;
    desc.subresourceRange.baseArrayLayer = 0;
    desc.subresourceRange.layerCount = 1;
    desc.subresourceRange.mipLevel = 0;
    desc.subresourceRange.mipLevelCount = 1;
    desc.subresourceRange.aspectMask = gfx::TextureAspect::Color;
    desc.renderTarget.shape = getGFXResourceType(dimension);
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createTextureView(nullptr, desc, handle.writeRef()));
    return ref<RenderTargetView>(new RenderTargetView(pDevice, nullptr, handle, 0, 0, 0));
}

FALCOR_SCRIPT_BINDING(ResourceView)
{
    pybind11::class_<ShaderResourceView, ref<ShaderResourceView>>(m, "ShaderResourceView");
    pybind11::class_<RenderTargetView, ref<RenderTargetView>>(m, "RenderTargetView");
    pybind11::class_<UnorderedAccessView, ref<UnorderedAccessView>>(m, "UnorderedAccessView");
    pybind11::class_<DepthStencilView, ref<DepthStencilView>>(m, "DepthStencilView");
}
} // namespace Falcor
