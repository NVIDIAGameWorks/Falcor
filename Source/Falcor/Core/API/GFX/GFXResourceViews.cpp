/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "GFXFormats.h"
#include "Core/API/ResourceViews.h"
#include "Core/API/Texture.h"
#include "Core/API/Buffer.h"
#include "Core/API/Device.h"
namespace Falcor
{
    template<typename T>
    ResourceView<T>::~ResourceView() = default;

    ShaderResourceView::SharedPtr ShaderResourceView::create(ConstTextureSharedPtrRef pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
    {
        Slang::ComPtr<gfx::IResourceView> handle;
        gfx::IResourceView::Desc desc = {};
        desc.format = getGFXFormat(depthToColorFormat(pTexture->getFormat()));
        desc.type = gfx::IResourceView::Type::ShaderResource;
        desc.subresourceRange.baseArrayLayer = firstArraySlice;
        desc.subresourceRange.layerCount = arraySize;
        desc.subresourceRange.mipLevel = mostDetailedMip;
        desc.subresourceRange.mipLevelCount = mipCount;
        gfx_call(gpDevice->getApiHandle()->createTextureView(static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get()), desc, handle.writeRef()));
        return SharedPtr(new ShaderResourceView(pTexture, handle, mostDetailedMip, mipCount, firstArraySlice, arraySize));
    }

    ShaderResourceView::SharedPtr ShaderResourceView::create(ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        Slang::ComPtr<gfx::IResourceView> handle;
        gfx::IResourceView::Desc desc = {};
        auto format = depthToColorFormat(pBuffer->getFormat());
        desc.format = getGFXFormat(format);
        desc.type = gfx::IResourceView::Type::ShaderResource;

        uint32_t bufferElementSize = 0;
        uint32_t bufferElementCount = 0;
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
            bufferElementSize = 0;
            bufferElementCount = (uint32_t)(pBuffer->getSize());
        }

        bool useDefaultCount = (elementCount == ShaderResourceView::kMaxPossible);
        FALCOR_ASSERT(useDefaultCount || (firstElement + elementCount) <= bufferElementCount); // Check range
        desc.bufferRange.firstElement = firstElement;
        desc.bufferRange.elementCount = useDefaultCount ? (bufferElementCount - firstElement) : elementCount;

        // Views that extend to close to 4GB or beyond the base address are not supported by D3D12, so
        // we report an error here.
        if (bufferElementSize > 0 && desc.bufferRange.firstElement + desc.bufferRange.elementCount > ((1ull << 32) / bufferElementSize - 8))
        {
            throw RuntimeError("Buffer SRV exceeds the maximum supported size");
        }

        gfx_call(gpDevice->getApiHandle()->createBufferView(static_cast<gfx::IBufferResource*>(pBuffer->getApiHandle().get()), desc, handle.writeRef()));
        return SharedPtr(new ShaderResourceView(pBuffer, handle, firstElement, elementCount));
    }

    ShaderResourceView::SharedPtr ShaderResourceView::create(Dimension dimension)
    {
        // Create a null view of the specified dimension.
        return SharedPtr(new ShaderResourceView(std::weak_ptr<Resource>(), nullptr, 0, 0));
    }

    DepthStencilView::SharedPtr DepthStencilView::create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        auto gfxTexture = static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get());

        Slang::ComPtr<gfx::IResourceView> handle;
        gfx::IResourceView::Desc desc = {};
        desc.format = getGFXFormat(pTexture->getFormat());
        desc.type = gfx::IResourceView::Type::DepthStencil;
        desc.subresourceRange.baseArrayLayer = firstArraySlice;
        desc.subresourceRange.layerCount = arraySize;
        desc.subresourceRange.mipLevel = mipLevel;
        desc.subresourceRange.mipLevelCount = 1;
        desc.renderTarget.shape = gfxTexture->getDesc()->type;
        desc.renderTarget.arrayIndex = firstArraySlice;
        desc.renderTarget.arraySize = arraySize;
        desc.renderTarget.mipSlice = mipLevel;
        desc.renderTarget.planeIndex = 0;
        gfx_call(gpDevice->getApiHandle()->createTextureView(static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get()), desc, handle.writeRef()));
        return SharedPtr(new DepthStencilView(pTexture, handle, mipLevel, firstArraySlice, arraySize));
    }

    DepthStencilView::SharedPtr DepthStencilView::create(Dimension dimension)
    {
        return SharedPtr(new DepthStencilView(std::weak_ptr<Resource>(), nullptr, 0, 0, 0));
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        Slang::ComPtr<gfx::IResourceView> handle;
        gfx::IResourceView::Desc desc = {};
        desc.format = getGFXFormat(pTexture->getFormat());
        desc.type = gfx::IResourceView::Type::UnorderedAccess;
        desc.subresourceRange.baseArrayLayer = firstArraySlice;
        desc.subresourceRange.layerCount = arraySize;
        desc.subresourceRange.mipLevel = mipLevel;
        desc.subresourceRange.mipLevelCount = 1;
        gfx_call(gpDevice->getApiHandle()->createTextureView(static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get()), desc, handle.writeRef()));
        return SharedPtr(new UnorderedAccessView(pTexture, handle, mipLevel, firstArraySlice, arraySize));
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::create(ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        Slang::ComPtr<gfx::IResourceView> handle;
        gfx::IResourceView::Desc desc = {};
        desc.format = getGFXFormat(pBuffer->getFormat());
        desc.type = gfx::IResourceView::Type::UnorderedAccess;
        desc.bufferRange.firstElement = firstElement;

        bool useDefaultCount = (elementCount == UnorderedAccessView::kMaxPossible);
        desc.bufferRange.elementCount = useDefaultCount ? pBuffer->getElementCount() - firstElement : elementCount;
        gfx_call(gpDevice->getApiHandle()->createBufferView(static_cast<gfx::IBufferResource*>(pBuffer->getApiHandle().get()), desc, handle.writeRef()));
        return SharedPtr(new UnorderedAccessView(pBuffer, handle, firstElement, elementCount));
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::create(Dimension dimension)
    {
        return SharedPtr(new UnorderedAccessView(std::weak_ptr<Resource>(), nullptr, 0, 0));
    }

    RenderTargetView::~RenderTargetView() = default;

    RenderTargetView::SharedPtr RenderTargetView::create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        auto gfxTexture = static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get());
        Slang::ComPtr<gfx::IResourceView> handle;
        gfx::IResourceView::Desc desc = {};
        desc.format = getGFXFormat(pTexture->getFormat());
        desc.type = gfx::IResourceView::Type::RenderTarget;
        desc.renderTarget.arrayIndex = firstArraySlice;
        desc.renderTarget.arraySize = arraySize;
        desc.renderTarget.mipSlice = mipLevel;
        desc.renderTarget.planeIndex = 0;
        desc.renderTarget.shape = gfxTexture->getDesc()->type;
        gfx_call(gpDevice->getApiHandle()->createTextureView(gfxTexture, desc, handle.writeRef()));
        return SharedPtr(new RenderTargetView(pTexture, handle, mipLevel, firstArraySlice, arraySize));
    }

    RenderTargetView::SharedPtr RenderTargetView::create(Dimension dimension)
    {
        return SharedPtr(new RenderTargetView(std::weak_ptr<Resource>(), nullptr, 0, 0, 0));
    }

    ConstantBufferView::SharedPtr ConstantBufferView::create(ConstBufferSharedPtrRef pBuffer)
    {
        // TODO: GFX doesn't support constant buffer view. Consider remove this from public interface.
        FALCOR_ASSERT(pBuffer);
        return nullptr;
    }

    ConstantBufferView::SharedPtr ConstantBufferView::create()
    {
        // TODO: GFX doesn't support constant buffer view. Consider remove this from public interface.
        return nullptr;
    }

    using ResourceViewImpl = ResourceView<Slang::ComPtr<gfx::IResourceView>>;
    template ResourceSharedPtr ResourceViewImpl::getResource() const;
    template const ResourceViewImpl::ApiHandle& ResourceViewImpl::getApiHandle() const;
    template const ResourceViewInfo& ResourceViewImpl::getViewInfo() const;
#if FALCOR_ENABLE_CUDA
    template void* ResourceViewImpl::getCUDADeviceAddress() const;
#endif
}
