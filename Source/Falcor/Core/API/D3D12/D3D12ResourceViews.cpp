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
#include "Core/API/ResourceViews.h"
#include "Core/API/Texture.h"
#include "Core/API/Buffer.h"
#include "Core/API/Device.h"

namespace Falcor
{
    template<typename T>
    ResourceView<T>::~ResourceView() = default;

    template<typename ViewType>
    ViewType getViewDimension(Resource::Type type, bool isArray);
    
    Texture::SharedPtr getEmptyTexture()
    {
        return Texture::SharedPtr();
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC createBufferSrvDesc(const Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        assert(pBuffer);
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};

        uint32_t bufferElementCount = ShaderResourceView::kMaxPossible;
        if (pBuffer->isTyped())
        {
            bufferElementCount = pBuffer->getElementCount();
            desc.Format = getDxgiFormat(pBuffer->getFormat());
        }
        else if (pBuffer->isStructured())
        {
            bufferElementCount = pBuffer->getElementCount();
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Buffer.StructureByteStride = pBuffer->getStructSize();
        }
        else
        {
            bufferElementCount = (uint32_t)pBuffer->getSize() / sizeof(float);
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        }

        bool useDefaultCount = (elementCount == ShaderResourceView::kMaxPossible);
        assert(useDefaultCount || (firstElement + elementCount) <= bufferElementCount); // Check range
        desc.Buffer.FirstElement = firstElement;
        desc.Buffer.NumElements = useDefaultCount ? (bufferElementCount - firstElement) : elementCount;

        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;

        return desc;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC createTextureSrvDesc(const Texture* pTexture, uint32_t firstArraySlice, uint32_t arraySize, uint32_t mostDetailedMip, uint32_t mipCount)
    {
        assert(pTexture);
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};

        //If not depth, returns input format
        ResourceFormat colorFormat = depthToColorFormat(pTexture->getFormat());
        desc.Format = getDxgiFormat(colorFormat);

        bool isTextureArray = pTexture->getArraySize() > 1;
        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(pTexture->getType(), isTextureArray);

        switch (pTexture->getType())
        {
        case Resource::Type::Texture1D:
            if (isTextureArray)
            {
                desc.Texture1DArray.MipLevels = mipCount;
                desc.Texture1DArray.MostDetailedMip = mostDetailedMip;
                desc.Texture1DArray.ArraySize = arraySize;
                desc.Texture1DArray.FirstArraySlice = firstArraySlice;
            }
            else
            {
                desc.Texture1D.MipLevels = mipCount;
                desc.Texture1D.MostDetailedMip = mostDetailedMip;
            }
            break;
        case Resource::Type::Texture2D:
            if (isTextureArray)
            {
                desc.Texture2DArray.MipLevels = mipCount;
                desc.Texture2DArray.MostDetailedMip = mostDetailedMip;
                desc.Texture2DArray.ArraySize = arraySize;
                desc.Texture2DArray.FirstArraySlice = firstArraySlice;
            }
            else
            {
                desc.Texture2D.MipLevels = mipCount;
                desc.Texture2D.MostDetailedMip = mostDetailedMip;
            }
            break;
        case Resource::Type::Texture2DMultisample:
            if (arraySize > 1)
            {
                desc.Texture2DMSArray.ArraySize = arraySize;
                desc.Texture2DMSArray.FirstArraySlice = firstArraySlice;
            }
            break;
        case Resource::Type::Texture3D:
            assert(arraySize == 1);
            desc.Texture3D.MipLevels = mipCount;
            desc.Texture3D.MostDetailedMip = mostDetailedMip;
            break;
        case Resource::Type::TextureCube:
            if (arraySize > 1)
            {
                desc.TextureCubeArray.First2DArrayFace = 0;
                desc.TextureCubeArray.NumCubes = arraySize;
                desc.TextureCubeArray.MipLevels = mipCount;
                desc.TextureCubeArray.MostDetailedMip = mostDetailedMip;
            }
            else
            {
                desc.TextureCube.MipLevels = mipCount;
                desc.TextureCube.MostDetailedMip = mostDetailedMip;
            }
            break;
        default:
            should_not_get_here();
        }

        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        return desc;
    }

    template<typename DescType, bool finalCall>
    DescType createDsvRtvUavDescCommon(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        DescType desc = {};
        const Texture* pTexture = dynamic_cast<const Texture*>(pResource);
        assert(pTexture);   // Buffers should not get here

        desc = {};
        uint32_t arrayMultiplier = (pResource->getType() == Resource::Type::TextureCube) ? 6 : 1;

        if (arraySize == Resource::kMaxPossible)
        {
            arraySize = pTexture->getArraySize() - firstArraySlice;
        }

        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(pTexture->getType(), pTexture->getArraySize() > 1);

        switch (pResource->getType())
        {
        case Resource::Type::Texture1D:
            if (pTexture->getArraySize() > 1)
            {
                desc.Texture1DArray.ArraySize = arraySize;
                desc.Texture1DArray.FirstArraySlice = firstArraySlice;
                desc.Texture1DArray.MipSlice = mipLevel;
            }
            else
            {
                desc.Texture1D.MipSlice = mipLevel;
            }
            break;
        case Resource::Type::Texture2D:
        case Resource::Type::TextureCube:
            if (pTexture->getArraySize() * arrayMultiplier > 1)
            {
                desc.Texture2DArray.ArraySize = arraySize * arrayMultiplier;
                desc.Texture2DArray.FirstArraySlice = firstArraySlice * arrayMultiplier;
                desc.Texture2DArray.MipSlice = mipLevel;
            }
            else
            {
                desc.Texture2D.MipSlice = mipLevel;
            }
            break;
        default:
            if (finalCall) should_not_get_here();
        }
        desc.Format = getDxgiFormat(pTexture->getFormat());

        return desc;
    }

    template<typename DescType>
    DescType createDsvRtvDesc(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        DescType desc = createDsvRtvUavDescCommon<DescType, false>(pResource, mipLevel, firstArraySlice, arraySize);

        if (pResource->getType() == Resource::Type::Texture2DMultisample)
        {
            const Texture* pTexture = dynamic_cast<const Texture*>(pResource);
            if (pTexture->getArraySize() > 1)
            {
                desc.Texture2DMSArray.ArraySize = arraySize;
                desc.Texture2DMSArray.FirstArraySlice = firstArraySlice;
            }
        }

        return desc;
    }

    D3D12_DEPTH_STENCIL_VIEW_DESC createDsvDesc(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        return createDsvRtvDesc<D3D12_DEPTH_STENCIL_VIEW_DESC>(pResource, mipLevel, firstArraySlice, arraySize);
    }

    D3D12_RENDER_TARGET_VIEW_DESC createRtvDesc(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        return createDsvRtvDesc<D3D12_RENDER_TARGET_VIEW_DESC>(pResource, mipLevel, firstArraySlice, arraySize);
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC createBufferUavDesc(const Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        assert(pBuffer);
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};

        desc = {};
        uint32_t bufferElementCount = UnorderedAccessView::kMaxPossible;
        if (pBuffer->isTyped())
        {
            bufferElementCount = pBuffer->getElementCount();
            desc.Format = getDxgiFormat(pBuffer->getFormat());
        }
        else if (pBuffer->isStructured())
        {
            bufferElementCount = pBuffer->getElementCount();
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Buffer.StructureByteStride = pBuffer->getStructSize();
        }
        else
        {
            bufferElementCount = ((uint32_t)pBuffer->getSize() / sizeof(float));
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        }

        bool useDefaultCount = (elementCount == UnorderedAccessView::kMaxPossible);
        assert(useDefaultCount || (firstElement + elementCount) <= bufferElementCount); // Check range
        desc.Buffer.FirstElement = firstElement;
        desc.Buffer.NumElements = useDefaultCount ? bufferElementCount - firstElement : elementCount;

        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;

        return desc;
    }

    ShaderResourceView::ApiHandle createSrvDescriptor(const D3D12_SHADER_RESOURCE_VIEW_DESC& desc, Resource::ApiHandle resHandle)
    {
        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::TextureSrv, 0, 1);
        ShaderResourceView::ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateShaderResourceView(resHandle, &desc, handle->getCpuHandle(0));

        return handle;
    }

    ShaderResourceView::SharedPtr ShaderResourceView::create(ConstTextureSharedPtrRef pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
    {
        if (!pTexture && getNullView()) return getNullView();

        D3D12_SHADER_RESOURCE_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        if(pTexture)
        {
            desc = createTextureSrvDesc(pTexture.get(), firstArraySlice, arraySize, mostDetailedMip, mipCount);
            resHandle = pTexture->getApiHandle();
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        }

        SharedPtr pObj = SharedPtr(new ShaderResourceView(pTexture, createSrvDescriptor(desc, resHandle), mostDetailedMip, mipCount, firstArraySlice, arraySize));
        return pObj;
    }

    ShaderResourceView::SharedPtr ShaderResourceView::create(ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        if (!pBuffer && getNullView()) return getNullView();

        D3D12_SHADER_RESOURCE_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        if (pBuffer)
        {
            desc = createBufferSrvDesc(pBuffer.get(), firstElement, elementCount);
            resHandle = pBuffer->getApiHandle();
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        }

        SharedPtr pObj = SharedPtr(new ShaderResourceView(pBuffer, createSrvDescriptor(desc, resHandle), firstElement, elementCount));
        return pObj;
    }

    DepthStencilView::SharedPtr DepthStencilView::create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        if (!pTexture && getNullView()) return getNullView();

        D3D12_DEPTH_STENCIL_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        if(pTexture)
        {
            desc = createDsvDesc(pTexture.get(), mipLevel, firstArraySlice, arraySize);
            resHandle = pTexture->getApiHandle();
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_D16_UNORM;
            desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        }

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::Dsv, 0, 1);
        ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateDepthStencilView(resHandle, &desc, handle->getCpuHandle(0));

        return SharedPtr(new DepthStencilView(pTexture, handle, mipLevel, firstArraySlice, arraySize));
    }

    UnorderedAccessView::ApiHandle createUavDescriptor(const D3D12_UNORDERED_ACCESS_VIEW_DESC& desc, Resource::ApiHandle resHandle, Resource::ApiHandle counterHandle)
    {
        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::TextureUav, 0, 1);
        UnorderedAccessView::ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateUnorderedAccessView(resHandle, counterHandle, &desc, handle->getCpuHandle(0));
        return handle;
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        if (!pTexture && getNullView()) return getNullView();

        D3D12_UNORDERED_ACCESS_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;

        if(pTexture != nullptr)
        {
            desc = createDsvRtvUavDescCommon<D3D12_UNORDERED_ACCESS_VIEW_DESC, true>(pTexture.get(), mipLevel, firstArraySlice, arraySize);
            resHandle = pTexture->getApiHandle();
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_R32_UINT;
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        }

        return SharedPtr(new UnorderedAccessView(pTexture, createUavDescriptor(desc, resHandle, nullptr), mipLevel, firstArraySlice, arraySize));
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::create(ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        if (!pBuffer && getNullView()) return getNullView();

        D3D12_UNORDERED_ACCESS_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        Resource::ApiHandle counterHandle = nullptr;

        if (pBuffer != nullptr)
        {
            desc = createBufferUavDesc(pBuffer.get(), firstElement, elementCount);
            resHandle = pBuffer->getApiHandle();

            if (pBuffer->getUAVCounter())
            {
                counterHandle = pBuffer->getUAVCounter()->getApiHandle();
            }
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_R32_UINT;
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        }

        return SharedPtr(new UnorderedAccessView(pBuffer, createUavDescriptor(desc, resHandle, counterHandle), firstElement, elementCount));
    }

    RenderTargetView::~RenderTargetView() = default;

    RenderTargetView::SharedPtr RenderTargetView::create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        if (!pTexture && getNullView()) return getNullView();

        D3D12_RENDER_TARGET_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        if(pTexture)
        {
            desc = createRtvDesc(pTexture.get(), mipLevel, firstArraySlice, arraySize);
            resHandle = pTexture->getApiHandle();
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;;
            desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        }

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::Rtv, 0, 1);
        ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateRenderTargetView(resHandle, &desc, handle->getCpuHandle(0));

        SharedPtr pObj = SharedPtr(new RenderTargetView(pTexture, handle, mipLevel, firstArraySlice, arraySize));
        return pObj;
    }

    ConstantBufferView::SharedPtr ConstantBufferView::create(ConstBufferSharedPtrRef pBuffer)
    {
        if (!pBuffer && getNullView()) return getNullView();

        D3D12_CONSTANT_BUFFER_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        if (pBuffer)
        {
            desc.BufferLocation = pBuffer->getGpuAddress();
            desc.SizeInBytes = (uint32_t)pBuffer->getSize();
            resHandle = pBuffer->getApiHandle();
        }
        else
        {
            desc = {};
        }

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::Cbv, 0, 1);
        ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateConstantBufferView(&desc, handle->getCpuHandle(0));

        SharedPtr pObj = SharedPtr(new ConstantBufferView(pBuffer, handle));
        return pObj;
    }
}
