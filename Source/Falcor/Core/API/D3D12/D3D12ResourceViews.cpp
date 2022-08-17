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
#include "Core/API/ResourceViews.h"
#include "Core/API/Texture.h"
#include "Core/API/Buffer.h"
#include "Core/API/Device.h"
#include "Core/API/D3D12/D3D12API.h"

namespace Falcor
{
    namespace
    {
        /** Translate resource type enum to Falcor view dimension.
        */
        ReflectionResourceType::Dimensions getDimension(Resource::Type type, bool isTextureArray)
        {
            switch (type)
            {
            case Resource::Type::Buffer:
                FALCOR_ASSERT(isTextureArray == false);
                return ReflectionResourceType::Dimensions::Buffer;
            case Resource::Type::Texture1D:
                return (isTextureArray) ? ReflectionResourceType::Dimensions::Texture1DArray : ReflectionResourceType::Dimensions::Texture1D;
            case Resource::Type::Texture2D:
                return (isTextureArray) ? ReflectionResourceType::Dimensions::Texture2DArray : ReflectionResourceType::Dimensions::Texture2D;
            case Resource::Type::Texture2DMultisample:
                return (isTextureArray) ? ReflectionResourceType::Dimensions::Texture2DMSArray : ReflectionResourceType::Dimensions::Texture2DMS;
            case Resource::Type::Texture3D:
                FALCOR_ASSERT(isTextureArray == false);
                return ReflectionResourceType::Dimensions::Texture3D;
            case Resource::Type::TextureCube:
                return (isTextureArray) ? ReflectionResourceType::Dimensions::TextureCubeArray : ReflectionResourceType::Dimensions::TextureCube;
            default:
                FALCOR_UNREACHABLE();
                return ReflectionResourceType::Dimensions::Unknown;
            }
        }

        /** Translate Falcor view dimension to D3D12 view dimension.
        */
        template<typename ViewType>
        ViewType getViewDimension(ReflectionResourceType::Dimensions dimension);

        template<>
        D3D12_SRV_DIMENSION getViewDimension<D3D12_SRV_DIMENSION>(ReflectionResourceType::Dimensions dimension)
        {
            switch (dimension)
            {
            case ReflectionResourceType::Dimensions::Buffer: return D3D12_SRV_DIMENSION_BUFFER;
            case ReflectionResourceType::Dimensions::Texture1D: return D3D12_SRV_DIMENSION_TEXTURE1D;
            case ReflectionResourceType::Dimensions::Texture1DArray: return D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            case ReflectionResourceType::Dimensions::Texture2D: return D3D12_SRV_DIMENSION_TEXTURE2D;
            case ReflectionResourceType::Dimensions::Texture2DArray: return D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            case ReflectionResourceType::Dimensions::Texture2DMS: return D3D12_SRV_DIMENSION_TEXTURE2DMS;
            case ReflectionResourceType::Dimensions::Texture2DMSArray: return D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
            case ReflectionResourceType::Dimensions::Texture3D: return D3D12_SRV_DIMENSION_TEXTURE3D;
            case ReflectionResourceType::Dimensions::TextureCube: return D3D12_SRV_DIMENSION_TEXTURECUBE;
            case ReflectionResourceType::Dimensions::TextureCubeArray: return D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
            case ReflectionResourceType::Dimensions::AccelerationStructure: return D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            default:
                FALCOR_UNREACHABLE();
                return D3D12_SRV_DIMENSION_UNKNOWN;
            }
        }

        template<>
        D3D12_UAV_DIMENSION getViewDimension<D3D12_UAV_DIMENSION>(ReflectionResourceType::Dimensions dimension)
        {
            switch (dimension)
            {
            case ReflectionResourceType::Dimensions::Buffer: return D3D12_UAV_DIMENSION_BUFFER;
            case ReflectionResourceType::Dimensions::Texture1D: return D3D12_UAV_DIMENSION_TEXTURE1D;
            case ReflectionResourceType::Dimensions::Texture1DArray: return D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            case ReflectionResourceType::Dimensions::Texture2D: return D3D12_UAV_DIMENSION_TEXTURE2D;
            case ReflectionResourceType::Dimensions::Texture2DArray: return D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            case ReflectionResourceType::Dimensions::Texture3D: return D3D12_UAV_DIMENSION_TEXTURE3D;
            default:
                FALCOR_UNREACHABLE();
                return D3D12_UAV_DIMENSION_UNKNOWN;
            }
        }

        template<>
        D3D12_DSV_DIMENSION getViewDimension<D3D12_DSV_DIMENSION>(ReflectionResourceType::Dimensions dimension)
        {
            switch (dimension)
            {
            case ReflectionResourceType::Dimensions::Texture1D: return D3D12_DSV_DIMENSION_TEXTURE1D;
            case ReflectionResourceType::Dimensions::Texture1DArray: return D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
            case ReflectionResourceType::Dimensions::Texture2D: return D3D12_DSV_DIMENSION_TEXTURE2D;
            case ReflectionResourceType::Dimensions::Texture2DArray: return D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
            case ReflectionResourceType::Dimensions::Texture2DMS: return D3D12_DSV_DIMENSION_TEXTURE2DMS;
            case ReflectionResourceType::Dimensions::Texture2DMSArray: return D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
            // TODO: Falcor previously mapped cube to 2D array. Not sure if needed anymore.
            //case ReflectionResourceType::Dimensions::TextureCube: return D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
            default:
                FALCOR_UNREACHABLE();
                return D3D12_DSV_DIMENSION_UNKNOWN;
            }
        }

        template<>
        D3D12_RTV_DIMENSION getViewDimension<D3D12_RTV_DIMENSION>(ReflectionResourceType::Dimensions dimension)
        {
            switch (dimension)
            {
            case ReflectionResourceType::Dimensions::Buffer: return D3D12_RTV_DIMENSION_BUFFER;
            case ReflectionResourceType::Dimensions::Texture1D: return D3D12_RTV_DIMENSION_TEXTURE1D;
            case ReflectionResourceType::Dimensions::Texture1DArray: return D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
            case ReflectionResourceType::Dimensions::Texture2D: return D3D12_RTV_DIMENSION_TEXTURE2D;
            case ReflectionResourceType::Dimensions::Texture2DArray: return D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            case ReflectionResourceType::Dimensions::Texture2DMS: return D3D12_RTV_DIMENSION_TEXTURE2DMS;
            case ReflectionResourceType::Dimensions::Texture2DMSArray: return D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
            case ReflectionResourceType::Dimensions::Texture3D: return D3D12_RTV_DIMENSION_TEXTURE3D;
            // TODO: Falcor previously mapped cube to 2D array. Not sure if needed anymore.
            //case ReflectionResourceType::Dimensions::TextureCube: return D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            default:
                FALCOR_UNREACHABLE();
                return D3D12_RTV_DIMENSION_UNKNOWN;
            }
        }
    }

    template<typename T>
    ResourceView<T>::~ResourceView() = default;

    D3D12_SHADER_RESOURCE_VIEW_DESC createBufferSrvDesc(const Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        FALCOR_ASSERT(pBuffer);
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};

        uint32_t bufferElementSize = 0;
        uint32_t bufferElementCount = 0;
        if (pBuffer->isTyped())
        {
            FALCOR_ASSERT(getFormatPixelsPerBlock(pBuffer->getFormat()) == 1);
            bufferElementSize = getFormatBytesPerBlock(pBuffer->getFormat());
            bufferElementCount = pBuffer->getElementCount();
            desc.Format = getDxgiFormat(pBuffer->getFormat());
        }
        else if (pBuffer->isStructured())
        {
            bufferElementSize = pBuffer->getStructSize();
            bufferElementCount = pBuffer->getElementCount();
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Buffer.StructureByteStride = pBuffer->getStructSize();
        }
        else
        {
            bufferElementSize = sizeof(uint32_t);
            bufferElementCount = (uint32_t)(pBuffer->getSize() / sizeof(uint32_t));
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        }

        bool useDefaultCount = (elementCount == ShaderResourceView::kMaxPossible);
        FALCOR_ASSERT(useDefaultCount || (firstElement + elementCount) <= bufferElementCount); // Check range
        desc.Buffer.FirstElement = firstElement;
        desc.Buffer.NumElements = useDefaultCount ? (bufferElementCount - firstElement) : elementCount;

        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;

        // D3D12 doesn't currently handle views that extend to close to 4GB or beyond the base address.
        // TODO: Revisit this check in the future.
        FALCOR_ASSERT(bufferElementSize > 0);
        if (desc.Buffer.FirstElement + desc.Buffer.NumElements > ((1ull << 32) / bufferElementSize - 8))
        {
            throw RuntimeError("Buffer SRV exceeds the maximum supported size");
        }

        return desc;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC createTextureSrvDesc(const Texture* pTexture, uint32_t firstArraySlice, uint32_t arraySize, uint32_t mostDetailedMip, uint32_t mipCount)
    {
        FALCOR_ASSERT(pTexture);
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};

        //If not depth, returns input format
        ResourceFormat colorFormat = depthToColorFormat(pTexture->getFormat());
        desc.Format = getDxgiFormat(colorFormat);

        bool isTextureArray = pTexture->getArraySize() > 1;
        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(getDimension(pTexture->getType(), isTextureArray));

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
            FALCOR_ASSERT(arraySize == 1);
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
            FALCOR_UNREACHABLE();
        }

        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        return desc;
    }

    template<typename DescType>
    DescType createDsvRtvUavDescCommon(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        const Texture* pTexture = dynamic_cast<const Texture*>(pResource);
        FALCOR_ASSERT(pTexture);   // Buffers should not get here

        uint32_t arrayMultiplier = (pResource->getType() == Resource::Type::TextureCube) ? 6 : 1;

        if (arraySize == Resource::kMaxPossible)
        {
            arraySize = pTexture->getArraySize() - firstArraySlice;
        }

        DescType desc = {};
        desc.Format = getDxgiFormat(pTexture->getFormat());
        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(getDimension(pTexture->getType(), pTexture->getArraySize() > 1));

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
        case Resource::Type::Texture2DMultisample:
            if constexpr (std::is_same_v<DescType, D3D12_DEPTH_STENCIL_VIEW_DESC> || std::is_same_v<DescType, D3D12_RENDER_TARGET_VIEW_DESC>)
            {
                if (pTexture->getArraySize() > 1)
                {
                    desc.Texture2DMSArray.ArraySize = arraySize;
                    desc.Texture2DMSArray.FirstArraySlice = firstArraySlice;
                }
            }
            else
            {
                throw RuntimeError("Texture2DMultisample does not support UAV views");
            }
            break;
        case Resource::Type::Texture3D:
            if constexpr (std::is_same_v<DescType, D3D12_UNORDERED_ACCESS_VIEW_DESC> || std::is_same_v<DescType, D3D12_RENDER_TARGET_VIEW_DESC>)
            {
                FALCOR_ASSERT(pTexture->getArraySize() == 1);
                desc.Texture3D.MipSlice = mipLevel;
                desc.Texture3D.FirstWSlice = 0;
                desc.Texture3D.WSize = pTexture->getDepth(mipLevel);
            }
            else
            {
                throw RuntimeError("Texture3D does not support DSV views");
            }
            break;
        default:
            FALCOR_UNREACHABLE();
        }

        return desc;
    }

    D3D12_DEPTH_STENCIL_VIEW_DESC createDsvDesc(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        return createDsvRtvUavDescCommon<D3D12_DEPTH_STENCIL_VIEW_DESC>(pResource, mipLevel, firstArraySlice, arraySize);
    }

    D3D12_RENDER_TARGET_VIEW_DESC createRtvDesc(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        return createDsvRtvUavDescCommon<D3D12_RENDER_TARGET_VIEW_DESC>(pResource, mipLevel, firstArraySlice, arraySize);
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC createBufferUavDesc(const Buffer* pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        FALCOR_ASSERT(pBuffer);
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};

        uint32_t bufferElementSize = 0;
        uint32_t bufferElementCount = 0;
        if (pBuffer->isTyped())
        {
            FALCOR_ASSERT(getFormatPixelsPerBlock(pBuffer->getFormat()) == 1);
            bufferElementSize = getFormatBytesPerBlock(pBuffer->getFormat());
            bufferElementCount = pBuffer->getElementCount();
            desc.Format = getDxgiFormat(pBuffer->getFormat());
        }
        else if (pBuffer->isStructured())
        {
            bufferElementSize = pBuffer->getStructSize();
            bufferElementCount = pBuffer->getElementCount();
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Buffer.StructureByteStride = pBuffer->getStructSize();
        }
        else
        {
            bufferElementSize = sizeof(uint32_t);
            bufferElementCount = (uint32_t)(pBuffer->getSize() / sizeof(uint32_t));
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        }

        bool useDefaultCount = (elementCount == UnorderedAccessView::kMaxPossible);
        FALCOR_ASSERT(useDefaultCount || (firstElement + elementCount) <= bufferElementCount); // Check range
        desc.Buffer.FirstElement = firstElement;
        desc.Buffer.NumElements = useDefaultCount ? bufferElementCount - firstElement : elementCount;

        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;

        // D3D12 doesn't currently handle views that extend to close to 4GB or beyond the base address.
        // TODO: Revisit this check in the future.
        FALCOR_ASSERT(bufferElementSize > 0);
        if (desc.Buffer.FirstElement + desc.Buffer.NumElements > ((1ull << 32) / bufferElementSize - 8))
        {
            throw RuntimeError("Buffer UAV exceeds the maximum supported size");
        }

        return desc;
    }

    ShaderResourceView::ApiHandle createSrvDescriptor(const D3D12_SHADER_RESOURCE_VIEW_DESC& desc, Resource::ApiHandle resHandle)
    {
        if (desc.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE &&
            !gpDevice->isFeatureSupported(Device::SupportedFeatures::Raytracing))
        {
            throw RuntimeError("Raytracing is not supported by the current device");
        }

        D3D12DescriptorSet::Layout layout;
        layout.addRange(ShaderResourceType::TextureSrv, 0, 1);
        ShaderResourceView::ApiHandle handle = D3D12DescriptorSet::create(gpDevice->getD3D12CpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateShaderResourceView(resHandle, &desc, handle->getCpuHandle(0));

        return handle;
    }

    ShaderResourceView::SharedPtr ShaderResourceView::create(ConstTextureSharedPtrRef pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
    {
        FALCOR_ASSERT(pTexture);
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = createTextureSrvDesc(pTexture.get(), firstArraySlice, arraySize, mostDetailedMip, mipCount);
        Resource::ApiHandle resHandle = pTexture->getApiHandle();

        return SharedPtr(new ShaderResourceView(pTexture, createSrvDescriptor(desc, resHandle), mostDetailedMip, mipCount, firstArraySlice, arraySize));
    }

    ShaderResourceView::SharedPtr ShaderResourceView::create(ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        FALCOR_ASSERT(pBuffer);
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = createBufferSrvDesc(pBuffer.get(), firstElement, elementCount);
        Resource::ApiHandle resHandle = pBuffer->getApiHandle();

        return SharedPtr(new ShaderResourceView(pBuffer, createSrvDescriptor(desc, resHandle), firstElement, elementCount));
    }

    ShaderResourceView::SharedPtr ShaderResourceView::create(Dimension dimension)
    {
        // Create a null view of the specified dimension.
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
        desc.Format = (dimension == Dimension::AccelerationStructure ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R32_UINT);
        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(dimension);
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

        return SharedPtr(new ShaderResourceView(std::weak_ptr<Resource>(), createSrvDescriptor(desc, nullptr), 0, 0));
    }

    ShaderResourceView::SharedPtr ShaderResourceView::createViewForAccelerationStructure(ConstBufferSharedPtrRef pBuffer)
    {
        // Views for acceleration structures pass the GPU VA as part of the view desc.
        // Note that in the call to CreateShaderResourceView() the resource ptr should be nullptr.
        FALCOR_ASSERT(pBuffer);
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.RaytracingAccelerationStructure.Location = pBuffer->getGpuAddress();

        D3D12DescriptorSet::Layout layout;
        layout.addRange(ShaderResourceType::AccelerationStructureSrv, 0, 1);
        ShaderResourceView::ApiHandle handle = D3D12DescriptorSet::create(gpDevice->getD3D12CpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateShaderResourceView(nullptr, &srvDesc, handle->getCpuHandle(0));

        return SharedPtr(new ShaderResourceView(pBuffer, handle));
    }

    D3D12DescriptorCpuHandle ShaderResourceView::getD3D12CpuHeapHandle() const
    {
        return mApiHandle->getCpuHandle(0);
    }

    DepthStencilView::ApiHandle createDsvDescriptor(const D3D12_DEPTH_STENCIL_VIEW_DESC& desc, Resource::ApiHandle resHandle)
    {
        D3D12DescriptorSet::Layout layout;
        layout.addRange(ShaderResourceType::Dsv, 0, 1);
        DepthStencilView::ApiHandle handle = D3D12DescriptorSet::create(gpDevice->getD3D12CpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateDepthStencilView(resHandle, &desc, handle->getCpuHandle(0));

        return handle;
    }

    DepthStencilView::SharedPtr DepthStencilView::create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        FALCOR_ASSERT(pTexture);
        D3D12_DEPTH_STENCIL_VIEW_DESC desc = createDsvDesc(pTexture.get(), mipLevel, firstArraySlice, arraySize);
        Resource::ApiHandle resHandle = pTexture->getApiHandle();

        return SharedPtr(new DepthStencilView(pTexture, createDsvDescriptor(desc, resHandle), mipLevel, firstArraySlice, arraySize));
    }

    DepthStencilView::SharedPtr DepthStencilView::create(Dimension dimension)
    {
        // Create a null view of the specified dimension.
        D3D12_DEPTH_STENCIL_VIEW_DESC desc = {};
        desc.Format = DXGI_FORMAT_D32_FLOAT;
        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(dimension);

        return SharedPtr(new DepthStencilView(std::weak_ptr<Resource>(), createDsvDescriptor(desc, nullptr), 0, 0, 1));
    }

    D3D12DescriptorCpuHandle DepthStencilView::getD3D12CpuHeapHandle() const
    {
        return mApiHandle->getCpuHandle(0);
    }

    UnorderedAccessView::ApiHandle createUavDescriptor(const D3D12_UNORDERED_ACCESS_VIEW_DESC& desc, Resource::ApiHandle resHandle, Resource::ApiHandle counterHandle)
    {
        D3D12DescriptorSet::Layout layout;
        layout.addRange(ShaderResourceType::TextureUav, 0, 1);
        UnorderedAccessView::ApiHandle handle = D3D12DescriptorSet::create(gpDevice->getD3D12CpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateUnorderedAccessView(resHandle, counterHandle, &desc, handle->getCpuHandle(0));

        return handle;
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        FALCOR_ASSERT(pTexture);
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc = createDsvRtvUavDescCommon<D3D12_UNORDERED_ACCESS_VIEW_DESC>(pTexture.get(), mipLevel, firstArraySlice, arraySize);
        Resource::ApiHandle resHandle = pTexture->getApiHandle();

        return SharedPtr(new UnorderedAccessView(pTexture, createUavDescriptor(desc, resHandle, nullptr), mipLevel, firstArraySlice, arraySize));
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::create(ConstBufferSharedPtrRef pBuffer, uint32_t firstElement, uint32_t elementCount)
    {
        FALCOR_ASSERT(pBuffer);
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc = createBufferUavDesc(pBuffer.get(), firstElement, elementCount);
        Resource::ApiHandle resHandle = pBuffer->getApiHandle();
        Resource::ApiHandle counterHandle = nullptr;
        if (pBuffer->getUAVCounter())
        {
            counterHandle = pBuffer->getUAVCounter()->getApiHandle();
        }

        return SharedPtr(new UnorderedAccessView(pBuffer, createUavDescriptor(desc, resHandle, counterHandle), firstElement, elementCount));
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::create(Dimension dimension)
    {
        // Create a null view of the specified dimension.
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
        desc.Format = DXGI_FORMAT_R32_UINT;
        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(dimension);

        return SharedPtr(new UnorderedAccessView(std::weak_ptr<Resource>(), createUavDescriptor(desc, nullptr, nullptr), 0, 0));
    }

    D3D12DescriptorCpuHandle UnorderedAccessView::getD3D12CpuHeapHandle() const
    {
        return mApiHandle->getCpuHandle(0);
    }

    RenderTargetView::~RenderTargetView() = default;

    RenderTargetView::ApiHandle createRtvDescriptor(const D3D12_RENDER_TARGET_VIEW_DESC& desc, Resource::ApiHandle resHandle)
    {
        D3D12DescriptorSet::Layout layout;
        layout.addRange(ShaderResourceType::Rtv, 0, 1);
        RenderTargetView::ApiHandle handle = D3D12DescriptorSet::create(gpDevice->getD3D12CpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateRenderTargetView(resHandle, &desc, handle->getCpuHandle(0));

        return handle;
    }

    RenderTargetView::SharedPtr RenderTargetView::create(ConstTextureSharedPtrRef pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        FALCOR_ASSERT(pTexture);
        D3D12_RENDER_TARGET_VIEW_DESC desc = createRtvDesc(pTexture.get(), mipLevel, firstArraySlice, arraySize);
        Resource::ApiHandle resHandle = pTexture->getApiHandle();

        return SharedPtr(new RenderTargetView(pTexture, createRtvDescriptor(desc, resHandle), mipLevel, firstArraySlice, arraySize));
    }

    RenderTargetView::SharedPtr RenderTargetView::create(Dimension dimension)
    {
        // Create a null view of the specified dimension.
        D3D12_RENDER_TARGET_VIEW_DESC desc = {};
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(dimension);

        return SharedPtr(new RenderTargetView(std::weak_ptr<Resource>(), createRtvDescriptor(desc, nullptr), 0, 0, 1));
    }

    D3D12DescriptorCpuHandle RenderTargetView::getD3D12CpuHeapHandle() const
    {
        return mApiHandle->getCpuHandle(0);
    }

    ConstantBufferView::ApiHandle createCbvDescriptor(const D3D12_CONSTANT_BUFFER_VIEW_DESC& desc, Resource::ApiHandle resHandle)
    {
        D3D12DescriptorSet::Layout layout;
        layout.addRange(ShaderResourceType::Cbv, 0, 1);
        ConstantBufferView::ApiHandle handle = D3D12DescriptorSet::create(gpDevice->getD3D12CpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateConstantBufferView(&desc, handle->getCpuHandle(0));

        return handle;
    }

    ConstantBufferView::SharedPtr ConstantBufferView::create(ConstBufferSharedPtrRef pBuffer)
    {
        FALCOR_ASSERT(pBuffer);
        D3D12_CONSTANT_BUFFER_VIEW_DESC desc = {};
        desc.BufferLocation = pBuffer->getGpuAddress();
        desc.SizeInBytes = (uint32_t)pBuffer->getSize();
        Resource::ApiHandle resHandle = pBuffer->getApiHandle();

        return SharedPtr(new ConstantBufferView(pBuffer, createCbvDescriptor(desc, resHandle)));
    }

    ConstantBufferView::SharedPtr ConstantBufferView::create()
    {
        // Create a null view.
        D3D12_CONSTANT_BUFFER_VIEW_DESC desc = {};

        return SharedPtr(new ConstantBufferView(std::weak_ptr<Resource>(), createCbvDescriptor(desc, nullptr)));
    }

    D3D12DescriptorCpuHandle ConstantBufferView::getD3D12CpuHeapHandle() const
    {
        return mApiHandle->getCpuHandle(0);
    }
}
