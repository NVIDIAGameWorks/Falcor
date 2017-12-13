/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "API/Resource.h"
#include "API/Buffer.h"
#include "API/Texture.h"
#include "API/TypedBuffer.h"

namespace Falcor
{
    template<typename ViewType>
    ViewType getViewDimension(Resource::Type type, bool isArray);

    template<typename ViewDesc>
    void initializeSrvDesc(const Resource* pResource, uint32_t firstArraySlice, uint32_t arraySize, uint32_t mostDetailedMip, uint32_t mipCount, ViewDesc& desc)
    {
        const Texture* pTexture = dynamic_cast<const Texture*>(pResource);
        const Buffer* pBuffer = dynamic_cast<const Buffer*>(pResource);

        desc = {};

        if (pBuffer)
        {
            const TypedBufferBase* pTypedBuffer = dynamic_cast<const TypedBufferBase*>(pResource);
            const StructuredBuffer* pStructuredBuffer = dynamic_cast<const StructuredBuffer*>(pResource);

            desc.Buffer.FirstElement = 0;
            if (pTypedBuffer)
            {
                desc.Format = getDxgiFormat(pTypedBuffer->getResourceFormat());
                desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
                desc.Buffer.NumElements = pTypedBuffer->getElementCount();
            }
            else if (pStructuredBuffer)
            {
                desc.Format = DXGI_FORMAT_UNKNOWN;
                desc.Buffer.NumElements = (uint32_t)pStructuredBuffer->getElementCount();
                desc.Buffer.StructureByteStride = (uint32_t)pStructuredBuffer->getElementSize();
            }
            else
            {
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
                desc.Buffer.NumElements = (uint32_t)pBuffer->getSize() / sizeof(float);
            }
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            return;
        }

        assert(pTexture);

        //If not depth, returns input format
        ResourceFormat colorFormat = depthToColorFormat(pTexture->getFormat());
        desc.Format = getDxgiFormat(colorFormat);

        bool isTextureArray = pTexture->getArraySize() > 1;
        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(pResource->getType(), isTextureArray);

        switch(pResource->getType())
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
            if(isTextureArray)
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
            if(arraySize > 1)
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
            if(arraySize > 1)
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
#ifdef FALCOR_D3D12
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
#endif
    }

    template<typename DescType, bool finalCall>
    inline void initializeDsvRtvUavDescCommon(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize, DescType& desc)
    {
        const Texture* pTexture = dynamic_cast<const Texture*>(pResource);
        assert(pTexture);   // Buffers should not get here

        desc = {};
        uint32_t arrayMultiplier = (pResource->getType() == Resource::Type::TextureCube) ? 6 : 1;

        if(arraySize == Resource::kMaxPossible)
        {
            arraySize = pTexture->getArraySize() - firstArraySlice;
        }

        desc.ViewDimension = getViewDimension<decltype(desc.ViewDimension)>(pTexture->getType(), pTexture->getArraySize() > 1);

        switch(pResource->getType())
        {
        case Resource::Type::Texture1D:
            if(pTexture->getArraySize() > 1)
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
            if(pTexture->getArraySize() * arrayMultiplier > 1)
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
            if(finalCall)
            {
                should_not_get_here();
            }
        }
        desc.Format = getDxgiFormat(pTexture->getFormat());
    }

    template<typename DescType>
    inline void initializeDsvRtvDesc(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize, DescType& desc)
    {
        initializeDsvRtvUavDescCommon<DescType, false>(pResource, mipLevel, firstArraySlice, arraySize, desc);
        
        if(pResource->getType() == Resource::Type::Texture2DMultisample)
        {
            const Texture* pTexture = dynamic_cast<const Texture*>(pResource);
            if (pTexture->getArraySize() > 1)
            {
                desc.Texture2DMSArray.ArraySize = arraySize;
                desc.Texture2DMSArray.FirstArraySlice = firstArraySlice;
            }
        }
    }

    template<typename DescType>
    inline void initializeDsvDesc(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize, DescType& desc)
    {
        return initializeDsvRtvDesc<DescType>(pResource, mipLevel, firstArraySlice, arraySize, desc);
    }

    template<typename DescType>
    inline void initializeRtvDesc(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize, DescType& desc)
    {
        return initializeDsvRtvDesc<DescType>(pResource, mipLevel, firstArraySlice, arraySize, desc);
    }

    template<typename DescType>
    inline void initializeUavDesc(const Resource* pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize, DescType& desc)
    {
        if (pResource->getType() == Resource::Type::Buffer)
        {
            const Buffer* pBuffer = dynamic_cast<const Buffer*>(pResource);
            const TypedBufferBase* pTypedBuffer = dynamic_cast<const TypedBufferBase*>(pBuffer);
            const StructuredBuffer* pStructuredBuffer = dynamic_cast<const StructuredBuffer*>(pBuffer);

            desc = {};
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;

            if (pTypedBuffer != nullptr)
            {
                desc.Format = getDxgiFormat(pTypedBuffer->getResourceFormat());
                desc.Buffer.NumElements = pTypedBuffer->getElementCount();
            }
            else if (pStructuredBuffer != nullptr)
            {
                desc.Format = DXGI_FORMAT_UNKNOWN;
                desc.Buffer.NumElements = (uint32_t)pStructuredBuffer->getElementCount();
                desc.Buffer.StructureByteStride = (uint32_t)pStructuredBuffer->getElementSize();
            }
            else
            {
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                desc.Buffer.NumElements = (uint32_t)pBuffer->getSize() / sizeof(float);
            }
        }
        else
        {
            initializeDsvRtvUavDescCommon<DescType, true>(pResource, mipLevel, firstArraySlice, arraySize, desc);
        }
    }
}
