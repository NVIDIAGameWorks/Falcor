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
#include "API/Texture.h"
#include "API/D3D/D3DViews.h"
#include <vector>

namespace Falcor
{
    ID3D11ShaderResourceViewPtr createShaderResourceView(ID3D11ResourcePtr pResource, ResourceFormat format, uint32_t arraySize, Texture::Type type)
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
        initializeSrvDesc(format, arraysize, type, srvDesc);

        ID3D11ShaderResourceViewPtr pSrv;
        d3d_call(getD3D11Device()->CreateShaderResourceView(pResource, &srvDesc, &pSrv));
        return pSrv;
    }

    Texture::~Texture()
    {
    }

    uint64_t Texture::makeResident(const Sampler* pSampler) const
    {
        UNSUPPORTED_IN_D3D11("Texture::makeResident()");
        return 0;
    }

    void Texture::evict(const Sampler* pSampler) const
    {
        UNSUPPORTED_IN_D3D11("Texture::evict()");
    }

    std::vector<D3D11_SUBRESOURCE_DATA> createInitDataVector(uint32_t width, uint32_t height, uint32_t depth, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData)
    {
        uint32_t subresourceCount = arraySize * mipLevels;
        std::vector<D3D11_SUBRESOURCE_DATA> initData(subresourceCount);

        const uint8_t* pSrc = (uint8_t*)pData;

        // Loop over the array slices. 3D textures have a single array-slice.
        for(uint32_t array = 0; array < arraySize; array++)
        {
            for(uint32_t mip = 0; mip < mipLevels; mip++)
            {
                auto& data = initData[D3D11CalcSubresource(mip, array, mipLevels)];
                data.pSysMem = pSrc;
                data.SysMemPitch = getFormatBytesPerBlock(format) * (width >> mip);
                data.SysMemSlicePitch = data.SysMemPitch * (height >> mip);
                pSrc += data.SysMemSlicePitch * depth;
            }
        }

        return initData;
    }

    ID3D11ResourcePtr createTexture1D(const D3D11_TEXTURE1D_DESC& desc, const D3D11_SUBRESOURCE_DATA* pInitData)
    {
        ID3D11Texture1DPtr pDxTex;
        d3d_call(getD3D11Device()->CreateTexture1D(&desc, pInitData, &pDxTex));
        return pDxTex;
    }

    ID3D11ResourcePtr createTexture2D(const D3D11_TEXTURE2D_DESC& desc, const D3D11_SUBRESOURCE_DATA* pInitData)
    {
        ID3D11Texture2DPtr pDxTex;
        d3d_call(getD3D11Device()->CreateTexture2D(&desc, pInitData, &pDxTex));
        return pDxTex;
    }

    ID3D11ResourcePtr createTexture3D(const D3D11_TEXTURE3D_DESC& desc, const D3D11_SUBRESOURCE_DATA* pInitData)
    {
        ID3D11Texture3DPtr pDxTex;
        d3d_call(getD3D11Device()->CreateTexture3D(&desc, pInitData, &pDxTex));
        return pDxTex;
    }

    Texture::SharedPtr Texture::create1D(uint32_t width, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData)
    {
        bool isDepth = isDepthStencilFormat(format);
        if(isDepth && mipLevels)
        {
            logWarning("Texture::create1D() - can't automatically generate mip levels for depth texture");
        }

        D3D11_TEXTURE1D_DESC desc;
        desc.ArraySize = arraySize;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.Format = getDxgiFormat(format);
        desc.BindFlags |= isDepth ? D3D11_BIND_DEPTH_STENCIL : D3D11_BIND_RENDER_TARGET;
        desc.MipLevels = (mipLevels == kMaxPossible) ? 1 : mipLevels;
        desc.MiscFlags = isDepth ? 0 : D3D11_RESOURCE_MISC_GENERATE_MIPS;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.Width = width;

        std::vector<D3D11_SUBRESOURCE_DATA> initData;
        const D3D11_SUBRESOURCE_DATA* pInit = nullptr;
        if(pData)
        {
            initData = createInitDataVector(width, 1, 1, format, arraySize, desc.MipLevels, pData);
            pInit = initData.data();
        }

        // create Falcor texture
        SharedPtr pTexture = SharedPtr(new Texture(width, 1, 1, arraySize, mipLevels, 1, format, Type::Texture1D));
        
        // create the DX texture
        pTexture->mApiHandle = createTexture1D(desc, pInit);

        return pTexture;
    }
    
    D3D11_TEXTURE2D_DESC CreateTexture2DDesc(uint32_t width, uint32_t height, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels)
    {
        bool isDepth = isDepthStencilFormat(format);
        if(isDepth && mipLevels)
        {
            logWarning("Texture::CreateTexture2DDesc() - can't automatically generate mip levels for depth texture");
        }
        D3D11_TEXTURE2D_DESC desc;
        desc.ArraySize = arraySize;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.BindFlags = isDepth ? D3D11_BIND_DEPTH_STENCIL : (D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET);
        desc.CPUAccessFlags = 0;
        desc.Format = getDxgiFormat(format);
        desc.MipLevels = (mipLevels == Texture::kMaxPossible) ? 1 : mipLevels;
        desc.MiscFlags = isDepth ? 0 : D3D11_RESOURCE_MISC_GENERATE_MIPS;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.Width = width;
        desc.Height = height;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;

        return desc;
    }

    Texture::SharedPtr Texture::create2D(uint32_t width, uint32_t height, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData)
    {
        D3D11_TEXTURE2D_DESC desc = CreateTexture2DDesc(width, height, format, arraySize, mipLevels);

        std::vector<D3D11_SUBRESOURCE_DATA> initData;
        const D3D11_SUBRESOURCE_DATA* pInit = nullptr;
        if(pData)
        {
            initData = createInitDataVector(width, height, 1, format, arraySize, desc.MipLevels, pData);
            pInit = initData.data();
        }

        // create Falcor texture
        SharedPtr pTexture = SharedPtr(new Texture(width, height, 1, arraySize, mipLevels, 1, format, Type::Texture2D));

        // create the DX texture
        pTexture->mApiHandle = createTexture2D(desc, pInit);

        return pTexture;
    }

    Texture::SharedPtr Texture::create3D(uint32_t width, uint32_t height, uint32_t depth, ResourceFormat format, uint32_t mipLevels, const void* pData, bool isSparse)
    {
        bool isDepth = isDepthStencilFormat(format);
        if(isDepth && mipLevels)
        {
            logWarning("Texture::create3D() - can't automatically generate mip levels for depth texture");
        }

        D3D11_TEXTURE3D_DESC desc;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.BindFlags |= isDepthStencilFormat(format) ? D3D11_BIND_DEPTH_STENCIL : D3D11_BIND_RENDER_TARGET;
        desc.CPUAccessFlags = 0;
        desc.Format = getDxgiFormat(format);
        desc.MipLevels = (mipLevels == kMaxPossible) ? 1 : mipLevels;
        desc.MiscFlags = isDepth ? 0 : D3D11_RESOURCE_MISC_GENERATE_MIPS;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.Width = width;
        desc.Height = height;
        desc.Depth = depth;

        std::vector<D3D11_SUBRESOURCE_DATA> InitData;
        const D3D11_SUBRESOURCE_DATA* pInit = nullptr;
        if(pData)
        {
            InitData = createInitDataVector(width, height, depth, format, 1, desc.MipLevels, pData);
            pInit = InitData.data();
        }

        // create Falcor texture
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, depth, 1, mipLevels, 1, format, Type::Texture3D));

        // create the DX texture
        pTexture->mApiHandle = createTexture3D(desc, pInit);

        return pTexture;
    }

    // Texture Cube
    Texture::SharedPtr Texture::createCube(uint32_t width, uint32_t height, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData)
    {
        D3D11_TEXTURE2D_DESC desc = CreateTexture2DDesc(width, height, format, arraySize, mipLevels);
        desc.ArraySize *= 6;
        desc.MiscFlags |= D3D11_RESOURCE_MISC_TEXTURECUBE;

        std::vector<D3D11_SUBRESOURCE_DATA> InitData;
        const D3D11_SUBRESOURCE_DATA* pInit = nullptr;
        if(pData)
        {
            InitData = createInitDataVector(width, height, 1, format, arraySize*6, desc.MipLevels, pData);
            pInit = InitData.data();
        }

        // create Falcor texture
        SharedPtr pTexture = SharedPtr(new Texture(width, 1, 1, arraySize, mipLevels, 1, format, Type::TextureCube));

        // create the DX texture
        pTexture->mApiHandle = createTexture2D(desc, pInit);

        return pTexture;
    }

    Texture::SharedPtr Texture::create2DMS(uint32_t width, uint32_t height, ResourceFormat format, uint32_t sampleCount, uint32_t arraySize, bool useFixedSampleLocations)
    {
        if(useFixedSampleLocations == false)
        {
            logWarning("DX11 multisampled textures only support fixed sample locations.");
        }

        D3D11_TEXTURE2D_DESC desc = CreateTexture2DDesc(width, width, format, arraySize, 1);

        // create Falcor texture
        SharedPtr pTexture = SharedPtr(new Texture(width, 1, 1, arraySize, 1, 1, format, Type::Texture2DMultisample));

        // create the DX texture
        pTexture->mApiHandle = createTexture2D(desc, nullptr);

        return pTexture;
    }

    Texture::SharedPtr Texture::create2DFromView(uint32_t apiHandle, uint32_t width, uint32_t height, ResourceFormat format)
    {
        auto pResource = SharedPtr(new Texture(width, height, 1u, 1u, 1u, 1u, format, Texture::Type::Texture2D));
        pResource->mApiHandle = apiHandle;
        return pResource;
    }

    ShaderResourceViewHandle Texture::getShaderResourceView() const
    {
        if(mpSRV == nullptr)
        {
            mpSRV = createShaderResourceView(mApiHandle, mFormat, mArraySize, mType);
        }
        return mpSRV;
    }

    void Texture::readSubresourceData(void* pData, uint32_t dataSize, uint32_t mipLevel, uint32_t arraySlice) const
    {
        UNSUPPORTED_IN_D3D11("Texture::readSubresourceData");
    }

    void Texture::uploadSubresourceData(const void* pData, uint32_t dataSize, uint32_t mipLevel, uint32_t arraySlice) const
    {
        UNSUPPORTED_IN_D3D11("Texture::uploadSubresourceData()");
    }

    void Texture::compress2DTexture()
    {
        UNSUPPORTED_IN_D3D11("Texture::compress2DTexture");
    }

	void Texture::generateMips() const
	{
		UNSUPPORTED_IN_D3D11("Texture::GenerateMips");
	}

    Texture::SharedPtr Texture::createView(uint32_t firstArraySlice, uint32_t arraySize, uint32_t mostDetailedMip, uint32_t mipCount) const
    {
        UNSUPPORTED_IN_D3D11("createView");
        return nullptr;
    }
}
