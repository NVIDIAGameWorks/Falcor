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
#include "Core/API/Texture.h"
#include "D3D12Resource.h"
#include "Core/API/Device.h"
#include "Core/API/Formats.h"
#include "Core/API/D3D12/D3D12API.h"
#include "Utils/Math/Common.h"

namespace Falcor
{
    D3D12_RESOURCE_DIMENSION getResourceDimension(Texture::Type type)
    {
        switch (type)
        {
        case Texture::Type::Texture1D:
            return D3D12_RESOURCE_DIMENSION_TEXTURE1D;

        case Texture::Type::Texture2D:
        case Texture::Type::Texture2DMultisample:
        case Texture::Type::TextureCube:
            return D3D12_RESOURCE_DIMENSION_TEXTURE2D;

        case Texture::Type::Texture3D:
            return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        default:
            FALCOR_UNREACHABLE();
            return D3D12_RESOURCE_DIMENSION_UNKNOWN;
        }
    }

    uint64_t Texture::getTextureSizeInBytes() const
    {
        ID3D12DevicePtr pDevicePtr = gpDevice->getApiHandle();
        ID3D12ResourcePtr pTexResource = this->getApiHandle();

        D3D12_RESOURCE_ALLOCATION_INFO d3d12ResourceAllocationInfo;
        D3D12_RESOURCE_DESC desc = pTexResource->GetDesc();

        FALCOR_ASSERT(desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D);

        d3d12ResourceAllocationInfo = pDevicePtr->GetResourceAllocationInfo(0, 1, &desc);
        FALCOR_ASSERT(d3d12ResourceAllocationInfo.SizeInBytes > 0);
        return d3d12ResourceAllocationInfo.SizeInBytes;
    }

    void Texture::apiInit(const void* pData, bool autoGenMips)
    {
        D3D12_RESOURCE_DESC desc = {};

        desc.MipLevels = mMipLevels;
        desc.Format = getDxgiFormat(mFormat);
        desc.Width = align_to(getFormatWidthCompressionRatio(mFormat), mWidth);
        desc.Height = align_to(getFormatHeightCompressionRatio(mFormat), mHeight);
        desc.Flags = getD3D12ResourceFlags(mBindFlags);
        desc.SampleDesc.Count = mSampleCount;
        desc.SampleDesc.Quality = 0;
        desc.Dimension = getResourceDimension(mType);
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Alignment = 0;

        if (mType == Texture::Type::TextureCube)
        {
            desc.DepthOrArraySize = mArraySize * 6;
        }
        else if (mType == Texture::Type::Texture3D)
        {
            desc.DepthOrArraySize = mDepth;
        }
        else
        {
            desc.DepthOrArraySize = mArraySize;
        }
        FALCOR_ASSERT(desc.Width > 0 && desc.Height > 0);
        FALCOR_ASSERT(desc.MipLevels > 0 && desc.DepthOrArraySize > 0 && desc.SampleDesc.Count > 0);

        D3D12_CLEAR_VALUE clearValue = {};
        D3D12_CLEAR_VALUE* pClearVal = nullptr;
        if ((mBindFlags & (Texture::BindFlags::RenderTarget | Texture::BindFlags::DepthStencil)) != Texture::BindFlags::None)
        {
            clearValue.Format = desc.Format;
            if ((mBindFlags & Texture::BindFlags::DepthStencil) != Texture::BindFlags::None)
            {
                clearValue.DepthStencil.Depth = 1.0f;
            }
            pClearVal = &clearValue;
        }

        //If depth and either ua or sr, set to typeless
        if (isDepthFormat(mFormat) && is_set(mBindFlags, Texture::BindFlags::ShaderResource | Texture::BindFlags::UnorderedAccess))
        {
            desc.Format = getTypelessFormatFromDepthFormat(mFormat);
            pClearVal = nullptr;
        }

        D3D12_HEAP_FLAGS heapFlags = is_set(mBindFlags, ResourceBindFlags::Shared) ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE;
        FALCOR_D3D_CALL(gpDevice->getApiHandle()->CreateCommittedResource(&kDefaultHeapProps, heapFlags, &desc, D3D12_RESOURCE_STATE_COMMON, pClearVal, IID_PPV_ARGS(&mApiHandle)));
        FALCOR_ASSERT(mApiHandle);

        if (pData)
        {
            uploadInitData(pData, autoGenMips);
        }
    }

    Texture::~Texture()
    {
        gpDevice->releaseResource(mApiHandle);
    }
}
