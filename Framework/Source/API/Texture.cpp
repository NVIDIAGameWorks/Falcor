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
#include "API/Device.h"
#include "Utils/ThreadPool.h"

namespace Falcor
{
    uint32_t Texture::tempDefaultUint = 0;

    Texture::BindFlags updateBindFlags(Texture::BindFlags flags, bool hasInitData, uint32_t mipLevels)
    {
        if ((mipLevels != Texture::kMaxPossible) || (hasInitData == false))
        {
            return flags;
        }

        flags |= Texture::BindFlags::RenderTarget;
        return flags;
    }

    Texture::SharedPtr Texture::create1D(uint32_t width, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData, BindFlags bindFlags)
    {
        bindFlags = updateBindFlags(bindFlags, pData != nullptr, mipLevels);
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, 1, 1, arraySize, mipLevels, 1, format, Type::Texture1D, bindFlags));
        pTexture->apinit(pData, (mipLevels == kMaxPossible));
        return pTexture->mApiHandle ? pTexture : nullptr;
    }

    Texture::SharedPtr Texture::create2D(uint32_t width, uint32_t height, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData, BindFlags bindFlags)
    {
        bindFlags = updateBindFlags(bindFlags, pData != nullptr, mipLevels);
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, 1, arraySize, mipLevels, 1, format, Type::Texture2D, bindFlags));
        pTexture->apinit(pData, (mipLevels == kMaxPossible));
        return pTexture->mApiHandle ? pTexture : nullptr;
    }

    Texture::SharedPtr Texture::create3D(uint32_t width, uint32_t height, uint32_t depth, ResourceFormat format, uint32_t mipLevels, const void* pData, BindFlags bindFlags, bool isSparse)
    {
        bindFlags = updateBindFlags(bindFlags, pData != nullptr, mipLevels);
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, depth, 1, mipLevels, 1, format, Type::Texture3D, bindFlags));
        pTexture->apinit(pData, (mipLevels == kMaxPossible));
        return pTexture->mApiHandle ? pTexture : nullptr;
    }

    // Texture Cube
    Texture::SharedPtr Texture::createCube(uint32_t width, uint32_t height, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData, BindFlags bindFlags)
    {
        bindFlags = updateBindFlags(bindFlags, pData != nullptr, mipLevels);
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, 1, arraySize, mipLevels, 1, format, Type::TextureCube, bindFlags));
        pTexture->apinit(pData, (mipLevels == kMaxPossible));
        return pTexture->mApiHandle ? pTexture : nullptr;
    }

    Texture::SharedPtr Texture::create2DMS(uint32_t width, uint32_t height, ResourceFormat format, uint32_t sampleCount, uint32_t arraySize, BindFlags bindFlags)
    {
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, 1, arraySize, 1, sampleCount, format, Type::Texture2DMultisample, bindFlags));
        pTexture->apinit(nullptr, false);
        return pTexture->mApiHandle ? pTexture : nullptr;
    }

    Texture::Texture(uint32_t width, uint32_t height, uint32_t depth, uint32_t arraySize, uint32_t mipLevels, uint32_t sampleCount, ResourceFormat format, Type type, BindFlags bindFlags)
        : Resource(type, bindFlags), mWidth(width), mHeight(height), mDepth(depth), mMipLevels(mipLevels), mSampleCount(sampleCount), mArraySize(arraySize), mFormat(format)
    {
        if(mMipLevels == kMaxPossible)
        {
            uint32_t dims = width | height | depth;
            mMipLevels = bitScanReverse(dims) + 1;
        }
    }

    void Texture::captureToFile(uint32_t mipLevel, uint32_t arraySlice, const std::string& filename, Bitmap::FileFormat format, Bitmap::ExportFlags exportFlags) const
    {
        uint32_t subresource = getSubresourceIndex(arraySlice, mipLevel);
        std::vector<uint8> textureData = gpDevice->getRenderContext()->readTextureSubresource(this, subresource);

        auto func = [=]()
        {
            Bitmap::saveImage(filename, getWidth(mipLevel), getHeight(mipLevel), format, exportFlags, getFormat(), true, (void*)textureData.data());
        };

        static ThreadPool<16> sThreadPool;
        sThreadPool.getAvailable() = std::thread(func);
    }

    void Texture::uploadInitData(const void* pData, bool autoGenMips)
    {
        auto& pRenderContext = gpDevice->getRenderContext();
        if (autoGenMips)
        {
            // Upload just the first mip-level
            size_t arraySliceSize = mWidth * mHeight * getFormatBytesPerBlock(mFormat);
            const uint8_t* pSrc = (uint8_t*)pData;
            uint32_t numFaces = (mType == Texture::Type::TextureCube) ? 6 : 1;
            for (uint32_t i = 0; i < mArraySize * numFaces; i++)
            {
                uint32_t subresource = getSubresourceIndex(i, 0);
                pRenderContext->updateTextureSubresource(this, subresource, pSrc);
                pSrc += arraySliceSize;
            }
        }
        else
        {
            pRenderContext->updateTexture(this, pData);
        }

        if (autoGenMips)
        {
            generateMips();
            invalidateViews();
        }
    }

    void Texture::generateMips()
    {
        if (mType != Type::Texture2D)
        {
            logWarning("Texture::generateMips() was only tested with Texture2Ds");
        }

        RenderContext* pContext = gpDevice->getRenderContext().get();

        for (uint32_t i = 0; i < mMipLevels - 1; i++)
        {
            auto srv = getSRV(i, 1, 0, mArraySize);
            auto rtv = getRTV(i + 1, 0, mArraySize);
            pContext->blit(srv, rtv);
        }

		if(mReleaseRtvsAfterGenMips)
		{
			// Releasing RTVs to free space on the heap.
			// We only do it once to handle the case that generateMips() was called during load. 
			// If it was called more then once, the texture is probably dynamic and it's better to keep the RTVs around
			mRtvs.clear();
			mReleaseRtvsAfterGenMips = false;
		}
    }
}
