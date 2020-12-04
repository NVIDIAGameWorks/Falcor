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
#include "Texture.h"
#include "Device.h"
#include "RenderContext.h"
#include "Utils/Threading.h"

#include <mutex>

namespace Falcor
{
    namespace
    {
        Texture::BindFlags updateBindFlags(Texture::BindFlags flags, bool hasInitData, uint32_t mipLevels, ResourceFormat format, const std::string& texType)
        {
            if ((mipLevels == Texture::kMaxPossible) && hasInitData)
            {
                flags |= Texture::BindFlags::RenderTarget;
            }

            Texture::BindFlags supported = getFormatBindFlags(format);
            supported |= ResourceBindFlags::Shared;
            if ((flags & supported) != flags)
            {
                logError("Error when creating " + texType + " of format " + to_string(format) + ". The requested bind-flags are not supported.\n"
                    "Requested = (" + to_string(flags) + "), supported = (" + to_string(supported) + ").\n\n"
                    "The texture will be created only with the supported bind flags, which may result in a crash or a rendering error.");
                flags = flags & supported;
            }

            return flags;
        }
    }

    Texture::SharedPtr Texture::createFromApiHandle(ApiHandle handle, Type type, uint32_t width, uint32_t height, uint32_t depth, ResourceFormat format, uint32_t sampleCount, uint32_t arraySize, uint32_t mipLevels, State initState, BindFlags bindFlags)
    {
        assert(handle);
        switch (type)
        {
            case Resource::Type::Texture1D:
                assert(height == 1 && depth == 1 && sampleCount == 1);
                break;
            case Resource::Type::Texture2D:
                assert(depth == 1 && sampleCount == 1);
                break;
            case Resource::Type::Texture2DMultisample:
                assert(depth == 1);
                break;
            case Resource::Type::Texture3D:
                assert(sampleCount == 1);
                break;
            case Resource::Type::TextureCube:
                assert(depth == 1 && sampleCount == 1);
                break;
            default:
                should_not_get_here();
                break;
        }
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, depth, arraySize, mipLevels, sampleCount, format, type, bindFlags));
        pTexture->mApiHandle = handle;
        pTexture->mState.global = initState;
        pTexture->mState.isGlobal = true;
        return pTexture;
    }

    Texture::SharedPtr Texture::create1D(uint32_t width, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData, BindFlags bindFlags)
    {
        bindFlags = updateBindFlags(bindFlags, pData != nullptr, mipLevels, format, "Texture1D");
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, 1, 1, arraySize, mipLevels, 1, format, Type::Texture1D, bindFlags));
        pTexture->apiInit(pData, (mipLevels == kMaxPossible));
        return pTexture;
    }

    Texture::SharedPtr Texture::create2D(uint32_t width, uint32_t height, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData, BindFlags bindFlags)
    {
        bindFlags = updateBindFlags(bindFlags, pData != nullptr, mipLevels, format, "Texture2D");
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, 1, arraySize, mipLevels, 1, format, Type::Texture2D, bindFlags));
        pTexture->apiInit(pData, (mipLevels == kMaxPossible));
        return pTexture;
    }

    Texture::SharedPtr Texture::create3D(uint32_t width, uint32_t height, uint32_t depth, ResourceFormat format, uint32_t mipLevels, const void* pData, BindFlags bindFlags, bool isSparse)
    {
        bindFlags = updateBindFlags(bindFlags, pData != nullptr, mipLevels, format, "Texture3D");
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, depth, 1, mipLevels, 1, format, Type::Texture3D, bindFlags));
        pTexture->apiInit(pData, (mipLevels == kMaxPossible));
        return pTexture;
    }

    Texture::SharedPtr Texture::createCube(uint32_t width, uint32_t height, ResourceFormat format, uint32_t arraySize, uint32_t mipLevels, const void* pData, BindFlags bindFlags)
    {
        bindFlags = updateBindFlags(bindFlags, pData != nullptr, mipLevels, format, "TextureCube");
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, 1, arraySize, mipLevels, 1, format, Type::TextureCube, bindFlags));
        pTexture->apiInit(pData, (mipLevels == kMaxPossible));
        return pTexture;
    }

    Texture::SharedPtr Texture::create2DMS(uint32_t width, uint32_t height, ResourceFormat format, uint32_t sampleCount, uint32_t arraySize, BindFlags bindFlags)
    {
        bindFlags = updateBindFlags(bindFlags, false, 1, format, "Texture2DMultisample");
        Texture::SharedPtr pTexture = SharedPtr(new Texture(width, height, 1, arraySize, 1, sampleCount, format, Type::Texture2DMultisample, bindFlags));
        pTexture->apiInit(nullptr, false);
        return pTexture;
    }

    Texture::Texture(uint32_t width, uint32_t height, uint32_t depth, uint32_t arraySize, uint32_t mipLevels, uint32_t sampleCount, ResourceFormat format, Type type, BindFlags bindFlags)
        : Resource(type, bindFlags, 0), mWidth(width), mHeight(height), mDepth(depth), mMipLevels(mipLevels), mSampleCount(sampleCount), mArraySize(arraySize), mFormat(format)
    {
        assert(width > 0 && height > 0 && depth > 0);
        assert(arraySize > 0 && mipLevels > 0 && sampleCount > 0);
        assert(format != ResourceFormat::Unknown);

        if (mMipLevels == kMaxPossible)
        {
            uint32_t dims = width | height | depth;
            mMipLevels = bitScanReverse(dims) + 1;
        }
        mState.perSubresource.resize(mMipLevels * mArraySize, mState.global);
    }

    template<typename ViewClass>
    using CreateFuncType = std::function<typename ViewClass::SharedPtr(Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)>;

    template<typename ViewClass, typename ViewMapType>
    typename ViewClass::SharedPtr findViewCommon(Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize, ViewMapType& viewMap, CreateFuncType<ViewClass> createFunc)
    {
        uint32_t resMipCount = 1;
        uint32_t resArraySize = 1;

        resArraySize = pTexture->getArraySize();
        resMipCount = pTexture->getMipCount();

        if (firstArraySlice >= resArraySize)
        {
            logWarning("First array slice is OOB when creating resource view. Clamping");
            firstArraySlice = resArraySize - 1;
        }

        if (mostDetailedMip >= resMipCount)
        {
            logWarning("Most detailed mip is OOB when creating resource view. Clamping");
            mostDetailedMip = resMipCount - 1;
        }

        if (mipCount == Resource::kMaxPossible)
        {
            mipCount = resMipCount - mostDetailedMip;
        }
        else if (mipCount + mostDetailedMip > resMipCount)
        {
            logWarning("Mip count is OOB when creating resource view. Clamping");
            mipCount = resMipCount - mostDetailedMip;
        }

        if (arraySize == Resource::kMaxPossible)
        {
            arraySize = resArraySize - firstArraySlice;
        }
        else if (arraySize + firstArraySlice > resArraySize)
        {
            logWarning("Array size is OOB when creating resource view. Clamping");
            arraySize = resArraySize - firstArraySlice;
        }

        ResourceViewInfo view = ResourceViewInfo(mostDetailedMip, mipCount, firstArraySlice, arraySize);

        if (viewMap.find(view) == viewMap.end())
        {
            viewMap[view] = createFunc(pTexture, mostDetailedMip, mipCount, firstArraySlice, arraySize);
        }

        return viewMap[view];
    }

    DepthStencilView::SharedPtr Texture::getDSV(uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        auto createFunc = [](Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
        {
            return DepthStencilView::create(std::static_pointer_cast<Texture>(pTexture->shared_from_this()), mostDetailedMip, firstArraySlice, arraySize);
        };

        return findViewCommon<DepthStencilView>(this, mipLevel, 1, firstArraySlice, arraySize, mDsvs, createFunc);
    }

    UnorderedAccessView::SharedPtr Texture::getUAV(uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        auto createFunc = [](Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
        {
            return UnorderedAccessView::create(std::static_pointer_cast<Texture>(pTexture->shared_from_this()), mostDetailedMip, firstArraySlice, arraySize);
        };

        return findViewCommon<UnorderedAccessView>(this, mipLevel, 1, firstArraySlice, arraySize, mUavs, createFunc);
    }

    ShaderResourceView::SharedPtr Texture::getSRV()
    {
        return getSRV(0);
    }

    UnorderedAccessView::SharedPtr Texture::getUAV()
    {
        return getUAV(0);
    }

#if _ENABLE_CUDA
    void* Texture::getCUDADeviceAddress() const
    {
        throw std::exception("Texture::getCUDADeviceAddress() - unimplemented");
    }

    void* Texture::getCUDADeviceAddress(ResourceViewInfo const& viewInfo) const
    {
        throw std::exception("Texture::getCUDADeviceAddress() - unimplemented");
    }
#endif

    RenderTargetView::SharedPtr Texture::getRTV(uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        auto createFunc = [](Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
        {
            return RenderTargetView::create(std::static_pointer_cast<Texture>(pTexture->shared_from_this()), mostDetailedMip, firstArraySlice, arraySize);
        };

        return findViewCommon<RenderTargetView>(this, mipLevel, 1, firstArraySlice, arraySize, mRtvs, createFunc);
    }

    ShaderResourceView::SharedPtr Texture::getSRV(uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
    {
        auto createFunc = [](Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
        {
            return ShaderResourceView::create(std::static_pointer_cast<Texture>(pTexture->shared_from_this()), mostDetailedMip, mipCount, firstArraySlice, arraySize);
        };

        return findViewCommon<ShaderResourceView>(this, mostDetailedMip, mipCount, firstArraySlice, arraySize, mSrvs, createFunc);
    }

    void Texture::captureToFile(uint32_t mipLevel, uint32_t arraySlice, const std::string& filename, Bitmap::FileFormat format, Bitmap::ExportFlags exportFlags)
    {
        if (format == Bitmap::FileFormat::DdsFile)
        {
            throw std::exception("Texture::captureToFile does not yet support saving to DDS.");
        }

        assert(mType == Type::Texture2D);
        RenderContext* pContext = gpDevice->getRenderContext();
        // Handle the special case where we have an HDR texture with less then 3 channels
        FormatType type = getFormatType(mFormat);
        uint32_t channels = getFormatChannelCount(mFormat);
        std::vector<uint8_t> textureData;
        ResourceFormat resourceFormat = mFormat;

        if (type == FormatType::Float && channels < 3)
        {
            Texture::SharedPtr pOther = Texture::create2D(getWidth(mipLevel), getHeight(mipLevel), ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);
            pContext->blit(getSRV(mipLevel, 1, arraySlice, 1), pOther->getRTV(0, 0, 1));
            textureData = pContext->readTextureSubresource(pOther.get(), 0);
            resourceFormat = ResourceFormat::RGBA32Float;
        }
        else
        {
            uint32_t subresource = getSubresourceIndex(arraySlice, mipLevel);
            textureData = pContext->readTextureSubresource(this, subresource);
        }

        uint32_t width = getWidth(mipLevel);
        uint32_t height = getHeight(mipLevel);
        auto func = [=]()
        {
            Bitmap::saveImage(filename, width, height, format, exportFlags, resourceFormat, true, (void*)textureData.data());
        };

        Threading::dispatchTask(func);
    }

    void Texture::uploadInitData(const void* pData, bool autoGenMips)
    {
        // TODO: This is a hack to allow multi-threaded texture loading using AsyncTextureLoader.
        // Replace with something better.
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        assert(gpDevice);
        auto pRenderContext = gpDevice->getRenderContext();
        if (autoGenMips)
        {
            // Upload just the first mip-level
            size_t arraySliceSize = mWidth * mHeight * getFormatBytesPerBlock(mFormat);
            const uint8_t* pSrc = (uint8_t*)pData;
            uint32_t numFaces = (mType == Texture::Type::TextureCube) ? 6 : 1;
            for (uint32_t i = 0; i < mArraySize * numFaces; i++)
            {
                uint32_t subresource = getSubresourceIndex(i, 0);
                pRenderContext->updateSubresourceData(this, subresource, pSrc);
                pSrc += arraySliceSize;
            }
        }
        else
        {
            pRenderContext->updateTextureData(this, pData);
        }

        if (autoGenMips)
        {
            generateMips(gpDevice->getRenderContext());
            invalidateViews();
        }
    }

    void Texture::generateMips(RenderContext* pContext)
    {
        if (mType != Type::Texture2D)
        {
            logWarning("Texture::generateMips() was only tested with Texture2Ds");
        }
        // #OPTME: should blit support arrays?
        for (uint32_t m = 0; m < mMipLevels - 1; m++)
        {
            for(uint32_t a = 0 ; a < mArraySize ; a++)
            {
                auto srv = getSRV(m, 1, a, 1);
                auto rtv = getRTV(m + 1, a, 1);
                pContext->blit(srv, rtv);
            }
        }

        if (mReleaseRtvsAfterGenMips)
        {
            // Releasing RTVs to free space on the heap.
            // We only do it once to handle the case that generateMips() was called during load.
            // If it was called more then once, the texture is probably dynamic and it's better to keep the RTVs around
            mRtvs.clear();
            mReleaseRtvsAfterGenMips = false;
        }
    }

    uint64_t Texture::getTexelCount() const
    {
        uint64_t count = 0;
        for (uint32_t i = 0; i < getMipCount(); i++)
        {
            uint64_t texelsInMip = (uint64_t)getWidth(i) * getHeight(i) * getDepth(i);
            assert(texelsInMip > 0);
            count += texelsInMip;
        }
        count *= getArraySize();
        assert(count > 0);
        return count;
    }

    uint64_t Texture::getTextureSizeInBytes() const
    {
        ID3D12DevicePtr pDevicePtr = gpDevice->getApiHandle();
        ID3D12ResourcePtr pTexResource = this->getApiHandle();

        D3D12_RESOURCE_ALLOCATION_INFO d3d12ResourceAllocationInfo;
        D3D12_RESOURCE_DESC desc = pTexResource->GetDesc();

        assert(desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D);

        d3d12ResourceAllocationInfo = pDevicePtr->GetResourceAllocationInfo(0, 1, &desc);
        assert(d3d12ResourceAllocationInfo.SizeInBytes > 0);
        return d3d12ResourceAllocationInfo.SizeInBytes;
    }

    SCRIPT_BINDING(Texture)
    {
        pybind11::class_<Texture, Texture::SharedPtr> texture(m, "Texture");
        texture.def_property_readonly("width", &Texture::getWidth);
        texture.def_property_readonly("height", &Texture::getHeight);
        texture.def_property_readonly("depth", &Texture::getDepth);
        texture.def_property_readonly("mipCount", &Texture::getMipCount);
        texture.def_property_readonly("arraySize", &Texture::getArraySize);
        texture.def_property_readonly("samples", &Texture::getSampleCount);
        texture.def_property_readonly("format", &Texture::getFormat);

        auto data = [](Texture* pTexture, uint32_t subresource)
        {
            return gpDevice->getRenderContext()->readTextureSubresource(pTexture, subresource);
        };
        texture.def("data", data, "subresource"_a);
    }
}
