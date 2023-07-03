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
#include "Texture.h"
#include "Device.h"
#include "Formats.h"
#include "RenderContext.h"
#include "GFXHelpers.h"
#include "GFXAPI.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/ObjectPython.h"
#include "Utils/Logger.h"
#include "Utils/Threading.h"
#include "Utils/Math/Common.h"
#include "Utils/Image/ImageIO.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Core/Pass/FullScreenPass.h"

#include <pybind11/numpy.h>

#include <mutex>

namespace Falcor
{
namespace
{
static constexpr bool kTopDown = true; // Memory layout when loading from file

Texture::BindFlags updateBindFlags(
    ref<Device> pDevice,
    Texture::BindFlags flags,
    bool hasInitData,
    uint32_t mipLevels,
    ResourceFormat format,
    const std::string& texType
)
{
    if ((mipLevels == Texture::kMaxPossible) && hasInitData)
    {
        flags |= Texture::BindFlags::RenderTarget;
    }

    Texture::BindFlags supported = pDevice->getFormatBindFlags(format);
    supported |= ResourceBindFlags::Shared;
    if ((flags & supported) != flags)
    {
        throw RuntimeError(
            "Error when creating {} of format {}. The requested bind-flags are not supported. Requested = ({}), supported = ({}).", texType,
            to_string(format), to_string(flags), to_string(supported)
        );
        flags = flags & supported;
    }

    return flags;
}
} // namespace

ref<Texture> Texture::createFromResource(
    ref<Device> pDevice,
    gfx::ITextureResource* pResource,
    Type type,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    ResourceFormat format,
    uint32_t sampleCount,
    uint32_t arraySize,
    uint32_t mipLevels,
    State initState,
    BindFlags bindFlags
)
{
    FALCOR_ASSERT(pResource);
    switch (type)
    {
    case Resource::Type::Texture1D:
        FALCOR_ASSERT(height == 1 && depth == 1 && sampleCount == 1);
        break;
    case Resource::Type::Texture2D:
        FALCOR_ASSERT(depth == 1 && sampleCount == 1);
        break;
    case Resource::Type::Texture2DMultisample:
        FALCOR_ASSERT(depth == 1);
        break;
    case Resource::Type::Texture3D:
        FALCOR_ASSERT(sampleCount == 1);
        break;
    case Resource::Type::TextureCube:
        FALCOR_ASSERT(depth == 1 && sampleCount == 1);
        break;
    default:
        FALCOR_UNREACHABLE();
        break;
    }
    ref<Texture> pTexture =
        ref<Texture>(new Texture(pDevice, width, height, depth, arraySize, mipLevels, sampleCount, format, type, bindFlags));
    pTexture->mGfxTextureResource = pResource;
    pTexture->mState.global = initState;
    pTexture->mState.isGlobal = true;
    return pTexture;
}

ref<Texture> Texture::create1D(
    ref<Device> pDevice,
    uint32_t width,
    ResourceFormat format,
    uint32_t arraySize,
    uint32_t mipLevels,
    const void* pData,
    BindFlags bindFlags
)
{
    bindFlags = updateBindFlags(pDevice, bindFlags, pData != nullptr, mipLevels, format, "Texture1D");
    ref<Texture> pTexture = ref<Texture>(new Texture(pDevice, width, 1, 1, arraySize, mipLevels, 1, format, Type::Texture1D, bindFlags));
    pTexture->apiInit(pData, (mipLevels == kMaxPossible));
    return pTexture;
}

ref<Texture> Texture::create2D(
    ref<Device> pDevice,
    uint32_t width,
    uint32_t height,
    ResourceFormat format,
    uint32_t arraySize,
    uint32_t mipLevels,
    const void* pData,
    BindFlags bindFlags
)
{
    bindFlags = updateBindFlags(pDevice, bindFlags, pData != nullptr, mipLevels, format, "Texture2D");
    ref<Texture> pTexture =
        ref<Texture>(new Texture(pDevice, width, height, 1, arraySize, mipLevels, 1, format, Type::Texture2D, bindFlags));
    pTexture->apiInit(pData, (mipLevels == kMaxPossible));
    return pTexture;
}

ref<Texture> Texture::create3D(
    ref<Device> pDevice,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    ResourceFormat format,
    uint32_t mipLevels,
    const void* pData,
    BindFlags bindFlags,
    bool isSparse
)
{
    bindFlags = updateBindFlags(pDevice, bindFlags, pData != nullptr, mipLevels, format, "Texture3D");
    ref<Texture> pTexture = ref<Texture>(new Texture(pDevice, width, height, depth, 1, mipLevels, 1, format, Type::Texture3D, bindFlags));
    pTexture->apiInit(pData, (mipLevels == kMaxPossible));
    return pTexture;
}

ref<Texture> Texture::createCube(
    ref<Device> pDevice,
    uint32_t width,
    uint32_t height,
    ResourceFormat format,
    uint32_t arraySize,
    uint32_t mipLevels,
    const void* pData,
    BindFlags bindFlags
)
{
    bindFlags = updateBindFlags(pDevice, bindFlags, pData != nullptr, mipLevels, format, "TextureCube");
    ref<Texture> pTexture =
        ref<Texture>(new Texture(pDevice, width, height, 1, arraySize, mipLevels, 1, format, Type::TextureCube, bindFlags));
    pTexture->apiInit(pData, (mipLevels == kMaxPossible));
    return pTexture;
}

ref<Texture> Texture::create2DMS(
    ref<Device> pDevice,
    uint32_t width,
    uint32_t height,
    ResourceFormat format,
    uint32_t sampleCount,
    uint32_t arraySize,
    BindFlags bindFlags
)
{
    bindFlags = updateBindFlags(pDevice, bindFlags, false, 1, format, "Texture2DMultisample");
    ref<Texture> pTexture =
        ref<Texture>(new Texture(pDevice, width, height, 1, arraySize, 1, sampleCount, format, Type::Texture2DMultisample, bindFlags));
    pTexture->apiInit(nullptr, false);
    return pTexture;
}

ref<Texture> Texture::createMippedFromFiles(
    ref<Device> pDevice,
    fstd::span<const std::filesystem::path> paths,
    bool loadAsSrgb,
    Texture::BindFlags bindFlags
)
{
    std::vector<Bitmap::UniqueConstPtr> mips;
    mips.reserve(paths.size());
    size_t combinedSize = 0;
    std::filesystem::path fullPathMip0;

    for (const auto& path : paths)
    {
        Bitmap::UniqueConstPtr pBitmap;
        if (hasExtension(path, "dds"))
        {
            pBitmap = ImageIO::loadBitmapFromDDS(path);
        }
        else
        {
            pBitmap = Bitmap::createFromFile(path, kTopDown);
        }
        if (!pBitmap)
        {
            logWarning("Error loading mip {}. Loading failed for image file '{}'.", mips.size(), path);
            break;
        }

        if (!mips.empty())
        {
            if (mips.back()->getFormat() != pBitmap->getFormat())
            {
                logWarning("Error loading mip {} from file {}. Texture format of all mip levels must match.", mips.size(), path);
                break;
            }
            if (std::max(mips.back()->getWidth() / 2, 1u) != pBitmap->getWidth() ||
                std::max(mips.back()->getHeight() / 2, 1u) != pBitmap->getHeight())
            {
                logWarning(
                    "Error loading mip {} from file {}. Image resolution must decrease by half. ({}, {}) != ({}, {})/2", mips.size(), path,
                    pBitmap->getWidth(), pBitmap->getHeight(), mips.back()->getWidth(), mips.back()->getHeight()
                );
                break;
            }
        }
        else
        {
            fullPathMip0 = path;
        }
        combinedSize += pBitmap->getSize();
        mips.emplace_back(std::move(pBitmap));
    }

    ref<Texture> pTex;
    if (!mips.empty())
    {
        // Combine all the mip data into a single buffer
        size_t copyDst = 0;
        std::unique_ptr<uint8_t[]> combinedData(new uint8_t[combinedSize]);
        for (auto& mip : mips)
        {
            std::memcpy(&combinedData[copyDst], mip->getData(), mip->getSize());
            copyDst += mip->getSize();
        }

        ResourceFormat texFormat = mips[0]->getFormat();
        if (loadAsSrgb)
            texFormat = linearToSrgbFormat(texFormat);

        // Create mip mapped latent texture
        pTex =
            Texture::create2D(pDevice, mips[0]->getWidth(), mips[0]->getHeight(), texFormat, 1, mips.size(), combinedData.get(), bindFlags);
    }

    if (pTex != nullptr)
    {
        pTex->setSourcePath(fullPathMip0);

        // Log debug info.
        std::string str = fmt::format(
            "Loaded texture: size={}x{} mips={} format={} path={}", pTex->getWidth(), pTex->getHeight(), pTex->getMipCount(),
            to_string(pTex->getFormat()), fullPathMip0
        );
        logDebug(str);
    }

    return pTex;
}

ref<Texture> Texture::createFromFile(
    ref<Device> pDevice,
    const std::filesystem::path& path,
    bool generateMipLevels,
    bool loadAsSrgb,
    Texture::BindFlags bindFlags
)
{
    std::filesystem::path fullPath;
    if (!findFileInDataDirectories(path, fullPath))
    {
        logWarning("Error when loading image file. Can't find image file '{}'.", path);
        return nullptr;
    }

    ref<Texture> pTex;
    if (hasExtension(fullPath, "dds"))
    {
        try
        {
            pTex = ImageIO::loadTextureFromDDS(pDevice, fullPath, loadAsSrgb);
        }
        catch (const std::exception& e)
        {
            logWarning("Error loading '{}': {}", fullPath, e.what());
        }
    }
    else
    {
        Bitmap::UniqueConstPtr pBitmap = Bitmap::createFromFile(fullPath, kTopDown);
        if (pBitmap)
        {
            ResourceFormat texFormat = pBitmap->getFormat();
            if (loadAsSrgb)
            {
                texFormat = linearToSrgbFormat(texFormat);
            }

            pTex = Texture::create2D(
                pDevice, pBitmap->getWidth(), pBitmap->getHeight(), texFormat, 1, generateMipLevels ? Texture::kMaxPossible : 1,
                pBitmap->getData(), bindFlags
            );
        }
    }

    if (pTex != nullptr)
    {
        pTex->setSourcePath(fullPath);

        // Log debug info.
        std::string str = fmt::format(
            "Loaded texture: size={}x{} mips={} format={} path={}", pTex->getWidth(), pTex->getHeight(), pTex->getMipCount(),
            to_string(pTex->getFormat()), fullPath
        );
        logDebug(str);
    }

    return pTex;
}

Texture::Texture(
    ref<Device> pDevice,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    uint32_t arraySize,
    uint32_t mipLevels,
    uint32_t sampleCount,
    ResourceFormat format,
    Type type,
    BindFlags bindFlags
)
    : Resource(pDevice, type, bindFlags, 0)
    , mWidth(width)
    , mHeight(height)
    , mDepth(depth)
    , mMipLevels(mipLevels)
    , mSampleCount(sampleCount)
    , mArraySize(arraySize)
    , mFormat(format)
{
    FALCOR_ASSERT(width > 0 && height > 0 && depth > 0);
    FALCOR_ASSERT(arraySize > 0 && mipLevels > 0 && sampleCount > 0);
    FALCOR_ASSERT(format != ResourceFormat::Unknown);

    if (mMipLevels == kMaxPossible)
    {
        uint32_t dims = width | height | depth;
        mMipLevels = bitScanReverse(dims) + 1;
    }
    mState.perSubresource.resize(mMipLevels * mArraySize, mState.global);
}

gfx::IResource* Texture::getGfxResource() const
{
    return mGfxTextureResource;
}

template<typename ViewClass>
using CreateFuncType = std::function<
    ref<ViewClass>(Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)>;

template<typename ViewClass, typename ViewMapType>
ref<ViewClass> findViewCommon(
    Texture* pTexture,
    uint32_t mostDetailedMip,
    uint32_t mipCount,
    uint32_t firstArraySlice,
    uint32_t arraySize,
    ViewMapType& viewMap,
    CreateFuncType<ViewClass> createFunc
)
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

ref<DepthStencilView> Texture::getDSV(uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
{
    auto createFunc = [](Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
    { return DepthStencilView::create(pTexture->getDevice().get(), pTexture, mostDetailedMip, firstArraySlice, arraySize); };

    return findViewCommon<DepthStencilView>(this, mipLevel, 1, firstArraySlice, arraySize, mDsvs, createFunc);
}

ref<UnorderedAccessView> Texture::getUAV(uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
{
    auto createFunc = [](Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
    { return UnorderedAccessView::create(pTexture->getDevice().get(), pTexture, mostDetailedMip, firstArraySlice, arraySize); };

    return findViewCommon<UnorderedAccessView>(this, mipLevel, 1, firstArraySlice, arraySize, mUavs, createFunc);
}

ref<ShaderResourceView> Texture::getSRV()
{
    return getSRV(0);
}

ref<UnorderedAccessView> Texture::getUAV()
{
    return getUAV(0);
}

ref<RenderTargetView> Texture::getRTV(uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
{
    auto createFunc = [](Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
    { return RenderTargetView::create(pTexture->getDevice().get(), pTexture, mostDetailedMip, firstArraySlice, arraySize); };

    return findViewCommon<RenderTargetView>(this, mipLevel, 1, firstArraySlice, arraySize, mRtvs, createFunc);
}

ref<ShaderResourceView> Texture::getSRV(uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
{
    auto createFunc = [](Texture* pTexture, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
    { return ShaderResourceView::create(pTexture->getDevice().get(), pTexture, mostDetailedMip, mipCount, firstArraySlice, arraySize); };

    return findViewCommon<ShaderResourceView>(this, mostDetailedMip, mipCount, firstArraySlice, arraySize, mSrvs, createFunc);
}

void Texture::captureToFile(
    uint32_t mipLevel,
    uint32_t arraySlice,
    const std::filesystem::path& path,
    Bitmap::FileFormat format,
    Bitmap::ExportFlags exportFlags,
    bool async
)
{
    if (format == Bitmap::FileFormat::DdsFile)
    {
        throw RuntimeError("Texture::captureToFile does not yet support saving to DDS.");
    }

    if (mType != Type::Texture2D)
        throw RuntimeError("Texture::captureToFile only supported for 2D textures.");

    RenderContext* pContext = mpDevice->getRenderContext();

    // Handle the special case where we have an HDR texture with less then 3 channels.
    FormatType type = getFormatType(mFormat);
    uint32_t channels = getFormatChannelCount(mFormat);
    std::vector<uint8_t> textureData;
    ResourceFormat resourceFormat = mFormat;

    if (type == FormatType::Float && channels < 3)
    {
        ref<Texture> pOther = Texture::create2D(
            mpDevice, getWidth(mipLevel), getHeight(mipLevel), ResourceFormat::RGBA32Float, 1, 1, nullptr,
            ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
        );
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

    auto func = [=]() { Bitmap::saveImage(path, width, height, format, exportFlags, resourceFormat, true, (void*)textureData.data()); };

    if (async)
        Threading::dispatchTask(func);
    else
        func();
}

void Texture::uploadInitData(RenderContext* pRenderContext, const void* pData, bool autoGenMips)
{
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
        generateMips(pRenderContext);
        invalidateViews();
    }
}

void Texture::generateMips(RenderContext* pContext, bool minMaxMips)
{
    if (mType != Type::Texture2D)
    {
        logWarning("Texture::generateMips() was only tested with Texture2Ds");
    }

    // #OPTME: should blit support arrays?
    for (uint32_t m = 0; m < mMipLevels - 1; m++)
    {
        for (uint32_t a = 0; a < mArraySize; a++)
        {
            auto srv = getSRV(m, 1, a, 1);
            auto rtv = getRTV(m + 1, a, 1);
            if (!minMaxMips)
            {
                pContext->blit(srv, rtv, RenderContext::kMaxRect, RenderContext::kMaxRect, Sampler::Filter::Linear);
            }
            else
            {
                const Sampler::ReductionMode redModes[] = {
                    Sampler::ReductionMode::Standard, Sampler::ReductionMode::Min, Sampler::ReductionMode::Max,
                    Sampler::ReductionMode::Standard};
                const float4 componentsTransform[] = {
                    float4(1.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 1.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 1.0f, 0.0f),
                    float4(0.0f, 0.0f, 0.0f, 1.0f)};
                pContext->blit(
                    srv, rtv, RenderContext::kMaxRect, RenderContext::kMaxRect, Sampler::Filter::Linear, redModes, componentsTransform
                );
            }
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
        FALCOR_ASSERT(texelsInMip > 0);
        count += texelsInMip;
    }
    count *= getArraySize();
    FALCOR_ASSERT(count > 0);
    return count;
}

bool Texture::compareDesc(const Texture* pOther) const
{
    return mWidth == pOther->mWidth && mHeight == pOther->mHeight && mDepth == pOther->mDepth && mMipLevels == pOther->mMipLevels &&
           mSampleCount == pOther->mSampleCount && mArraySize == pOther->mArraySize && mFormat == pOther->mFormat &&
           mIsSparse == pOther->mIsSparse && all(mSparsePageRes == pOther->mSparsePageRes);
}

gfx::IResource::Type getResourceType(Texture::Type type)
{
    switch (type)
    {
    case Texture::Type::Texture1D:
        return gfx::IResource::Type::Texture1D;
    case Texture::Type::Texture2D:
    case Texture::Type::Texture2DMultisample:
        return gfx::IResource::Type::Texture2D;
    case Texture::Type::TextureCube:
        return gfx::IResource::Type::TextureCube;
    case Texture::Type::Texture3D:
        return gfx::IResource::Type::Texture3D;
    default:
        FALCOR_UNREACHABLE();
        return gfx::IResource::Type::Unknown;
    }
}

uint64_t Texture::getTextureSizeInBytes() const
{
    // get allocation info for resource description
    size_t outSizeBytes = 0, outAlignment = 0;

    gfx::ITextureResource::Desc* desc = mGfxTextureResource->getDesc();

    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->getTextureAllocationInfo(*desc, &outSizeBytes, &outAlignment));
    FALCOR_ASSERT(outSizeBytes > 0);

    return outSizeBytes;
}

void Texture::apiInit(const void* pData, bool autoGenMips)
{
    // WARNING: This is a hack to allow parallel texture loading in TextureManager.
    std::lock_guard<std::mutex> lock(mpDevice->getGlobalGfxMutex());

    // create resource description
    gfx::ITextureResource::Desc desc = {};

    // base description

    // type
    desc.type = getResourceType(mType); // same as resource dimension in D3D12

    // default state and allowed states
    gfx::ResourceState defaultState;
    getGFXResourceState(mBindFlags, defaultState, desc.allowedStates);

    // Always set texture to general(common) state upon creation.
    desc.defaultState = gfx::ResourceState::General;

    desc.memoryType = gfx::MemoryType::DeviceLocal;
    // texture resource specific description attributes

    // size
    desc.size.width = align_to(getFormatWidthCompressionRatio(mFormat), mWidth);
    desc.size.height = align_to(getFormatHeightCompressionRatio(mFormat), mHeight);
    desc.size.depth = mDepth; // relevant for Texture3D

    // array size
    if (mType == Texture::Type::TextureCube)
    {
        desc.arraySize = mArraySize * 6;
    }
    else
    {
        desc.arraySize = mArraySize;
    }

    // mip map levels
    desc.numMipLevels = mMipLevels;

    // format
    desc.format = getGFXFormat(mFormat); // lookup can result in Unknown / unsupported format

    // sample description
    desc.sampleDesc.numSamples = mSampleCount;
    desc.sampleDesc.quality = 0;

    // clear value
    gfx::ClearValue clearValue;
    if ((mBindFlags & (Texture::BindFlags::RenderTarget | Texture::BindFlags::DepthStencil)) != Texture::BindFlags::None)
    {
        if ((mBindFlags & Texture::BindFlags::DepthStencil) != Texture::BindFlags::None)
        {
            clearValue.depthStencil.depth = 1.0f;
        }
        desc.optimalClearValue = &clearValue;
    }

    // shared resource
    if (is_set(mBindFlags, Resource::BindFlags::Shared))
    {
        desc.isShared = true;
    }

    // validate description
    FALCOR_ASSERT(desc.size.width > 0 && desc.size.height > 0);
    FALCOR_ASSERT(desc.numMipLevels > 0 && desc.size.depth > 0 && desc.arraySize > 0 && desc.sampleDesc.numSamples > 0);

    // create resource
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createTextureResource(desc, nullptr, mGfxTextureResource.writeRef()));
    FALCOR_ASSERT(mGfxTextureResource);

    // upload init data through texture class
    if (pData)
    {
        uploadInitData(mpDevice->getRenderContext(), pData, autoGenMips);
    }
}

Texture::~Texture()
{
    mpDevice->releaseResource(mGfxTextureResource);
}

/**
 * Python binding wrapper for returning the content of a texture as a numpy array.
 */
pybind11::array pyTextureGetImage(const Texture& texture, uint32_t mipLevel = 0, uint32_t arraySlice = 0)
{
    checkArgument(
        mipLevel < texture.getMipCount(), "'mipLevel' ({}) is out of bounds. Only {} level(s) available.", mipLevel, texture.getMipCount()
    );
    checkArgument(
        arraySlice < texture.getArraySize(), "'arraySlice' ({}) is out of bounds. Only {} slice(s) available.", arraySlice,
        texture.getArraySize()
    );
    checkArgument(texture.getSampleCount() == 1, "Texture is multi-sampled.");

    ResourceFormat format = texture.getFormat();
    checkArgument(isCompressedFormat(format) == false, "Texture uses a compressed format.");
    checkArgument(isStencilFormat(format) == false, "Texture uses a stencil format.");

    FormatType formatType = getFormatType(format);
    checkArgument(formatType != FormatType::Unknown, "Texture uses an unknown pixel format.");

    uint32_t channelCount = getFormatChannelCount(format);
    checkArgument(channelCount > 0, "Texture has zero channels.");

    // Make sure all channels have the same number of bits.
    uint32_t channelBits = getNumChannelBits(format, 0);
    for (uint32_t i = 1; i < channelCount; ++i)
        checkArgument(getNumChannelBits(format, i) == channelBits, "Texture uses different bit depths per channel.");
    checkArgument(
        channelBits == 8 || channelBits == 16 || channelBits == 32, "Texture uses bit depth {}. Only 8, 16 and 32 are supported.",
        channelBits
    );

    // Generate numpy dtype.
    std::string dtype;
    switch (formatType)
    {
    case FormatType::Float:
        dtype = "float";
        break;
    case FormatType::Sint:
    case FormatType::Snorm:
        dtype = "int";
        break;
    case FormatType::Uint:
    case FormatType::Unorm:
    case FormatType::UnormSrgb:
        dtype = "uint";
        break;
    }
    dtype += std::to_string(channelBits);

    // Get image dimensions.
    uint32_t width = texture.getWidth(mipLevel);
    uint32_t height = texture.getHeight(mipLevel);
    uint32_t depth = texture.getDepth(mipLevel);

    // Generate numpy array shape.
    std::vector<pybind11::ssize_t> shape{width, channelCount};
    if (height > 1)
        shape.insert(shape.begin(), height);
    if (depth > 1)
        shape.insert(shape.begin(), depth);

    pybind11::array result(pybind11::dtype(dtype), shape);
    // TODO: We should add support for writing directly to a prepared buffer.
    uint32_t subresource = texture.getSubresourceIndex(arraySlice, mipLevel);
    auto data = texture.getDevice()->getRenderContext()->readTextureSubresource(&texture, mipLevel);
    auto request = result.request();
    FALCOR_ASSERT_EQ(data.size(), request.size * request.itemsize);
    std::memcpy(request.ptr, data.data(), data.size());

    return result;
}

/**
 * Python binding wrapper for returning the content of texture as raw memory.
 */
std::vector<uint8_t> pyTextureGetData(const Texture& texture, uint32_t mipLevel = 0, uint32_t arraySlice = 0)
{
    checkArgument(
        mipLevel < texture.getMipCount(), "'mipLevel' ({}) is out of bounds. Only {} level(s) available.", mipLevel, texture.getMipCount()
    );
    checkArgument(
        arraySlice < texture.getArraySize(), "'arraySlice' ({}) is out of bounds. Only {} slice(s) available.", arraySlice,
        texture.getArraySize()
    );

    uint32_t subresource = texture.getSubresourceIndex(arraySlice, mipLevel);
    return texture.getDevice()->getRenderContext()->readTextureSubresource(&texture, subresource);
}

FALCOR_SCRIPT_BINDING(Texture)
{
    using namespace pybind11::literals;

    FALCOR_SCRIPT_BINDING_DEPENDENCY(Resource)

    pybind11::class_<Texture, Resource, ref<Texture>> texture(m, "Texture");
    texture.def_property_readonly("width", [](const Texture& texture) { return texture.getWidth(); });
    texture.def_property_readonly("height", [](const Texture& texture) { return texture.getHeight(); });
    texture.def_property_readonly("depth", [](const Texture& texture) { return texture.getDepth(); });
    texture.def_property_readonly("mip_count", &Texture::getMipCount);
    texture.def_property_readonly("array_size", &Texture::getArraySize);
    texture.def_property_readonly("samples", &Texture::getSampleCount);
    texture.def_property_readonly("format", &Texture::getFormat);

    texture.def("get_image", pyTextureGetImage, "mip_level"_a = 0, "array_slice"_a = 0);
    texture.def("get_data", pyTextureGetData, "mip_level"_a = 0, "array_slice"_a = 0);

    // PYTHONDEPRECATED BEGIN
    texture.def("getImage", pyTextureGetImage, "mipLevel"_a = 0, "arraySlice"_a = 0);
    texture.def("getData", pyTextureGetData, "mipLevel"_a = 0, "arraySlice"_a = 0);
    // PYTHONDEPRECATED END
}
} // namespace Falcor
