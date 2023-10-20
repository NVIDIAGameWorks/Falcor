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
#include "PythonHelpers.h"
#include "Core/Error.h"
#include "Core/ObjectPython.h"
#include "Utils/Logger.h"
#include "Utils/Threading.h"
#include "Utils/Math/Common.h"
#include "Utils/Image/ImageIO.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Scripting/ndarray.h"
#include "Core/Pass/FullScreenPass.h"

#include <mutex>

namespace Falcor
{
namespace
{
static constexpr bool kTopDown = true; // Memory layout when loading from file

gfx::IResource::Type getGfxResourceType(Texture::Type type)
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

} // namespace

Texture::Texture(
    ref<Device> pDevice,
    Type type,
    ResourceFormat format,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    uint32_t arraySize,
    uint32_t mipLevels,
    uint32_t sampleCount,
    ResourceBindFlags bindFlags,
    const void* pInitData
)
    : Resource(std::move(pDevice), type, bindFlags, 0)
    , mFormat(format)
    , mWidth(width)
    , mHeight(height)
    , mDepth(depth)
    , mMipLevels(mipLevels)
    , mArraySize(arraySize)
    , mSampleCount(sampleCount)
{
    FALCOR_ASSERT(mType != Type::Buffer);
    FALCOR_ASSERT(mFormat != ResourceFormat::Unknown);
    FALCOR_ASSERT(mWidth > 0 && mHeight > 0 && mDepth > 0);
    switch (mType)
    {
    case Resource::Type::Texture1D:
        FALCOR_ASSERT(mHeight == 1 && mDepth == 1 && mSampleCount == 1);
        break;
    case Resource::Type::Texture2D:
        FALCOR_ASSERT(mDepth == 1 && mSampleCount == 1);
        break;
    case Resource::Type::Texture2DMultisample:
        FALCOR_ASSERT(mDepth == 1);
        break;
    case Resource::Type::Texture3D:
        FALCOR_ASSERT(mSampleCount == 1);
        break;
    case Resource::Type::TextureCube:
        FALCOR_ASSERT(mDepth == 1 && mSampleCount == 1);
        break;
    default:
        FALCOR_UNREACHABLE();
        break;
    }

    FALCOR_ASSERT(mArraySize > 0 && mMipLevels > 0 && mSampleCount > 0);

    bool autoGenerateMips = pInitData && (mMipLevels == Texture::kMaxPossible);

    if (autoGenerateMips)
        mBindFlags |= ResourceBindFlags::RenderTarget;

    if (mMipLevels == kMaxPossible)
    {
        uint32_t dims = width | height | depth;
        mMipLevels = bitScanReverse(dims) + 1;
    }

    mState.perSubresource.resize(mMipLevels * mArraySize, mState.global);

    ResourceBindFlags supported = mpDevice->getFormatBindFlags(mFormat);
    supported |= ResourceBindFlags::Shared;
    if ((mBindFlags & supported) != mBindFlags)
    {
        FALCOR_THROW(
            "Error when creating {} of format {}. The requested bind-flags are not supported. Requested = ({}), supported = ({}).",
            to_string(mType),
            to_string(mFormat),
            to_string(mBindFlags),
            to_string(supported)
        );
    }

    gfx::ITextureResource::Desc desc = {};

    desc.type = getGfxResourceType(mType);

    // Default state and allowed states.
    gfx::ResourceState defaultState;
    getGFXResourceState(mBindFlags, defaultState, desc.allowedStates);

    // Always set texture to general(common) state upon creation.
    desc.defaultState = gfx::ResourceState::General;

    desc.memoryType = gfx::MemoryType::DeviceLocal;

    desc.size.width = align_to(getFormatWidthCompressionRatio(mFormat), mWidth);
    desc.size.height = align_to(getFormatHeightCompressionRatio(mFormat), mHeight);
    desc.size.depth = mDepth;

    desc.arraySize = mType == Texture::Type::TextureCube ? mArraySize * 6 : mArraySize;
    desc.numMipLevels = mMipLevels;

    desc.format = getGFXFormat(mFormat); // lookup can result in Unknown / unsupported format

    desc.sampleDesc.numSamples = mSampleCount;
    desc.sampleDesc.quality = 0;

    // Clear value.
    gfx::ClearValue clearValue;
    if ((mBindFlags & (ResourceBindFlags::RenderTarget | ResourceBindFlags::DepthStencil)) != ResourceBindFlags::None)
    {
        if ((mBindFlags & ResourceBindFlags::DepthStencil) != ResourceBindFlags::None)
        {
            clearValue.depthStencil.depth = 1.0f;
        }
        desc.optimalClearValue = &clearValue;
    }

    // Shared resource.
    if (is_set(mBindFlags, ResourceBindFlags::Shared))
    {
        desc.isShared = true;
    }

    // Validate description.
    FALCOR_ASSERT(desc.size.width > 0 && desc.size.height > 0);
    FALCOR_ASSERT(desc.numMipLevels > 0 && desc.size.depth > 0 && desc.arraySize > 0 && desc.sampleDesc.numSamples > 0);

    // Create & upload resource.
    {
        // WARNING: This is a hack to allow parallel texture loading in TextureManager.
        std::lock_guard<std::mutex> lock(mpDevice->getGlobalGfxMutex());

        FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createTextureResource(desc, nullptr, mGfxTextureResource.writeRef()));
        FALCOR_ASSERT(mGfxTextureResource);

        if (pInitData)
        {
            // Prevent the texture from being destroyed while uploading the data.
            incRef();
            uploadInitData(mpDevice->getRenderContext(), pInitData, autoGenerateMips);
            decRef(false);
        }
    }
}

Texture::Texture(
    ref<Device> pDevice,
    gfx::ITextureResource* pResource,
    Type type,
    ResourceFormat format,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    uint32_t arraySize,
    uint32_t mipLevels,
    uint32_t sampleCount,
    ResourceBindFlags bindFlags,
    Resource::State initState
)
    : Texture(std::move(pDevice), type, format, width, height, depth, arraySize, mipLevels, sampleCount, bindFlags, nullptr)
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

    mGfxTextureResource = pResource;
    mState.global = initState;
    mState.isGlobal = true;
}

Texture::~Texture()
{
    mpDevice->releaseResource(mGfxTextureResource);
}

ref<Texture> Texture::createMippedFromFiles(
    ref<Device> pDevice,
    fstd::span<const std::filesystem::path> paths,
    bool loadAsSrgb,
    ResourceBindFlags bindFlags
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
                    "Error loading mip {} from file {}. Image resolution must decrease by half. ({}, {}) != ({}, {})/2",
                    mips.size(),
                    path,
                    pBitmap->getWidth(),
                    pBitmap->getHeight(),
                    mips.back()->getWidth(),
                    mips.back()->getHeight()
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
            pDevice->createTexture2D(mips[0]->getWidth(), mips[0]->getHeight(), texFormat, 1, mips.size(), combinedData.get(), bindFlags);
    }

    if (pTex != nullptr)
    {
        pTex->setSourcePath(fullPathMip0);

        // Log debug info.
        std::string str = fmt::format(
            "Loaded texture: size={}x{} mips={} format={} path={}",
            pTex->getWidth(),
            pTex->getHeight(),
            pTex->getMipCount(),
            to_string(pTex->getFormat()),
            fullPathMip0
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
    ResourceBindFlags bindFlags
)
{
    if (!std::filesystem::exists(path))
    {
        logWarning("Error when loading image file. File '{}' does not exist.", path);
        return nullptr;
    }

    ref<Texture> pTex;
    if (hasExtension(path, "dds"))
    {
        try
        {
            pTex = ImageIO::loadTextureFromDDS(pDevice, path, loadAsSrgb);
        }
        catch (const std::exception& e)
        {
            logWarning("Error loading '{}': {}", path, e.what());
        }
    }
    else
    {
        Bitmap::UniqueConstPtr pBitmap = Bitmap::createFromFile(path, kTopDown);
        if (pBitmap)
        {
            ResourceFormat texFormat = pBitmap->getFormat();
            if (loadAsSrgb)
            {
                texFormat = linearToSrgbFormat(texFormat);
            }

            pTex = pDevice->createTexture2D(
                pBitmap->getWidth(),
                pBitmap->getHeight(),
                texFormat,
                1,
                generateMipLevels ? Texture::kMaxPossible : 1,
                pBitmap->getData(),
                bindFlags
            );
        }
    }

    if (pTex != nullptr)
    {
        pTex->setSourcePath(path);

        // Log debug info.
        std::string str = fmt::format(
            "Loaded texture: size={}x{} mips={} format={} path={}",
            pTex->getWidth(),
            pTex->getHeight(),
            pTex->getMipCount(),
            to_string(pTex->getFormat()),
            path
        );
        logDebug(str);
    }

    return pTex;
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

Texture::SubresourceLayout Texture::getSubresourceLayout(uint32_t subresource) const
{
    FALCOR_CHECK(subresource < getSubresourceCount(), "subresource out of range");

    gfx::ITextureResource* gfxTexture = getGfxTextureResource();
    gfx::FormatInfo gfxFormatInfo;
    FALCOR_GFX_CALL(gfx::gfxGetFormatInfo(gfxTexture->getDesc()->format, &gfxFormatInfo));

    SubresourceLayout layout;
    uint32_t mipLevel = getSubresourceMipLevel(subresource);
    layout.rowSize = div_round_up(getWidth(mipLevel), uint32_t(gfxFormatInfo.blockWidth)) * gfxFormatInfo.blockSizeInBytes;
    layout.rowSizeAligned = align_to(mpDevice->getTextureRowAlignment(), layout.rowSize);
    layout.rowCount = div_round_up(getHeight(mipLevel), uint32_t(gfxFormatInfo.blockHeight));
    layout.depth = getDepth();

    return layout;
}

void Texture::setSubresourceBlob(uint32_t subresource, const void* pData, size_t size)
{
    FALCOR_CHECK(subresource < getSubresourceCount(), "'subresource' ({}) is out of range ({})", subresource, getSubresourceCount());
    SubresourceLayout layout = getSubresourceLayout(subresource);
    FALCOR_CHECK(
        size == layout.getTotalByteSize(), "'size' ({}) does not match the subresource size ({})", size, layout.getTotalByteSize()
    );

    mpDevice->getRenderContext()->updateSubresourceData(this, subresource, pData);
}

void Texture::getSubresourceBlob(uint32_t subresource, void* pData, size_t size) const
{
    FALCOR_CHECK(subresource < getSubresourceCount(), "'subresource' ({}) is out of range ({})", subresource, getSubresourceCount());
    SubresourceLayout layout = getSubresourceLayout(subresource);
    FALCOR_CHECK(
        size == layout.getTotalByteSize(), "'size' ({}) does not match the subresource size ({})", size, layout.getTotalByteSize()
    );

    auto data = mpDevice->getRenderContext()->readTextureSubresource(this, subresource);
    FALCOR_ASSERT(data.size() == size);
    std::memcpy(pData, data.data(), data.size());
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
        FALCOR_THROW("Texture::captureToFile does not yet support saving to DDS.");
    }

    if (mType != Type::Texture2D)
        FALCOR_THROW("Texture::captureToFile only supported for 2D textures.");

    RenderContext* pContext = mpDevice->getRenderContext();

    // Handle the special case where we have an HDR texture with less then 3 channels.
    FormatType type = getFormatType(mFormat);
    uint32_t channels = getFormatChannelCount(mFormat);
    std::vector<uint8_t> textureData;
    ResourceFormat resourceFormat = mFormat;

    if (type == FormatType::Float && channels < 3)
    {
        ref<Texture> pOther = mpDevice->createTexture2D(
            getWidth(mipLevel),
            getHeight(mipLevel),
            ResourceFormat::RGBA32Float,
            1,
            1,
            nullptr,
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
                pContext->blit(srv, rtv, RenderContext::kMaxRect, RenderContext::kMaxRect, TextureFilteringMode::Linear);
            }
            else
            {
                const TextureReductionMode redModes[] = {
                    TextureReductionMode::Standard,
                    TextureReductionMode::Min,
                    TextureReductionMode::Max,
                    TextureReductionMode::Standard,
                };
                const float4 componentsTransform[] = {
                    float4(1.0f, 0.0f, 0.0f, 0.0f),
                    float4(0.0f, 1.0f, 0.0f, 0.0f),
                    float4(0.0f, 0.0f, 1.0f, 0.0f),
                    float4(0.0f, 0.0f, 0.0f, 1.0f),
                };
                pContext->blit(
                    srv, rtv, RenderContext::kMaxRect, RenderContext::kMaxRect, TextureFilteringMode::Linear, redModes, componentsTransform
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

uint64_t Texture::getTextureSizeInBytes() const
{
    // get allocation info for resource description
    size_t outSizeBytes = 0, outAlignment = 0;

    gfx::ITextureResource::Desc* desc = mGfxTextureResource->getDesc();

    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->getTextureAllocationInfo(*desc, &outSizeBytes, &outAlignment));
    FALCOR_ASSERT(outSizeBytes > 0);

    return outSizeBytes;
}

/**
 * Python binding wrapper for returning the content of a texture as a numpy array.
 */
inline pybind11::ndarray<pybind11::numpy> texture_to_numpy(const Texture& self, uint32_t mip_level, uint32_t array_slice)
{
    FALCOR_CHECK(
        mip_level < self.getMipCount(), "'mip_level' ({}) is out of bounds. Only {} level(s) available.", mip_level, self.getMipCount()
    );
    FALCOR_CHECK(
        array_slice < self.getArraySize(),
        "'array_slice' ({}) is out of bounds. Only {} slice(s) available.",
        array_slice,
        self.getArraySize()
    );

    // Get image dimensions.
    uint32_t width = self.getWidth(mip_level);
    uint32_t height = self.getHeight(mip_level);
    uint32_t depth = self.getDepth(mip_level);

    uint32_t subresource = self.getSubresourceIndex(array_slice, mip_level);
    Texture::SubresourceLayout layout = self.getSubresourceLayout(subresource);

    size_t subresourceSize = layout.getTotalByteSize();
    void* cpuData = new uint8_t[subresourceSize];
    self.getSubresourceBlob(subresource, cpuData, subresourceSize);

    pybind11::capsule owner(cpuData, [](void* p) noexcept { delete[] reinterpret_cast<uint8_t*>(p); });

    if (auto dtype = resourceFormatToDtype(self.getFormat()))
    {
        uint32_t channelCount = getFormatChannelCount(self.getFormat());
        std::vector<pybind11::size_t> shape;
        if (depth > 1)
            shape.push_back(depth);
        if (height > 1)
            shape.push_back(height);
        shape.push_back(width);
        if (channelCount > 1)
            shape.push_back(channelCount);
        return pybind11::ndarray<pybind11::numpy>(
            cpuData, shape.size(), shape.data(), owner, nullptr, *dtype, pybind11::device::cpu::value
        );
    }
    else
    {
        pybind11::size_t shape[1] = {subresourceSize};
        return pybind11::ndarray<pybind11::numpy>(
            cpuData, 1, shape, owner, nullptr, pybind11::dtype<uint8_t>(), pybind11::device::cpu::value
        );
    }
}

inline void texture_from_numpy(Texture& self, pybind11::ndarray<pybind11::numpy> data, uint32_t mip_level, uint32_t array_slice)
{
    FALCOR_CHECK(
        mip_level < self.getMipCount(), "'mip_level' ({}) is out of bounds. Only {} level(s) available.", mip_level, self.getMipCount()
    );
    FALCOR_CHECK(
        array_slice < self.getArraySize(),
        "'array_slice' ({}) is out of bounds. Only {} slice(s) available.",
        array_slice,
        self.getArraySize()
    );
    FALCOR_CHECK(isNdarrayContiguous(data), "numpy array is not contiguous");

    uint32_t subresource = self.getSubresourceIndex(array_slice, mip_level);
    Texture::SubresourceLayout layout = self.getSubresourceLayout(subresource);

    size_t subresourceSize = layout.getTotalByteSize();
    size_t dataSize = getNdarrayByteSize(data);
    FALCOR_CHECK(dataSize == subresourceSize, "numpy array is doesn't match the subresource size ({} != {})", dataSize, subresourceSize);

    self.setSubresourceBlob(subresource, data.data(), dataSize);
}

FALCOR_SCRIPT_BINDING(Texture)
{
    using namespace pybind11::literals;

    FALCOR_SCRIPT_BINDING_DEPENDENCY(Resource)

    pybind11::class_<Texture, Resource, ref<Texture>> texture(m, "Texture");
    texture.def_property_readonly("format", &Texture::getFormat);
    texture.def_property_readonly("width", [](const Texture& texture) { return texture.getWidth(); });
    texture.def_property_readonly("height", [](const Texture& texture) { return texture.getHeight(); });
    texture.def_property_readonly("depth", [](const Texture& texture) { return texture.getDepth(); });
    texture.def_property_readonly("mip_count", &Texture::getMipCount);
    texture.def_property_readonly("array_size", &Texture::getArraySize);
    texture.def_property_readonly("sample_count", &Texture::getSampleCount);

    texture.def("to_numpy", texture_to_numpy, "mip_level"_a = 0, "array_slice"_a = 0);
    texture.def("from_numpy", texture_from_numpy, "data"_a, "mip_level"_a = 0, "array_slice"_a = 0);
}
} // namespace Falcor
