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
#pragma once
#include "fwd.h"
#include "Handles.h"
#include "Formats.h"
#include "Resource.h"
#include "ResourceViews.h"
#include "Core/Macros.h"
#include "Utils/Image/Bitmap.h"
#include <filesystem>
#include <fstd/span.h>

namespace Falcor
{
class Sampler;
class RenderContext;

/**
 * Abstracts the API texture objects
 */
class FALCOR_API Texture : public Resource
{
    FALCOR_OBJECT(Texture)
public:
    struct SubresourceLayout
    {
        /// Size of a single row in bytes (unaligned).
        size_t rowSize;
        /// Size of a single row in bytes (aligned to device texture alignment).
        size_t rowSizeAligned;
        /// Number of rows.
        size_t rowCount;
        /// Number of depth slices.
        size_t depth;

        /// Get the total size of the subresource in bytes (unaligned).
        size_t getTotalByteSize() const { return rowSize * rowCount * depth; }

        /// Get the total size of the subresource in bytes (aligned to device texture alignment).
        size_t getTotalByteSizeAligned() const { return rowSizeAligned * rowCount * depth; }
    };

    Texture(
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
    );

    Texture(
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
    );

    ~Texture();

    /**
     * Get a mip-level width
     */
    uint32_t getWidth(uint32_t mipLevel = 0) const
    {
        return (mipLevel == 0) || (mipLevel < mMipLevels) ? std::max(1U, mWidth >> mipLevel) : 0;
    }

    /**
     * Get a mip-level height
     */
    uint32_t getHeight(uint32_t mipLevel = 0) const
    {
        return (mipLevel == 0) || (mipLevel < mMipLevels) ? std::max(1U, mHeight >> mipLevel) : 0;
    }

    /**
     * Get a mip-level depth
     */
    uint32_t getDepth(uint32_t mipLevel = 0) const
    {
        return (mipLevel == 0) || (mipLevel < mMipLevels) ? std::max(1U, mDepth >> mipLevel) : 0;
    }

    /**
     * Get the number of mip-levels
     */
    uint32_t getMipCount() const { return mMipLevels; }

    /**
     * Get the sample count
     */
    uint32_t getSampleCount() const { return mSampleCount; }

    /**
     * Get the array size
     */
    uint32_t getArraySize() const { return mArraySize; }

    /**
     * Get the array index of a subresource
     */
    uint32_t getSubresourceArraySlice(uint32_t subresource) const { return subresource / mMipLevels; }

    /**
     * Get the mip-level of a subresource
     */
    uint32_t getSubresourceMipLevel(uint32_t subresource) const { return subresource % mMipLevels; }

    /**
     * Get the subresource index
     */
    uint32_t getSubresourceIndex(uint32_t arraySlice, uint32_t mipLevel) const { return mipLevel + arraySlice * mMipLevels; }

    /**
     * Get the number of subresources
     */
    uint32_t getSubresourceCount() const { return mMipLevels * mArraySize; }

    /**
     * Get the resource format
     */
    ResourceFormat getFormat() const { return mFormat; }

    /**
     * Create a new texture object with mips specified explicitly from individual files.
     * @param[in] paths List of full paths of all mips, starting from mip0.
     * @param[in] loadAsSrgb Load the texture using sRGB format. Only valid for 3 or 4 component textures.
     * @param[in] bindFlags The bind flags to create the texture with.
     * @return A new texture, or nullptr if the texture failed to load.
     */
    static ref<Texture> createMippedFromFiles(
        ref<Device> pDevice,
        fstd::span<const std::filesystem::path> paths,
        bool loadAsSrgb,
        ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource
    );

    /**
     * Create a new texture object from a file.
     * @param[in] path File path of the image (absolute or relative to working directory).
     * @param[in] generateMipLevels Whether the mip-chain should be generated.
     * @param[in] loadAsSrgb Load the texture using sRGB format. Only valid for 3 or 4 component textures.
     * @param[in] bindFlags The bind flags to create the texture with.
     * @return A new texture, or nullptr if the texture failed to load.
     */
    static ref<Texture> createFromFile(
        ref<Device> pDevice,
        const std::filesystem::path& path,
        bool generateMipLevels,
        bool loadAsSrgb,
        ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource
    );

    gfx::ITextureResource* getGfxTextureResource() const { return mGfxTextureResource; }

    virtual gfx::IResource* getGfxResource() const override;

    /**
     * Get a shader-resource view for the entire resource
     */
    virtual ref<ShaderResourceView> getSRV() override;

    /**
     * Get an unordered access view for the entire resource
     */
    virtual ref<UnorderedAccessView> getUAV() override;

    /**
     * Get a shader-resource view.
     * @param[in] mostDetailedMip The most detailed mip level of the view
     * @param[in] mipCount The number of mip-levels to bind. If this is equal to Texture#kMaxPossible, will create a view ranging from
     * mostDetailedMip to the texture's mip levels count
     * @param[in] firstArraySlice The first array slice of the view
     * @param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the
     * texture's array size
     */
    ref<ShaderResourceView> getSRV(
        uint32_t mostDetailedMip,
        uint32_t mipCount = kMaxPossible,
        uint32_t firstArraySlice = 0,
        uint32_t arraySize = kMaxPossible
    );

    /**
     * Get a render-target view.
     * @param[in] mipLevel The requested mip-level
     * @param[in] firstArraySlice The first array slice of the view
     * @param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the
     * texture's array size
     */
    ref<RenderTargetView> getRTV(uint32_t mipLevel = 0, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);

    /**
     * Get a depth stencil view.
     * @param[in] mipLevel The requested mip-level
     * @param[in] firstArraySlice The first array slice of the view
     * @param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the
     * texture's array size
     */
    ref<DepthStencilView> getDSV(uint32_t mipLevel = 0, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);

    /**
     * Get an unordered access view.
     * @param[in] mipLevel The requested mip-level
     * @param[in] firstArraySlice The first array slice of the view
     * @param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the
     * texture's array size
     */
    ref<UnorderedAccessView> getUAV(uint32_t mipLevel, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);

    /**
     * Get the data layout of a subresource.
     * @param[in] subresource The subresource index.
     */
    SubresourceLayout getSubresourceLayout(uint32_t subresource) const;

    /**
     * Set the data of a subresource.
     * @param[in] subresource The subresource index.
     * @param[in] pData The data to write.
     * @param[in] size The size of the data (must match the actual subresource size).
     */
    void setSubresourceBlob(uint32_t subresource, const void* pData, size_t size);

    /**
     * Get the data of a subresource.
     * @param[in] subresource The subresource index.
     * @param[in] pData The data buffer to read to.
     * @param[in] size The size of the data (must match the actual subresource size).
     */
    void getSubresourceBlob(uint32_t subresource, void* pData, size_t size) const;

    /**
     * Capture the texture to an image file.
     * @param[in] mipLevel Requested mip-level
     * @param[in] arraySlice Requested array-slice
     * @param[in] path Path of the file to save.
     * @param[in] fileFormat Destination image file format (e.g., PNG, PFM, etc.)
     * @param[in] exportFlags Save flags, see Bitmap::ExportFlags
     * @param[in] async Save asynchronously, otherwise the function blocks until the texture is saved.
     */
    void captureToFile(
        uint32_t mipLevel,
        uint32_t arraySlice,
        const std::filesystem::path& path,
        Bitmap::FileFormat format = Bitmap::FileFormat::PngFile,
        Bitmap::ExportFlags exportFlags = Bitmap::ExportFlags::None,
        bool async = true
    );

    /**
     * Generates mipmaps for a specified texture object.
     * @param[in] pContext Used render context.
     * @param[in] minMaxMips Generate a min/max mipmap pyramid. Each RGBA texel of levels >0 in the resulting MIP pyramid will cointain
     * {Avg, Min, Max, Avg} of the 4 coresponding texels from the immediatly larger MIP level.
     */
    void generateMips(RenderContext* pContext, bool minMaxMips = false);

    /**
     * In case the texture was loaded from a file, use this to set the file path
     */
    void setSourcePath(const std::filesystem::path& path) { mSourcePath = path; }

    /**
     * In case the texture was loaded from a file, get the source file path
     */
    const std::filesystem::path& getSourcePath() const { return mSourcePath; }

    /**
     * Returns the total number of texels across all mip levels and array slices.
     */
    uint64_t getTexelCount() const;

    /**
     * Returns the size of the texture in bytes as allocated in GPU memory.
     */
    uint64_t getTextureSizeInBytes() const;

    /**
     * Compares the texture description to another texture.
     * @return True if all fields (size/format/etc) are identical.
     */
    bool compareDesc(const Texture* pOther) const;

protected:
    void uploadInitData(RenderContext* pRenderContext, const void* pData, bool autoGenMips);

    Slang::ComPtr<gfx::ITextureResource> mGfxTextureResource;

    bool mReleaseRtvsAfterGenMips = true;
    std::filesystem::path mSourcePath;

    ResourceFormat mFormat = ResourceFormat::Unknown;
    uint32_t mWidth = 0;
    uint32_t mHeight = 0;
    uint32_t mDepth = 0;
    uint32_t mMipLevels = 0;
    uint32_t mArraySize = 0;
    uint32_t mSampleCount = 0;
    bool mIsSparse = false;
    int3 mSparsePageRes = int3(0);

    friend class Device;
};
} // namespace Falcor
