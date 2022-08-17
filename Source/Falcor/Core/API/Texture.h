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
#pragma once
#include "Handles.h"
#include "Formats.h"
#include "Resource.h"
#include "ResourceViews.h"
#include "Core/Macros.h"
#include "Utils/Image/Bitmap.h"
#include <memory>
#include <filesystem>

namespace Falcor
{
    class Sampler;
    class Device;
    class RenderContext;

    /** Abstracts the API texture objects
    */
    class FALCOR_API Texture : public Resource
    {
    public:
        using SharedPtr = std::shared_ptr<Texture>;
        using SharedConstPtr = std::shared_ptr<const Texture>;
        using WeakPtr = std::weak_ptr<Texture>;
        using WeakConstPtr = std::weak_ptr<const Texture>;

        ~Texture();

        /** Get a mip-level width
        */
        uint32_t getWidth(uint32_t mipLevel = 0) const { return (mipLevel == 0) || (mipLevel < mMipLevels) ? std::max(1U, mWidth >> mipLevel) : 0; }

        /** Get a mip-level height
        */
        uint32_t getHeight(uint32_t mipLevel = 0) const { return (mipLevel == 0) || (mipLevel < mMipLevels) ? std::max(1U, mHeight >> mipLevel) : 0; }

        /** Get a mip-level depth
        */
        uint32_t getDepth(uint32_t mipLevel = 0) const { return (mipLevel == 0) || (mipLevel < mMipLevels) ? std::max(1U, mDepth >> mipLevel) : 0; }

        /** Get the number of mip-levels
        */
        uint32_t getMipCount() const { return mMipLevels; }

        /** Get the sample count
        */
        uint32_t getSampleCount() const { return mSampleCount; }

        /** Get the array size
        */
        uint32_t getArraySize() const { return mArraySize; }

        /** Get the array index of a subresource
        */
        uint32_t getSubresourceArraySlice(uint32_t subresource) const { return subresource / mMipLevels; }

        /** Get the mip-level of a subresource
        */
        uint32_t getSubresourceMipLevel(uint32_t subresource) const { return subresource % mMipLevels; }

        /** Get the subresource index
        */
        uint32_t getSubresourceIndex(uint32_t arraySlice, uint32_t mipLevel) const { return mipLevel + arraySlice * mMipLevels; }

        /** Get the resource format
        */
        ResourceFormat getFormat() const { return mFormat; }

        /** Create a new texture from an existing API handle.
            \param[in] handle Handle of already allocated resource.
            \param[in] type The type of texture.
            \param[in] width The width of the texture.
            \param[in] height The height of the texture.
            \param[in] depth The depth of the texture.
            \param[in] format The format of the texture.
            \param[in] sampleCount The sample count of the texture.
            \param[in] arraySize The array size of the texture.
            \param[in] mipLevels The number of mip levels.
            \param[in] initState The initial resource state.
            \param[in] bindFlags Texture bind flags. Flags must match the bind flags of the original resource.
            \return A pointer to a new texture, or throws an exception if creation failed.
        */
        static SharedPtr createFromApiHandle(ApiHandle handle, Type type, uint32_t width, uint32_t height, uint32_t depth, ResourceFormat format, uint32_t sampleCount, uint32_t arraySize, uint32_t mipLevels, State initState, BindFlags bindFlags);

        /** Create a 1D texture.
            \param[in] width The width of the texture.
            \param[in] format The format of the texture.
            \param[in] arraySize The array size of the texture.
            \param[in] mipLevels If equal to kMaxPossible then an entire mip chain will be generated from mip level 0. If any other value is given then the data for at least that number of miplevels must be provided.
            \param[in] pInitData If different than nullptr, pointer to a buffer containing data to initialize the texture with.
            \param[in] bindFlags The requested bind flags for the resource.
            \return A pointer to a new texture, or throws an exception if creation failed.
        */
        static SharedPtr create1D(uint32_t width, ResourceFormat format, uint32_t arraySize = 1, uint32_t mipLevels = kMaxPossible, const void* pInitData = nullptr, BindFlags bindFlags = BindFlags::ShaderResource);

        /** Create a 2D texture.
            \param[in] width The width of the texture.
            \param[in] height The height of the texture.
            \param[in] format The format of the texture.
            \param[in] arraySize The array size of the texture.
            \param[in] mipLevels If equal to kMaxPossible then an entire mip chain will be generated from mip level 0. If any other value is given then the data for at least that number of miplevels must be provided.
            \param[in] pInitData If different than nullptr, pointer to a buffer containing data to initialize the texture with.
            \param[in] bindFlags The requested bind flags for the resource.
            \return A pointer to a new texture, or throws an exception if creation failed.
        */
        static SharedPtr create2D(uint32_t width, uint32_t height, ResourceFormat format, uint32_t arraySize = 1, uint32_t mipLevels = kMaxPossible, const void* pInitData = nullptr, BindFlags bindFlags = BindFlags::ShaderResource);

        /** Create a 3D texture.
            \param[in] width The width of the texture.
            \param[in] height The height of the texture.
            \param[in] depth The depth of the texture.
            \param[in] format The format of the texture.
            \param[in] mipLevels If equal to kMaxPossible then an entire mip chain will be generated from mip level 0. If any other value is given then the data for at least that number of miplevels must be provided.
            \param[in] pInitData If different than nullptr, pointer to a buffer containing data to initialize the texture with.
            \param[in] bindFlags The requested bind flags for the resource.
            \param[in] isSparse If true, the texture is created using sparse texture options supported by the API.
            \return A pointer to a new texture, or throws an exception if creation failed.
        */
        static SharedPtr create3D(uint32_t width, uint32_t height, uint32_t depth, ResourceFormat format, uint32_t mipLevels = kMaxPossible, const void* pInitData = nullptr, BindFlags bindFlags = BindFlags::ShaderResource, bool isSparse = false);

        /** Create a cube texture.
            \param[in] width The width of the texture.
            \param[in] height The height of the texture.
            \param[in] format The format of the texture.
            \param[in] arraySize The array size of the texture.
            \param[in] mipLevels If equal to kMaxPossible then an entire mip chain will be generated from mip level 0. If any other value is given then the data for at least that number of miplevels must be provided.
            \param[in] pInitData If different than nullptr, pointer to a buffer containing data to initialize the texture with.
            \param[in] bindFlags The requested bind flags for the resource.
            \return A pointer to a new texture, or throws an exception if creation failed.
        */
        static SharedPtr createCube(uint32_t width, uint32_t height, ResourceFormat format, uint32_t arraySize = 1, uint32_t mipLevels = kMaxPossible, const void* pInitData = nullptr, BindFlags bindFlags = BindFlags::ShaderResource);

        /** Create a multi-sampled 2D texture.
            \param[in] width The width of the texture.
            \param[in] height The height of the texture.
            \param[in] format The format of the texture.
            \param[in] sampleCount The sample count of the texture.
            \param[in] arraySize The array size of the texture.
            \param[in] bindFlags The requested bind flags for the resource.
            \return A pointer to a new texture, or throws an exception if creation failed.
        */
        static SharedPtr create2DMS(uint32_t width, uint32_t height, ResourceFormat format, uint32_t sampleCount, uint32_t arraySize = 1, BindFlags bindFlags = BindFlags::ShaderResource);

        /** Create a new texture object from a file.
            \param[in] path File path of the image. Can also include a full path or relative path from a data directory.
            \param[in] generateMipLevels Whether the mip-chain should be generated.
            \param[in] loadAsSrgb Load the texture using sRGB format. Only valid for 3 or 4 component textures.
            \param[in] bindFlags The bind flags to create the texture with.
            \return A new texture, or nullptr if the texture failed to load.
        */
        static SharedPtr createFromFile(const std::filesystem::path& path, bool generateMipLevels, bool loadAsSrgb, BindFlags bindFlags = BindFlags::ShaderResource);

        /** Get a shader-resource view for the entire resource
        */
        virtual ShaderResourceView::SharedPtr getSRV() override;

        /** Get an unordered access view for the entire resource
        */
        virtual UnorderedAccessView::SharedPtr getUAV() override;

#if FALCOR_HAS_CUDA
        /** Get the CUDA device address for this resource.
            \return CUDA device address.
            Throws an exception if the resource is not (or cannot be) shared with CUDA.
        */
        virtual void* getCUDADeviceAddress() const override;

        /** Get the CUDA device address for a view of this resource.
            Throws an exception if the resource is not (or cannot be) shared with CUDA.
        */
        virtual void* getCUDADeviceAddress(ResourceViewInfo const& viewInfo) const override;
#endif

        /** Get a shader-resource view.
            \param[in] mostDetailedMip The most detailed mip level of the view
            \param[in] mipCount The number of mip-levels to bind. If this is equal to Texture#kMaxPossible, will create a view ranging from mostDetailedMip to the texture's mip levels count
            \param[in] firstArraySlice The first array slice of the view
            \param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the texture's array size
        */
        ShaderResourceView::SharedPtr getSRV(uint32_t mostDetailedMip, uint32_t mipCount = kMaxPossible, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);

        /** Get a render-target view.
            \param[in] mipLevel The requested mip-level
            \param[in] firstArraySlice The first array slice of the view
            \param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the texture's array size
        */
        RenderTargetView::SharedPtr getRTV(uint32_t mipLevel = 0, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);

        /** Get a depth stencil view.
            \param[in] mipLevel The requested mip-level
            \param[in] firstArraySlice The first array slice of the view
            \param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the texture's array size
        */
        DepthStencilView::SharedPtr getDSV(uint32_t mipLevel = 0, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);

        /** Get an unordered access view.
            \param[in] mipLevel The requested mip-level
            \param[in] firstArraySlice The first array slice of the view
            \param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the texture's array size
        */
        UnorderedAccessView::SharedPtr getUAV(uint32_t mipLevel, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);

        /** Capture the texture to an image file.
            \param[in] mipLevel Requested mip-level
            \param[in] arraySlice Requested array-slice
            \param[in] path Path of the file to save.
            \param[in] fileFormat Destination image file format (e.g., PNG, PFM, etc.)
            \param[in] exportFlags Save flags, see Bitmap::ExportFlags
        */
        void captureToFile(uint32_t mipLevel, uint32_t arraySlice, const std::filesystem::path& path, Bitmap::FileFormat format = Bitmap::FileFormat::PngFile, Bitmap::ExportFlags exportFlags = Bitmap::ExportFlags::None);

        /** Generates mipmaps for a specified texture object.
            \param[in] pContext Used render context.
            \param[in] minMaxMips Generate a min/max mipmap pyramid. Each RGBA texel of levels >0 in the resulting MIP pyramid will cointain {Avg, Min, Max, Avg} of the 4 coresponding texels from the immediatly larger MIP level.
        */
        void generateMips(RenderContext* pContext, bool minMaxMips = false);

        /** In case the texture was loaded from a file, use this to set the file path
        */
        void setSourcePath(const std::filesystem::path& path) { mSourcePath = path; }

        /** In case the texture was loaded from a file, get the source file path
        */
        const std::filesystem::path& getSourcePath() const { return mSourcePath; }

        /** Returns the total number of texels across all mip levels and array slices.
        */
        uint64_t getTexelCount() const;

        /** Returns the size of the texture in bytes as allocated in GPU memory.
        */
        uint64_t getTextureSizeInBytes() const;

        /** Compares the texture description to another texture.
            \return True if all fields (size/format/etc) are identical.
        */
        bool compareDesc(const Texture* pOther) const;

    protected:
        Texture(uint32_t width, uint32_t height, uint32_t depth, uint32_t arraySize, uint32_t mipLevels, uint32_t sampleCount, ResourceFormat format, Type Type, BindFlags bindFlags);
        void apiInit(const void* pData, bool autoGenMips);
        void uploadInitData(const void* pData, bool autoGenMips);

        bool mReleaseRtvsAfterGenMips = true;
        std::filesystem::path mSourcePath;

        uint32_t mWidth = 0;
        uint32_t mHeight = 0;
        uint32_t mDepth = 0;
        uint32_t mMipLevels = 0;
        uint32_t mSampleCount = 0;
        uint32_t mArraySize = 0;
        ResourceFormat mFormat = ResourceFormat::Unknown;
        bool mIsSparse = false;
        int3 mSparsePageRes = int3(0);

        friend class Device;
    };
}
