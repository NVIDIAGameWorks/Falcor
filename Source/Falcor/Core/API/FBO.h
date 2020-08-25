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
#pragma once
#include "Core/API/Texture.h"
#include "Core/API/ResourceViews.h"

namespace Falcor
{
    /** Low level framebuffer object.
        This class abstracts the API's framebuffer creation and management.
    */
    class dlldecl Fbo : public std::enable_shared_from_this<Fbo>
    {
    public:
        using SharedPtr = std::shared_ptr<Fbo>;
        using SharedConstPtr = std::shared_ptr<const Fbo>;
        using ApiHandle = FboHandle;

        class dlldecl Desc
        {
        public:
            Desc();

            /** Set a render target to be a color target.
                \param[in] rtIndex Index of render target
                \param[in] format Texture resource format
                \param[in] allowUav Whether the resource can be a UAV
            */
            Desc& setColorTarget(uint32_t rtIndex, ResourceFormat format, bool allowUav = false) { mColorTargets[rtIndex] = TargetDesc(format, allowUav); return *this; }

            /** Set the format of the depth-stencil target.
                \param[in] format Texture resource format
                \param[in] allowUav Whether the resource can be a UAV
            */
            Desc& setDepthStencilTarget(ResourceFormat format, bool allowUav = false) { mDepthStencilTarget = TargetDesc(format, allowUav); return *this; }

            /** Set the resource sample count.
            */
            Desc& setSampleCount(uint32_t sampleCount) { mSampleCount = sampleCount ? sampleCount : 1; return *this; }

            /** Get the resource format of a render target
            */
            ResourceFormat getColorTargetFormat(uint32_t rtIndex) const { return mColorTargets[rtIndex].format; }

            /** Get whether a color target resource can be a UAV.
            */
            bool isColorTargetUav(uint32_t rtIndex) const { return mColorTargets[rtIndex].allowUav; }

            /** Get the depth-stencil resource format.
            */
            ResourceFormat getDepthStencilFormat() const { return mDepthStencilTarget.format; }

            /** Get whether depth-stencil resource can be a UAV.
            */
            bool isDepthStencilUav() const { return mDepthStencilTarget.allowUav; }

            /** Get sample count of the targets.
            */
            uint32_t getSampleCount() const { return mSampleCount; }

            /** Comparison operator
            */
            bool operator==(const Desc& other) const;

        private:
            struct TargetDesc
            {
                TargetDesc() = default;
                TargetDesc(ResourceFormat f, bool uav) : format(f), allowUav(uav) {}
                ResourceFormat format = ResourceFormat::Unknown;
                bool allowUav = false;
                
                bool operator==(const TargetDesc& other) const {return (format == other.format) && (allowUav == other.allowUav); }

                bool operator!=(const TargetDesc& other) const { return !(*this == other); }
            };

            std::vector<TargetDesc> mColorTargets;
            TargetDesc mDepthStencilTarget;
            uint32_t mSampleCount = 1;
        };

        /** Used to tell some functions to attach all array slices of a specific mip-level.
        */
        static const uint32_t kAttachEntireMipLevel = uint32_t(-1);

        /** Destructor. Releases the API object
        */
        ~Fbo();

        /** Get a FBO representing the default framebuffer object
        */
        static SharedPtr getDefault();

        /** Create a new empty FBO.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create();

        /** Create an FBO from a list of textures. It will bind mip 0 and the all of the array slices.
            \param[in] colors A vector with color textures. The index in the vector corresponds to the render target index in the shader. You can use nullptr for unused indices.
            \param[in] depth An optional depth buffer texture.
            \return A new object. An exception is thrown if creation failed, for example due to texture size mismatch, bind flags issues, illegal formats, etc.
        */
        static SharedPtr create(const std::vector<Texture::SharedPtr>& colors, const Texture::SharedPtr& pDepth = nullptr);

        /** Create a color-only 2D framebuffer.
            \param[in] width Width of the render targets.
            \param[in] height Height of the render targets.
            \param[in] fboDesc Struct specifying the frame buffer's attachments and formats.
            \param[in] arraySize Optional. The number of array slices in the texture.
            \param[in] mipLevels Optional. The number of mip levels to create. You can use Texture#kMaxPossible to create the entire chain.
            \return A new object. An exception is thrown if creation failed, for example due to invalid parameters.
        */
        static SharedPtr create2D(uint32_t width, uint32_t height, const Desc& fboDesc, uint32_t arraySize = 1, uint32_t mipLevels = 1);

        /** Create a color-only cubemap framebuffer.
            \param[in] width width of the render targets.
            \param[in] height height of the render targets.
            \param[in] fboDesc Struct specifying the frame buffer's attachments and formats.
            \param[in] arraySize Optional. The number of cubes in the texture.
            \param[in] mipLevels Optional. The number of mip levels to create. You can use Texture#kMaxPossible to create the entire chain.
            \return A new object. An exception is thrown if creation failed, for example due to invalid parameters.
        */
        static SharedPtr createCubemap(uint32_t width, uint32_t height, const Desc& fboDesc, uint32_t arraySize = 1, uint32_t mipLevels = 1);

        /** Creates an FBO with a single color texture (single mip, single array slice), and optionally a depth buffer.
            \param[in] width Width of the render targets.
            \param[in] height Height of the render targets.
            \param[in] color The color format.
            \param[in] depth The depth-format. If a depth-buffer is not required, use ResourceFormat::Unknown.
            \return A new object. An exception is thrown if creation failed, for example due to invalid parameters.
        */
        static SharedPtr create2D(uint32_t width, uint32_t height, ResourceFormat color, ResourceFormat depth = ResourceFormat::Unknown);

        /** Attach a depth-stencil texture.
            An exception is thrown if the texture can't be used as a depth-buffer (usually a format or bind flags issue).
            \param pDepthStencil The depth-stencil texture.
            \param mipLevel The selected mip-level to attach.
            \param firstArraySlice The first array-slice to bind
            \param arraySize The number of array sliced to bind, or Fbo#kAttachEntireMipLevel to attach the range [firstArraySlice, pTexture->getArraySize()]
        */
        void attachDepthStencilTarget(const Texture::SharedPtr& pDepthStencil, uint32_t mipLevel = 0, uint32_t firstArraySlice = 0, uint32_t arraySize = kAttachEntireMipLevel);

        /** Attach a color texture.
            An exception is thrown if the texture can't be used as a color-target (usually a format or bind flags issue).
            \param pColorTexture The color texture.
            \param rtIndex The render-target index to attach the texture to.
            \param mipLevel The selected mip-level to attach.
            \param firstArraySlice The first array-slice to bind
            \param arraySize The number of array sliced to bind, or Fbo#kAttachEntireMipLevel to attach the range [firstArraySlice, pTexture->getArraySize()]
        */
        void attachColorTarget(const Texture::SharedPtr& pColorTexture, uint32_t rtIndex, uint32_t mipLevel = 0, uint32_t firstArraySlice = 0, uint32_t arraySize = kAttachEntireMipLevel);

        /** Get the object's API handle.      
        */
        const ApiHandle& getApiHandle() const;

        /** Get the maximum number of color targets
        */
        static uint32_t getMaxColorTargetCount();

        /** Get an attached color texture. If no texture is attached will return nullptr.
        */
        Texture::SharedPtr getColorTexture(uint32_t index) const;

        /** Get the attached depth-stencil texture, or nullptr if no texture is attached.
        */
        const Texture::SharedPtr& getDepthStencilTexture() const;

        /** Get the width of the FBO
        */
        uint32_t getWidth() const { finalize(); return mWidth; }

        /** Get the height of the FBO
        */
        uint32_t getHeight() const { finalize(); return mHeight; }

        /** Get the sample-count of the FBO
        */
        uint32_t getSampleCount() const { finalize(); return mpDesc->getSampleCount(); }

        /** Get the FBO format descriptor
        */
        const Desc& getDesc() const { finalize();  return *mpDesc; }

        /** Get a depth-stencil view to the depth-stencil target.
        */
        DepthStencilView::SharedPtr getDepthStencilView() const;

        /** Get a render target view to a color target.
        */
        RenderTargetView::SharedPtr getRenderTargetView(uint32_t rtIndex) const;

        struct SamplePosition
        {
            int8_t xOffset = 0;
            int8_t yOffset = 0;
        };

        /**  Configure the sample positions used by multi-sampled buffers.
            \param[in] samplesPerPixel The number of samples-per-pixel. This value has to match the FBO's sample count
            \param[in] pixelCount the number if pixels the sample pattern is specified for
            \param[in] positions The sample positions. (0,0) is a pixel's center. The size of this array should be samplesPerPixel*pixelCount
            To reset the positions to their original location pass `nullptr` for positions
        */
        void setSamplePositions(uint32_t samplesPerPixel, uint32_t pixelCount, const SamplePosition positions[]);

        /** Get the sample positions
        */
        const std::vector<SamplePosition> getSamplePositions() const { return mSamplePositions; }

        /** Get the number of pixels the sample positions are configured for
        */
        uint32_t getSamplePositionsPixelCount() const { return mSamplePositionsPixelCount; }

        struct Attachment
        {
            Texture::SharedPtr pTexture = nullptr;
            uint32_t mipLevel = 0;
            uint32_t arraySize = 1;
            uint32_t firstArraySlice = 0;
        };

        struct DescHash
        {
            std::size_t operator()(const Desc& d) const;
        };

    private:
        static std::unordered_set<Desc, DescHash> sDescs;

        bool verifyAttachment(const Attachment& attachment) const;
        bool calcAndValidateProperties() const;

        void applyColorAttachment(uint32_t rtIndex);
        void applyDepthAttachment();
        void initApiHandle() const;

        /** Validates that the framebuffer attachments are OK. Throws an exception on error.
            This function causes the actual HW resources to be generated (RTV/DSV).
        */
        void finalize() const;

        Fbo();
        std::vector<Attachment> mColorAttachments;
        std::vector<SamplePosition> mSamplePositions;
        uint32_t mSamplePositionsPixelCount = 0;

        Attachment mDepthStencil;

        mutable Desc mTempDesc;
        mutable const Desc* mpDesc = nullptr;
        mutable uint32_t mWidth  = (uint32_t)-1;
        mutable uint32_t mHeight = (uint32_t)-1;
        mutable uint32_t mDepth = (uint32_t)-1;
        mutable bool mHasDepthAttachment = false;
        mutable bool mIsLayered = false;
        mutable bool mIsZeroAttachment = false;

        mutable ApiHandle mApiHandle = {};
        void* mpPrivateData = nullptr;
    };
}
