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
#include "ImageLoader.h"
#include "RenderGraph/RenderPassLibrary.h"

const RenderPass::Info ImageLoader::kInfo { "ImageLoader", "Load an image into a texture." };

namespace
{
    const std::string kDst = "dst";

    const std::string kOutputSize = "outputSize";
    const std::string kOutputFormat = "outputFormat";
    const std::string kImage = "filename";
    const std::string kMips = "mips";
    const std::string kSrgb = "srgb";
    const std::string kArraySlice = "arrayIndex";
    const std::string kMipLevel = "mipLevel";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(ImageLoader::kInfo, ImageLoader::create);
}

RenderPassReflection ImageLoader::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    uint2 fixedSize = mpTex ? uint2(mpTex->getWidth(), mpTex->getHeight()) : uint2(0);
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, fixedSize, compileData.defaultTexDims);

    reflector.addOutput(kDst, "Destination texture").format(mOutputFormat).texture2D(sz.x, sz.y);
    return reflector;
}

ImageLoader::SharedPtr ImageLoader::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new ImageLoader(dict));
}

ImageLoader::ImageLoader(const Dictionary& dict)
    : RenderPass(kInfo)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kOutputSize) mOutputSizeSelection = value;
        else if (key == kOutputFormat) mOutputFormat = value;
        else if (key == kImage) mImagePath = value.operator std::filesystem::path();
        else if (key == kSrgb) mLoadSRGB = value;
        else if (key == kMips) mGenerateMips = value;
        else if (key == kArraySlice) mArraySlice = value;
        else if (key == kMipLevel) mMipLevel = value;
        else logWarning("Unknown field '{}' in a ImageLoader dictionary.", key);
    }

    if (!mImagePath.empty())
    {
        if (!loadImage(mImagePath))
        {
            throw RuntimeError("ImageLoader: Failed to load image from '{}'", mImagePath);
        }
    }
}

Dictionary ImageLoader::getScriptingDictionary()
{
    Dictionary dict;
    dict[kOutputSize] = mOutputSizeSelection;
    if (mOutputFormat != ResourceFormat::Unknown) dict[kOutputFormat] = mOutputFormat;
    dict[kImage] = stripDataDirectories(mImagePath);
    dict[kMips] = mGenerateMips;
    dict[kSrgb] = mLoadSRGB;
    dict[kArraySlice] = mArraySlice;
    dict[kMipLevel] = mMipLevel;
    return dict;
}

void ImageLoader::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!mpTex) throw RuntimeError("ImageLoader: No image loaded");
}

void ImageLoader::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pDstTex = renderData.getTexture(kDst);
    FALCOR_ASSERT(pDstTex);
    mOutputFormat = pDstTex->getFormat();
    mOutputSize = { pDstTex->getWidth(), pDstTex->getHeight() };

    if (!mpTex)
    {
        pRenderContext->clearRtv(pDstTex->getRTV().get(), float4(0, 0, 0, 0));
        return;
    }

    mMipLevel = std::min(mMipLevel, mpTex->getMipCount() - 1);
    mArraySlice = std::min(mArraySlice, mpTex->getArraySize() - 1);
    pRenderContext->blit(mpTex->getSRV(mMipLevel, 1, mArraySlice, 1), pDstTex->getRTV());
}

void ImageLoader::renderUI(Gui::Widgets& widget)
{
    // When output size requirements change, we'll trigger a graph recompile to update the render pass I/O sizes.
    if (widget.dropdown("Output size", RenderPassHelpers::kIOSizeList, (uint32_t&)mOutputSizeSelection)) requestRecompile();
    widget.tooltip("Specifies the pass output size.\n"
        "'Default' means that the output is sized based on requirements of connected passes.\n"
        "'Fixed' means the output is always at the image's native size.\n"
        "If the output is of a different size than the native image resolution, the image will be rescaled bilinearly." , true);

    widget.text("Image File: " + mImagePath.string());
    bool reloadImage = false;
    reloadImage |= widget.checkbox("Load As SRGB", mLoadSRGB);
    reloadImage |= widget.checkbox("Generate Mipmaps", mGenerateMips);

    if (widget.button("Load File"))
    {
        reloadImage |= openFileDialog({}, mImagePath);
    }

    if (mpTex)
    {
        if (mpTex->getMipCount() > 1) widget.slider("Mip Level", mMipLevel, 0u, mpTex->getMipCount() - 1);
        if (mpTex->getArraySize() > 1) widget.slider("Array Slice", mArraySlice, 0u, mpTex->getArraySize() - 1);

        widget.image(mImagePath.string().c_str(), mpTex, { 320, 320 });
        widget.text("Image format: " + to_string(mpTex->getFormat()));
        widget.text("Image size: (" + std::to_string(mpTex->getWidth()) + ", " + std::to_string(mpTex->getHeight()) + ")");
        widget.text("Output format: " + to_string(mOutputFormat));
        widget.text("Output size: (" + std::to_string(mOutputSize.x) + ", " + std::to_string(mOutputSize.y) + ")");
    }

    if (reloadImage && !mImagePath.empty())
    {
        uint2 prevSize = {};
        if (mpTex) prevSize = { mpTex->getWidth(), mpTex->getHeight() };

        if (!loadImage(mImagePath))
        {
            msgBox(fmt::format("Failed to load image from '{}'", mImagePath), MsgBoxType::Ok, MsgBoxIcon::Warning);
        }

        // If output is set to native size and image dimensions have changed, we'll trigger a graph recompile to update the render pass I/O sizes.
        if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed && mpTex != nullptr &&
            (mpTex->getWidth() != prevSize.x || mpTex->getHeight() != prevSize.y))
        {
            requestRecompile();
        }
    }
}

bool ImageLoader::loadImage(const std::filesystem::path& path)
{
    if (path.empty()) return false;

    // Find the full path of the specified image.
    // We retain this for later as the search paths may change during execution.
    std::filesystem::path fullPath;
    if (findFileInDataDirectories(mImagePath, fullPath))
    {
        mImagePath = fullPath;
        mpTex = Texture::createFromFile(mImagePath, mGenerateMips, mLoadSRGB);
        return mpTex != nullptr;
    }
    else
    {
        return false;
    }
}
