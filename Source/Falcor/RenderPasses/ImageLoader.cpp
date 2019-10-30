/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "ImageLoader.h"

namespace Falcor
{
    const char* ImageLoader::kDesc = "Load an image into a texture";

    namespace
    {
        const std::string kDst = "dst";
        const std::string kImage = "filename";
        const std::string kMips = "mips";
        const std::string kSrgb = "srgb";
        const std::string kArraySlice = "arrayIndex";
        const std::string kMipLevel = "mipLevel";
    }
    
    RenderPassReflection ImageLoader::reflect(const CompileData& compileData)
    {
        RenderPassReflection reflector;
        reflector.addOutput(kDst, "Destination texture");
        return reflector;
    }

    ImageLoader::SharedPtr ImageLoader::create(RenderContext* pRenderContext, const Dictionary& dict)
    {
        SharedPtr pPass = SharedPtr(new ImageLoader);

        for (const auto& v : dict)
        {
            if (v.key() == kImage) pPass->mImageName = v.val().operator std::string();
            else if(v.key() == kSrgb) pPass->mLoadSRGB = v.val();
            else if (v.key() == kMips) pPass->mGenerateMips = v.val();
            else if (v.key() == kArraySlice) pPass->mArraySlice = v.val();
            else if (v.key() == kMipLevel) pPass->mMipLevel = v.val();
            else logWarning("Unknown field `" + v.key() + "` in a ImageLoader dictionary");
        }

        if (pPass->mImageName.size())
        {
            pPass->mpTex = Texture::createFromFile(pPass->mImageName, pPass->mGenerateMips, pPass->mLoadSRGB);
        }

        return pPass;
    }

    Dictionary ImageLoader::getScriptingDictionary()
    {
        Dictionary dict;
        dict[kImage] = mImageName;
        dict[kMips] = mGenerateMips;
        dict[kSrgb] = mLoadSRGB;
        dict[kArraySlice] = mArraySlice;
        dict[kMipLevel] = mMipLevel;
        return dict;
    }

    ImageLoader::ImageLoader()
    {
    }

    void ImageLoader::compile(RenderContext* pContext, const CompileData& compileData)
    {
        if (!mpTex) throw std::runtime_error("ImageLoader::compile - No image loaded!");
    }

    void ImageLoader::execute(RenderContext* pContext, const RenderData& renderData)
    {
        const auto& pDstTex = renderData[kDst]->asTexture();
        if (!mpTex)
        {
            pContext->clearRtv(pDstTex->getRTV().get(), glm::vec4(0, 0, 0, 0));
            return;
        }
        pContext->blit(mpTex->getSRV(mMipLevel, 1, mArraySlice, 1), pDstTex->getRTV());
    }

    void ImageLoader::renderUI(Gui::Widgets& widget)
    {
        bool reloadImage = widget.textbox("Image File", mImageName);
        reloadImage |= widget.checkbox("Load As SRGB", mLoadSRGB);
        reloadImage |= widget.checkbox("Generate Mipmaps", mGenerateMips);
        if (mGenerateMips)
        {
            reloadImage |= widget.slider("Mip Level", mMipLevel, 0u, mpTex ? mpTex->getMipCount() : 0u);
        }
        reloadImage |= widget.slider("Array Slice", mArraySlice, 0u, mpTex ? mpTex->getArraySize() : 0u);
        
        if (widget.button("Load File")) { reloadImage |= openFileDialog({}, mImageName); }

        if (mpTex)
        {
            widget.image(mImageName.c_str(), mpTex, { 320, 320 });
        }

        if (reloadImage && mImageName.size())
        {
            mImageName = stripDataDirectories(mImageName);
            mpTex = Texture::createFromFile(mImageName, mGenerateMips, mLoadSRGB);
        }
    }
}
