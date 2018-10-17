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
#include "Framework.h"
#include "ImageLoader.h"
#include "Graphics/TextureHelper.h"
#include "API/RenderContext.h"
#include "Utils/Gui.h"
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace Falcor
{
    static const std::string kDst = "dst";
    static const std::string kImage = "fileName";
    static const std::string kMips = "mips";
    static const std::string kSrgb = "srgb";
    static const std::string kDefaultImage = "image";

    RenderPassReflection ImageLoader::reflect() const
    {
        RenderPassReflection reflector;
        reflector.addOutput(kDst);

        return reflector;
    }

    ImageLoader::SharedPtr ImageLoader::create(const Dictionary& dict)
    {
        SharedPtr pPass = SharedPtr(new ImageLoader);

        for (const auto& v : dict)
        {
            if (v.key() == kImage)
            {
                pPass->mImageName = v.val().operator std::string();
            }
            else if(v.key() == kSrgb)
            {
                pPass->mLoadSRGB = v.val();
            }
            else if (v.key() == kMips)
            {
                pPass->mGenerateMips = v.val();
            }
            else
            {
                logWarning("Unknown field `" + v.key() + "` in a ImageLoader dictionary");
            }
        }
    
        return pPass;
    }

    Dictionary ImageLoader::getScriptingDictionary() const
    {
        Dictionary dict;
        dict[kImage] = mImageName;
        dict[kMips] = mGenerateMips;
        dict[kSrgb] = mLoadSRGB;
        return dict;
    }

    ImageLoader::ImageLoader() : RenderPass("ImageLoader")
    {
    }

    void ImageLoader::execute(RenderContext* pContext, const RenderData* pRenderData)
    {
        if (!mpTex)
        {
            // attempt to load default image
            auto& dict = pRenderData->getDictionary();
            if (!mImageName.size() && dict.keyExists(kDefaultImage))
            {
                std::string defaultImageName = dict[kDefaultImage].operator std::string();
                mImageName = defaultImageName;
            }
            if (mImageName.size())
            {
                if (dict.keyExists(mImageName)) { mpTex = dict[mImageName]; }
                else
                {
                    mImageName = stripDataDirectories(mImageName);
                    mpTex = createTextureFromFile(mImageName, mGenerateMips, mLoadSRGB);
                    // if updatePass is called, the image will be unloaded
                    // save the image pointer in the shared dictionary to avoid this
                    dict[mImageName] = mpTex;
                }
            }
            if (!mpTex) { logWarning("No image loaded! Not able to execute image loader pass."); return; }
        }
       
        const auto& pDstTex = pRenderData->getTexture(kDst);
        if (pDstTex)
        {
            pContext->blit(mpTex->getSRV(), pDstTex->getRTV());
        }
        else
        {
            logWarning("ImageLoader::execute() - missing an input or output resource");
        }
    }

    void ImageLoader::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (!uiGroup || pGui->beginGroup(uiGroup))
        {
            bool reloadImage =  pGui->addTextBox("Image File", mImageName); ;
            reloadImage |= pGui->addCheckBox("Load As SRGB", mLoadSRGB);
            reloadImage |= pGui->addCheckBox("Generate Mipmaps", mGenerateMips);
            if (pGui->addButton("loadFile")) { reloadImage |= openFileDialog("", mImageName); }
            if (pGui->addButton("clear"))
            {
                mpTex = nullptr; mImageName.clear();
            }
            
            if (mpTex)
            {
                pGui->addImage(mImageName.c_str(), mpTex, { 320, 320 });
            }

            if (reloadImage && mImageName.size())
            {
                mImageName = stripDataDirectories(mImageName);
                mpTex = createTextureFromFile(mImageName, mGenerateMips, mLoadSRGB);
            }

            if (uiGroup) pGui->endGroup();
        }
    }
}