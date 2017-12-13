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
#include "API/Formats.h"
#include "FboHelper.h"
#include "API/FBO.h"
#include "API/Texture.h"

namespace Falcor
{
    namespace FboHelper
    {
        bool CheckParams(const std::string& Func, uint32_t width, uint32_t height, uint32_t arraySize, uint32_t mipLevels, uint32_t sampleCount)
        {
            std::string msg = "FboHelper::" + Func + "() - ";
            std::string param;

            if(mipLevels == 0)
                param = "mipLevels";
            else if(width == 0)
                param = "width";
            else if(height == 0)
                param = "height";
            else if(arraySize == 0)
                param = "arraySize";
            else
            {
                if(sampleCount > 1 && mipLevels > 1)
                {
                    logError(msg + "can't create multi-sampled texture with more than one mip-level. sampleCount = " + std::to_string(sampleCount) + ", mipLevels = " + std::to_string(mipLevels) + ".");
                    return false;
                }
                return true;
            }

            logError(msg + param + " can't be zero.");
            return false;
        }

        static Texture::SharedPtr createTexture2D(uint32_t w, uint32_t h, ResourceFormat format, uint32_t sampleCount, uint32_t arraySize, uint32_t mipLevels, Texture::BindFlags flags)
        {
            if (format == ResourceFormat::Unknown)
            {
                logError("Can't create Texture2D with an unknown resource format");
                return nullptr;
            }

            Texture::SharedPtr pTex;
            if (sampleCount > 1)
            {
                pTex = Texture::create2DMS(w, h, format, sampleCount, arraySize, flags);
            }
            else
            {
                pTex = Texture::create2D(w, h, format, arraySize, mipLevels, nullptr, flags);
            }

            return pTex;
        }

        static Texture::BindFlags getBindFlags(bool isDepth, bool allowUav)
        {
            Texture::BindFlags flags = Texture::BindFlags::ShaderResource;
            flags |= isDepth ? Texture::BindFlags::DepthStencil : Texture::BindFlags::RenderTarget;

            if (allowUav)
            {
                flags |= Texture::BindFlags::UnorderedAccess;
            }
            return flags;
        }

        Fbo::SharedPtr create2D(uint32_t width, uint32_t height, const Fbo::Desc& fboDesc, uint32_t arraySize, uint32_t mipLevels)
        {
            uint32_t sampleCount = fboDesc.getSampleCount();
            if(CheckParams("Create2D", width, height, arraySize, mipLevels, sampleCount) == false)
            {
                return nullptr;
            }

            Fbo::SharedPtr pFbo = Fbo::create();

            // create the color targets
            for(uint32_t i = 0; i < Fbo::getMaxColorTargetCount(); i++)
            {
                if(fboDesc.getColorTargetFormat(i) != ResourceFormat::Unknown)
                {
                    Texture::BindFlags flags = getBindFlags(false, fboDesc.isColorTargetUav(i));
                    Texture::SharedPtr pTex = createTexture2D(width, height, fboDesc.getColorTargetFormat(i), sampleCount, arraySize, mipLevels, flags);
                    pFbo->attachColorTarget(pTex, i, 0, 0, Fbo::kAttachEntireMipLevel);
                }
            }

            if(fboDesc.getDepthStencilFormat() != ResourceFormat::Unknown)
            {
                Texture::BindFlags flags = getBindFlags(true, fboDesc.isDepthStencilUav());
                Texture::SharedPtr pDepth = createTexture2D(width, height, fboDesc.getDepthStencilFormat(), sampleCount, arraySize, mipLevels, flags);
                pFbo->attachDepthStencilTarget(pDepth, 0, 0, Fbo::kAttachEntireMipLevel);
            }

            return pFbo;
        }

        Fbo::SharedPtr createCubemap(uint32_t width, uint32_t height, const Fbo::Desc& fboDesc, uint32_t arraySize, uint32_t mipLevels)
        {
            if (fboDesc.getSampleCount() > 1)
            {
                logError("creatceCubemap() - can't create a multisampled FBO");
                return nullptr;
            }
            if(CheckParams("CreateCubemap", width, height, arraySize, mipLevels, 0) == false)
            {
                return nullptr;
            }

            Fbo::SharedPtr pFbo = Fbo::create();

            // create the color targets
            for(uint32_t i = 0; i < Fbo::getMaxColorTargetCount(); i++)
            {
                Texture::BindFlags flags = getBindFlags(false, fboDesc.isColorTargetUav(i));
                auto pTex = Texture::createCube(width, height, fboDesc.getColorTargetFormat(i), arraySize, mipLevels, nullptr, flags);
                pFbo->attachColorTarget(pTex, i, 0, Fbo::kAttachEntireMipLevel);
            }

            if(fboDesc.getDepthStencilFormat() != ResourceFormat::Unknown)
            {
                Texture::BindFlags flags = getBindFlags(true, fboDesc.isDepthStencilUav());
                auto pDepth = Texture::createCube(width, height, fboDesc.getDepthStencilFormat(), arraySize, mipLevels, nullptr, flags);
                pFbo->attachDepthStencilTarget(pDepth, 0, Fbo::kAttachEntireMipLevel);
            }

            return pFbo;
        }
    }
}