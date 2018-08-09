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
#include "DepthOfField.h"
#include "Graphics/Program/ProgramVars.h"
#include "Graphics/Camera/Camera.h"
#include "API/RenderContext.h"
#include "Utils/Gui.h"

namespace Falcor
{
    const uint32_t kMaxKernelSize = 15;

    DepthOfField::UniquePtr DepthOfField::create(const Camera::SharedConstPtr& pCamera)
    {
        return DepthOfField::UniquePtr(new DepthOfField(pCamera));
    }

    DepthOfField::UniquePtr DepthOfField::create(float mPlaneOfFocus, float mAperture, float mFocalLength, float mNearZ, float mFarZ)
    {
        return DepthOfField::UniquePtr(new DepthOfField(mPlaneOfFocus, mAperture, mFocalLength, mNearZ, mFarZ));
    }

    DepthOfField::DepthOfField(const Camera::SharedConstPtr& pCamera)
    {
        mpBlurPass = GaussianBlur::create();
        
        mpBlitPass = FullScreenPass::create("Framework/Shaders/Blit.vs.slang", "Effects/DepthOfField.ps.slang");
        mSrcTexLoc = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("gSrcTex");
        mSrcDepthLoc = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("gSrcDepthTex");
        mpVars = GraphicsVars::create(mpBlitPass->getProgram()->getReflector());
        mpVars["SrcRectCB"]["gOffset"] = vec2(0.0f);
        mpVars["SrcRectCB"]["gScale"] = vec2(1.0f);

        BlendState::Desc desc;
        desc.setRtBlend(0, true);
        desc.setRtParams(0,
            BlendState::BlendOp::Add, BlendState::BlendOp::Add,
            BlendState::BlendFunc::One, BlendState::BlendFunc::Zero,
            BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha);

        mpAdditiveBlend = BlendState::create(desc);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpSampler = Sampler::create(samplerDesc);
        mpVars->setSampler("gSampler", mpSampler);
        
        mpBlurredFbo = Fbo::create();
        
        setCamera(pCamera);
    }

    DepthOfField::DepthOfField(float planeOfFocus, float aperture, float focalLength, float nearZ, float farZ)
        : DepthOfField(nullptr)
    {
        mpVars["DepthOfField"]["planeOfFocus"] = planeOfFocus;
        mpVars["DepthOfField"]["aperture"] = aperture;
        mpVars["DepthOfField"]["focalLength"] = focalLength;
        mpVars["DepthOfField"]["nearZ"] = nearZ;
        mpVars["DepthOfField"]["farZ"] = farZ;
    }

    void DepthOfField::setCamera(const Camera::SharedConstPtr& pCamera)
    {
        mpCamera = pCamera;
        updateFromCamera();
    }

    void DepthOfField::updateFromCamera()
    {
        if (mpCamera)
        {
            mpVars["DepthOfField"]["planeOfFocus"] = mpCamera->getFocalDistance();
            mpVars["DepthOfField"]["aperture"] = mpCamera->getApertureRadius();
            mpVars["DepthOfField"]["focalLength"] = mpCamera->getFocalLength();
            mpVars["DepthOfField"]["nearZ"] = mpCamera->getNearPlane();
            mpVars["DepthOfField"]["farZ"] = mpCamera->getFarPlane();
        }
    }

    void DepthOfField::updateTextures(const Texture::SharedPtr& pTexture)
    {
        uint32_t imageWidth = pTexture->getWidth(), imageHeight = pTexture->getHeight();

        // resize the images so they will fit with the blit
        for (uint32_t i = 0; i < mpBlurredImages.size(); ++i)
        {
            if (!mpBlurredImages[i] ||
                (mpBlurredImages[i]->getWidth() != imageWidth ||
                    mpBlurredImages[i]->getHeight() != imageHeight))
            {
                mpBlurredImages[i] = Texture::create2D(
                    imageWidth, imageHeight,
                    pTexture->getFormat(), 1, 1, nullptr,
                    Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);

                if (i >= 1)
                {
                    if (!mpTempImages[i - 1] ||
                        (mpBlurredImages[i]->getWidth() != mpTempImages[i - 1]->getWidth() ||
                            mpBlurredImages[i]->getHeight() != mpTempImages[i - 1]->getHeight()))
                    {
                        mpTempImages[i - 1] = Texture::create2D(
                            imageWidth, imageHeight,
                            pTexture->getFormat(), 1, 1, nullptr,
                            Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);
                    }
                }
            }

            imageWidth /= 2;
            imageHeight /= 2;
        }
    }

    void DepthOfField::execute(RenderContext* pRenderContext, Fbo::SharedPtr pFbo)
    {
        Texture::SharedPtr pSrcColorTexture = pFbo->getColorTexture(0);
        
        if (!mOverrideCamera)
        {
            updateFromCamera();
        }

        updateTextures(pFbo->getColorTexture(0)); mpCamera->getNearPlane();
        mpVars["DepthOfField"]["nearZ"] = 0.1f;

        pRenderContext->blit(pSrcColorTexture->getSRV(), mpBlurredImages[0]->getRTV());
        
        // down scale the screen 4 times
        pRenderContext->blit(mpBlurredImages[0]->getSRV(), mpTempImages[0]->getRTV());
        pRenderContext->blit(mpTempImages[0]->getSRV(), mpTempImages[1]->getRTV());
        pRenderContext->blit(mpTempImages[1]->getSRV(), mpTempImages[2]->getRTV());
        pRenderContext->blit(mpTempImages[2]->getSRV(), mpTempImages[3]->getRTV());

        // blur the last three images
        for (uint32_t i = 0; i < 4; ++i)
        {
            mpBlurredFbo->attachColorTarget(mpBlurredImages[i + 1], 0);
            mpBlurPass->execute(pRenderContext, mpTempImages[i], mpBlurredFbo);
        }

        for (uint32_t i = 0; i < mpBlurredImages.size(); ++i)
        {
            mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, i, mpBlurredImages[i]->getSRV());
        }
        mpVars->getDefaultBlock()->setSrv(mSrcDepthLoc, 0, pFbo->getDepthStencilTexture()->getSRV());
        
        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
        pState->pushFbo(pFbo);
        pRenderContext->pushGraphicsVars(mpVars);
        mpBlitPass->execute(pRenderContext, nullptr, mpAdditiveBlend);
        pRenderContext->popGraphicsVars();
        pState->popFbo();
    }

    void DepthOfField::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (uiGroup == nullptr || pGui->beginGroup(uiGroup))
        {
            if (mpCamera)
            {
                pGui->addCheckBox("Override Camera", mOverrideCamera);
            }

            if (!mpCamera || mOverrideCamera)
            {
                mpVars["DepthOfField"]->renderUI(pGui, "");
            }
            
            pGui->endGroup();
        }
    }

}
