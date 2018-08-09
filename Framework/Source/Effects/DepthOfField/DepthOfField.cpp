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
        for (uint32_t i = 0; i < mpBlurPasses.size(); ++i)
        {
            mpBlurPasses[i] = GaussianBlur::create(5 + 2 * i, (7.0f + 2.0f * i) / 2.0f);
        }
        
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
        mpTempFbo = Fbo::create();
        

        setCamera(pCamera);
    }

    DepthOfField::DepthOfField(float planeOfFocus, float aperture, float focalLength, float nearZ, float farZ)
        : mPlaneOfFocus(planeOfFocus), mAperture(aperture), mFocalLength(focalLength), mNearZ(nearZ), mFarZ(farZ)
    {
    }

    void DepthOfField::setCamera(const Camera::SharedConstPtr& pCamera)
    {
        mpCamera = pCamera;

        if (pCamera)
        {
            mPlaneOfFocus = pCamera->getFocalDistance();
            mAperture = pCamera->getApertureRadius();
            mFocalLength = pCamera->getFocalLength();
            mNearZ = pCamera->getNearPlane();
            mFarZ = pCamera->getFarPlane();
        }

        mpVars["DepthOfField"]["planeOfFocus"] = mPlaneOfFocus;
        mpVars["DepthOfField"]["aperture"] = mAperture;
        mpVars["DepthOfField"]["focalLength"] = mFocalLength;
        mpVars["DepthOfField"]["nearZ"] = mNearZ;
        mpVars["DepthOfField"]["farZ"] = mFarZ;
    }

    void DepthOfField::execute(RenderContext* pRenderContext, Fbo::SharedPtr pFbo)
    {
        Texture::SharedPtr pSrcColorTexture = pFbo->getColorTexture(0);
        mpVars["DepthOfField"]["nearZ"] = mNearZ;

        if (!mpBlurredFbo->getColorTexture(0) ||
            (mpBlurredFbo->getColorTexture(0)->getWidth()  != pSrcColorTexture->getWidth() ||
             mpBlurredFbo->getColorTexture(0)->getHeight() != pSrcColorTexture->getHeight()))
        {
            Texture::SharedPtr pBlurredArrayImage = Texture::create2D(
                pSrcColorTexture->getWidth(), pSrcColorTexture->getHeight(),
                pFbo->getColorTexture(0)->getFormat(), static_cast<uint32_t>(mpBlurPasses.size()), 
                1, nullptr,
                Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);

            mpBlurredFbo->attachColorTarget(pBlurredArrayImage, 0, 0, 0, 1);
            mpTempFbo->attachColorTarget(pBlurredArrayImage, 0, 0, 0, 1);
        }

        pRenderContext->blit(pSrcColorTexture->getSRV(), mpBlurredFbo->getColorTexture(0)->getRTV());
        
        // blur the entire array texture
        for (uint32_t i = 0; i < mpBlurredFbo->getColorTexture(0)->getArraySize() - 1; ++i)
        {
            mpTempFbo->attachColorTarget(mpTempFbo->getColorTexture(0), 0, 0, i, 1);
            mpBlurredFbo->attachColorTarget(mpBlurredFbo->getColorTexture(0), 0, 0, i + 1, 1);
            mpBlurPasses[i]->execute(pRenderContext, mpTempFbo->getColorTexture(0), mpBlurredFbo, i);
        }

        mpBlurredFbo->attachColorTarget(mpBlurredFbo->getColorTexture(0), 0);

        mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, mpBlurredFbo->getColorTexture(0)->getSRV());
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
            //if (!mpCamera)
            {
                mpVars["DepthOfField"]->renderUI(pGui, "");
            }
            
            pGui->endGroup();
        }
    }

}
