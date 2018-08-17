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
#include "SubsurfaceScattering.h"
#include "API/RenderContext.h"
#include "Graphics/FboHelper.h"
#include "Utils/Gui.h"
#define _USE_MATH_DEFINES
#include "glm/gtx/compatibility.hpp"
#include <math.h>

namespace Falcor
{
    static std::string kShaderFilename("Effects/SubsurfaceScattering.ps.slang");
    static Gui::DropdownList sScatteringTypeDropdown;

    SubsurfaceScattering::~SubsurfaceScattering() = default;

    SubsurfaceScattering::SubsurfaceScattering(uint32_t kernelWidth, float scatteringWidth, const glm::vec3& color)
        : mKernelWidth(kernelWidth), mScatteringWidth(scatteringWidth), mColor(color)
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpSampler = Sampler::create(samplerDesc);

        Gui::DropdownValue value;
        value.label = "Translucent";
        value.value = SubsurfaceScatteringMode::Translucent;
        sScatteringTypeDropdown.push_back(value);
        value.label = "Skin";
        value.value = SubsurfaceScatteringMode::Skin;
        sScatteringTypeDropdown.push_back(value);

        mDirty = true;
    }

    SubsurfaceScattering::UniquePtr SubsurfaceScattering::create(uint32_t kernelSize, float scatteringWidth, const glm::vec3& color)
    {
        SubsurfaceScattering* pBlur = new SubsurfaceScattering(kernelSize, scatteringWidth, color);
        return SubsurfaceScattering::UniquePtr(pBlur);
    }

    void SubsurfaceScattering::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (uiGroup == nullptr || pGui->beginGroup(uiGroup))
        {
            if (pGui->addRgbColor("Color", mColor, 0))
            {
                updateDiffusionProfile();
                mpVars->setTypedBuffer("gaussianWeights", mpWeights);
            }
            if (pGui->addFloatVar("Strength", mStrength, 0.01f))
            {
                updateDiffusionProfile();
                mpVars->setTypedBuffer("gaussianWeights", mpWeights);
            }
            int32_t kernelWidth = (int32_t)mKernelWidth;
            if (pGui->addIntVar("Kernel Width", kernelWidth, 1))
            {
                mKernelWidth = kernelWidth;
                mDirty = true;
            }
            uint32_t dropdownValue = static_cast<uint32_t>(mMode);
            if (pGui->addDropdown("", sScatteringTypeDropdown, dropdownValue))
            {
                mMode = static_cast<SubsurfaceScatteringMode>(dropdownValue);
                updateDiffusionProfile();
                mpVars->setTypedBuffer("gaussianWeights", mpWeights);
            }

            if (uiGroup) pGui->endGroup();
        }
    }

    float getCoefficient(float sigma, float x)
    {
        float sigmaSquared = sigma * sigma;
        float p = -(x*x) / (2 * sigmaSquared);
        float e = exp(p);
    
        float a = 2 * (float)M_PI * sigmaSquared;
        return e / a;
    }
    
    float getProfile(float s, float r)
    {
        // guassian from seperable paper
        float profileValue = 0.1f * getCoefficient(0.0484f, r);
        profileValue += 0.118f * getCoefficient(0.1847f, r);
        profileValue += 0.113f * getCoefficient(0.567f, r);
        profileValue += 0.358f * getCoefficient(1.99f, r);
        profileValue += 0.078f * getCoefficient(7.41f, r);
        return profileValue;
        // float sr = s * r;
        // float p1 = -sr;
        // float p2 = -sr / 3.0f;
        // float e1 = exp(p1);
        // float e2 = exp(p2);
        // return (e1 + e2) / (8.0f * static_cast<float>(M_PI) * r);
    }

    void SubsurfaceScattering::updateDiffusionProfile()
    {
        mpWeights = TypedBuffer<float>::create(mKernelWidth * 4, Resource::BindFlags::ShaderResource);

        // set per sample offset for gaussian
        for (uint32_t i = 0; i < mKernelWidth; ++i)
        {
            float offset = -2.0f + i * 6.0f / (float(mKernelWidth) - 1.0f);
            float sign = glm::sign(offset);
            mpWeights[i * 4 + 3] = sign * offset * offset / 3.0f;
        }

        glm::vec3 sum{0.0f, 0.0f, 0.0f};

        // calculate the gaussian weights using the profile described in the paper
        for (uint32_t i = 0; i < mKernelWidth; ++i)
        {
            float offsetArea = 0.0f;
            if(i < mKernelWidth - 1) offsetArea += std::abs(mpWeights[i * 4 + 3] - mpWeights[(i + 1) * 4 + 3]);
            if (i > 0) offsetArea += std::abs(mpWeights[i * 4 + 3] - mpWeights[(i - 1) * 4 + 3]);
            offsetArea /= 2;
            for (uint32_t j = 0; j < 3; ++j)
            {
                // float m = (mStrength - 0.33f);
                // mScatteringWidth = (3.5f + 100.0f * m * m * m * m);
                float m = (mStrength - 0.8f);
                mScatteringWidth = (1.9f - mStrength + 3.5f * m * m);

                float weight = offsetArea * getProfile(mScatteringWidth, mpWeights[i * 4 + 3]);
                mpWeights[i * 4 + j] = weight;
                sum[j] += weight;
            }
        }

        mpWeights[3] = mStrength;

        int32_t halfKernelWidth = mKernelWidth / 2;
        glm::vec4 middleWeight{ mpWeights[halfKernelWidth * 4],
            mpWeights[halfKernelWidth * 4 + 1], 
            mpWeights[halfKernelWidth * 4 + 2],
            mpWeights[halfKernelWidth * 4 + 3] };

        for (int32_t i = halfKernelWidth; i > 0; --i)
        {
            mpWeights[i * 4] = mpWeights[(i - 1) * 4];
            mpWeights[i * 4 + 1] = mpWeights[(i - 1) * 4 + 1];
            mpWeights[i * 4 + 2] = mpWeights[(i - 1) * 4 + 2];
            mpWeights[i * 4 + 3] = mpWeights[(i - 1) * 4 + 3];
        }

        mpWeights[0] = glm::lerp(1.0f, static_cast<float>(middleWeight.x), mStrength);
        mpWeights[1] = glm::lerp(1.0f, static_cast<float>(middleWeight.y), mStrength);
        mpWeights[2] = glm::lerp(1.0f, static_cast<float>(middleWeight.z), mStrength);


        // normalize the diffusion weights similar to the old paper
        for (uint32_t i = 0; i < mKernelWidth; ++i)
        {
            mpWeights[i * 4 ] = mpWeights[i * 4] * mStrength / sum.x;
            mpWeights[i * 4 + 1] = mpWeights[i * 4 + 1] * mStrength  / sum.y;
            mpWeights[i * 4 + 2] = mpWeights[i * 4 + 2] * mStrength  / sum.z;
        }
    }

    void SubsurfaceScattering::createTmpFbo(const Texture* pSrc)
    {
        bool createFbo = mpTmpFbo == nullptr;
        ResourceFormat srcFormat = pSrc->getFormat();

        if (createFbo == false)
        {
            createFbo = (pSrc->getWidth() != mpTmpFbo->getWidth()) ||
                (pSrc->getHeight() != mpTmpFbo->getHeight()) ||
                (srcFormat != mpTmpFbo->getColorTexture(0)->getFormat()) ||
                pSrc->getArraySize() != mpTmpFbo->getColorTexture(0)->getArraySize();
        }

        if (createFbo)
        {
            Fbo::Desc fboDesc;
            fboDesc.setColorTarget(0, srcFormat);
            mpTmpFbo = FboHelper::create2D(pSrc->getWidth(), pSrc->getHeight(), fboDesc, pSrc->getArraySize());
        }
    }

    void SubsurfaceScattering::createProgram()
    {
        Program::DefineList defines;
        defines.add("_KERNEL_WIDTH", std::to_string(mKernelWidth));
        
        uint32_t arraySize = mpTmpFbo->getColorTexture(0)->getArraySize();
        mpBlurPass = FullScreenPass::create(kShaderFilename, defines, true, true);
        
        ProgramReflection::SharedConstPtr pReflector = mpBlurPass->getProgram()->getReflector();
        mpVars = GraphicsVars::create(pReflector);

        mSrcTexLoc = pReflector->getDefaultParameterBlock()->getResourceBinding("gSrcTex");
        mSrcDepthTexLoc = pReflector->getDefaultParameterBlock()->getResourceBinding("gSrcDepthTex");
        mSrcOcclTexLoc = pReflector->getDefaultParameterBlock()->getResourceBinding("gSrcOcclusionTex");
        mpVars->setSampler("gSampler", mpSampler);

        mpVars->setTypedBuffer("gaussianWeights", mpWeights);

        mpVars["SubsurfaceParams"]["scatteringWidth"] = mScatteringWidth;
    }

    void SubsurfaceScattering::execute(RenderContext* pRenderContext, Texture::SharedPtr pDiffuseSrc, Texture::SharedPtr pSrcDepth, Fbo::SharedPtr pDst, Texture::SharedPtr pSrcMaskTex)
    {
        createTmpFbo(pDiffuseSrc.get());
        if (mDirty)
        {
            updateDiffusionProfile();
            createProgram();
            mDirty = false;
        }

        uint32_t arraySize = pDiffuseSrc->getArraySize();
        GraphicsState::Viewport vp;
        vp.originX = 0;
        vp.originY = 0;
        vp.height = (float)mpTmpFbo->getHeight();
        vp.width = (float)mpTmpFbo->getWidth();
        vp.minDepth = 0;
        vp.maxDepth = 1;
         
        GraphicsState* pState = pRenderContext->getGraphicsState().get();
        for (uint32_t i = 0; i < arraySize; i++)
        {
            pState->pushViewport(i, vp);
        }
        
        mpVars["SubsurfaceParams"]["pixelSize"] = glm::vec2(1.0f / pDst->getWidth(), 1.0f / pDst->getHeight());

        mpVars["SubsurfaceParams"]["dir"] = glm::vec2(1.0f, 0.0f);
        mpVars->getDefaultBlock()->setSrv(mSrcDepthTexLoc, 0, pSrcDepth->getSRV());
        mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, pDiffuseSrc->getSRV());
        mpVars->getDefaultBlock()->setSrv(mSrcOcclTexLoc, 0, pSrcMaskTex->getSRV());

        // Horizontal pass
        pState->pushFbo(mpTmpFbo);
        pRenderContext->pushGraphicsVars(mpVars);
        mpBlurPass->execute(pRenderContext);

        mpVars->setTypedBuffer("gaussianWeights", mpWeights);
        mpVars["SubsurfaceParams"]["scatteringWidth"] = mScatteringWidth;

        // set direction
        mpVars["SubsurfaceParams"]["dir"] = glm::vec2(0.0f, 1.0f);

        // Vertical pass
        mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, mpTmpFbo->getColorTexture(0)->getSRV());
        mpVars->getDefaultBlock()->setSrv(mSrcDepthTexLoc, 0, pSrcDepth->getSRV());

        pRenderContext->setGraphicsVars(mpVars);
        pState->setFbo(pDst);
        mpBlurPass->execute(pRenderContext);

        pState->popFbo();
        for (uint32_t i = 0; i < arraySize; i++)
        {
            pState->popViewport(i);
        }

        pRenderContext->popGraphicsVars();
    }
}