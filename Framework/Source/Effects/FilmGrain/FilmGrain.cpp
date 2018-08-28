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
#include "FilmGrain.h"
#include "Graphics/Program/ProgramVars.h"
#include "Graphics/Camera/Camera.h"
#include "API/RenderContext.h"
#include "Utils/Gui.h"
#include "Graphics/RenderGraph/RenderPassSerializer.h"
#include "Externals/GLM/glm/gtc/noise.hpp"

namespace Falcor
{
    static const char* kInputOutputName = "color";
    const uint32_t kMaxKernelSize = 15;


    FilmGrain::SharedPtr FilmGrain::create(float grainSize, float intensity,
        const glm::vec3& grainColor, const glm::vec2& luminanceRange, bool useLuminanceRange, bool useColoredNoise)
    {
        return SharedPtr(new FilmGrain(grainSize, intensity, grainColor,
            luminanceRange, useLuminanceRange, useColoredNoise));
    }

    FilmGrain::FilmGrain(float grainSize, float intensity, const glm::vec3& grainColor, 
        const glm::vec2& luminanceRange, bool useLuminanceRange, bool useColoredNoise)
        : RenderPass("FilmGrain"), mGrainSize(grainSize), mIntensity(intensity), 
        mGrainColor(grainColor), mLuminanceRange(luminanceRange), mUseLuminanceRange(useLuminanceRange), mUseColoredNoise(useColoredNoise)
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap);
        mpSampler = Sampler::create(samplerDesc);

        createProgram();
    }

    void FilmGrain::createProgram()
    {
        Program::DefineList defineList;
        defineList.add("_LUMINANCE_RANGE_MIN", std::to_string(mLuminanceRange.x));
        defineList.add("_LUMINANCE_RANGE_MAX", std::to_string(mLuminanceRange.y));
        mpBlitPass = FullScreenPass::create("Effects/FilmGrain.ps.slang", defineList);

        ProgramReflection::SharedConstPtr pReflector = mpBlitPass->getProgram()->getReflector();
        mpVars = GraphicsVars::create(pReflector);
        mSrcTexLoc = pReflector->getDefaultParameterBlock()->getResourceBinding("srcTex");
        mNoiseTexLoc = pReflector->getDefaultParameterBlock()->getResourceBinding("noiseTex");
        mpVars = GraphicsVars::create(pReflector);
        mpVars->setSampler("gSampler", mpSampler);

        mpVars["filmGrain"]["lumRange"] = mLuminanceRange;
        mpVars["filmGrain"]["intensity"] = mIntensity;
        mpVars["filmGrain"]["grainSize"] = 1.0f / mGrainSize;
        mpVars["filmGrain"]["grainTint"] = mUseColoredNoise ? mGrainColor : glm::vec3(1.0f, 1.0f, 1.0f);
        
        if (mUseLuminanceRange)
        {
            mpBlitPass->getProgram()->addDefine("_USE_LUMINANCE_RANGE");
        }
    }

    FilmGrain::SharedPtr FilmGrain::deserialize(const RenderPassSerializer& serializer)
    {
        Scene::UserVariable grainSizeVar = serializer.getValue("filmGrain.grainSize");
        if (grainSizeVar.type == Scene::UserVariable::Type::Unknown)
        {
            return create();
        }

        float grainSize = static_cast<float>(grainSizeVar.d64);
        float intensity = static_cast<float>(serializer.getValue("filmGrain.intensity").d64);
        glm::vec3 grainColor = serializer.getValue("filmGrain.grainColor").vec3;
        glm::vec2 luminanceRange = serializer.getValue("filmGrain.luminanceRange").vec2;
        bool useColoredNoise = serializer.getValue("filmGrain.useColoredNoise").b;
        bool useLuminanceRange = serializer.getValue("filmGrain.useLuminanceRange").b;

        return create(grainSize, intensity, grainColor, luminanceRange,
            useColoredNoise, useLuminanceRange);
    }

    void FilmGrain::reflect(RenderPassReflection& reflector) const
    {
        reflector.addInputOutput("color");
    }

    void FilmGrain::serialize(RenderPassSerializer& renderPassSerializer)
    {
        renderPassSerializer.addVariable("filmGrain.grainSize", mGrainSize);
        renderPassSerializer.addVariable("filmGrain.intensity", mIntensity);
        renderPassSerializer.addVariable("filmGrain.grainColor", mGrainColor);
        renderPassSerializer.addVariable("filmGrain.luminanceRange", mLuminanceRange);
        renderPassSerializer.addVariable("filmGrain.useColoredNoise", mUseColoredNoise);
        renderPassSerializer.addVariable("filmGrain.useLuminanceRange", mUseLuminanceRange);
    }

    void FilmGrain::execute(RenderContext* pRenderContext, const RenderData* pData)
    {
        if (!mpTargetFbo) mpTargetFbo = Fbo::create();

        mpTargetFbo->attachColorTarget(pData->getTexture(kInputOutputName), 0);
        execute(pRenderContext, mpTargetFbo);
    }

    void FilmGrain::createNoiseTexture()
    {
        const uint32_t noiseTexHeight = 512;
        const uint32_t noiseTexWidth = static_cast<uint32_t>(512.0f * (mResolution.x / mResolution.y));
        std::vector<float> data;
        data.resize(noiseTexHeight * noiseTexWidth);
        const float denomY = glm::max(0.001f, mGrainSize * mResolution.y / (512.0f));
        const float denomX = glm::max(0.001f, mGrainSize * mResolution.x / (static_cast<float>(noiseTexWidth)));
        
        for (uint32_t i = 0; i < noiseTexHeight; ++i)
        {
            for (uint32_t j = 0; j < noiseTexWidth; ++j)
            {
                float2 noiseInput{ static_cast<float>(j) / denomX, static_cast<float>(i) / denomY };
                float2 noiseInput1{ static_cast<float>(noiseTexWidth - j - 1) / denomX, static_cast<float>(noiseTexHeight - i - 1) / denomY };
                data[i * noiseTexWidth + j] = poisson(mIntensity * 2.0f, glm::simplex(noiseInput1) * glm::simplex(noiseInput));
            }
        }
        
        mpNoiseTex = Texture::create2D(
            noiseTexWidth, noiseTexHeight, ResourceFormat::R32Float, 1, 1, (void*)data.data(), Resource::BindFlags::ShaderResource);
    }

    void FilmGrain::execute(RenderContext* pRenderContext, Fbo::SharedPtr pFbo)
    {
        Texture::SharedPtr pSrcColorTexture = pFbo->getColorTexture(0);

        if (mDirty)
        {
            createProgram();
            mDirty = false;
        }

        if (!mpNoiseTex || mResolution.x != pFbo->getWidth() || mResolution.y != pFbo->getHeight())
        {
            mResolution.x = static_cast<float>(pFbo->getWidth());
            mResolution.y = static_cast<float>(pFbo->getHeight());
            createNoiseTexture();
        }

        mpVars["filmGrain"]["intensity"] = mIntensity;
        if (!mPaused)
        {
            mpVars["filmGrain"]["randOffset"] = glm::vec2(std::rand() % 2000 / 2000.0f, std::rand() % 2000 / 2000.0f);
        }
        
        mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, pFbo->getColorTexture(0)->getSRV());
        mpVars->getDefaultBlock()->setSrv(mNoiseTexLoc, 0, mpNoiseTex->getSRV());

        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
        pState->pushFbo(pFbo);
        pRenderContext->pushGraphicsVars(mpVars);
        mpBlitPass->execute(pRenderContext, nullptr);
        pRenderContext->popGraphicsVars();
        pState->popFbo();
    }

    void FilmGrain::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (uiGroup == nullptr || pGui->beginGroup(uiGroup))
        {
            if (pGui->addFloatVar("Grain Size", mGrainSize, 0.001f))
            {
                mpVars["filmGrain"]["grainSize"] = 1.0f / mGrainSize;
                createNoiseTexture();
            }

            if (pGui->addFloatVar("Intensity", mIntensity, 0.001f, 1.0f))
            {
                mpVars["filmGrain"]["intensity"] = mIntensity;
            }

            if (pGui->addCheckBox("Use Luminance Range", mUseLuminanceRange))
            {
                if (mUseLuminanceRange)
                {
                    mpBlitPass->getProgram()->addDefine("_USE_LUMINANCE_RANGE");
                }
                else
                {
                    mpBlitPass->getProgram()->removeDefine("_USE_LUMINANCE_RANGE");
                }
            }

            if (pGui->addCheckBox("Use Colored Noise", mUseColoredNoise))
            {
                mpVars["filmGrain"]["grainTint"] = mUseColoredNoise ? mGrainColor : glm::vec3(1.0f, 1.0f, 1.0f);
            }

            if (mUseColoredNoise)
            {
                if (pGui->addRgbColor("Grain Color", mGrainColor))
                {
                    mpVars["filmGrain"]["grainTint"] = mGrainColor;
                }
            }

            pGui->addCheckBox("Pause", mPaused);

            if (mUseLuminanceRange)
            {
                mDirty |= pGui->addFloat2Var("Luminance Range", mLuminanceRange, 0.0f);
            }

            pGui->endGroup();
        }
    }

}
