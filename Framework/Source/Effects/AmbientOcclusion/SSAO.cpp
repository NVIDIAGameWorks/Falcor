/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "Effects/AmbientOcclusion/SSAO.h"
#include "Graphics/FboHelper.h"
#include "API/RenderContext.h"
#include "Graphics/Camera/Camera.h"
#include "glm/gtc/random.hpp"
#include "glm/gtc/packing.hpp"
#include "Utils/Math/FalcorMath.h"
#include "Graphics/Scene/Scene.h"

namespace Falcor
{
    const Gui::DropdownList SSAO::kDistributionDropdown = 
    {
        { (int32_t)SampleDistribution::Random, "Random" },
        { (int32_t)SampleDistribution::UniformHammersley, "Uniform Hammersley" },
        { (int32_t)SampleDistribution::CosineHammersley, "Cosine Hammersley" }
    };

    SSAO::SharedPtr SSAO::create(const uvec2& aoMapSize, uint32_t kernelSize, uint32_t blurSize, float blurSigma, const uvec2& noiseSize, SampleDistribution distribution)
    {
        return SharedPtr(new SSAO(aoMapSize, kernelSize, blurSize, blurSigma, noiseSize, distribution));
    }

    void SSAO::renderUI(Gui* pGui, const char* uiGroup)
    {
        if(!uiGroup || pGui->beginGroup(uiGroup))
        {
            if (pGui->addDropdown("Kernel Distribution", kDistributionDropdown, mHemisphereDistribution))
            {
                setKernel(mData.kernelSize, (SampleDistribution)mHemisphereDistribution);
            }

            int32_t size = mData.kernelSize;
            if (pGui->addIntVar("Kernel Size", size, 1, MAX_SAMPLES))
            {
                setKernel((uint32_t)size, (SampleDistribution)mHemisphereDistribution);
            }

            if (pGui->addFloatVar("Sample Radius", mData.radius, 0.001f, FLT_MAX))
            {
                mDirty = true;
            }

            pGui->addCheckBox("Apply Blur", mApplyBlur);

            if (mApplyBlur)
            {
                mpBlur->renderUI(pGui, "Blur Settings");
            }

            if (uiGroup) pGui->endGroup();
        }
    }

    Texture::SharedPtr SSAO::generateAOMap(RenderContext* pContext, const Camera* pCamera, const Texture::SharedPtr& pDepthTexture, const Texture::SharedPtr& pNormalTexture)
    {
        upload();

        // Update state/vars
        mpSSAOState->setFbo(mpAOFbo);
        ParameterBlock* pDefaultBlock = mpSSAOVars->getDefaultBlock().get();
        pDefaultBlock->setSampler(mBindLocations.noiseSampler, 0, mpNoiseSampler);
        pDefaultBlock->setSampler(mBindLocations.textureSampler, 0, mpTextureSampler);
        pDefaultBlock->setSrv(mBindLocations.depthTex, 0, pDepthTexture->getSRV());
        pDefaultBlock->setSrv(mBindLocations.noiseTex, 0, mpNoiseTexture->getSRV());
        pDefaultBlock->setSrv(mBindLocations.normalTex, 0, pNormalTexture->getSRV());

        ConstantBuffer* pCB = pDefaultBlock->getConstantBuffer(mBindLocations.internalPerFrameCB, 0).get();
        if (pCB != nullptr)
        {
            pCamera->setIntoConstantBuffer(pCB, 0);
        }

        // Generate AO
        pContext->pushGraphicsState(mpSSAOState);
        pContext->pushGraphicsVars(mpSSAOVars);
        mpSSAOPass->execute(pContext);
        pContext->popGraphicsVars();
        pContext->popGraphicsState();

        // Blur
        if (mApplyBlur)
        {
            mpBlur->execute(pContext, mpAOFbo->getColorTexture(0), mpAOFbo);
        }

        return mpAOFbo->getColorTexture(0);
    }

    SSAO::SSAO(const uvec2& aoMapSize, uint32_t kernelSize, uint32_t blurSize, float blurSigma, const uvec2& noiseSize, SampleDistribution distribution) : RenderPass("SSAO")
    {
        Fbo::Desc fboDesc;
        fboDesc.setColorTarget(0, Falcor::ResourceFormat::R8Unorm);
        mpAOFbo = FboHelper::create2D(aoMapSize.x, aoMapSize.y, fboDesc);

        initShader();

        mpSSAOState = GraphicsState::create();

        mpBlur = GaussianBlur::create(5, 2.0f);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap);
        mpNoiseSampler = Sampler::create(samplerDesc);

        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpTextureSampler = Sampler::create(samplerDesc);

        setKernel(kernelSize, distribution);
        setNoiseTexture(noiseSize.x, noiseSize.y);

        mComposeData.pApplySSAOPass = FullScreenPass::create("ApplyAO.ps.slang");
        mComposeData.pVars = GraphicsVars::create(mComposeData.pApplySSAOPass->getProgram()->getReflector());

        Sampler::Desc desc;
        desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        mComposeData.pVars->setSampler("gSampler", Sampler::create(desc));
        mComposeData.pState = GraphicsState::create();
        mComposeData.pState->setFbo(Fbo::create());
    }

    void SSAO::upload()
    {
        if (mDirty)
        {
            ConstantBuffer* pCB = mpSSAOVars->getDefaultBlock()->getConstantBuffer(mBindLocations.ssaoCB, 0).get();
            if (pCB != nullptr)
            {
                pCB->setBlob(&mData, 0, sizeof(mData));
            }

            mDirty = false;
        }
    }

    void SSAO::initShader()
    {
        mpSSAOPass = FullScreenPass::create("Effects/SSAO.ps.slang");
        mpSSAOVars = GraphicsVars::create(mpSSAOPass->getProgram()->getReflector());

        const ParameterBlockReflection* pReflector = mpSSAOPass->getProgram()->getReflector()->getDefaultParameterBlock().get();
        mBindLocations.internalPerFrameCB = pReflector->getResourceBinding("InternalPerFrameCB");
        mBindLocations.ssaoCB = pReflector->getResourceBinding("SSAOCB");
        mBindLocations.noiseSampler = pReflector->getResourceBinding("gNoiseSampler");
        mBindLocations.textureSampler = pReflector->getResourceBinding("gTextureSampler");
        mBindLocations.depthTex = pReflector->getResourceBinding("gDepthTex");
        mBindLocations.normalTex = pReflector->getResourceBinding("gNormalTex");
        mBindLocations.noiseTex = pReflector->getResourceBinding("gNoiseTex");
    }

    void SSAO::setKernel(uint32_t kernelSize, SampleDistribution distribution)
    {
        kernelSize = glm::clamp(kernelSize, (uint32_t)1, (uint32_t)MAX_SAMPLES);
        mData.kernelSize = kernelSize;

        mHemisphereDistribution = (uint32_t)distribution;

        for (uint32_t i = 0; i < kernelSize; i++)
        {
            // Hemisphere in the Z+ direction
            glm::vec3 p;
            switch (distribution)
            {
            case SampleDistribution::Random:
                p = glm::normalize(glm::linearRand(glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)));
                break;

            case SampleDistribution::UniformHammersley:
                p = hammersleyUniform(i, kernelSize);
                break;

            case SampleDistribution::CosineHammersley:
                p = hammersleyCosine(i, kernelSize);
                break;
            }

            mData.sampleKernel[i] = glm::vec4(p, 0.0f);

            // Skew sample point distance on a curve so more cluster around the origin
            float dist = (float)i / (float)kernelSize;
            dist = glm::mix(0.1f, 1.0f, dist * dist);
            mData.sampleKernel[i] *= dist;
        }

        mDirty = true;
    }

    void SSAO::setNoiseTexture(uint32_t width, uint32_t height)
    {
        std::vector<uint32_t> data;
        data.resize(width * height);

        for (uint32_t i = 0; i < width * height; i++)
        {
            // Random directions on the XY plane
            glm::vec2 dir = glm::normalize(glm::linearRand(glm::vec2(-1), glm::vec2(1))) * 0.5f + 0.5f;
            data[i] = glm::packUnorm4x8(glm::vec4(dir, 0.0f, 1.0f));
        }

        mpNoiseTexture = Texture::create2D(width, height, ResourceFormat::RGBA8Unorm, 1, Texture::kMaxPossible, data.data());

        mData.noiseScale = glm::vec2(mpAOFbo->getWidth(), mpAOFbo->getHeight()) / glm::vec2(width, height);

        mDirty = true;
    }

    static const std::string kColorIn = "colorIn";
    static const std::string kColorOut = "colorOut";
    static const std::string kDepth = "depth";
    static const std::string kNormals = "normals";

    void SSAO::reflect(RenderPassReflection& reflector) const
    {
        reflector.addInput(kColorIn);
        reflector.addOutput(kColorOut);
        reflector.addInput(kDepth);
        reflector.addInput(kNormals).setFlags(RenderPassReflection::Field::Flags::Optional);
    }

    void SSAO::execute(RenderContext* pContext, const Camera* pCamera, const Texture::SharedPtr& pColorIn, const Texture::SharedPtr& pColorOut, const Texture::SharedPtr& pDepthTexture, const Texture::SharedPtr& pNormalTexture)
    {
        assert(pColorOut != pColorIn);
        auto& pAoMap = generateAOMap(pContext, mpScene->getActiveCamera().get(), pDepthTexture, pNormalTexture);

        mComposeData.pVars->setTexture("gColor", pColorIn);
        mComposeData.pVars->setTexture("gAOMap", pAoMap);
        auto& pFbo = mComposeData.pState->getFbo();
        pFbo->attachColorTarget(pColorOut, 0);
        mComposeData.pState->setFbo(pFbo);

        pContext->pushGraphicsState(mComposeData.pState);
        pContext->pushGraphicsVars(mComposeData.pVars);

        mComposeData.pApplySSAOPass->execute(pContext);

        pContext->popGraphicsState();
        pContext->popGraphicsVars();
    }

    void SSAO::execute(RenderContext* pRenderContext, const RenderData* pData)
    {
        // Run the AO pass
        auto& pDepth = pData->getTexture(kDepth);
        auto& pNormals = pData->getTexture(kNormals);        
        auto& pColorOut = pData->getTexture(kColorOut);
        auto& pColorIn = pData->getTexture(kColorIn);
        execute(pRenderContext, mpScene->getActiveCamera().get(), pColorIn, pColorOut, pDepth, pNormals);
    }
}
