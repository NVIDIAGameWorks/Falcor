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
#include "SSAOPass.h"
#include "glm/gtc/random.hpp"

namespace Falcor
{
    const char* SSAOPass::kDesc = "Screen-space ambient occlusion. Can be used with and without a normal-map";

    namespace
    {
        const Gui::DropdownList kDistributionDropdown =
        {
            { (uint32_t)SSAOPass::SampleDistribution::Random, "Random" },
            { (uint32_t)SSAOPass::SampleDistribution::UniformHammersley, "Uniform Hammersley" },
            { (uint32_t)SSAOPass::SampleDistribution::CosineHammersley, "Cosine Hammersley" }
        };

        const std::string kAoMapSize = "aoMapSize";
        const std::string kKernelSize = "kernelSize";
        const std::string kNoiseSize = "noiseSize";
        const std::string kDistribution = "distribution";
        const std::string kBlurDict = "blurDict";

        const std::string kColorIn = "colorIn";
        const std::string kColorOut = "colorOut";
        const std::string kDepth = "depth";
        const std::string kNormals = "normals";
        const std::string kAoMap = "AoMap";

        const std::string kSSAOShader = "Effects/SSAO.ps.slang";
        const std::string kApplySSAOShader = "ApplyAO.ps.slang";
    }

    SSAOPass::SSAOPass()
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap);
        mpNoiseSampler = Sampler::create(samplerDesc);

        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpTextureSampler = Sampler::create(samplerDesc);

        mpSSAOPass = FullScreenPass::create(kSSAOShader);
        mComposeData.pApplySSAOPass = FullScreenPass::create(kApplySSAOShader);
        Sampler::Desc desc;
        desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        mComposeData.pApplySSAOPass["gSampler"] = Sampler::create(desc);
        mComposeData.pFbo = Fbo::create();
    }

    SSAOPass::SharedPtr SSAOPass::create(RenderContext* pRenderContext, const Dictionary& dict)
    {
        SharedPtr pSSAO = SharedPtr(new SSAOPass);
        Dictionary blurDict;
        for (const auto& v : dict)
        {
            if (v.key() == kAoMapSize) pSSAO->mAoMapSize = v.val();
            if (v.key() == kKernelSize) pSSAO->mData.kernelSize = v.val();
            if (v.key() == kNoiseSize) pSSAO->mNoiseSize = v.val();
            if (v.key() == kDistribution) pSSAO->mHemisphereDistribution = v.val();
            if (v.key() == kBlurDict) pSSAO->mBlurDict = v.val();
            else logWarning("Unknown field '" + v.key() + "' in a SSAOPass dictionary");
        }
        return pSSAO;
    }

    Dictionary SSAOPass::getScriptingDictionary()
    {
        Dictionary dict;
        dict[kAoMapSize] = mAoMapSize;
        dict[kKernelSize] = mData.kernelSize;
        dict[kNoiseSize] = mNoiseSize;
        dict[kDistribution] = mHemisphereDistribution;
        dict[kBlurDict] = mpBlurGraph->getPass("GaussianBlur")->getScriptingDictionary();
        return dict;
    }

    RenderPassReflection SSAOPass::reflect(const CompileData& compileData)
    {
        RenderPassReflection reflector;
        reflector.addInput(kColorIn, "Color buffer");
        reflector.addOutput(kColorOut, "Color-buffer with AO applied to it");
        reflector.addInput(kDepth, "Depth-buffer");
        reflector.addInput(kNormals, "World space normals, [0, 1] range").flags(RenderPassReflection::Field::Flags::Optional);
        reflector.addInternal(kAoMapSize, "AO Map");
        return reflector;
    }

    void SSAOPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
    {
        Fbo::Desc fboDesc;
        fboDesc.setColorTarget(0, Falcor::ResourceFormat::R8Unorm);
        mpAOFbo = Fbo::create2D(mAoMapSize.x, mAoMapSize.y, fboDesc);

        setKernel();
        setNoiseTexture(mNoiseSize.x, mNoiseSize.y);

        mpBlurGraph = RenderGraph::create("Gaussian Blur");
        GaussianBlurPass::SharedPtr pBlurPass = GaussianBlurPass::create(pRenderContext, mBlurDict);
        mpBlurGraph->addPass(pBlurPass, "GaussianBlur");
        mpBlurGraph->markOutput("GaussianBlur.dst");
        mpBlurGraph->setScene(mpScene);
    }

    void SSAOPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
    {
        if (!mpScene) return;

        // Run the AO pass
        auto pDepth = renderData[kDepth]->asTexture();
        auto pNormals = renderData[kNormals]->asTexture();
        auto pColorOut = renderData[kColorOut]->asTexture();
        auto pColorIn = renderData[kColorIn]->asTexture();
        auto pAoMap = renderData[kAoMap]->asTexture();

        assert(pColorOut != pColorIn);
        pAoMap = generateAOMap(pRenderContext, mpScene->getCamera().get(), pDepth, pNormals);

        if (mApplyBlur)
        {
            mpBlurGraph->setInput("GaussianBlur.src", pAoMap);
            mpBlurGraph->execute(pRenderContext);
            pAoMap = mpBlurGraph->getOutput("GaussianBlur.dst")->asTexture();
        }

        mComposeData.pApplySSAOPass["gColor"] = pColorIn;
        mComposeData.pApplySSAOPass["gAOMap"] = pAoMap;
        mComposeData.pFbo->attachColorTarget(pColorOut, 0);
        mComposeData.pApplySSAOPass->execute(pRenderContext, mComposeData.pFbo);
    }

    Texture::SharedPtr SSAOPass::generateAOMap(RenderContext* pContext, const Camera* pCamera, const Texture::SharedPtr& pDepthTexture, const Texture::SharedPtr& pNormalTexture)
    {
        if (mDirty)
        {
            ConstantBuffer::SharedPtr pCB = mpSSAOPass["SSAOCB"];
            if (pCB != nullptr) pCB->setBlob(&mData, 0, sizeof(mData));
            mDirty = false;
        }

        // Update state/vars
        mpSSAOPass["gNoiseSampler"] = mpNoiseSampler;
        mpSSAOPass["gTextureSampler"] = mpTextureSampler;
        mpSSAOPass["gDepthTex"] = pDepthTexture;
        mpSSAOPass["gNoiseTex"] = mpNoiseTexture;
        mpSSAOPass["gNormalTex"] = pNormalTexture;

        // Generate AO
        mpSSAOPass->execute(pContext, mpAOFbo);
        return mpAOFbo->getColorTexture(0);
    }

    void SSAOPass::renderUI(Gui::Widgets& widget)
    {
        uint32_t distribution = (uint32_t)mHemisphereDistribution;
        if (widget.dropdown("Kernel Distribution", kDistributionDropdown, distribution)) setDistribution(distribution);

        int32_t size = mData.kernelSize;
        if (widget.var("Kernel Size", size, 1, MAX_SAMPLES)) setKernelSize(size);

        float radius = mData.radius;
        if (widget.var("Sample Radius", radius, 0.001f, FLT_MAX, 0.001f)) setSampleRadius(radius);

        widget.checkbox("Apply Blur", mApplyBlur);
        if (mApplyBlur)
        {
            auto blurGroup = Gui::Group(widget, "Blur Settings");
            if (blurGroup.open())
            {
                mpBlurGraph->getPass("GaussianBlur")->renderUI(blurGroup);
                blurGroup.release();
            }   
        }
    }

    void SSAOPass::setSampleRadius(float radius)
    {
        mData.radius = radius;
        mDirty = true;
    }

    void SSAOPass::setKernelSize(uint32_t kernelSize)
    {
        kernelSize = glm::clamp(kernelSize, (uint32_t)1, (uint32_t)MAX_SAMPLES);
        mData.kernelSize = kernelSize;
        setKernel();
    }

    void SSAOPass::setDistribution(uint32_t distribution)
    {
        mHemisphereDistribution = (SampleDistribution)distribution;
        setKernel();
    }

    void SSAOPass::setKernel()
    {
        for (uint32_t i = 0; i < mData.kernelSize; i++)
        {
            // Hemisphere in the Z+ direction
            glm::vec3 p;
            switch (mHemisphereDistribution)
            {
            case SampleDistribution::Random:
                p = glm::normalize(glm::linearRand(glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)));
                break;

            case SampleDistribution::UniformHammersley:
                p = hammersleyUniform(i, mData.kernelSize);
                break;

            case SampleDistribution::CosineHammersley:
                p = hammersleyCosine(i, mData.kernelSize);
                break;
            }

            mData.sampleKernel[i] = glm::vec4(p, 0.0f);

            // Skew sample point distance on a curve so more cluster around the origin
            float dist = (float)i / (float)mData.kernelSize;
            dist = glm::mix(0.1f, 1.0f, dist * dist);
            mData.sampleKernel[i] *= dist;
        }

        mDirty = true;
    }

    void SSAOPass::setNoiseTexture(uint32_t width, uint32_t height)
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

    SCRIPT_BINDING(SSAOPass)
    {
        auto c = m.regClass(SSAOPass);
        c.func_("kernelRadius", &SSAOPass::setKernelSize);
        c.func_("kernelRadius", &SSAOPass::getKernelSize);
        c.func_("distribution", &SSAOPass::setDistribution);
        c.func_("distribution", &SSAOPass::getDistribution);
        c.func_("sampleRadius", &SSAOPass::setSampleRadius);
        c.func_("sampleRadius", &SSAOPass::getSampleRadius);
    }
}
