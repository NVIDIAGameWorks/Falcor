/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Falcor.h"
#include "FalcorExperimental.h"
#include "SSAOData.slang"
#include "../Utils/GaussianBlur/GaussianBlur.h"

using namespace Falcor;

class SSAO : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<SSAO>;

    static const char* kDesc;

    enum class SampleDistribution : uint32_t
    {
        Random,
        UniformHammersley,
        CosineHammersley
    };

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    std::string getDesc() override { return kDesc; }
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override { mpScene = pScene; }
    virtual void renderUI(Gui::Widgets& widget) override;

    void setSampleRadius(float radius);
    void setKernelSize(uint32_t kernelSize);
    void setDistribution(uint32_t distribution);
    float getSampleRadius() { return mData.radius; }
    uint32_t getKernelSize() { return mData.kernelSize; }
    uint32_t getDistribution() { return (uint32_t)mHemisphereDistribution; }

private:
    SSAO();
    Texture::SharedPtr generateAOMap(RenderContext* pContext, const Camera* pCamera, const Texture::SharedPtr& pDepthTexture, const Texture::SharedPtr& pNormalTexture);
    void setNoiseTexture(uint32_t width, uint32_t height);
    void setKernel();

    SSAOData mData;
    bool mDirty = false;

    Fbo::SharedPtr mpAOFbo;
    uint2 mAoMapSize = uint2(1024);

    Sampler::SharedPtr mpNoiseSampler;
    Texture::SharedPtr mpNoiseTexture;
    uint2 mNoiseSize = uint2(16);

    Sampler::SharedPtr mpTextureSampler;
    SampleDistribution mHemisphereDistribution = SampleDistribution::CosineHammersley;

    FullScreenPass::SharedPtr mpSSAOPass;
    RenderGraph::SharedPtr mpBlurGraph;
    Dictionary mBlurDict;
    bool mApplyBlur = true;

    Scene::SharedPtr mpScene;

    struct
    {
        FullScreenPass::SharedPtr pApplySSAOPass;
        Fbo::SharedPtr pFbo;
    } mComposeData;
};
