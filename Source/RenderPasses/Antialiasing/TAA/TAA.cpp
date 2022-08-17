/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "TAA.h"

const RenderPass::Info TAA::kInfo { "TAA", "Temporal Anti-Aliasing." };

namespace
{
    const std::string kMotionVec = "motionVecs";
    const std::string kColorIn = "colorIn";
    const std::string kColorOut = "colorOut";

    const std::string kAlpha = "alpha";
    const std::string kColorBoxSigma = "colorBoxSigma";

    const std::string kShaderFilename = "RenderPasses/Antialiasing/TAA/TAA.ps.slang";
}

TAA::TAA()
    : RenderPass(kInfo)
{
    mpPass = FullScreenPass::create(kShaderFilename);
    mpFbo = Fbo::create();
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpLinearSampler = Sampler::create(samplerDesc);
}

TAA::SharedPtr TAA::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pTAA = SharedPtr(new TAA());
    for (const auto& [key, value] : dict)
    {
        if (key == kAlpha) pTAA->mControls.alpha = value;
        else if (key == kColorBoxSigma) pTAA->mControls.colorBoxSigma = value;
        else logWarning("Unknown field '{}' in a TemporalAA dictionary.", key);
    }
    return pTAA;
}

Dictionary TAA::getScriptingDictionary()
{
    Dictionary dict;
    dict[kAlpha] = mControls.alpha;
    dict[kColorBoxSigma] = mControls.colorBoxSigma;
    return dict;
}

RenderPassReflection TAA::reflect(const CompileData& compileData)
{
    RenderPassReflection reflection;
    reflection.addInput(kMotionVec, "Screen-space motion vectors");
    reflection.addInput(kColorIn, "Color-buffer of the current frame");
    reflection.addOutput(kColorOut, "Anti-aliased color buffer");
    return reflection;
}

void TAA::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pColorIn = renderData.getTexture(kColorIn);
    const auto& pColorOut = renderData.getTexture(kColorOut);
    const auto& pMotionVec = renderData.getTexture(kMotionVec);
    allocatePrevColor(pColorOut.get());
    mpFbo->attachColorTarget(pColorOut, 0);

    // Make sure the dimensions match
    FALCOR_ASSERT((pColorIn->getWidth() == mpPrevColor->getWidth()) && (pColorIn->getWidth() == pMotionVec->getWidth()));
    FALCOR_ASSERT((pColorIn->getHeight() == mpPrevColor->getHeight()) && (pColorIn->getHeight() == pMotionVec->getHeight()));
    FALCOR_ASSERT(pColorIn->getSampleCount() == 1 && mpPrevColor->getSampleCount() == 1 && pMotionVec->getSampleCount() == 1);

    mpPass["PerFrameCB"]["gAlpha"] = mControls.alpha;
    mpPass["PerFrameCB"]["gColorBoxSigma"] = mControls.colorBoxSigma;
    mpPass["gTexColor"] = pColorIn;
    mpPass["gTexMotionVec"] = pMotionVec;
    mpPass["gTexPrevColor"] = mpPrevColor;
    mpPass["gSampler"] = mpLinearSampler;

    mpPass->execute(pRenderContext, mpFbo);
    pRenderContext->blit(pColorOut->getSRV(), mpPrevColor->getRTV());
}

void TAA::allocatePrevColor(const Texture* pColorOut)
{
    bool allocate = mpPrevColor == nullptr;
    allocate = allocate || (mpPrevColor->getWidth() != pColorOut->getWidth());
    allocate = allocate || (mpPrevColor->getHeight() != pColorOut->getHeight());
    allocate = allocate || (mpPrevColor->getDepth() != pColorOut->getDepth());
    allocate = allocate || (mpPrevColor->getFormat() != pColorOut->getFormat());
    FALCOR_ASSERT(pColorOut->getSampleCount() == 1);

    if (allocate) mpPrevColor = Texture::create2D(pColorOut->getWidth(), pColorOut->getHeight(), pColorOut->getFormat(), 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
}

void TAA::renderUI(Gui::Widgets& widget)
{
    widget.var("Alpha", mControls.alpha, 0.f, 1.0f, 0.001f);
    widget.var("Color-Box Sigma", mControls.colorBoxSigma, 0.f, 15.f, 0.001f);
}
