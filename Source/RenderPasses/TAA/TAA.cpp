/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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

namespace
{
const std::string kMotionVec = "motionVecs";
const std::string kColorIn = "colorIn";
const std::string kColorOut = "colorOut";

const std::string kAlpha = "alpha";
const std::string kColorBoxSigma = "colorBoxSigma";
const std::string kAntiFlicker = "antiFlicker";

const std::string kShaderFilename = "RenderPasses/TAA/TAA.ps.slang";
} // namespace

static void regTAA(pybind11::module& m)
{
    pybind11::class_<TAA, RenderPass, ref<TAA>> pass(m, "TAA");
    pass.def_property("alpha", &TAA::getAlpha, &TAA::setAlpha);
    pass.def_property("sigma", &TAA::getColorBoxSigma, &TAA::setColorBoxSigma);
    pass.def_property("antiFlicker", &TAA::getAntiFlicker, &TAA::setAntiFlicker);
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, TAA>();
    ScriptBindings::registerBinding(regTAA);
}

TAA::TAA(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mpPass = FullScreenPass::create(mpDevice, kShaderFilename);
    mpFbo = Fbo::create(mpDevice);
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    mpLinearSampler = mpDevice->createSampler(samplerDesc);

    for (const auto& [key, value] : props)
    {
        if (key == kAlpha)
            mControls.alpha = value;
        else if (key == kColorBoxSigma)
            mControls.colorBoxSigma = value;
        else if (key == kAntiFlicker)
            mControls.antiFlicker = value;
        else
            logWarning("Unknown property '{}' in a TemporalAA properties.", key);
    }
}

Properties TAA::getProperties() const
{
    Properties props;
    props[kAlpha] = mControls.alpha;
    props[kColorBoxSigma] = mControls.colorBoxSigma;
    props[kAntiFlicker] = mControls.antiFlicker;
    return props;
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

    auto var = mpPass->getRootVar();
    var["PerFrameCB"]["gAlpha"] = mControls.alpha;
    var["PerFrameCB"]["gColorBoxSigma"] = mControls.colorBoxSigma;
    var["PerFrameCB"]["gAntiFlicker"] = mControls.antiFlicker;
    var["gTexColor"] = pColorIn;
    var["gTexMotionVec"] = pMotionVec;
    var["gTexPrevColor"] = mpPrevColor;
    var["gSampler"] = mpLinearSampler;

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

    if (allocate)
        mpPrevColor = mpDevice->createTexture2D(
            pColorOut->getWidth(),
            pColorOut->getHeight(),
            pColorOut->getFormat(),
            1,
            1,
            nullptr,
            ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
        );
}

void TAA::renderUI(Gui::Widgets& widget)
{
    widget.var("Alpha", mControls.alpha, 0.f, 1.0f, 0.001f);
    widget.var("Color-Box Sigma", mControls.colorBoxSigma, 0.f, 15.f, 0.001f);
    widget.checkbox("Anti Flicker", mControls.antiFlicker);
}
