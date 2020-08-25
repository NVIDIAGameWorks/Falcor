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
#include "FXAA.h"

const char* FXAA::kDesc = "Fast Approximate Anti-Aliasing";

namespace
{
    const std::string kSrc = "src";
    const std::string kDst = "dst";

    const std::string kQualitySubPix = "qualitySubPix";
    const std::string kQualityEdgeThreshold = "qualityEdgeThreshold";
    const std::string kQualityEdgeThresholdMin = "qualityEdgeThresholdMin";
    const std::string kEarlyOut = "earlyOut";

    const std::string kShaderFilename = "RenderPasses/Antialiasing/FXAA/FXAA.slang";
}

FXAA::FXAA()
{
    mpPass = FullScreenPass::create(kShaderFilename);
    mpFbo = Fbo::create();
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point);
    mpPass["gSampler"] = Sampler::create(samplerDesc);
}

FXAA::SharedPtr FXAA::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pFXAA = SharedPtr(new FXAA);
    for (const auto& [key, value] : dict)
    {
        if (key == kQualitySubPix) pFXAA->mQualitySubPix = value;
        else if (key == kQualityEdgeThreshold) pFXAA->mQualityEdgeThreshold = value;
        else if (key == kQualityEdgeThresholdMin) pFXAA->mQualityEdgeThresholdMin = value;
        else if (key == kEarlyOut) pFXAA->mEarlyOut = value;
        else logWarning("Unknown field '" + key + "' in an FXAA dictionary");
    }
    return pFXAA;
}

Dictionary FXAA::getScriptingDictionary()
{
    Dictionary dict;
    dict[kQualitySubPix] = mQualitySubPix;
    dict[kQualityEdgeThreshold] = mQualityEdgeThreshold;
    dict[kQualityEdgeThresholdMin] = mQualityEdgeThresholdMin;
    dict[kEarlyOut] = mEarlyOut;
    return dict;
}

RenderPassReflection FXAA::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kSrc, "Source color-buffer");
    reflector.addOutput(kDst, "Destination color-buffer");
    return reflector;
}

void FXAA::execute(RenderContext* pContext, const RenderData& renderData)
{
    auto pSrc = renderData[kSrc]->asTexture();
    auto pDst = renderData[kDst]->asTexture();
    mpFbo->attachColorTarget(pDst, 0);

    mpPass["gSrc"] = pSrc;
    float2 rcpFrame = 1.0f / float2(pSrc->getWidth(), pSrc->getHeight());

    auto pCB = mpPass["PerFrameCB"];
    pCB["rcpTexDim"] = rcpFrame;
    pCB["qualitySubPix"] = mQualitySubPix;
    pCB["qualityEdgeThreshold"] = mQualityEdgeThreshold;
    pCB["qualityEdgeThresholdMin"] = mQualityEdgeThresholdMin;
    pCB["earlyOut"] = mEarlyOut;

    mpPass->execute(pContext, mpFbo);
}

void FXAA::renderUI(Gui::Widgets& widget)
{
    widget.var("Sub-Pixel Quality", mQualitySubPix, 0.f, 1.f, 0.001f);
    widget.var("Edge Threshold", mQualityEdgeThreshold, 0.f, 1.f, 0.001f);
    widget.var("Edge Threshold Min", mQualityEdgeThresholdMin, 0.f, 1.f, 0.001f);
    widget.checkbox("Early out", mEarlyOut);
}
