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
#include "InvalidPixelDetectionPass.h"

const char* InvalidPixelDetectionPass::kDesc = "Pass that marks all NaN pixels red and Inf pixels green in an image.";

namespace
{
    const std::string kSrc = "src";
    const std::string kDst = "dst";
}

InvalidPixelDetectionPass::InvalidPixelDetectionPass()
{
    mpInvalidPixelDetectPass = FullScreenPass::create("RenderPasses/DebugPasses/InvalidPixelDetectionPass/InvalidPixelDetection.ps.slang");
    mpFbo = Fbo::create();
}

InvalidPixelDetectionPass::SharedPtr InvalidPixelDetectionPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new InvalidPixelDetectionPass());
    return pPass;
}

RenderPassReflection InvalidPixelDetectionPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    mReady = false;
    if (compileData.connectedResources.getFieldCount() > 0)
    {
        const RenderPassReflection::Field* edge = compileData.connectedResources.getField(kSrc);
        RenderPassReflection::Field::Type srcType = edge->getType();
        ResourceFormat srcFormat = edge->getFormat();
        uint32_t srcWidth = edge->getWidth();
        uint32_t srcHeight = edge->getHeight();
        uint32_t srcDepth = edge->getDepth();
        uint32_t srcSampleCount = edge->getSampleCount();
        uint32_t srcMipCount = edge->getMipCount();
        uint32_t srcArraySize = edge->getArraySize();

        auto formatField = [=](RenderPassReflection::Field& f) {
            return f.resourceType(srcType, srcWidth, srcHeight, srcDepth, srcSampleCount, srcMipCount, srcArraySize);
        };

        formatField(r.addInput(kSrc, "Input image to be checked")).format(srcFormat);
        formatField(r.addOutput(kDst, "Output where pixels are red if NaN, green if Inf, and black otherwise"));
        mReady = true;
    }
    else
    {
        r.addInput(kSrc, "Input image to be checked");
        r.addOutput(kDst, "Output where pixels are red if NaN, green if Inf, and black otherwise");
    }
    return r;
}

void InvalidPixelDetectionPass::compile(RenderContext* pContext, const CompileData& compileData)
{
    if (!mReady) throw std::runtime_error("InvalidPixelDetectionPass::compile - missing incoming reflection data");
}

void InvalidPixelDetectionPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpInvalidPixelDetectPass["gTexture"] = renderData[kSrc]->asTexture();
    mpFbo->attachColorTarget(renderData[kDst]->asTexture(), 0);
    mpInvalidPixelDetectPass->getState()->setFbo(mpFbo);
    mpInvalidPixelDetectPass->execute(pRenderContext, mpFbo);
}
