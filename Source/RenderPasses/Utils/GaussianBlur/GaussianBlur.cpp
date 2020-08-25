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
#include "GaussianBlur.h"
#define _USE_MATH_DEFINES
#include <math.h>

const char* GaussianBlur::kDesc = "Gaussian Blur";

namespace
{
    const char kSrc[] = "src";
    const char kDst[] = "dst";

    const char kKernelWidth[] = "kernelWidth";
    const char kSigma[] = "sigma";

    const char kShaderFilename[] = "RenderPasses/Utils/GaussianBlur/GaussianBlur.ps.slang";
}

void GaussianBlur::registerBindings(pybind11::module& m)
{
    pybind11::class_<GaussianBlur, RenderPass, GaussianBlur::SharedPtr> pass(m, "GaussianBlur");
    pass.def_property(kKernelWidth, &GaussianBlur::getKernelWidth, &GaussianBlur::setKernelWidth);
    pass.def_property(kSigma, &GaussianBlur::getSigma, &GaussianBlur::setSigma);
}

GaussianBlur::GaussianBlur()
{
    mpFbo = Fbo::create();
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpSampler = Sampler::create(samplerDesc);
}

GaussianBlur::SharedPtr GaussianBlur::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pBlur = SharedPtr(new GaussianBlur);
    for (const auto& [key, value] : dict)
    {
        if (key == kKernelWidth) pBlur->mKernelWidth = value;
        else if (key == kSigma) pBlur->mSigma = value;
        else logWarning("Unknown field '" + key + "' in a GaussianBlur dictionary");
    }
    return pBlur;
}

Dictionary GaussianBlur::getScriptingDictionary()
{
    Dictionary dict;
    dict[kKernelWidth] = mKernelWidth;
    dict[kSigma] = mSigma;
    return dict;
}

RenderPassReflection GaussianBlur::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
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
            return f.format(srcFormat).resourceType(srcType, srcWidth, srcHeight, srcDepth, srcSampleCount, srcMipCount, srcArraySize);
        };

        formatField(reflector.addInput(kSrc, "input image to be blurred"));
        formatField(reflector.addOutput(kDst, "output blurred image"));
        mReady = true;
    }
    else
    {
        reflector.addInput(kSrc, "input image to be blurred");
        reflector.addOutput(kDst, "output blurred image");
    }
    return reflector;
}

void GaussianBlur::compile(RenderContext* pContext, const CompileData& compileData)
{
    if (!mReady) throw std::runtime_error("GaussianBlur::compile - missing incoming reflection information");

    uint32_t arraySize = compileData.connectedResources.getField(kSrc)->getArraySize();
    Program::DefineList defines;
    defines.add("_KERNEL_WIDTH", std::to_string(mKernelWidth));
    if (arraySize > 1) defines.add("_USE_TEX2D_ARRAY");

    uint32_t layerMask = (arraySize > 1) ? ((1 << arraySize) - 1) : 0;
    defines.add("_HORIZONTAL_BLUR");
    mpHorizontalBlur = FullScreenPass::create(kShaderFilename, defines, layerMask);
    defines.remove("_HORIZONTAL_BLUR");
    defines.add("_VERTICAL_BLUR");
    mpVerticalBlur = FullScreenPass::create(kShaderFilename, defines, layerMask);

    // Make the programs share the vars
    mpVerticalBlur->setVars(mpHorizontalBlur->getVars());

    updateKernel();
}

void GaussianBlur::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pSrc = renderData[kSrc]->asTexture();
    mpFbo->attachColorTarget(renderData[kDst]->asTexture(), 0);
    createTmpFbo(pSrc.get());

    // Horizontal pass
    mpHorizontalBlur["gSampler"] = mpSampler;
    mpHorizontalBlur["gSrcTex"] = pSrc;
    mpHorizontalBlur->execute(pRenderContext, mpTmpFbo);

    // Vertical pass
    mpVerticalBlur["gSrcTex"] = mpTmpFbo->getColorTexture(0);
    mpVerticalBlur->execute(pRenderContext, mpFbo);
}

void GaussianBlur::createTmpFbo(const Texture* pSrc)
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
        mpTmpFbo = Fbo::create2D(pSrc->getWidth(), pSrc->getHeight(), fboDesc, pSrc->getArraySize());
    }
}

void GaussianBlur::renderUI(Gui::Widgets& widget)
{
    if (widget.var("Kernel Width", (int&)mKernelWidth, 1, 15, 2)) setKernelWidth(mKernelWidth);
    if (widget.slider("Sigma", mSigma, 0.001f, mKernelWidth / 2.f)) setSigma(mSigma);
}

void GaussianBlur::setKernelWidth(uint32_t kernelWidth)
{
    mKernelWidth = kernelWidth | 1; // Make sure the kernel width is an odd number
    mPassChangedCB();
}

void GaussianBlur::setSigma(float sigma)
{
    mSigma = sigma;
    mPassChangedCB();
}

float getCoefficient(float sigma, float kernelWidth, float x)
{
    float sigmaSquared = sigma * sigma;
    float p = -(x*x) / (2 * sigmaSquared);
    float e = std::exp(p);

    float a = 2 * (float)M_PI * sigmaSquared;
    return e / a;
}

void GaussianBlur::updateKernel()
{
    uint32_t center = mKernelWidth / 2;
    float sum = 0;
    std::vector<float> weights(center + 1);
    for (uint32_t i = 0; i <= center; i++)
    {
        weights[i] = getCoefficient(mSigma, (float)mKernelWidth, (float)i);
        sum += (i == 0) ? weights[i] : 2 * weights[i];
    }

    Buffer::SharedPtr pBuf = Buffer::createTyped<float>(mKernelWidth, Resource::BindFlags::ShaderResource);

    for (uint32_t i = 0; i <= center; i++)
    {
        float w = weights[i] / sum;
        pBuf->setElement(center + i, w);
        pBuf->setElement(center - i, w);
    }

    mpHorizontalBlur["weights"] = pBuf;
}
