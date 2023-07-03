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
#pragma once
#include "Falcor.h"
#include "Core/Pass/FullScreenPass.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

class GaussianBlur : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(GaussianBlur, "GaussianBlur", "Gaussian blur.");

    static ref<GaussianBlur> create(ref<Device> pDevice, const Properties& props) { return make_ref<GaussianBlur>(pDevice, props); }

    GaussianBlur(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

    void setKernelWidth(uint32_t kernelWidth);
    void setSigma(float sigma);
    uint32_t getKernelWidth() { return mKernelWidth; }
    float getSigma() { return mSigma; }

    static void registerBindings(pybind11::module& m);

private:
    uint32_t mKernelWidth = 5;
    float mSigma = 2.0f;
    bool mReady = false;
    void createTmpFbo(const Texture* pSrc);
    void updateKernel();

    ref<FullScreenPass> mpHorizontalBlur;
    ref<FullScreenPass> mpVerticalBlur;
    ref<Fbo> mpFbo;
    ref<Fbo> mpTmpFbo;
    ref<Sampler> mpSampler;
};
