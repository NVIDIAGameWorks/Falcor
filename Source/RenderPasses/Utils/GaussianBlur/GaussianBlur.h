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

using namespace Falcor;

#ifdef BUILD_GAUSSIAN_PASS
#define dllpassdecl __declspec(dllexport)
#else
#define dllpassdecl __declspec(dllimport)
#endif

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib);

class dllpassdecl GaussianBlur : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<GaussianBlur>;

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    std::string getDesc() override { return kDesc; }
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

    void setKernelWidth(uint32_t kernelWidth);
    void setSigma(float sigma);
    uint32_t getKernelWidth() { return mKernelWidth; }
    float getSigma() { return mSigma; }

private:
    GaussianBlur();
    uint32_t mKernelWidth = 5;
    float mSigma = 2.0f;
    bool mReady = false;
    void createTmpFbo(const Texture* pSrc);
    void updateKernel();

    FullScreenPass::SharedPtr mpHorizontalBlur;
    FullScreenPass::SharedPtr mpVerticalBlur;
    Fbo::SharedPtr mpFbo;
    Fbo::SharedPtr mpTmpFbo;
    Sampler::SharedPtr mpSampler;

    static const char* kDesc;
    static void registerBindings(pybind11::module& m);
    friend void getPasses(Falcor::RenderPassLibrary& lib);
};
