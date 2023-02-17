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
#include "RenderGraph/BasePasses/FullScreenPass.h"

using namespace Falcor;

/** Temporal AA class
*/
class TAA : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(TAA, "TAA", "Temporal Anti-Aliasing.");

    using SharedPtr = std::shared_ptr<TAA>;

    static SharedPtr create(std::shared_ptr<Device> pDevice, const Dictionary& dict);

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

    void setAlpha(float alpha) { mControls.alpha = alpha; }
    void setColorBoxSigma(float sigma) { mControls.colorBoxSigma = sigma; }
    void setAntiFlicker(bool antiFlicker) { mControls.antiFlicker = antiFlicker; }
    float getAlpha() { return mControls.alpha; }
    float getColorBoxSigma() { return mControls.colorBoxSigma; }
    bool getAntiFlicker() { return mControls.antiFlicker; }

private:
    TAA(std::shared_ptr<Device> pDevice);
    void allocatePrevColor(const Texture* pColorOut);

    FullScreenPass::SharedPtr mpPass;
    Fbo::SharedPtr mpFbo;
    Sampler::SharedPtr mpLinearSampler;

    struct
    {
        float alpha = 0.1f;
        float colorBoxSigma = 1.0f;
        bool antiFlicker = true;
    } mControls;

    Texture::SharedPtr mpPrevColor;
};
