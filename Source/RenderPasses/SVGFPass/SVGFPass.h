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
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/FullScreenPass.h"

using namespace Falcor;

class SVGFPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SVGFPass, "SVGFPass", "SVGF denoising pass.");

    static ref<SVGFPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<SVGFPass>(pDevice, props); }

    SVGFPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

private:
    void allocateFbos(uint2 dim, RenderContext* pRenderContext);
    void clearBuffers(RenderContext* pRenderContext, const RenderData& renderData);

    void computeLinearZAndNormal(RenderContext* pRenderContext, ref<Texture> pLinearZTexture, ref<Texture> pWorldNormalTexture);
    void computeReprojection(
        RenderContext* pRenderContext,
        ref<Texture> pAlbedoTexture,
        ref<Texture> pColorTexture,
        ref<Texture> pEmissionTexture,
        ref<Texture> pMotionVectorTexture,
        ref<Texture> pPositionNormalFwidthTexture,
        ref<Texture> pPrevLinearZAndNormalTexture
    );
    void computeFilteredMoments(RenderContext* pRenderContext);
    void computeAtrousDecomposition(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture);

    bool mBuffersNeedClear = false;

    // SVGF parameters
    bool mFilterEnabled = true;
    int32_t mFilterIterations = 4;
    int32_t mFeedbackTap = 1;
    float mVarainceEpsilon = 1e-4f;
    float mPhiColor = 10.0f;
    float mPhiNormal = 128.0f;
    float mAlpha = 0.05f;
    float mMomentsAlpha = 0.2f;

    // SVGF passes
    ref<FullScreenPass> mpPackLinearZAndNormal;
    ref<FullScreenPass> mpReprojection;
    ref<FullScreenPass> mpFilterMoments;
    ref<FullScreenPass> mpAtrous;
    ref<FullScreenPass> mpFinalModulate;

    // Intermediate framebuffers
    ref<Fbo> mpPingPongFbo[2];
    ref<Fbo> mpLinearZAndNormalFbo;
    ref<Fbo> mpFilteredPastFbo;
    ref<Fbo> mpCurReprojFbo;
    ref<Fbo> mpPrevReprojFbo;
    ref<Fbo> mpFilteredIlluminationFbo;
    ref<Fbo> mpFinalFbo;
};
