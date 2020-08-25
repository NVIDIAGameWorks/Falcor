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

using namespace Falcor;

class SVGFPass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<SVGFPass>;

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual std::string getDesc() override { return "SVGF Denoising Pass"; }
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void compile(RenderContext* pContext, const CompileData& compileData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

private:
    SVGFPass(const Dictionary& dict);

    bool init(const Dictionary& dict);
    void allocateFbos(uint2 dim, RenderContext* pRenderContext);
    void clearBuffers(RenderContext* pRenderContext, const RenderData& renderData);

    void computeLinearZAndNormal(RenderContext* pRenderContext, Texture::SharedPtr pLinearZTexture,
                                 Texture::SharedPtr pWorldNormalTexture);
    void computeReprojection(RenderContext* pRenderContext, Texture::SharedPtr pAlbedoTexture,
                             Texture::SharedPtr pColorTexture, Texture::SharedPtr pEmissionTexture,
                             Texture::SharedPtr pMotionVectorTexture,
                             Texture::SharedPtr pPositionNormalFwidthTexture,
                             Texture::SharedPtr pPrevLinearZAndNormalTexture);
    void computeFilteredMoments(RenderContext* pRenderContext);
    void computeAtrousDecomposition(RenderContext* pRenderContext, Texture::SharedPtr pAlbedoTexture);

    bool mBuffersNeedClear = false;

    // SVGF parameters
    bool    mFilterEnabled       = true;
    int32_t mFilterIterations    = 4;
    int32_t mFeedbackTap         = 1;
    float   mVarainceEpsilon     = 1e-4f;
    float   mPhiColor            = 10.0f;
    float   mPhiNormal           = 128.0f;
    float   mAlpha               = 0.05f;
    float   mMomentsAlpha        = 0.2f;

    // SVGF passes
    FullScreenPass::SharedPtr mpPackLinearZAndNormal;
    FullScreenPass::SharedPtr mpReprojection;
    FullScreenPass::SharedPtr mpFilterMoments;
    FullScreenPass::SharedPtr mpAtrous;
    FullScreenPass::SharedPtr mpFinalModulate;

    // Intermediate framebuffers
    Fbo::SharedPtr mpPingPongFbo[2];
    Fbo::SharedPtr mpLinearZAndNormalFbo;
    Fbo::SharedPtr mpFilteredPastFbo;
    Fbo::SharedPtr mpCurReprojFbo;
    Fbo::SharedPtr mpPrevReprojFbo;
    Fbo::SharedPtr mpFilteredIlluminationFbo;
    Fbo::SharedPtr mpFinalFbo;
};
