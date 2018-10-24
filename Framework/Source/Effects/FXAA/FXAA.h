/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include "Graphics/RenderGraph/RenderPass.h"
#include "Graphics/FullScreenPass.h"
#include "Graphics/Program/ProgramVars.h"

namespace Falcor
{
    class Gui;
    class Texture;
    class Fbo;

    /** Temporal AA class
    */
    class FXAA : public RenderPass, public inherit_shared_from_this<RenderPass, FXAA>
    {
    public:
        using SharedPtr = std::shared_ptr<FXAA>;

        /** Destructor
        */
        ~FXAA();

        /** Create a new instance
        */
        static SharedPtr create(const Dictionary& dict = {});

        /** Render UI controls for this effect.
            \param[in] pGui GUI object to render UI elements with
        */
        void renderUI(Gui* pGui, const char* uiGroup);

        /** Run the effect
            \param[in] pRenderContext Render context with the destination FBO already set
            \param[in] pCurColor Current frame color buffer
            \param[in] pPrevColor Previous frame color buffer
            \param[in] pMotionVec Motion vector buffer
        */
        void execute(RenderContext* pRenderContext, const std::shared_ptr<Texture>& pSrcTex, const std::shared_ptr<Fbo>& pDstFbo);

        virtual RenderPassReflection reflect() const override;
        virtual void execute(RenderContext* pContext, const RenderData* pData) override;
    private:
        FXAA();

        FullScreenPass::UniquePtr mpPass;
        GraphicsVars::SharedPtr mpGraphicsVars;
        Sampler::SharedPtr mpLinearSampler;

        float mQualitySubPix = 0.75f;
        float mQualityEdgeThreshold = 0.166f;
        float mQualityEdgeThresholdMin = 0.0833f;
        bool mEarlyOut = true;
    };
}