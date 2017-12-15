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
#include "Falcor.h"

namespace Falcor
{
    class Gui;

    /** Temporal AA class
    */
    class TemporalAA
    {
    public:
        using UniquePtr = std::unique_ptr<TemporalAA>;

        /** Destructor
        */
        ~TemporalAA();

        /** Create a new instance
        */
        static UniquePtr create();

        /** Render UI controls for this effect.
            \param[in] pGui GUI object to render UI elements with
        */
        void renderUI(Gui* pGui);

        /** Run the effect
            \param[in] pRenderContext Render context with the destination FBO already set
            \param[in] pCurColor Current frame color buffer
            \param[in] pPrevColor Previous frame color buffer
            \param[in] pMotionVec Motion vector buffer
        */
        void execute(RenderContext* pRenderContext, const Texture::SharedPtr & pCurColor, const Texture::SharedPtr & pPrevColor, const Texture::SharedPtr & pMotionVec);

        /** Sets the alpha value used to blend the previous frame with the current frame. Lower values means previous frame has more weight
        */
        void setAlphaValue(float alpha) { mControls.alpha = alpha; }

        /** Sets the sigma value
        */
        void setColorBoxSigma(float sigma) { mControls.colorBoxSigma = sigma; }

    private:
        TemporalAA();

        //  Create the Program.
        void createProgram();

        //  Set the Variable Data needed for Rendering.
        void setVarsData(const Texture::SharedPtr & pCurColor, const Texture::SharedPtr & pPrevColor, const Texture::SharedPtr & pMotionVec);

        FullScreenPass::UniquePtr mpProgram;
        GraphicsVars::SharedPtr mpProgVars;
        ConstantBuffer::SharedPtr mpCB;
        Sampler::SharedPtr mpLinearSampler;

        struct
        {
            ParameterBlock::BindLocation colorTex;
            ParameterBlock::BindLocation prevColorTex;
            ParameterBlock::BindLocation motionVecTex;
            ParameterBlock::BindLocation sampler;
            size_t alpha;
            size_t colorBoxSigma;
        } mVarLocations;

        struct Controls
        {
            float alpha = 0.1f;
            float colorBoxSigma = 1.0f;
        };

        //  
        Controls mControls;
    };
}