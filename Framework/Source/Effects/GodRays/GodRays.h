/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include <memory>
#include "API/FBO.h"
#include "Graphics/FullScreenPass.h"
#include "Effects/Utils/PassFilter/PassFilter.h"

namespace Falcor
{
    class RenderContext;

    class GodRays
    {
    public:
        using UniquePtr = std::unique_ptr<GodRays>;

        static UniquePtr create(float threshold, float mediumDensity = 100, float mediumDecay = 0.995f, float mediumWeight = 0.015f, int32_t numSamples = 1000);

        void execute(RenderContext* pRenderContext, Fbo::SharedPtr pFbo);


        /** Render UI controls for bloom settings.
        \param[in] pGui GUI instance to render UI elements with
        \param[in] uiGroup Optional name. If specified, UI elements will be rendered within a named group
        */
        void renderUI(Gui* pGui, const char* uiGroup = nullptr);

        // move this back to private
        GraphicsVars::SharedPtr mpVars;

    private:
        GodRays(float threshold, float mediumDensity, float mediumDecay, float mediumWeight, int32_t numSamples);
        void updateLowResTexture(const Texture::SharedPtr& pTexture);

        float mMediumDensity;
        float mMediumDecay;
        float mMediumWeight;
        float mThreshold = 1.0f;
        int32_t mNumSamples;
        int32_t mLightIndex = 0;
        TypedBuffer<uint>::SharedPtr mpBuf;

        //Scene::SharedPtr mpScene;
        PassFilter::UniquePtr mpFilter;
        Fbo::SharedPtr mpFilterResultFbo;
        Texture::SharedPtr mpLowResTexture;
        FullScreenPass::UniquePtr mpBlitPass;
        ParameterBlockReflection::BindLocation mSrcTexLoc;
        BlendState::SharedPtr mpAdditiveBlend;
        Sampler::SharedPtr mpSampler;
    };
}
