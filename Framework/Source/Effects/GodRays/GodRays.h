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
#include "Graphics/Scene/Scene.h"
#include "Graphics/FullScreenPass.h"
#include "Effects/Utils/PassFilter/PassFilter.h"

namespace Falcor
{
    class RenderContext;

    class GodRays
    {
    public:
        using UniquePtr = std::unique_ptr<GodRays>;

        static UniquePtr create(float threshold = 1.0f, float mediumDensity = 1000.0f, float mediumDecay = 0.964f, float mediumWeight = 0.196f, float exposer = 0.259f, int32_t numSamples = 250);

        void execute(RenderContext* pRenderContext, Fbo::SharedPtr pFbo);
        void execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrcTex, const Texture::SharedPtr& pSrcDepthTex, Fbo::SharedPtr pFbo);

        /** Render UI controls for bloom settings.
        \param[in] pGui GUI instance to render UI elements with
        \param[in] uiGroup Optional name. If specified, UI elements will be rendered within a named group
        */
        void renderUI(Gui* pGui, const char* uiGroup = nullptr, const Scene::SharedPtr& pScene = nullptr);

        void setNumSamples(int32_t numSamples);

        // move this back to private
        GraphicsVars::SharedPtr mpVars;

    private:
        GodRays(float threshold, float mediumDensity, float mediumDecay, float mediumWeight, float exposer, int32_t numSamples);
        void updateLowResTexture(const Texture::SharedPtr& pTexture);
        void createShader();

        float mMediumDensity;
        float mMediumDecay;
        float mMediumWeight;
        float mThreshold = 1.0f;
        float mExposer = 1.0f;
        int32_t mNumSamples;
        int32_t mLightIndex = 0;
        bool mDirty = false;

        //Scene::SharedPtr mpScene;
        PassFilter::UniquePtr mpFilter;
        Fbo::SharedPtr mpFilterResultFbo;
        Texture::SharedPtr mpLowResTexture;
        FullScreenPass::UniquePtr mpBlitPass;
        ParameterBlockReflection::BindLocation mSrcTexLoc;
        ParameterBlockReflection::BindLocation mSrcDepthLoc;
        BlendState::SharedPtr mpAdditiveBlend;
        Sampler::SharedPtr mpSampler;
    };
}
