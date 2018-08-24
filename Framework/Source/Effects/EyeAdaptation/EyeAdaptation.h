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
#include <array>
#include "Graphics/RenderGraph/RenderPass.h"
#include "API/FBO.h"
#include "Graphics/FullScreenPass.h"
#include "Graphics/Camera/Camera.h"
#include "Effects/Utils/GaussianBlur/GaussianBlur.h"
#include "Effects/Utils/PassFilter/PassFilter.h"

namespace Falcor
{
    class RenderContext;

    class EyeAdaptation : public RenderPass, public inherit_shared_from_this<RenderPass, EyeAdaptation>
    {
    public:
        using UniquePtr = std::unique_ptr<EyeAdaptation>;
        using SharedPtr = std::shared_ptr<EyeAdaptation>;

        static SharedPtr create();

        static SharedPtr deserialize(const RenderPassSerializer& serializer);

        virtual void reflect(RenderPassReflection& reflector) const override;

        virtual void execute(RenderContext* pRenderContext, const RenderData* pData) override;

        void execute(RenderContext* pRenderContext, Fbo::SharedPtr pFbo);

        void serialize(RenderPassSerializer& renderPassSerializer) override;

        /** Render UI controls for bloom settings.
        \param[in] pGui GUI instance to render UI elements with
        \param[in] uiGroup Optional name. If specified, UI elements will be rendered within a named group
        */
        virtual void renderUI(Gui* pGui, const char* uiGroup = nullptr) override;

        virtual void setScene(const Scene::SharedPtr& pScene) override { mpScene = pScene; }

    private:
        EyeAdaptation();

        void createShader();

        void updateValuesFromCamera();

        GraphicsVars::SharedPtr mpVars;
        glm::vec2 mLumaRange;
        
        bool mDirty = false;
        Scene::SharedPtr mpScene;

        Texture::SharedPtr mpSrcTex = nullptr;
        Texture::SharedPtr mpNoiseTex = nullptr;
        Fbo::SharedPtr mpTargetFbo;
        FullScreenPass::UniquePtr mpBlitPass;
        ParameterBlockReflection::BindLocation mSrcTexLoc;
        ParameterBlockReflection::BindLocation mNoiseTexLoc;
        Sampler::SharedPtr mpSampler;
    };
}
