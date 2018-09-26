/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Graphics/Program/GraphicsProgram.h"
#include "Graphics/Program/ProgramVars.h"
#include "Graphics/GraphicsState.h"
#include "Graphics/Scene/SceneRenderer.h"

namespace Falcor
{
    class DepthPass : public RenderPass, inherit_shared_from_this<RenderPass, DepthPass>
    {
    public:
        using SharedPtr = std::shared_ptr<DepthPass>;

        /** Create a new object
        */
        static SharedPtr create(const Dictionary& dict = {});

        virtual RenderPassReflection reflect() const override;
        virtual void execute(RenderContext* pContext, const RenderData* pData) override;
        virtual void setScene(const Scene::SharedPtr& pScene);
        virtual void renderUI(Gui* pGui, const char* uiGroup) override;
        virtual Dictionary getScriptingDictionary() const;

        DepthPass& setDepthBufferFormat(ResourceFormat format);
        DepthPass& setDepthStencilState(const DepthStencilState::SharedPtr& pDsState);
    private:
        DepthPass();
        Fbo::SharedPtr mpFbo;
        GraphicsState::SharedPtr mpState;
        GraphicsVars::SharedPtr mpVars;
        SceneRenderer::SharedPtr mpSceneRenderer;
        ResourceFormat mDepthFormat = ResourceFormat::D32Float;
    };
}