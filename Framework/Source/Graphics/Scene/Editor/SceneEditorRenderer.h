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

#include "Graphics/Scene/SceneRenderer.h"
#include "Graphics/Scene/Editor/Gizmo.h"
#include <unordered_set>

namespace Falcor
{
    /** Used by the SceneEditor to render helper objects.
    */
    class SceneEditorRenderer : public SceneRenderer
    {
    public:
        using UniquePtr = std::unique_ptr<SceneEditorRenderer>;
        using UniqueConstPtr = std::unique_ptr<const SceneEditorRenderer>;

        static UniquePtr create(const Scene::SharedPtr& pScene);

        void renderScene(RenderContext* pContext, Camera* pCamera);

        /** Informs the renderer what gizmos exist in order to color them appropriately when rendering
        */
        void registerGizmos(const Gizmo::Gizmos& gizmos);

    private:
        SceneEditorRenderer(const Scene::SharedPtr& pScene);

        virtual void setPerFrameData(const CurrentWorkingData& currentData) override;
        virtual bool setPerModelInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, uint32_t instanceID);
        Gizmo::Gizmos mGizmos;

        GraphicsProgram::SharedPtr mpProgram;
        GraphicsProgram::SharedPtr mpRotGizmoProgram;
        GraphicsVars::SharedPtr mpProgramVars;
        GraphicsState::SharedPtr mpGraphicsState;

        DepthStencilState::SharedPtr mpSetStencilDS;
        DepthStencilState::SharedPtr mpExcludeStencilDS;
    };
}

