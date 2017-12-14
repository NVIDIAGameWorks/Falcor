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
#include "Framework.h"
#include "Graphics/Scene/Editor/SceneEditorRenderer.h"

namespace Falcor
{
    SceneEditorRenderer::UniquePtr SceneEditorRenderer::create(const Scene::SharedPtr& pScene)
    {
        return UniquePtr(new SceneEditorRenderer(pScene));
    }

    void SceneEditorRenderer::renderScene(RenderContext* pContext, Camera* pCamera)
    {
        mpGraphicsState->setFbo(pContext->getGraphicsState()->getFbo());
        pContext->setGraphicsState(mpGraphicsState);
        pContext->setGraphicsVars(mpProgramVars);
        SceneRenderer::renderScene(pContext, pCamera);
    }

    void SceneEditorRenderer::registerGizmos(const Gizmo::Gizmos& gizmos)
    {
        mGizmos = gizmos;
    }

    SceneEditorRenderer::SceneEditorRenderer(const Scene::SharedPtr& pScene)
        : SceneRenderer(pScene)
    {
        mpGraphicsState = GraphicsState::create();

        // Solid Rasterizer state
        RasterizerState::Desc rsDesc;
        rsDesc.setFillMode(RasterizerState::FillMode::Solid).setCullMode(RasterizerState::CullMode::Back);
        mpGraphicsState->setRasterizerState(RasterizerState::create(rsDesc));

        // Depth test
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthTest(false).setStencilTest(true).setStencilRef(1);
        dsDesc.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Replace);
        mpSetStencilDS = DepthStencilState::create(dsDesc);

        dsDesc.setDepthTest(true).setStencilTest(true).setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::NotEqual);
        dsDesc.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep);
        mpExcludeStencilDS = DepthStencilState::create(dsDesc);

        // Shader
        Program::DefineList defines;
        defines.add("SHADING");
        mpProgram = GraphicsProgram::createFromFile("Framework/Shaders/SceneEditorVS.slang", "Framework/Shaders/SceneEditorPS.slang", defines);
        mpProgramVars = GraphicsVars::create(mpProgram->getActiveVersion()->getReflector());

        defines.add("CULL_REAR_SECTION");
        mpRotGizmoProgram = GraphicsProgram::createFromFile("Framework/Shaders/SceneEditorVS.slang", "Framework/Shaders/SceneEditorPS.slang", defines);
    }

    void SceneEditorRenderer::setPerFrameData(const CurrentWorkingData& currentData)
    {
        if (currentData.pCamera)
        {
            // Set camera for regular shader
            ConstantBuffer* pCB = mpProgramVars->getConstantBuffer(kPerFrameCbName).get();
            currentData.pCamera->setIntoConstantBuffer(pCB, sCameraDataOffset);
        }
    }

    bool SceneEditorRenderer::setPerModelInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, uint32_t instanceID)
    {
        const Gizmo::Type gizmoType = Gizmo::getGizmoType(mGizmos, currentData.pModel);

        if (gizmoType != Gizmo::Type::Invalid)
        {
            assert(instanceID < 3);
            glm::vec3 color;
            color[instanceID] = 1.0f;

            mpGraphicsState->setDepthStencilState(mpSetStencilDS);
            mpProgramVars["ConstColorCB"]["gColor"] = color;

            // For rotation gizmo, set shader to cut out away-facing parts
            if (gizmoType == Gizmo::Type::Rotate)
            {
                mpGraphicsState->setProgram(mpRotGizmoProgram);
                return true;
            }
        }
        else
        {
            mpGraphicsState->setDepthStencilState(mpExcludeStencilDS);
            mpProgramVars["ConstColorCB"]["gColor"] = glm::vec3(0.6f);
        }

        mpGraphicsState->setProgram(mpProgram);

        return true;
    }
}
