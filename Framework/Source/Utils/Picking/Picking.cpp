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
#include "Utils/Picking/Picking.h"
#include "Graphics/FboHelper.h"

namespace Falcor
{
    Picking::UniquePtr Picking::create(const Scene::SharedPtr& pScene, uint32_t fboWidth, uint32_t fboHeight)
    {
        return UniquePtr(new Picking(pScene, fboWidth, fboHeight));
    }

    bool Picking::pick(RenderContext* pContext, const glm::vec2& mousePos, const Camera::SharedPtr& pCamera)
    {
        calculateScissor(mousePos);
        renderScene(pContext, pCamera.get());
        readPickResults(pContext);
        return mPickResult.pModelInstance != nullptr;
    }

    ObjectInstance<Mesh>::SharedPtr Picking::getPickedMeshInstance() const
    {
        return mPickResult.pMeshInstance;
    }

    ObjectInstance<Model>::SharedPtr Picking::getPickedModelInstance() const
    {
        return mPickResult.pModelInstance;
    }

    void Picking::resizeFBO(uint32_t width, uint32_t height)
    {
        Fbo::Desc fboDesc;
        fboDesc.setColorTarget(0, Falcor::ResourceFormat::R16Int).setDepthStencilTarget(Falcor::ResourceFormat::D24UnormS8);

        mpFBO = FboHelper::create2D(width, height, fboDesc);
    }

    void Picking::registerGizmos(const Gizmo::Gizmos& gizmos)
    {
        mSceneGizmos = gizmos;
    }

    Picking::Picking(const Scene::SharedPtr& pScene, uint32_t fboWidth, uint32_t fboHeight)
        : SceneRenderer(pScene)
    {
        mpGraphicsState = GraphicsState::create();

        // Create FBO
        resizeFBO(fboWidth, fboHeight);
        mpGraphicsState->setFbo(mpFBO);

        // Compile shaders
        Program::DefineList defines;
        defines.add("PICKING");
        mpProgram = GraphicsProgram::createFromFile("Framework/Shaders/SceneEditorVS.slang", "Framework/Shaders/SceneEditorPS.slang", defines);
        mpProgramVars = GraphicsVars::create(mpProgram->getActiveVersion()->getReflector());

        defines.add("CULL_REAR_SECTION");
        mpRotGizmoProgram = GraphicsProgram::createFromFile("Framework/Shaders/SceneEditorVS.slang", "Framework/Shaders/SceneEditorPS.slang", defines);

        // Depth State
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthTest(false).setStencilTest(true).setStencilRef(1);
        dsDesc.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Replace);
        mpSetStencilDS = DepthStencilState::create(dsDesc);

        dsDesc.setDepthTest(true).setStencilTest(true).setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::NotEqual);
        dsDesc.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep);
        mpExcludeStencilDS = DepthStencilState::create(dsDesc);

        // Rasterizer State
        RasterizerState::Desc rsDesc;
        rsDesc.setCullMode(RasterizerState::CullMode::None);
        mpGraphicsState->setRasterizerState(RasterizerState::create(rsDesc));
    }

    void Picking::renderScene(RenderContext* pContext, Camera* pCamera)
    {
        mDrawIDToInstance.clear();

        const glm::vec4 clearColor(-1.0f);
        pContext->clearFbo(mpFBO.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

        // Save state
        auto pPrevGraphicsState = pContext->getGraphicsState();

        mpGraphicsState->setScissors(0, mScissor);

        // Render
        pContext->setGraphicsState(mpGraphicsState);
        pContext->setGraphicsVars(mpProgramVars);

        SceneRenderer::renderScene(pContext, pCamera);

        // Restore state
        pContext->setGraphicsState(pPrevGraphicsState);
    }

    void Picking::readPickResults(RenderContext* pContext)
    {
        std::vector<uint8_t> textureData = pContext->readTextureSubresource(mpFBO->getColorTexture(0).get(), 0);
        uint16_t* pData = (uint16_t*)textureData.data();

        uint32_t i = mScissor.top * mpFBO->getWidth() + mScissor.left;

        if (pData[i] >= 0)
        {
            mPickResult = mDrawIDToInstance[pData[i]];
        }
        else
        {
            // Nullptrs
            mPickResult = Instance();
        }
    }

    void Picking::setPerFrameData(const CurrentWorkingData& currentData)
    {
        if (currentData.pCamera)
        {
            // Set camera for regular shader
            ConstantBuffer* pCB = mpProgramVars->getConstantBuffer(kPerFrameCbName).get();
            currentData.pCamera->setIntoConstantBuffer(pCB, sCameraDataOffset);
        }
    }

    bool Picking::setPerModelData(const CurrentWorkingData& currentData)
    {
        const Gizmo::Type gizmoType = Gizmo::getGizmoType(mSceneGizmos, currentData.pModel);

        // If rendering a gizmo
        if (gizmoType != Gizmo::Type::Invalid)
        {
            mpGraphicsState->setDepthStencilState(mpSetStencilDS);

            // For rotation gizmo, set shader to cut out away-facing parts
            if (gizmoType == Gizmo::Type::Rotate)
            {
                mpGraphicsState->setProgram(gizmoType == Gizmo::Type::Rotate ? mpRotGizmoProgram : mpProgram);
                return SceneRenderer::setPerModelData(currentData);
            }
        }
        else
        {
            mpGraphicsState->setDepthStencilState(mpExcludeStencilDS);
        }

        mpGraphicsState->setProgram(mpProgram);
        currentData.pContext->setGraphicsVars(mpProgramVars);

        return SceneRenderer::setPerModelData(currentData);
    }

    bool Picking::setPerMeshInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, const Model::MeshInstance* pMeshInstance, uint32_t drawInstanceID)
    {
        ConstantBuffer* pCB = currentData.pContext->getGraphicsVars()->getConstantBuffer(kPerMeshCbName).get();
        pCB->setBlob(&currentData.drawID, sDrawIDOffset + drawInstanceID * sizeof(uint32_t), sizeof(uint32_t));

        mDrawIDToInstance[currentData.drawID] = Instance(const_cast<Scene::ModelInstance*>(pModelInstance)->shared_from_this(), const_cast<Model::MeshInstance*>(pMeshInstance)->shared_from_this());

        return SceneRenderer::setPerMeshInstanceData(currentData, pModelInstance, pMeshInstance, drawInstanceID);
    }

    bool Picking::setPerMaterialData(const CurrentWorkingData& currentData, const Material* pMaterial)
    {
        return true;
    }

    void Picking::calculateScissor(const glm::vec2& mousePos)
    {
        glm::vec2 mouseCoords = mousePos * glm::vec2(mpFBO->getWidth(), mpFBO->getHeight());;

        mScissor.top = (int32_t)mouseCoords.y;
        mScissor.left = (int32_t)mouseCoords.x;
        mScissor.right = mScissor.left + 1;
        mScissor.bottom = mScissor.top + 1;
    }
}
