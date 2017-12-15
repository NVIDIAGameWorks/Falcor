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
#include "Graphics/Model/ObjectInstance.h"
#include "Graphics/Scene/Editor/Gizmo.h"
#include <unordered_set>

namespace Falcor
{
    /** SceneRenderer extended to add picking capabilities. Determines which object in the scene was clicked by the mouse.
    */
    class Picking : public SceneRenderer
    {
    public:
        using UniquePtr = std::unique_ptr<Picking>;
        using UniqueConstPtr = std::unique_ptr<const Picking>;

        /** Creates an instance of the scene picker.
            \param[in] pScene Scene to pick.
            \param[in] fboWidth Size of internal FBO used for picking.
            \param[in] fboHeight Size of internal FBO used for picking.
            \return New Picking instance for pScene.
        */
        static UniquePtr create(const Scene::SharedPtr& pScene, uint32_t fboWidth, uint32_t fboHeight);

        /** Performs a picking operation on the scene and stores the result.
            \param[in] mousePos Mouse position in the range [0,1] with (0,0) being the top left corner. Same coordinate space as in MouseEvent.
            \param[in] pContext Render context to render scene with.
            \param[in] pCamera Active camera to pick from.
            \return Whether an object was picked or not.
        */
        bool pick(RenderContext* pContext, const glm::vec2& mousePos, const Camera::SharedPtr& pCamera);

        /** Gets the picked mesh instance.
            \return Pointer to the picked mesh instance, otherwise nullptr if nothing was picked.
        */
        Model::MeshInstance::SharedPtr getPickedMeshInstance() const;

        /** Gets the picked model instance.
            \return Pointer to the picked model instance, otherwise nullptr if nothing was picked.
        */
        Scene::ModelInstance::SharedPtr getPickedModelInstance() const;

        /** Resize the internal FBO used for picking.
            \param[in] width Width of the FBO.
            \param[in] height Height of the FBO.
        */
        void resizeFBO(uint32_t width, uint32_t height);

        // #HACK For picking the editor scene, register gizmos to conditionally set states
        void registerGizmos(const Gizmo::Gizmos& gizmos);

    private:

        Picking(const Scene::SharedPtr& pScene, uint32_t fboWidth, uint32_t fboHeight);

        void renderScene(RenderContext* pContext, Camera* pCamera);
        void readPickResults(RenderContext* pContext);

        virtual void setPerFrameData(const CurrentWorkingData& currentData) override;
        virtual bool setPerModelData(const CurrentWorkingData& currentData) override;
        virtual bool setPerMeshInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, const Model::MeshInstance* pMeshInstance, uint32_t drawInstanceID) override;
        virtual bool setPerMaterialData(const CurrentWorkingData& currentData, const Material* pMaterial) override;

        void calculateScissor(const glm::vec2& mousePos);

        struct Instance
        {
            Scene::ModelInstance::SharedPtr pModelInstance;
            Model::MeshInstance::SharedPtr pMeshInstance;

            Instance() {}

            Instance(Scene::ModelInstance::SharedPtr pModelInstance, Model::MeshInstance::SharedPtr pMeshInstance)
                : pModelInstance(pModelInstance), pMeshInstance(pMeshInstance) {}
        };

        Gizmo::Gizmos mSceneGizmos;

        std::unordered_map<uint32_t, Instance> mDrawIDToInstance;
        Instance mPickResult;

        Fbo::SharedPtr mpFBO;
        GraphicsState::SharedPtr mpGraphicsState;

        GraphicsProgram::SharedPtr mpProgram;
        GraphicsVars::SharedPtr mpProgramVars;

        // Separate program for rotation gizmos because away-facing parts are discarded :(
        GraphicsProgram::SharedPtr mpRotGizmoProgram;

        DepthStencilState::SharedPtr mpSetStencilDS;
        DepthStencilState::SharedPtr mpExcludeStencilDS;

        GraphicsState::Scissor mScissor;
    };
}
