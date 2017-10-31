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
#include <vector>
#include "Utils/Gui.h"
#include "Graphics/Camera/CameraController.h"
#include "Graphics/Scene/Scene.h"
#include "utils/CpuTimer.h"
#include "API/ConstantBuffer.h"
#include "Utils/DebugDrawer.h"

namespace Falcor
{
    class Model;
    class GraphicsState;
    class RenderContext;
    class Material;
    class Mesh;
    class Camera;

    class SceneRenderer
    {
    public:
        using SharedPtr = std::shared_ptr<SceneRenderer>;
        using SharedConstPtr = std::shared_ptr<const SceneRenderer>;

        /** Create a renderer instance
            \param[in] pScene Scene this renderer is responsible for rendering
        */
        static SharedPtr create(const Scene::SharedPtr& pScene);
        virtual ~SceneRenderer() = default;

        /** Renders the full scene using the scene's active camera.
            Call update() before using this function, otherwise camera will not move and models will not be animated.
        */
        virtual void renderScene(RenderContext* pContext);

        /** Renders the full scene, overriding the internal camera.
            Call update() before using this function otherwise model animation will not work
        */
        virtual void renderScene(RenderContext* pContext, Camera* pCamera);

        /** Update the camera and model animation.
            Should be called before renderScene(), unless not animations are used and you update the camera manually
        */
        bool update(double currentTime);

        bool onKeyEvent(const KeyboardEvent& keyEvent);
        bool onMouseEvent(const MouseEvent& mouseEvent);

        /** Enable/disable mesh culling. Culling does not always result in performance gain, especially when there are a lot of meshes to process with low rejection rate.
        */
        void setObjectCullState(bool enable) { mCullEnabled = enable; }

        /** Set the maximal number of mesh instance to dispatch in a single draw call.
        */
        void setMaxInstanceCount(uint32_t instanceCount) { mMaxInstanceCount = instanceCount; }

        enum class CameraControllerType
        {
            FirstPerson,
            SixDof,
            Hmd
        };

        void setCameraControllerType(CameraControllerType type);

        void detachCameraController();

        Scene::SharedPtr getScene() const { return mpScene; }

        void toggleStaticMaterialCompilation(bool on) { mCompileMaterialWithProgram = on; }

    protected:

        struct CurrentWorkingData
        {
            RenderContext* pContext = nullptr;
            GraphicsVars* pVars = nullptr;
            GraphicsState* pState = nullptr;
            const Camera* pCamera = nullptr;
            const Model* pModel = nullptr;
            const Material* pMaterial = nullptr;

            uint32_t drawID; // Zero-based mesh instance draw order/ID. Resets at the beginning of renderScene, and increments per mesh instance drawn.
        };

        SceneRenderer(const Scene::SharedPtr& pScene);
        Scene::SharedPtr mpScene;

        static const char* kPerMaterialCbName;
        static const char* kPerFrameCbName;
        static const char* kPerMeshCbName;

        static size_t sBonesOffset;
        static size_t sCameraDataOffset;
        static size_t sLightCountOffset;
        static size_t sLightArrayOffset;
        static size_t sAmbientLightOffset;
        static size_t sWorldMatArraySize;
        static size_t sWorldMatOffset;
        static size_t sPrevWorldMatOffset;
        static size_t sWorldInvTransposeMatOffset;
        static size_t sMeshIdOffset;
        static size_t sDrawIDOffset;

        static void updateVariableOffsets(const ProgramReflection* pReflector);

        virtual void setPerFrameData(const CurrentWorkingData& currentData);
        virtual bool setPerModelData(const CurrentWorkingData& currentData);
        virtual bool setPerModelInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, uint32_t instanceID);
        virtual bool setPerMeshData(const CurrentWorkingData& currentData, const Mesh* pMesh);
        virtual bool setPerMeshInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, const Model::MeshInstance* pMeshInstance, uint32_t drawInstanceID);
        virtual bool setPerMaterialData(const CurrentWorkingData& currentData, const Material* pMaterial);
        virtual void executeDraw(const CurrentWorkingData& currentData, uint32_t indexCount, uint32_t instanceCount);
        virtual void postFlushDraw(const CurrentWorkingData& currentData);

        void renderModelInstance(CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance);
        void renderMeshInstances(CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, uint32_t meshID);
        void draw(CurrentWorkingData& currentData, const Mesh* pMesh, uint32_t instanceCount);

        void renderScene(CurrentWorkingData& currentData);

        CameraControllerType mCamControllerType = CameraControllerType::SixDof;
        CameraController::SharedPtr mpCameraController;

        uint32_t mMaxInstanceCount = 64;
        const Material* mpLastMaterial = nullptr;
        bool mCullEnabled = true;
        bool mCompileMaterialWithProgram = true;
    };
}
