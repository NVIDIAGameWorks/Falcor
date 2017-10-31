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
#include "Framework.h"
#include "SceneRenderer.h"
#include "Graphics/Program.h"
#include "Utils/Gui.h"
#include "API/ConstantBuffer.h"
#include "API/RenderContext.h"
#include "Scene.h"
#include "Utils/OS.h"
#include "VR/OpenVR/VRSystem.h"
#include "API/Device.h"
#include "glm/matrix.hpp"
#include "Graphics/Material/MaterialSystem.h"

namespace Falcor
{
    size_t SceneRenderer::sBonesOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sCameraDataOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sWorldMatArraySize = 0;
    size_t SceneRenderer::sWorldMatOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sPrevWorldMatOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sWorldInvTransposeMatOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sMeshIdOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sDrawIDOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sLightCountOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sLightArrayOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sAmbientLightOffset = ConstantBuffer::kInvalidOffset;

    const char* SceneRenderer::kPerMaterialCbName = "InternalPerMaterialCB";
    const char* SceneRenderer::kPerFrameCbName = "InternalPerFrameCB";
    const char* SceneRenderer::kPerMeshCbName = "InternalPerMeshCB";

    SceneRenderer::SharedPtr SceneRenderer::create(const Scene::SharedPtr& pScene)
    {
        return SharedPtr(new SceneRenderer(pScene));
    }

    SceneRenderer::SceneRenderer(const Scene::SharedPtr& pScene) : mpScene(pScene)
    {
        setCameraControllerType(CameraControllerType::SixDof);
    }

    void SceneRenderer::updateVariableOffsets(const ProgramReflection* pReflector)
    {
        if (sWorldMatOffset == ConstantBuffer::kInvalidOffset)
        {
            const auto pPerMeshCbData = pReflector->getBufferDesc(kPerMeshCbName, ProgramReflection::BufferReflection::Type::Constant);

            if (pPerMeshCbData != nullptr)
            {
                assert(pPerMeshCbData->getVariableData("gWorldMat[0]")->isRowMajor == false); // We copy into CBs as column-major
                assert(pPerMeshCbData->getVariableData("gWorldInvTransposeMat[0]")->isRowMajor == false);
                assert(pPerMeshCbData->getVariableData("gWorldMat")->arraySize == pPerMeshCbData->getVariableData("gWorldInvTransposeMat")->arraySize);

                sWorldMatArraySize = pPerMeshCbData->getVariableData("gWorldMat")->arraySize;
                sWorldMatOffset = pPerMeshCbData->getVariableData("gWorldMat[0]")->location;
                sWorldInvTransposeMatOffset = pPerMeshCbData->getVariableData("gWorldInvTransposeMat[0]")->location;
                sMeshIdOffset = pPerMeshCbData->getVariableData("gMeshId")->location;
                sDrawIDOffset = pPerMeshCbData->getVariableData("gDrawId[0]")->location;
                sPrevWorldMatOffset = pPerMeshCbData->getVariableData("gPrevWorldMat[0]")->location;
            }
        }

        if (sCameraDataOffset == ConstantBuffer::kInvalidOffset)
        {
            const auto pPerFrameCbData = pReflector->getBufferDesc(kPerFrameCbName, ProgramReflection::BufferReflection::Type::Constant);

            if (pPerFrameCbData != nullptr)
            {
                sCameraDataOffset = pPerFrameCbData->getVariableData("gCam.viewMat")->location;
                const auto& pCountOffset = pPerFrameCbData->getVariableData("gLightsCount");
                sLightCountOffset = pCountOffset ? pCountOffset->location : ConstantBuffer::kInvalidOffset;
                const auto& pLightOffset = pPerFrameCbData->getVariableData("gLights[0].worldPos");
                sLightArrayOffset = pLightOffset ? pLightOffset->location : ConstantBuffer::kInvalidOffset;
                const auto& pAmbientOffset = pPerFrameCbData->getVariableData("gAmbientLighting");
                sAmbientLightOffset = pAmbientOffset ? pAmbientOffset->location : ConstantBuffer::kInvalidOffset;
            }
        }
    }

    void SceneRenderer::setPerFrameData(const CurrentWorkingData& currentData)
    {
        ConstantBuffer* pCB = currentData.pVars->getConstantBuffer(kPerFrameCbName).get();
        if (pCB)
        {
            // Set camera
            if (currentData.pCamera)
            {
                currentData.pCamera->setIntoConstantBuffer(pCB, sCameraDataOffset);
            }

            // Set lights
            if (sLightArrayOffset != ConstantBuffer::kInvalidOffset)
            {
                assert(mpScene->getLightCount() <= MAX_LIGHT_SOURCES);  // Max array size in the shader
                for (uint_t i = 0; i < mpScene->getLightCount(); i++)
                {
                    mpScene->getLight(i)->setIntoConstantBuffer(pCB, i * Light::getShaderStructSize() + sLightArrayOffset);
                }
            }
            if (sLightCountOffset != ConstantBuffer::kInvalidOffset)
            {
                pCB->setVariable(sLightCountOffset, mpScene->getLightCount());
            }
            if (sAmbientLightOffset != ConstantBuffer::kInvalidOffset)
            {
                pCB->setVariable(sAmbientLightOffset, mpScene->getAmbientIntensity());
            }
        }
    }

    bool SceneRenderer::setPerModelData(const CurrentWorkingData& currentData)
    {
        // Set bones
        if (currentData.pModel->hasBones())
        {
            ConstantBuffer* pCB = currentData.pVars->getConstantBuffer(kPerMeshCbName).get();
            if (pCB)
            {
                if (sBonesOffset == ConstantBuffer::kInvalidOffset)
                {
                    sBonesOffset = pCB->getVariableOffset("gWorldMat[0]");
                }

                pCB->setVariableArray(sBonesOffset, currentData.pModel->getBonesMatrices(), currentData.pModel->getBonesCount());
            }
        }
        return true;
    }

    bool SceneRenderer::setPerModelInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, uint32_t instanceID)
    {
        return true;
    }

    bool SceneRenderer::setPerMeshData(const CurrentWorkingData& currentData, const Mesh* pMesh)
    {
        return true;
    }

    bool SceneRenderer::setPerMeshInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, const Model::MeshInstance* pMeshInstance, uint32_t drawInstanceID)
    {
        ConstantBuffer* pCB = currentData.pVars->getConstantBuffer(kPerMeshCbName).get();
        if (pCB)
        {
            const Mesh* pMesh = pMeshInstance->getObject().get();

            assert(drawInstanceID == 0 || !pMesh->hasBones()); // The same array is reused for bone and instance matrices, both cannot be active
            if (pMesh->hasBones() == false)
            {
                glm::mat4 worldMat = pModelInstance->getTransformMatrix() * pMeshInstance->getTransformMatrix();
                glm::mat3x4 worldInvTransposeMat = transpose(inverse(glm::mat3(worldMat)));
                glm::mat4 prevWorldMat = pModelInstance->getPrevTransformMatrix() * pMeshInstance->getPrevTransformMatrix();

                assert(drawInstanceID < sWorldMatArraySize);
                pCB->setBlob(&worldMat, sWorldMatOffset + drawInstanceID * sizeof(glm::mat4), sizeof(glm::mat4));
                pCB->setBlob(&worldInvTransposeMat, sWorldInvTransposeMatOffset + drawInstanceID * sizeof(glm::mat3x4), sizeof(glm::mat3x4)); // HLSL uses column-major and packing rules require 16B alignment, hence use glm:mat3x4
                pCB->setBlob(&prevWorldMat, sPrevWorldMatOffset + drawInstanceID * sizeof(glm::mat4), sizeof(glm::mat4));
            }

            // Set mesh id
            pCB->setVariable(sMeshIdOffset, pMesh->getId());
        }

        return true;
    }

    bool SceneRenderer::setPerMaterialData(const CurrentWorkingData& currentData, const Material* pMaterial)
    {
        ConstantBuffer* pCB = currentData.pVars->getConstantBuffer(kPerMaterialCbName).get();
        if (pCB)
        {
            pMaterial->setIntoProgramVars(currentData.pVars, pCB, "gMaterial");
        }

        return true;
    }

    void SceneRenderer::executeDraw(const CurrentWorkingData& currentData, uint32_t indexCount, uint32_t instanceCount)
    {
        // Draw
        currentData.pContext->drawIndexedInstanced(indexCount, instanceCount, 0, 0, 0);
    }

    void SceneRenderer::draw(CurrentWorkingData& currentData, const Mesh* pMesh, uint32_t instanceCount)
    {
        currentData.pMaterial = pMesh->getMaterial().get();
        // Bind material
        if(mpLastMaterial != pMesh->getMaterial().get())
        {
            if (setPerMaterialData(currentData, currentData.pMaterial) == false)
            {
                return;
            }
            mpLastMaterial = pMesh->getMaterial().get();

            if(mCompileMaterialWithProgram)
            {
                MaterialSystem::patchProgram(currentData.pState->getProgram().get(), mpLastMaterial);
            }
        }

        executeDraw(currentData, pMesh->getIndexCount(), instanceCount);
        postFlushDraw(currentData);
        currentData.pState->getProgram()->removeDefine("_MS_STATIC_MATERIAL_DESC");
    }

    void SceneRenderer::postFlushDraw(const CurrentWorkingData& currentData)
    {

    }

    void SceneRenderer::renderMeshInstances(CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, uint32_t meshID)
    {
        const Model* pModel = currentData.pModel;
        const Mesh* pMesh = pModel->getMesh(meshID).get();

        if (setPerMeshData(currentData, pMesh))
        {
            // Bind VAO and set topology
            currentData.pState->setVao(pMesh->getVao());

            uint32_t activeInstances = 0;

            const uint32_t instanceCount = pModel->getMeshInstanceCount(meshID);
            for (uint32_t instanceID = 0; instanceID < instanceCount; instanceID++)
            {
                const Model::MeshInstance* pMeshInstance = pModel->getMeshInstance(meshID, instanceID).get();
                BoundingBox box = pMeshInstance->getBoundingBox().transform(pModelInstance->getTransformMatrix());

                if ((mCullEnabled == false) || (currentData.pCamera->isObjectCulled(box) == false))
                {
                    if (pMeshInstance->isVisible())
                    {
                        if (setPerMeshInstanceData(currentData, pModelInstance, pMeshInstance, activeInstances))
                        {
                            currentData.drawID++;
                            activeInstances++;

                            if (activeInstances == mMaxInstanceCount)
                            {
                                // DISABLED_FOR_D3D12
                                //pContext->setProgram(currentData.pProgram->getActiveProgramVersion());
                                draw(currentData, pMesh, activeInstances);
                                activeInstances = 0;
                            }
                        }
                    }
                }
            }
            if(activeInstances != 0)
            {
                draw(currentData, pMesh, activeInstances);
            }
        }
    }

    void SceneRenderer::renderModelInstance(CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance)
    {
        const Model* pModel = pModelInstance->getObject().get();

        if (setPerModelData(currentData))
        {
            Program* pProgram = currentData.pState->getProgram().get();
            // Bind the program
            if(pModel->hasBones())
            {
                pProgram->addDefine("_VERTEX_BLENDING");
            }

            mpLastMaterial = nullptr;

            // Loop over the meshes
            for (uint32_t meshID = 0; meshID < pModel->getMeshCount(); meshID++)
            {
                renderMeshInstances(currentData, pModelInstance, meshID);
            }

            // Restore the program state
            if(pModel->hasBones())
            {
                pProgram->removeDefine("_VERTEX_BLENDING");
            }
        }

    }

    bool SceneRenderer::update(double currentTime)
    {
        return mpScene->update(currentTime, mpCameraController.get());
    }

    void SceneRenderer::renderScene(RenderContext* pContext)
    {
        renderScene(pContext, mpScene->getActiveCamera().get());
    }

    void SceneRenderer::renderScene(CurrentWorkingData& currentData)
    {
        setPerFrameData(currentData);

        for (uint32_t modelID = 0; modelID < mpScene->getModelCount(); modelID++)
        {
            currentData.pModel = mpScene->getModel(modelID).get();

            for (uint32_t instanceID = 0; instanceID < mpScene->getModelInstanceCount(modelID); instanceID++)
            {
                const auto pInstance = mpScene->getModelInstance(modelID, instanceID).get();
                if (pInstance->isVisible())
                {
                    if (setPerModelInstanceData(currentData, pInstance, instanceID))
                    {
                        renderModelInstance(currentData, pInstance);
                    }
                }
            }
        }
    }

    void SceneRenderer::renderScene(RenderContext* pContext, Camera* pCamera)
    {
        updateVariableOffsets(pContext->getGraphicsVars()->getReflection().get());

        CurrentWorkingData currentData;
        currentData.pContext = pContext;
        currentData.pState = pContext->getGraphicsState().get();
        currentData.pVars = pContext->getGraphicsVars().get();
        currentData.pCamera = pCamera;
        currentData.pMaterial = nullptr;
        currentData.pModel = nullptr;
        currentData.drawID = 0;
        renderScene(currentData);
    }

    static CameraController::SharedPtr createHmdCameraController()
    {

    }

    void SceneRenderer::setCameraControllerType(CameraControllerType type)
    {
        switch(type)
        {
        case CameraControllerType::FirstPerson:
            mpCameraController = CameraController::SharedPtr(new FirstPersonCameraController);
            break;
        case CameraControllerType::SixDof:
            mpCameraController = CameraController::SharedPtr(new SixDoFCameraController);
            break;
        case CameraControllerType::Hmd:
            mpCameraController = CameraController::SharedPtr(new HmdCameraController);
            break;
        default:
            should_not_get_here();
        }
        mCamControllerType = type;
    }

    void SceneRenderer::detachCameraController()
    {
        mpCameraController->attachCamera(nullptr);
    }

    bool SceneRenderer::onMouseEvent(const MouseEvent& mouseEvent)
    {
        return mpCameraController->onMouseEvent(mouseEvent);
    }

    bool SceneRenderer::onKeyEvent(const KeyboardEvent& keyEvent)
    {
        return mpCameraController->onKeyEvent(keyEvent);
    }
}
