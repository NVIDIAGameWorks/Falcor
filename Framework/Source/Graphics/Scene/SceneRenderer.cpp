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
#include "Graphics/Program/Program.h"
#include "Utils/Gui.h"
#include "API/ConstantBuffer.h"
#include "API/RenderContext.h"
#include "Scene.h"
#include "Utils/Platform/OS.h"
#include "VR/OpenVR/VRSystem.h"
#include "API/Device.h"
#include "glm/matrix.hpp"
#include "Graphics/Material/MaterialSystem.h"
#include "Graphics/Light.h"
#include "Effects/Shadows/CSM.h"

namespace Falcor
{
    size_t SceneRenderer::sBonesOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sBonesInvTransposeOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sCameraDataOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sWorldMatArraySize = 0;
    size_t SceneRenderer::sWorldMatOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sPrevWorldMatOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sWorldInvTransposeMatOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sMeshIdOffset = ConstantBuffer::kInvalidOffset;
    size_t SceneRenderer::sDrawIDOffset = ConstantBuffer::kInvalidOffset;

    const char* SceneRenderer::kPerMaterialCbName = "InternalPerMaterialCB";
    const char* SceneRenderer::kPerFrameCbName = "InternalPerFrameCB";
    const char* SceneRenderer::kPerMeshCbName = "InternalPerMeshCB";
    const char* SceneRenderer::kBoneCbName = "InternalBoneCB";
    const char* SceneRenderer::kAreaLightCbName = "InternalAreaLightCB";


    SceneRenderer::SharedPtr SceneRenderer::create(Scene* pScene)
    {
        return SharedPtr(new SceneRenderer(pScene));
    }

    SceneRenderer::SharedPtr SceneRenderer::create(Scene::SharedPtr pScene)
    {
        return SharedPtr(new SceneRenderer(pScene.get()));
    }

    SceneRenderer::SceneRenderer(Scene* pScene) : mpScene(pScene)
    {
        setCameraControllerType(CameraControllerType::SixDof);
    }

    void SceneRenderer::updateVariableOffsets(const ProgramReflection* pReflector)
    {
        const ParameterBlockReflection* pBlock = pReflector->getDefaultParameterBlock().get();
        if (sWorldMatOffset == ConstantBuffer::kInvalidOffset)
        {
            const ReflectionVar* pVar = pBlock->getResource(kPerMeshCbName).get();
            assert(pVar->getType()->asResourceType()->getType() == ReflectionResourceType::Type::ConstantBuffer);

            if (pVar != nullptr)
            {
                const ReflectionType* pType = pVar->getType().get();

                assert(pType->findMember("gWorldMat[0]")->getType()->asBasicType()->isRowMajor() == false); // We copy into CBs as column-major
                assert(pType->findMember("gWorldInvTransposeMat[0]")->getType()->asBasicType()->isRowMajor() == false);
                assert(pType->findMember("gWorldMat")->getType()->getTotalArraySize() == pType->findMember("gWorldInvTransposeMat")->getType()->getTotalArraySize());

                sWorldMatArraySize = pType->findMember("gWorldMat")->getType()->getTotalArraySize();
                sWorldMatOffset = pType->findMember("gWorldMat[0]")->getOffset();
                sWorldInvTransposeMatOffset = pType->findMember("gWorldInvTransposeMat[0]")->getOffset();
                sMeshIdOffset = pType->findMember("gMeshId")->getOffset();
                sDrawIDOffset = pType->findMember("gDrawId[0]")->getOffset();
                sPrevWorldMatOffset = pType->findMember("gPrevWorldMat[0]")->getOffset();
            }
        }

        if (sCameraDataOffset == ConstantBuffer::kInvalidOffset)
        {
            const ReflectionVar* pVar = pBlock->getResource(kPerFrameCbName).get();
            assert(pVar->getType()->asResourceType()->getType() == ReflectionResourceType::Type::ConstantBuffer);

            if (pVar != nullptr)
            {
                const ReflectionType* pType = pVar->getType().get();
                sCameraDataOffset = pType->findMember("gCamera.viewMat")->getOffset();
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
            pCB->setVariable("gCamVpAtLastCsmUpdate", camVpAtLastCsmUpdate);
        }
        
        currentData.pVars->setParameterBlock("gLightEnv", currentData.pLightEnv->getParameterBlock());

        if (mpScene->getAreaLightCount() > 0)
        {
            const ParameterBlockReflection* pBlock = currentData.pVars->getReflection()->getDefaultParameterBlock().get();

            // If area lights have been declared
            const ReflectionVar* pVar = pBlock->getResource(kAreaLightCbName).get();
            if (pVar != nullptr)
            {
                const ReflectionVar* pAreaLightVar = pVar->getType()->findMember("gAreaLights").get();
                assert(pAreaLightVar != nullptr);

                uint32_t areaLightArraySize = pAreaLightVar->getType()->asArrayType()->getArraySize();
                for (uint32_t i = 0; i < min(areaLightArraySize, mpScene->getAreaLightCount()); i++)
                {
                    std::string varName = "gAreaLights[" + std::to_string(i) + "]";
                    mpScene->getAreaLight(i)->setIntoProgramVars(currentData.pVars, currentData.pVars->getConstantBuffer(kAreaLightCbName).get(), varName.c_str());
                }
            }
        }
    }

    bool SceneRenderer::setPerModelData(const CurrentWorkingData& currentData)
    {
        const Model* pModel = currentData.pModel;

        // Set bones
        if (pModel->hasBones())
        {
            ConstantBuffer* pCB = currentData.pVars->getConstantBuffer(kBoneCbName).get();
            if (pCB != nullptr)
            {
                if (sBonesOffset == ConstantBuffer::kInvalidOffset || sBonesInvTransposeOffset == ConstantBuffer::kInvalidOffset)
                {
                    sBonesOffset = pCB->getVariableOffset("gBoneMat[0]");
                    sBonesInvTransposeOffset = pCB->getVariableOffset("gInvTransposeBoneMat[0]");
                }

                assert(pModel->getBoneCount() <= MAX_BONES);
                pCB->setVariableArray(sBonesOffset, pModel->getBoneMatrices(), pModel->getBoneCount());
                pCB->setVariableArray(sBonesInvTransposeOffset, pModel->getBoneInvTransposeMatrices(), pModel->getBoneCount());
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

            glm::mat4 worldMat = pModelInstance->getTransformMatrix();
            glm::mat4 prevWorldMat = pModelInstance->getPrevTransformMatrix();

            if (pMesh->hasBones() == false)
            {
                worldMat = worldMat * pMeshInstance->getTransformMatrix();
                prevWorldMat = prevWorldMat * pMeshInstance->getPrevTransformMatrix();
            }

            glm::mat3x4 worldInvTransposeMat = transpose(inverse(glm::mat3(worldMat)));

            assert(drawInstanceID < sWorldMatArraySize);
            pCB->setBlob(&worldMat, sWorldMatOffset + drawInstanceID * sizeof(glm::mat4), sizeof(glm::mat4));
            pCB->setBlob(&worldInvTransposeMat, sWorldInvTransposeMatOffset + drawInstanceID * sizeof(glm::mat3x4), sizeof(glm::mat3x4)); // HLSL uses column-major and packing rules require 16B alignment, hence use glm:mat3x4
            pCB->setBlob(&prevWorldMat, sPrevWorldMatOffset + drawInstanceID * sizeof(glm::mat4), sizeof(glm::mat4));

            // Set mesh id
            pCB->setVariable(sMeshIdOffset, pMesh->getId());
        }

        return true;
    }

    bool SceneRenderer::setPerMaterialData(const CurrentWorkingData& currentData, const Material* pMaterial)
    {
        currentData.pVars->setParameterBlock("gMaterial", pMaterial->getParameterBlock());
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
        }

        executeDraw(currentData, pMesh->getIndexCount(), instanceCount);
        postFlushDraw(currentData);
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
            Program* pProgram = currentData.pState->getProgram().get();
            if (pMesh->hasBones())
            {
                pProgram->addDefine("_VERTEX_BLENDING");
            }

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

            // Restore the program state
            if (pMesh->hasBones())
            {
                pProgram->removeDefine("_VERTEX_BLENDING");
            }
        }
    }

    void SceneRenderer::renderModelInstance(CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance)
    {
        mpLastMaterial = nullptr;

        // Loop over the meshes
        for (uint32_t meshID = 0; meshID < pModelInstance->getObject()->getMeshCount(); meshID++)
        {
            renderMeshInstances(currentData, pModelInstance, meshID);
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
        currentData.pLightEnv = mpScene->getLightEnv().get();
        setPerFrameData(currentData);

        for (uint32_t modelID = 0; modelID < mpScene->getModelCount(); modelID++)
        {
            currentData.pModel = mpScene->getModel(modelID).get();

            if (setPerModelData(currentData))
            {
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

    void SceneRenderer::runShadowPass(RenderContext * pContext, Camera * pCamera, Texture::SharedPtr pDepthBuffer)
    {
        camVpAtLastCsmUpdate = pCamera->getViewProjMatrix();
        for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
        {
            auto light = mpScene->getLight(i);
            if (light->getTypeId() & LightType_ShadowBit)
            {
                auto infLight = dynamic_cast<InfinitesimalLight*>(light.get());
                infLight->getCsm()->setup(pContext, mpScene->getActiveCamera().get(), pDepthBuffer);
            }
        }
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
