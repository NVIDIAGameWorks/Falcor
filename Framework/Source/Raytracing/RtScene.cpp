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
#include "Framework.h"
#include "RtScene.h"
#include "Graphics/Scene/SceneImporter.h"
#include "API/DescriptorSet.h"
#include "API/Device.h"

namespace Falcor
{
    RtScene::SharedPtr RtScene::loadFromFile(const std::string& filename, RtBuildFlags rtFlags, Model::LoadFlags modelLoadFlags, Scene::LoadFlags sceneLoadFlags)
    {
        RtScene::SharedPtr pRtScene = create(rtFlags);
        if (SceneImporter::loadScene(*pRtScene, filename, modelLoadFlags | Model::LoadFlags::BuffersAsShaderResource, sceneLoadFlags) == false)
        {
            pRtScene = nullptr;
        }

        int count = 0;
        for (auto& path : pRtScene->mpPaths)
        {
            for (uint32_t objIdx = 0u; objIdx < path->getAttachedObjectCount(); objIdx++)
            {
                const auto& it = pRtScene->mModelInstanceToRtModelInstance.find(path->getAttachedObject(objIdx).get());
                if (it != pRtScene->mModelInstanceToRtModelInstance.end())
                {
                    path->attachObject(it->second);
                }
            }
        }

        pRtScene->mModelInstanceToRtModelInstance.clear();
        return pRtScene;
    }

    RtScene::SharedPtr RtScene::create(RtBuildFlags rtFlags)
    {
        return SharedPtr(new RtScene(rtFlags));
    }

    RtScene::SharedPtr RtScene::createFromModel(RtModel::SharedPtr pModel)
    {
        SharedPtr pScene = RtScene::create(pModel->getBuildFlags());
        pScene->addModelInstance(pModel, "instance0");

        return pScene;
    }

    bool RtScene::update(double currentTime, CameraController* cameraController)
    {
        bool changed = Scene::update(currentTime, cameraController);
        mTlasHitProgCount = mExtentsDirty ? -1 : mTlasHitProgCount;

        if (mEnableRefit)
        {
            mRefit = true;
        }
        return changed;
    }

    void RtScene::addModelInstance(const ModelInstance::SharedPtr& pInstance)
    {
        RtModel::SharedPtr pRtModel = std::dynamic_pointer_cast<RtModel>(pInstance->getObject());
        if (pRtModel)
        {
            Scene::addModelInstance(pInstance);
        }
        else
        {
            // Check if we need to create a new model
            const auto& it = mModelToRtModel.find(pInstance->getObject().get());
            if (it == mModelToRtModel.end())
            {
                pRtModel = RtModel::createFromModel(*pInstance->getObject(), mRtFlags);
                mModelToRtModel[pInstance->getObject().get()] = pRtModel;
            }
            else
            {
                pRtModel = it->second;
            }
            ModelInstance::SharedPtr pRtInstance = ModelInstance::create(pRtModel, pInstance->getTranslation(), pInstance->getTarget(), pInstance->getUpVector(), pInstance->getScaling(), pInstance->getName());
            Scene::addModelInstance(pRtInstance);

            // any paths attached to this ModelInstance need to be updated
            int count = 0;
            auto pMovable = std::dynamic_pointer_cast<IMovableObject>(pInstance);
            auto pRtMovable = std::dynamic_pointer_cast<IMovableObject>(pRtInstance);

            mModelInstanceToRtModelInstance[pMovable.get()] = pRtMovable;
        }

        // If we have skinned models, attach a skinning cache and animate the scene once to trigger a VB update
        if (pRtModel->hasBones())
        {
            pRtModel->attachSkinningCache(mpSkinningCache);
            pRtModel->animate(0);
        }
    }

    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> RtScene::createInstanceDesc(const RtScene* pScene, uint32_t hitProgCount)
    {
        mGeometryCount = 0;
        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDesc;
        mModelInstanceData.resize(pScene->getModelCount());

        uint32_t tlasIndex = 0;
        uint32_t instanceContributionToHitGroupIndex = 0;
        // Loop over all the models
        for (uint32_t modelId = 0; modelId < pScene->getModelCount(); modelId++)
        {
            auto& modelInstanceData = mModelInstanceData[modelId];
            const RtModel* pModel = dynamic_cast<RtModel*>(pScene->getModel(modelId).get());
            assert(pModel); // Can't work on regular models
            modelInstanceData.modelBase = tlasIndex;
            modelInstanceData.meshInstancesPerModelInstance = 0;
            modelInstanceData.meshBase.resize(pModel->getMeshCount());

            for (uint32_t modelInstance = 0; modelInstance < pScene->getModelInstanceCount(modelId); modelInstance++)
            {
                const auto& pModelInstance = pScene->getModelInstance(modelId, modelInstance);
                // Loop over the meshes
                for (uint32_t blasId = 0; blasId < pModel->getBottomLevelDataCount(); blasId++)
                {
                    // Initialize the instance desc
                    const auto& blasData = pModel->getBottomLevelData(blasId);
                    D3D12_RAYTRACING_INSTANCE_DESC idesc = {};
                    idesc.AccelerationStructure = blasData.pBlas->getGpuAddress();

                    // Set the meshes tlas offset
                    if (modelInstance == 0)
                    {
                        for (uint32_t i = 0; i < blasData.meshCount; i++)
                        {
                            assert(blasData.meshCount == 1 || pModel->getMeshInstanceCount(blasData.meshBaseIndex + i) == 1);   // A BLAS shouldn't have multiple instanced meshes
                            modelInstanceData.meshBase[blasData.meshBaseIndex + i] = modelInstanceData.meshInstancesPerModelInstance + i;   // If i>0 each mesh has a single instance
                        }
                    }

                    uint32_t meshInstanceCount = pModel->getMeshInstanceCount(blasData.meshBaseIndex);
                    for (uint32_t meshInstance = 0; meshInstance < meshInstanceCount; meshInstance++)
                    {
                        idesc.InstanceID = uint32_t(instanceDesc.size());
                        idesc.InstanceContributionToHitGroupIndex = instanceContributionToHitGroupIndex;
                        instanceContributionToHitGroupIndex += hitProgCount * blasData.meshCount;
                        idesc.InstanceMask = 0xff;
                        const auto& pMaterial = pModel->getMeshInstance(blasData.meshBaseIndex, meshInstance)->getObject()->getMaterial();
                        idesc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;

                        // TODO: This code is incorrect since a BLAS can have multiple meshes with different materials and hence different doubleSided flags.
                        if (pMaterial->getDoubleSided())
                        {
                            idesc.Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
                        }

                        // Only apply mesh-instance transform on non-skinned meshes
                        mat4 transform = pModelInstance->getTransformMatrix();
                        if (blasData.isStatic)
                        {
                            transform = transform * pModel->getMeshInstance(blasData.meshBaseIndex, meshInstance)->getTransformMatrix();    // If there are multiple meshes in a BLAS, they all have the same transform
                        }
                        transform = transpose(transform);
                        memcpy(idesc.Transform, &transform, sizeof(idesc.Transform));
                        instanceDesc.push_back(idesc);
                        mGeometryCount += blasData.meshCount;
                        if (modelInstance == 0) modelInstanceData.meshInstancesPerModelInstance += blasData.meshCount;
                        tlasIndex += blasData.meshCount;
                        assert(tlasIndex * hitProgCount == instanceContributionToHitGroupIndex);
                    }
                }
            }
        }
        assert(tlasIndex == mGeometryCount);

        // Validate that our getInstanceId() helper returns contigous indices.
        uint32_t instanceId = 0;
        for (uint32_t model = 0; model < getModelCount(); model++)
        {
            for (uint32_t modelInstance = 0; modelInstance < getModelInstanceCount(model); modelInstance++)
            {
                for (uint32_t mesh = 0; mesh < getModel(model)->getMeshCount(); mesh++)
                {
                    for (uint32_t meshInstance = 0; meshInstance < getModel(model)->getMeshInstanceCount(mesh); meshInstance++)
                    {
                        assert(getInstanceId(model, modelInstance, mesh, meshInstance) == instanceId++);
                    }
                }
            }
        }
        assert(instanceId == mGeometryCount);

        return instanceDesc;
    }

    // TODO: Cache TLAS per hitProgCount, as some render pipelines need multiple TLAS:es with different #hit progs in same frame, currently that trigger rebuild every frame. See issue #365.
    void RtScene::createTlas(uint32_t hitProgCount)
    {
        if (mTlasHitProgCount == hitProgCount) return;
        mTlasHitProgCount = hitProgCount;

        // Early out if hit program count is zero or if scene is empty.
        if (hitProgCount == 0 || getModelCount() == 0)
        {
            mModelInstanceData.clear();
            mpTopLevelAS = nullptr;
            mTlasSrv = nullptr;
            mGeometryCount = 0;
            mInstanceCount = 0;
            mRefit = false;
            return;
        }

        // todo: move this somewhere fair.
        mRtFlags |= RtBuildFlags::AllowUpdate;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS dxrFlags = getDxrBuildFlags(mRtFlags);
        RenderContext* pContext = gpDevice->getRenderContext().get();
        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDesc = createInstanceDesc(this, hitProgCount);

        // todo: improve this check - make sure things have not changed much and update was enabled last time
        bool isRefitPossible = mRefit && mpTopLevelAS && (mInstanceCount == (uint32_t)instanceDesc.size());

        mInstanceCount = (uint32_t)instanceDesc.size();

        // Create the top-level acceleration buffers
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
        inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.NumDescs = mInstanceCount;
        inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info;
        GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
        pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

        Buffer::SharedPtr pScratchBuffer = Buffer::create(align_to(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, info.ScratchDataSizeInBytes), Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);

        if (!isRefitPossible)
        {
            mpTopLevelAS = Buffer::create(align_to(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, info.ResultDataMaxSizeInBytes), Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
        }
        else
        {
            pContext->uavBarrier(mpTopLevelAS.get());
        }

        Buffer::SharedPtr pInstanceData = Buffer::create(mInstanceCount * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), Buffer::BindFlags::None, Buffer::CpuAccess::None, instanceDesc.data());
        assert((mInstanceCount != 0) && pInstanceData->getApiHandle() && mpTopLevelAS->getApiHandle() && pScratchBuffer->getApiHandle());

        // Create the TLAS
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
        asDesc.Inputs = inputs;
        asDesc.Inputs.InstanceDescs = pInstanceData->getGpuAddress();
        asDesc.DestAccelerationStructureData = mpTopLevelAS->getGpuAddress();
        asDesc.ScratchAccelerationStructureData = pScratchBuffer->getGpuAddress();

        if (isRefitPossible)
        {
            asDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            asDesc.SourceAccelerationStructureData = asDesc.DestAccelerationStructureData;
        }

        GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
        pContext->resourceBarrier(pInstanceData.get(), Resource::State::NonPixelShader);
        pList4->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);
        pContext->uavBarrier(mpTopLevelAS.get());

        // Create the SRV
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.RaytracingAccelerationStructure.Location = mpTopLevelAS->getGpuAddress();

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::TextureSrv, 0, 1);
        DescriptorSet::SharedPtr pSet = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        assert(pSet);
        gpDevice->getApiHandle()->CreateShaderResourceView(nullptr, &srvDesc, pSet->getCpuHandle(0));

        ResourceWeakPtr pWeak = mpTopLevelAS;
        mTlasSrv = std::make_shared<ShaderResourceView>(pWeak, pSet, 0, 1, 0, 1);

        mRefit = false;
    }
}