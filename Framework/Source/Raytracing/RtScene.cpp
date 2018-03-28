/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
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
        if (SceneImporter::loadScene(*pRtScene, filename, modelLoadFlags, sceneLoadFlags) == false)
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

        if(mEnableRefit)
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
            if(it == mModelToRtModel.end())
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
        if(pRtModel->hasBones())
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
                    uint32_t meshInstanceCount = pModel->getMeshInstanceCount(blasData.meshBaseIndex);
                    assert(meshInstanceCount == 1 || blasData.meshCount == 1); // A BLAS shouldn't have multiple instanced meshes
                    if (modelInstance == 0)                        
                    {
                        for(uint32_t i = 0 ; i < blasData.meshCount ; i++)
                        {
                            modelInstanceData.meshBase[blasData.meshBaseIndex + i] = modelInstanceData.meshInstancesPerModelInstance + i * meshInstanceCount;
                        }
                    }

                    for (uint32_t meshInstance = 0; meshInstance < meshInstanceCount; meshInstance++)
                    {
                        idesc.InstanceID = uint32_t(instanceDesc.size());
                        idesc.InstanceContributionToHitGroupIndex = instanceContributionToHitGroupIndex;
                        instanceContributionToHitGroupIndex += hitProgCount * blasData.meshCount;
                        idesc.InstanceMask = 0xff;
                        idesc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
                        // Only apply mesh-instance transform on non-skinned meshes
                        mat4 transform = pModelInstance->getTransformMatrix();
                        if (blasData.isStatic)
                        {
                            transform = transform * pModel->getMeshInstance(blasData.meshBaseIndex, meshInstance)->getTransformMatrix();    // PETRIK: If there are multiple meshes in a BLAS, they all have the same transform so this is OK.
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

    void RtScene::createTlas(uint32_t hitProgCount)
    {
        if (mTlasHitProgCount == hitProgCount) return;
        mTlasHitProgCount = hitProgCount;

        // todo: move this somewhere fair.
        mRtFlags |= RtBuildFlags::AllowUpdate;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS dxrFlags = getDxrBuildFlags(mRtFlags);
        ID3D12CommandListRaytracingPrototypePtr pRtCmdList = gpDevice->getRenderContext()->getLowLevelData()->getCommandList();
        ID3D12DeviceRaytracingPrototypePtr pRtDevice = gpDevice->getApiHandle();
        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDesc = createInstanceDesc(this, hitProgCount);

        // todo: improve this check - make sure things have not changed much and update was enabled last time
        bool isRefitPossible = mRefit && mpTopLevelAS && (mInstanceCount == (uint32_t)instanceDesc.size());

        mInstanceCount = (uint32_t)instanceDesc.size();

        // Create the top-level acceleration buffers
        D3D12_GET_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO_DESC prebuildDesc = {};
        prebuildDesc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        prebuildDesc.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        prebuildDesc.NumDescs = mInstanceCount;
        prebuildDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info;
        pRtDevice->GetRaytracingAccelerationStructurePrebuildInfo(&prebuildDesc, &info);

        Buffer::SharedPtr pScratchBuffer = Buffer::create(align_to(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, info.ScratchDataSizeInBytes), Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);

        if (!isRefitPossible)
        {
            mpTopLevelAS = Buffer::create(align_to(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, info.ResultDataMaxSizeInBytes), Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
        }
        Buffer::SharedPtr pInstanceData = Buffer::create(mInstanceCount * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), Buffer::BindFlags::None, Buffer::CpuAccess::None, instanceDesc.data());

        assert((mInstanceCount != 0) && pInstanceData->getApiHandle() && mpTopLevelAS->getApiHandle() && pScratchBuffer->getApiHandle());

        if (isRefitPossible) dxrFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;

        // Create the TLAS
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
        asDesc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        asDesc.DestAccelerationStructureData.StartAddress = mpTopLevelAS->getGpuAddress();
        asDesc.DestAccelerationStructureData.SizeInBytes = mpTopLevelAS->getSize();
        asDesc.Flags = dxrFlags;
        asDesc.InstanceDescs = pInstanceData->getGpuAddress();
        asDesc.NumDescs = mInstanceCount;
        asDesc.ScratchAccelerationStructureData.StartAddress = pScratchBuffer->getGpuAddress();
        asDesc.ScratchAccelerationStructureData.SizeInBytes = pScratchBuffer->getSize();
        asDesc.SourceAccelerationStructureData = isRefitPossible ? asDesc.DestAccelerationStructureData.StartAddress : 0;

        asDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        pRtCmdList->BuildRaytracingAccelerationStructure(&asDesc);

        // Create the SRV
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.RaytracingAccelerationStructure.Location = mpTopLevelAS->getGpuAddress();

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::TextureSrv, 0, 1);
        DescriptorSet::SharedPtr pSet = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateShaderResourceView(nullptr, &srvDesc, pSet->getCpuHandle(0));

        ResourceWeakPtr pWeak = mpTopLevelAS;
        mTlasSrv = std::make_shared<ShaderResourceView>(pWeak, pSet, 0, 1, 0, 1);

        mRefit = false;
    }
}
