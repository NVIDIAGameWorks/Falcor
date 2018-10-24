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
#pragma once
#include "Graphics/Scene/Scene.h"
#include "RtModel.h"
#include <map>

namespace Falcor
{
    class RtScene : public Scene, inherit_shared_from_this<Scene, RtScene>
    {
    public:
        using SharedPtr = std::shared_ptr<RtScene>;
        using SharedConstPtr = std::shared_ptr<const RtScene>;
        SharedPtr shared_from_this() { return inherit_shared_from_this<Scene, RtScene>::shared_from_this(); }

        static RtScene::SharedPtr loadFromFile(const std::string& filename, RtBuildFlags rtFlags = RtBuildFlags::None, Model::LoadFlags modelLoadFlags = Model::LoadFlags::None, Scene::LoadFlags sceneLoadFlags = LoadFlags::None);
        static RtScene::SharedPtr create(RtBuildFlags rtFlags);
        static RtScene::SharedPtr createFromModel(RtModel::SharedPtr pModel);

        ShaderResourceView::SharedPtr getTlasSrv(uint32_t hitProgCount) { createTlas(hitProgCount); return mTlasSrv; }
        void addModelInstance(const ModelInstance::SharedPtr& pInstance) override;
        using Scene::addModelInstance;
        uint32_t getGeometryCount(uint32_t rayCount) { createTlas(rayCount); return mGeometryCount; }
        uint32_t getInstanceCount(uint32_t rayCount) { createTlas(rayCount); return mInstanceCount; }
        uint32_t getInstanceId(uint32_t model, uint32_t modelInstance, uint32_t mesh, uint32_t meshInstance) const 
        {
            assert(model < mModelInstanceData.size() && mesh < mModelInstanceData[model].meshBase.size());
            uint32_t modelBase = mModelInstanceData[model].modelBase + mModelInstanceData[model].meshInstancesPerModelInstance * modelInstance;
            modelBase += mModelInstanceData[model].meshBase[mesh] + meshInstance;
            assert(modelBase < mGeometryCount);
            return modelBase;
        }
        virtual bool update(double currentTime, CameraController* cameraController = nullptr) override;

        void setRefit(bool enableRefit) { mEnableRefit = enableRefit; }

    protected:
        RtScene(RtBuildFlags rtFlags) : mRtFlags(rtFlags), mpSkinningCache(SkinningCache::create()) {}
        uint32_t mTlasHitProgCount = -1;
        RtBuildFlags mRtFlags;

        Buffer::SharedPtr mpTopLevelAS;             // The top-level acceleration structure for the model
        ShaderResourceView::SharedPtr mTlasSrv;
        void createTlas(uint32_t rayCount);
        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> createInstanceDesc(const RtScene* pScene, uint32_t hitProgCount);

        uint32_t mGeometryCount = 0;    // The total number of geometries in the scene
        uint32_t mInstanceCount = 0;    // The total number of TLAS instances in the scene

        struct ModelInstanceData
        {
            uint32_t modelBase = 0;
            uint32_t meshInstancesPerModelInstance = 0;
            std::vector<uint32_t> meshBase;
        };

        std::vector<ModelInstanceData> mModelInstanceData;
        std::unordered_map<const Model*, RtModel::SharedPtr> mModelToRtModel;
        std::unordered_map<IMovableObject*, IMovableObject::SharedPtr> mModelInstanceToRtModelInstance;

        SkinningCache::SharedPtr mpSkinningCache;

        bool mEnableRefit = false;
        bool mRefit = false;
    };
}