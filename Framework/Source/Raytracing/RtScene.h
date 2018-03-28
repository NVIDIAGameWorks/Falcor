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
