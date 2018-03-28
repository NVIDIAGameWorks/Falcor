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
#include "RtScene.h"
#include "Graphics/Scene/SceneRenderer.h"

namespace Falcor
{
    class RtProgramVars;

    class RtSceneRenderer : public SceneRenderer, inherit_shared_from_this<SceneRenderer, RtSceneRenderer>
    {
    public:
        using SharedPtr = std::shared_ptr<RtSceneRenderer>;
        using SharedConstPtr = std::shared_ptr<const RtSceneRenderer>;

        static SharedPtr create(RtScene::SharedPtr pScene);
        void renderScene(RenderContext* pContext, RtProgramVars::SharedPtr pRtVars, RtState::SharedPtr pState, uvec2 targetDim, Camera* pCamera = nullptr);
    protected:
        RtSceneRenderer(RtScene::SharedPtr pScene) : SceneRenderer(pScene) {}
        struct InstanceData
        {
            CurrentWorkingData currentData;
            uint32_t model;
            uint32_t modelInstance;
            uint32_t mesh;
            uint32_t meshInstance;
            uint32_t progId;
        };

        virtual void setPerFrameData(RtProgramVars* pRtVars, InstanceData& data);
        virtual bool setPerMeshInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, const Model::MeshInstance* pMeshInstance, uint32_t drawInstanceID) override;
        virtual void setHitShaderData(RtProgramVars* pRtVars, InstanceData& data);
        virtual void setMissShaderData(RtProgramVars* pRtVars, InstanceData& data);
        virtual void setRayGenShaderData(RtProgramVars* pRtVars, InstanceData& data);

        void initializeMeshBufferLocation(const ProgramReflection* pReflection);

        struct MeshBufferLocations
        {
            ParameterBlockReflection::BindLocation indices;
            ParameterBlockReflection::BindLocation normal;
            ParameterBlockReflection::BindLocation bitangent;
            ParameterBlockReflection::BindLocation position;
            ParameterBlockReflection::BindLocation prevPosition;
            ParameterBlockReflection::BindLocation texC;
            ParameterBlockReflection::BindLocation lightmapUVs;
        };
        MeshBufferLocations mMeshBufferLocations;
    };
}
