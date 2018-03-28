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
#include "RtSceneRenderer.h"
#include "RtProgramVars.h"
#include "RtState.h"

namespace Falcor
{
    RtSceneRenderer::SharedPtr RtSceneRenderer::create(RtScene::SharedPtr pScene)
    {
        return SharedPtr(new RtSceneRenderer(pScene));
    }

    void RtSceneRenderer::setHitShaderData(RtProgramVars* pRtVars, InstanceData& data)
    {
        const RtScene* pScene = dynamic_cast<RtScene*>(mpScene.get());
        uint32_t instanceId = pScene->getInstanceId(data.model, data.modelInstance, data.mesh, data.meshInstance);
        data.currentData.pVars = pRtVars->getHitVars(data.progId)[instanceId].get();
        if(data.currentData.pVars)
        {
            const Model* pModel = mpScene->getModel(data.model).get();
            const Scene::ModelInstance* pModelInstance = mpScene->getModelInstance(data.model, data.modelInstance).get();
            const Mesh* pMesh = pModel->getMesh(data.mesh).get();
            const Model::MeshInstance* pMeshInstance = pModel->getMeshInstance(data.mesh, data.meshInstance).get();

            setPerFrameData(pRtVars, data);
            setPerModelData(data.currentData);
            setPerModelInstanceData(data.currentData, pModelInstance, data.modelInstance);
            setPerMeshData(data.currentData, pMesh);
            setPerMeshInstanceData(data.currentData, pModelInstance, pMeshInstance, 0);
            setPerMaterialData(data.currentData, pMesh->getMaterial().get());
        }
    }

    void RtSceneRenderer::initializeMeshBufferLocation(const ProgramReflection* pReflection)
    {
        mMeshBufferLocations.indices = pReflection->getDefaultParameterBlock()->getResourceBinding("gIndices");
        mMeshBufferLocations.texC = pReflection->getDefaultParameterBlock()->getResourceBinding("gTexCrds");
        mMeshBufferLocations.lightmapUVs = pReflection->getDefaultParameterBlock()->getResourceBinding("gLightMapUVs");
        mMeshBufferLocations.normal = pReflection->getDefaultParameterBlock()->getResourceBinding("gNormals");
        mMeshBufferLocations.position = pReflection->getDefaultParameterBlock()->getResourceBinding("gPositions");
        mMeshBufferLocations.prevPosition = pReflection->getDefaultParameterBlock()->getResourceBinding("gPrevPositions");
        mMeshBufferLocations.bitangent = pReflection->getDefaultParameterBlock()->getResourceBinding("gBitangents");
    }

    static bool setVertexBuffer(ParameterBlockReflection::BindLocation bindLocation, uint32_t vertexLoc, const Vao* pVao, GraphicsVars* pVars)
    {
        if (bindLocation.setIndex != ProgramReflection::kInvalidLocation)
        {
            const auto& elemDesc = pVao->getElementIndexByLocation(vertexLoc);
            if (elemDesc.elementIndex == Vao::ElementDesc::kInvalidIndex)
            {
                pVars->getDefaultBlock()->setSrv(bindLocation, 0, nullptr);
            }
            else
            {
                assert(elemDesc.elementIndex == 0);
                pVars->getDefaultBlock()->setSrv(bindLocation, 0, pVao->getVertexBuffer(elemDesc.vbIndex)->getSRV());
                return true;
            }
        }
        return false;
    }

    bool RtSceneRenderer::setPerMeshInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, const Model::MeshInstance* pMeshInstance, uint32_t drawInstanceID)
    {
        const Mesh* pMesh = pMeshInstance->getObject().get();
        const Vao* pVao = pModelInstance->getObject()->getMeshVao(pMesh).get();
        auto& pVars = currentData.pVars;

        if (mMeshBufferLocations.indices.setIndex != ProgramReflection::kInvalidLocation)
        {
            auto pSrv = pVao->getIndexBuffer() ? pVao->getIndexBuffer()->getSRV() : nullptr;
            pVars->getDefaultBlock()->setSrv(mMeshBufferLocations.indices, 0, pSrv);
        }

        setVertexBuffer(mMeshBufferLocations.lightmapUVs, VERTEX_LIGHTMAP_UV_LOC, pVao, pVars);
        setVertexBuffer(mMeshBufferLocations.texC, VERTEX_TEXCOORD_LOC, pVao, pVars);
        setVertexBuffer(mMeshBufferLocations.normal, VERTEX_NORMAL_LOC, pVao, pVars);
        setVertexBuffer(mMeshBufferLocations.position, VERTEX_POSITION_LOC, pVao, pVars);
        setVertexBuffer(mMeshBufferLocations.bitangent, VERTEX_BITANGENT_LOC, pVao, pVars);

        // Bind vertex buffer for previous positions if it exists. If not, we bind the current positions.
        if (!setVertexBuffer(mMeshBufferLocations.prevPosition, VERTEX_PREV_POSITION_LOC, pVao, pVars))
        {
            setVertexBuffer(mMeshBufferLocations.prevPosition, VERTEX_POSITION_LOC, pVao, pVars);
        }

        return SceneRenderer::setPerMeshInstanceData(currentData, pModelInstance, pMeshInstance, drawInstanceID);
    }

    void RtSceneRenderer::setPerFrameData(RtProgramVars* pRtVars, InstanceData& data)
    {
        GraphicsVars* pVars = data.currentData.pVars;
        ParameterBlockReflection::BindLocation loc = pVars->getReflection()->getDefaultParameterBlock()->getResourceBinding("gRtScene");
        if (loc.setIndex != ProgramReflection::kInvalidLocation)
        {
            RtScene* pRtScene = dynamic_cast<RtScene*>(mpScene.get());
            pVars->getDefaultBlock()->setSrv(loc, 0, pRtScene->getTlasSrv(pRtVars->getHitProgramsCount()));
        }

        ConstantBuffer::SharedPtr pSbtCB = pVars->getConstantBuffer("DxrPerFrame");
        if (pSbtCB)
        {
            pSbtCB["hitProgramCount"] = pRtVars->getHitProgramsCount();
        }

        SceneRenderer::setPerFrameData(data.currentData);
    }

    void RtSceneRenderer::setRayGenShaderData(RtProgramVars* pRtVars, InstanceData& data)
    {
        data.currentData.pVars = pRtVars->getRayGenVars().get();
        setPerFrameData(pRtVars, data);
    }

    void RtSceneRenderer::setMissShaderData(RtProgramVars* pRtVars, InstanceData& data)
    {
        data.currentData.pVars = pRtVars->getMissVars(data.progId).get();
        if(data.currentData.pVars)
        {
            setPerFrameData(pRtVars, data);
        }
    }

    void RtSceneRenderer::renderScene(RenderContext* pContext, RtProgramVars::SharedPtr pRtVars, RtState::SharedPtr pState, uvec2 targetDim, Camera* pCamera)
    {
        InstanceData data;
        data.currentData.pCamera = pCamera;
        if(pRtVars->getMissProgramsCount())
        {
            updateVariableOffsets(pRtVars->getMissVars(0)->getReflection().get());
        }
        setRayGenShaderData(pRtVars.get(), data);

        // Set the miss-shader data
        for (data.progId = 0; data.progId < pRtVars->getMissProgramsCount(); data.progId++)
        {
            setMissShaderData(pRtVars.get(), data);
        }

        // Set the hit-shader data
        uint32_t hitCount = pRtVars->getHitProgramsCount();
        if (hitCount)
        {
            initializeMeshBufferLocation(pState->getProgram()->getHitProgram(0)->getActiveVersion()->getReflector().get());
        }

        for(data.progId = 0 ; data.progId < hitCount ; data.progId++)
        {
            for (data.model = 0; data.model < mpScene->getModelCount(); data.model++)
            {
                const Model* pModel = mpScene->getModel(data.model).get();
                data.currentData.pModel = pModel;
                for (data.modelInstance = 0; data.modelInstance < mpScene->getModelInstanceCount(data.model); data.modelInstance++)
                {
                    for (data.mesh = 0; data.mesh < pModel->getMeshCount(); data.mesh++)
                    {
                        const Mesh* pMesh = pModel->getMesh(data.mesh).get();
                        for (data.meshInstance = 0; data.meshInstance < pModel->getMeshInstanceCount(data.mesh); data.meshInstance++)
                        {
                            setHitShaderData(pRtVars.get(), data);
                        }
                    }
                }
            }
        }

        if (!pRtVars->apply(pContext, pState->getRtso().get()))
        {
            logError("RtSceneRenderer::renderScene() - applying RtProgramVars failed, most likely because we ran out of descriptors.", true);
            assert(false);
        }
        raytrace(pContext, pRtVars, pState, targetDim.x, targetDim.y);
    }
}
