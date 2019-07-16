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
#include "../RtSceneRenderer.h"
#include "../RtProgramVars.h"

namespace Falcor
{
    void RtSceneRenderer::setMaterialDataForGeometry(RtProgramVars* pRtVars, const Material* pMaterial, uint32_t geometryID)
    {
        if (mMaterialResourceLocations.baseColor.setIndex == ProgramReflection::BindLocation::kInvalidLocation)
        {
            auto pBlockReflection = pRtVars->getGlobalVars()->getReflection()->getParameterBlock("gRtMaterials");
            mMaterialResourceLocations.baseColor = pBlockReflection->getResourceBinding("baseColor");
            mMaterialResourceLocations.specular = pBlockReflection->getResourceBinding("specular");
            mMaterialResourceLocations.emissive = pBlockReflection->getResourceBinding("emissive");
            mMaterialResourceLocations.normalMap = pBlockReflection->getResourceBinding("normalMap");
            mMaterialResourceLocations.occlusionMap = pBlockReflection->getResourceBinding("occlusionMap");
            mMaterialResourceLocations.lightMap = pBlockReflection->getResourceBinding("lightMap");
            mMaterialResourceLocations.heightMap = pBlockReflection->getResourceBinding("heightMap");
            mMaterialResourceLocations.samplerState = pBlockReflection->getResourceBinding("samplerState");
        }

        RtScene* pScene = dynamic_cast<RtScene*>(mpScene.get());
        uint32_t geometryCount = pScene->getGeometryCount(pRtVars->getHitProgramsCount());
        if (!mpMaterialBlock || mGeometryCount != geometryCount)
        {
            mGeometryCount = geometryCount;

            auto pProgramReflection = pRtVars->getGlobalVars()->getReflection();
            auto pBlockReflection = pProgramReflection->getParameterBlock("gRtMaterials");
            auto pStructuredBufferReflection = std::dynamic_pointer_cast<const ReflectionResourceType>(pBlockReflection->getResource("constants")->getType());

            mpMaterialConstantsBuffer = StructuredBuffer::create("MaterialConstants", pStructuredBufferReflection, geometryCount);

            mpMaterialBlock = ParameterBlock::create(pBlockReflection, false);
            mpMaterialBlock->setStructuredBuffer("constants", mpMaterialConstantsBuffer);
        }

        static const size_t materialConstantsSize = sizeof(MaterialData) - sizeof(MaterialResources);
        mpMaterialConstantsBuffer->setBlob(&pMaterial->mData, geometryID * materialConstantsSize, materialConstantsSize);

        mpMaterialBlock->setSrv(mMaterialResourceLocations.baseColor, geometryID, pMaterial->getBaseColorTexture() ? pMaterial->getBaseColorTexture()->getSRV() : ShaderResourceView::getNullView());
        mpMaterialBlock->setSrv(mMaterialResourceLocations.specular, geometryID, pMaterial->getSpecularTexture() ? pMaterial->getSpecularTexture()->getSRV() : ShaderResourceView::getNullView());
        mpMaterialBlock->setSrv(mMaterialResourceLocations.emissive, geometryID, pMaterial->getEmissiveTexture() ? pMaterial->getEmissiveTexture()->getSRV() : ShaderResourceView::getNullView());
        mpMaterialBlock->setSrv(mMaterialResourceLocations.normalMap, geometryID, pMaterial->getNormalMap() ? pMaterial->getNormalMap()->getSRV() : ShaderResourceView::getNullView());
        mpMaterialBlock->setSrv(mMaterialResourceLocations.occlusionMap, geometryID, pMaterial->getOcclusionMap() ? pMaterial->getOcclusionMap()->getSRV() : ShaderResourceView::getNullView());
        mpMaterialBlock->setSrv(mMaterialResourceLocations.lightMap, geometryID, pMaterial->getLightMap() ? pMaterial->getLightMap()->getSRV() : ShaderResourceView::getNullView());
        mpMaterialBlock->setSrv(mMaterialResourceLocations.heightMap, geometryID, pMaterial->getHeightMap() ? pMaterial->getHeightMap()->getSRV() : ShaderResourceView::getNullView());
        mpMaterialBlock->setSampler(mMaterialResourceLocations.samplerState, geometryID, pMaterial->getSampler());

        pRtVars->getGlobalVars()->setParameterBlock("gRtMaterials", mpMaterialBlock);
    }

    bool RtSceneRenderer::setPerModelData(const CurrentWorkingData& currentData)
    {
        // Note: Skinning not supported
        return true;
    }

    bool RtSceneRenderer::setPerMeshInstanceData(const CurrentWorkingData& currentData, const Scene::ModelInstance* pModelInstance, const Model::MeshInstance* pMeshInstance, uint32_t instanceId)
    {
        const Mesh* pMesh = pMeshInstance->getObject().get();
        assert(!pMesh->hasBones());

        glm::mat4 worldMat = pModelInstance->getTransformMatrix() * pMeshInstance->getTransformMatrix();
        glm::mat4 prevWorldMat = pModelInstance->getPrevTransformMatrix() * pMeshInstance->getTransformMatrix();
        glm::mat3x4 worldInvTransposeMat = transpose(inverse(glm::mat3(worldMat)));

        // Populate shader records
        auto pCB = currentData.pVars->getDefaultBlock()->getConstantBuffer("InternalShaderRecord");
        pCB["gWorldMatLocal"] = worldMat;
        pCB["gPrevWorldMatLocal"] = prevWorldMat;
        pCB["gWorldInvTransposeMatLocal"] = worldInvTransposeMat;
        pCB["gGeometryID"] = instanceId;

        return true;
    }
}
