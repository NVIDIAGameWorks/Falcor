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
#include "ForwardRendererSceneRenderer.h"

static bool isMaterialTransparent(const Material* pMaterial)
{
    return pMaterial->getBaseColor().a < 1.0f;
}

ForwardRendererSceneRenderer::ForwardRendererSceneRenderer(const Scene::SharedPtr& pScene) : SceneRenderer(pScene)
{
    for (uint32_t model = 0; model < mpScene->getModelCount(); model++)
    {
        const auto& pModel = mpScene->getModel(model);
        for (uint32_t mesh = 0; mesh < pModel->getMeshCount(); mesh++)
        {
            const auto& pMesh = pModel->getMesh(mesh);
            uint32_t id = pMesh->getId();
            if (mTransparentMeshes.size() <= id) mTransparentMeshes.resize(id + 1);
            bool transparent = isMaterialTransparent(pMesh->getMaterial().get());
            mHasOpaqueObjects = mHasOpaqueObjects || (transparent == false);
            mHasTransparentObject = mHasTransparentObject || transparent;
            mTransparentMeshes[id] = transparent;
        }
    }

    RasterizerState::Desc rsDesc;
    mpDefaultRS = RasterizerState::create(rsDesc);
    rsDesc.setCullMode(RasterizerState::CullMode::None);
    mpNoCullRS = RasterizerState::create(rsDesc);
}

ForwardRendererSceneRenderer::SharedPtr ForwardRendererSceneRenderer::create(const Scene::SharedPtr& pScene)
{
    return SharedPtr(new ForwardRendererSceneRenderer(pScene));
}

bool ForwardRendererSceneRenderer::setPerMeshData(const CurrentWorkingData& currentData, const Mesh* pMesh)
{
    switch (mRenderMode)
    {
    case Mode::All:
        return true;
    case Mode::Opaque:
        return mTransparentMeshes[pMesh->getId()] == false;
    case Mode::Transparent:
        return mTransparentMeshes[pMesh->getId()];
    default:
        should_not_get_here();
        return false;
    }
}

void ForwardRendererSceneRenderer::renderScene(RenderContext* pContext)
{
    switch (mRenderMode)
    {
    case Mode::Opaque:
        if (mHasOpaqueObjects == false) return;
        break;
    case Mode::Transparent:
        if (mHasTransparentObject == false) return;
    }
    SceneRenderer::renderScene(pContext);
}

RasterizerState::SharedPtr ForwardRendererSceneRenderer::getRasterizerState(const Material* pMaterial)
{
    if (pMaterial->getAlphaMode() == AlphaModeMask)
    {
        return mpNoCullRS;
    }
    else
    {
        return mpDefaultRS;
    }
}

bool ForwardRendererSceneRenderer::setPerMaterialData(const CurrentWorkingData& currentData, const Material* pMaterial)
{
    const auto& pRsState = getRasterizerState(currentData.pMaterial);
    if (pRsState != mpLastSetRs)
    {
        currentData.pContext->getGraphicsState()->setRasterizerState(pRsState);
        mpLastSetRs = pRsState;
    }

    return SceneRenderer::setPerMaterialData(currentData, pMaterial);
}