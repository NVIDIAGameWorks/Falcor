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
#include "FeatureDemoSceneRenderer.h"

static bool isMaterialTransparent(const Material* pMaterial)
{
    if (pMaterial->getAlphaMap()) return true;

    for (uint32_t i = 0; i < pMaterial->getNumLayers(); i++)
    {
        const auto& layer = pMaterial->getLayer(i);
        if (layer.type == Material::Layer::Type::Lambert && layer.albedo.a < 1.0f) return true;
    }
    return false;
}

FeatureDemoSceneRenderer::FeatureDemoSceneRenderer(const Scene::SharedPtr& pScene) : SceneRenderer(pScene)
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
}

FeatureDemoSceneRenderer::SharedPtr FeatureDemoSceneRenderer::create(const Scene::SharedPtr& pScene)
{
    return SharedPtr(new FeatureDemoSceneRenderer(pScene));
}
bool FeatureDemoSceneRenderer::setPerMeshData(const CurrentWorkingData& currentData, const Mesh* pMesh)
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

void FeatureDemoSceneRenderer::renderScene(RenderContext* pContext)
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