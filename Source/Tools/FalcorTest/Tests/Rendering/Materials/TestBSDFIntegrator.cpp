/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#include "Testing/UnitTest.h"
#include "Scene/Scene.h"
#include "Scene/Material/StandardMaterial.h"
#include "Rendering/Materials/BSDFIntegrator.h"

namespace Falcor
{
    namespace
    {
        const float3 kExpectedResults[] =
        {
            { 0.271488f, 0.666471f, 0.745583f },
            { 0.230911f, 0.580707f, 0.650769f },
            { 0.220602f, 0.562734f, 0.631260f },
            { 0.218110f, 0.560894f, 0.629551f },
        };

        const float kMaxL2 = 1e-6f;
    }

    GPU_TEST(BSDFIntegrator)
    {
        // Create material.
        StandardMaterial::SharedPtr pMaterial = StandardMaterial::create("testMaterial");
        pMaterial->setBaseColor(float4(0.3f, 0.8f, 0.9f, 1.f));
        pMaterial->setMetallic(0.f);
        pMaterial->setRoughness(1.f);
        pMaterial->setSpecularTransmission(0.f);

        // Create and update scene containing the material.
        Scene::SceneData sceneData;
        sceneData.pMaterials = MaterialSystem::create();
        MaterialID materialID = sceneData.pMaterials->addMaterial(pMaterial);

        Scene::SharedPtr pScene = Scene::create(std::move(sceneData));
        auto updateFlags = pScene->update(ctx.getRenderContext(), 0.0);

        // Create BSDF integrator utility.
        auto pIntegrator = BSDFIntegrator::create(ctx.getRenderContext(), pScene);

        // Integrate BSDF.
        std::vector<float> cosThetas = { 0.25f, 0.5f, 0.75f, 1.f };
        auto results = pIntegrator->integrateIsotropic(ctx.getRenderContext(), materialID, cosThetas);

        // Validate results.
        for (size_t i = 0; i < cosThetas.size(); i++)
        {
            float3 e = results[i] - kExpectedResults[i];
            float l2 = std::sqrt(glm::dot(e, e));
            EXPECT_LE(l2, kMaxL2) << " result=" << to_string(results[i]) << " expected=" << to_string(kExpectedResults[i]) << " cosTheta=" << cosThetas[i];
        }
    }
}
