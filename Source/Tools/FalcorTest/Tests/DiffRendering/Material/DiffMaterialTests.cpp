/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Scene/Material/PBRT/PBRTDiffuseMaterial.h"
#include "DiffRendering/SceneGradients.h"
#include <iostream>

namespace Falcor
{
namespace
{
const char kShaderFile[] = "Tests/DiffRendering/Material/DiffMaterialTests.cs.slang";

struct BsdfConfig
{
    float3 wi;
    float3 wo;
    float4 baseColor;
};

void testDiffPBRTDiffuse(GPUUnitTestContext& ctx, const BsdfConfig& bsdfConfig)
{
    // Create material.
    ref<PBRTDiffuseMaterial> pMaterial = PBRTDiffuseMaterial::create(ctx.getDevice(), "PBRTDiffuse");
    pMaterial->setBaseColor(bsdfConfig.baseColor);

    // Create and update scene containing the material.
    Scene::SceneData sceneData;
    sceneData.pMaterials = std::make_unique<MaterialSystem>(ctx.getDevice());
    MaterialID materialID = sceneData.pMaterials->addMaterial(pMaterial);

    ref<Scene> pScene = Scene::create(ctx.getDevice(), std::move(sceneData));
    auto updateFlags = pScene->update(ctx.getRenderContext(), 0.0);

    // Create scene gradients.
    const uint32_t gradDim = 3;
    std::unique_ptr<SceneGradients> pSceneGradients = std::make_unique<SceneGradients>(ctx.getDevice(), uint2(gradDim), uint2(1));

    // Create program.
    ProgramDesc desc;
    desc.addShaderModules(pScene->getShaderModules());
    desc.addShaderLibrary(kShaderFile);
    desc.addTypeConformances(pScene->getTypeConformances());
    desc.csEntry("testDiffPBRTDiffuse");
    ctx.createProgram(desc, pScene->getSceneDefines());

    pScene->bindShaderData(ctx["gScene"]);
    pSceneGradients->bindShaderData(ctx["gSceneGradients"]);
    ctx["CB"]["gWi"] = bsdfConfig.wi;
    ctx["CB"]["gWo"] = bsdfConfig.wo;
    ctx.allocateStructuredBuffer("materialGrad", 3);
    ctx.allocateStructuredBuffer("geometryGrad", 3);
    ctx.runProgram(1, 1, 1);

    // Material gradient w.r.t. the diffuse albedo.
    std::vector<float> materialGrad = ctx.readBuffer<float>("materialGrad");
    const float kExpectedMaterialGrad[] = {0.3003115f, 0.3003115f, 0.3003115f};

    // Geometry gradient w.r.t. the outgoing direction wo.
    std::vector<float> geometryGrad = ctx.readBuffer<float>("geometryGrad");
    const float kExpectedGeometryGrad[] = {0.f, 0.f, 0.5411268f};

    for (uint32_t i = 0; i < 3; i++)
    {
        float l1 = std::abs(materialGrad[i] - kExpectedMaterialGrad[i]);
        EXPECT_LE(l1, 1e-3f);

        l1 = std::abs(geometryGrad[i] - kExpectedGeometryGrad[i]);
        EXPECT_LE(l1, 1e-3f);
    }
}
} // namespace

// Disabled on Vulkan for now as the compiler generates invalid code for atomic add.
GPU_TEST(DiffPBRTDiffuse, Device::Type::D3D12)
{
    BsdfConfig bsdfConfig = {
        normalize(float3(0.3f, 0.2f, 0.8f)),   // wi
        normalize(float3(-0.1f, -0.3f, 0.9f)), // wo
        float4(0.9f, 0.6f, 0.2f, 1.f),         // baseColor
    };
    testDiffPBRTDiffuse(ctx, bsdfConfig);
}
} // namespace Falcor
