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
#include "Rendering/Materials/RGLAcquisition.h"

namespace Falcor
{
    GPU_TEST(RGLAcquisition)
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

        // Create acquisition class.
        auto pAcquisition = RGLAcquisition::create(ctx.getRenderContext(), pScene);

        // Acquire BSDF.
        pAcquisition->acquireIsotropic(ctx.getRenderContext(), materialID);

        auto file = pAcquisition->toRGLFile();

        // There is no good way to test correctness of the output data, but
        // if we get here, we at least know the acquisition can complete without
        // crashing.
    }
}
