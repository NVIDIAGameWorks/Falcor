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
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Utils.Sampling.UniformSampleGenerator;
import DiffRendering.SceneGradients;
import DiffRendering.GradientIOWrapper;

RWStructuredBuffer<float> materialGrad;
RWStructuredBuffer<float> geometryGrad;

cbuffer CB
{
    float3 gWi;
    float3 gWo;
}

[Differentiable]
IMaterialInstance getDiffMaterialInstance(out DiffMaterialData diffData, const ShadingData sd)
{
    let lod = ExplicitLodTextureSampler(0.f);
    let material = gScene.materials.getMaterial(sd.materialID);
    let mi = material.setupDiffMaterialInstance(diffData, gScene.materials, sd, lod);
    return mi;
}

[Differentiable]
float3 evalAD_PBRTDiffuse(uint2 pixelID)
{
    // Setup shading data struct.
    ShadingData sd = {};
    sd.frame = ShadingFrame::createIdentity();
    sd.faceN = detach(sd.frame.N);
    sd.frontFacing = true;

    uint materialID = 0;
    sd.mtl = gScene.materials.getMaterialHeader(materialID);
    sd.materialID = materialID;
    sd.IoR = 1.f;

    sd.V = gWi;

    sd.materialGradOffset = 0;
    sd.geometryGradOffset = 0;
    sd.threadID = 0;

    // Create MaterialInstance.
    DiffMaterialData diffData = DiffMaterialData();
    IMaterialInstance mi = getDiffMaterialInstance(diffData, sd);

    UniformSampleGenerator sg = UniformSampleGenerator(pixelID, 0);

    // Setup differentiable wo.
    uint hashIndex = hashFunction(sd.threadID, gSceneGradients.getHashSize(GradientType::Geometry));
    GradientIOWrapper gradIO = GradientIOWrapper(GradientType::Geometry, sd.geometryGradOffset, hashIndex);
    float3 wo = gradIO.getFloat(gWo, 0);

    float3 value = mi.evalAD(diffData, sd, wo, sg);
    return value;
}

[numthreads(1, 1, 1)]
void testDiffPBRTDiffuse(uint3 threadID: SV_DispatchThreadID)
{
    bwd_diff(evalAD_PBRTDiffuse)(threadID.xy, float3(1.f));

    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
    {
        materialGrad[i] = gSceneGradients.getGrad(GradientType::Material, i, 0);
        geometryGrad[i] = gSceneGradients.getGrad(GradientType::Geometry, i, 0);
    }
}
