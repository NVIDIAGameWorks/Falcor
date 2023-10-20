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
import Scene.Material.ShadingUtils;
import Utils.Sampling.SampleGenerator;
import Utils.Math.BitTricks;
import Utils.Math.MathHelpers;
import BSDFOptimizerParams;
import BSDFOptimizerHelpers;

cbuffer CB
{
    BSDFOptimizerParams params;
};

struct SurfaceData : IDifferentiable
{
    ShadingData sd;
    float3 wo;
};

[Differentiable]
SurfaceData prepareShadingData(const VertexData v, const float3 viewDir, const uint materialID, const ITextureSampler lod)
{
    SurfaceData data = {};

    // Setup Falcor's ShadingData based on the selected scene material and lobes.
    data.sd = no_diff gScene.materials.prepareShadingData(v, materialID, viewDir, lod);
    return data;
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
float3 evalBSDFSlice(bool isRef, const uint2 pixel)
{
    float2 uv = no_diff getViewportCoord(pixel, float2(0.f), params.viewPortScale);
    let lod = ExplicitLodTextureSampler(0.f);
    SampleGenerator sg = SampleGenerator(pixel, params.frameCount);

    // Calculate geometry and incident/outgoing directions.
    VertexData v;
    float3 viewDir;
    float3 lightDir = no_diff calculateSliceGeometry(uv, v, viewDir);

    // Setup shading data.
    SurfaceData data = prepareShadingData(v, viewDir, isRef ? params.refMaterialID : params.initMaterialID, lod);
    data.wo = detach(lightDir);

    // Set offset for writing gradients.
    data.sd.materialGradOffset = 0;
    data.sd.threadID = 0;

    // Create differentiable BSDF instance and evaluate BSDF at shading point.
    DiffMaterialData diffData = DiffMaterialData();
    let mi = getDiffMaterialInstance(diffData, data.sd);
    float3 f = isRef ? no_diff mi.eval(data.sd, data.wo, sg) : mi.evalAD(diffData, data.sd, data.wo, sg);

    // Remove cosine term.
    float NdotL = abs(dot(data.sd.frame.N, data.wo));
    f = NdotL > 0.f ? f / NdotL : float3(0);

    return f;
}

[Differentiable]
float3 computeLoss(const uint2 pixel)
{
    float3 fRef = evalBSDFSlice(true, pixel);
    float3 fCur = evalBSDFSlice(false, pixel);

    float3 diff = fCur - fRef;
    return 0.5 * diff * diff;
}

void execute(const uint2 pixel)
{
    if (any(pixel >= params.bsdfTableDim))
        return;
    bwd_diff(computeLoss)(pixel, float3(1.f));
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    execute(dispatchThreadID.xy);
}
