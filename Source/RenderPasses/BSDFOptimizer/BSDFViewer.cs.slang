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

struct BSDFViewer
{
    BSDFOptimizerParams params;
    RWTexture2D<float4> output;

    struct SurfaceData
    {
        ShadingData sd;
        BSDFProperties bsdfProperties;
        float3 wo;
    };

    SurfaceData prepareShadingData(const VertexData v, const float3 viewDir, const uint materialID, const ITextureSampler lod)
    {
        SurfaceData data = {};

        // Setup Falcor's ShadingData based on the selected scene material and lobes.
        data.sd = gScene.materials.prepareShadingData(v, materialID, viewDir, lod);

        return data;
    }

    float3 evalBSDFSlice(const float2 uv, const uint materialID, const ITextureSampler lod, inout SampleGenerator sg)
    {
        // Calculate geometry and incident/outgoing directions.
        VertexData v;
        float3 viewDir;
        float3 lightDir = calculateSliceGeometry(uv, v, viewDir);

        // Setup shading data.
        SurfaceData data = prepareShadingData(v, viewDir, materialID, lod);
        data.wo = lightDir;

        // Create BSDF instance.
        uint hints = 0;
        let mi = gScene.materials.getMaterialInstance(data.sd, lod, hints);
        data.bsdfProperties = mi.getProperties(data.sd);

        // Evaluate BSDF at shading point.
        float3 f = mi.eval(data.sd, data.wo, sg);

        // Remove cosine term.
        float NdotL = abs(dot(data.sd.frame.N, data.wo));
        f = NdotL > 0.f ? f / NdotL : float3(0);

        return f;
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;

        uint viewportID = 0;
        float2 viewportOffset = params.initViewPortOffset;

        if (pixel.x >= params.refViewPortOffset.x)
        {
            viewportID = 2;
            viewportOffset = params.refViewPortOffset;
        }
        else if (pixel.x >= params.diffViewPortOffset.x)
        {
            viewportID = 1;
            viewportOffset = params.diffViewPortOffset;
        }

        float2 uv = getViewportCoord(pixel, viewportOffset, params.viewPortScale);
        if (any(uv < 0.f || uv >= 1.f))
        {
            output[pixel] = float4(0.f, 0.f, 0.f, 1.f);
            return;
        }

        let lod = ExplicitLodTextureSampler(0.f);
        SampleGenerator sg = SampleGenerator(pixel, params.frameCount);

        float3 fInitVal = evalBSDFSlice(uv, params.initMaterialID, lod, sg);
        float3 fRefVal = evalBSDFSlice(uv, params.refMaterialID, lod, sg);
        float3 fDiffVal = abs(fInitVal - fRefVal);

        if (viewportID == 0)
            output[pixel] = float4(fInitVal, 1.f);
        else if (viewportID == 1)
            output[pixel] = float4(fDiffVal, 1.f);
        else
            output[pixel] = float4(fRefVal, 1.f);
    }
};

ParameterBlock<BSDFViewer> gBSDFViewer;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    gBSDFViewer.execute(dispatchThreadID.xy);
}
