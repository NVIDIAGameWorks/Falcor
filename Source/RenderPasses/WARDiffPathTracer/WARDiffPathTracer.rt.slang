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

import Rendering.Lights.EmissiveLightSampler;
import Scene.Raytracing;
import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;
import DiffRendering.SharedTypes;
import DiffRendering.DiffSceneIO;
import DiffRendering.DiffSceneQuery;
import DiffRendering.DiffDebug;

import PTUtils;
import WarpedAreaReparam;

static const float kPixelGaussianSigma = 0.5f;

struct DiffPTData
{
    WARDiffPathTracerParams params;
    EmissiveLightSampler emissiveSampler;
};

ParameterBlock<DiffPTData> gDiffPTData;

StructuredBuffer<float3> dLdI;

// Outputs
RWTexture2D<float4> gOutputColor;

// Convert square samples to Gaussian samples
float2 sampleGaussian(float2 xy, float sigma)
{
    return (sqrt(-2.f * log(xy.x)) * float2(cos(M_2PI * xy.y), sin(M_2PI * xy.y))) * sigma;
}

// Evaluate Gaussian at xy
[Differentiable]
[PreferRecompute]
float evalGaussian(float2 xy, no_diff float sigma)
{
    const float alpha = (-1.f / (2.f * sigma * sigma));
    return exp(alpha * dot(xy, xy)) * (1.f / (2.f * M_PI * sigma * sigma));
}

[Differentiable]
float3 tracePath(SceneQueryAD sceneQuery, float2 pixel, no_diff float2 pixelCenter, inout SampleGenerator sg)
{
    PathData pathData = PathData();
    IntersectionAD isect = IntersectionAD();

    // Warped-area reparameterization for primary rays.
    if (kUseWAR)
    {
        float weight = reparameterizeScreenSample(sceneQuery, gDiffPTData.params.frameDim, sg, pixel);
        pathData.thp *= weight;
    }

    // Use Gaussian pixel filter.
    float pixelWeight = evalGaussian(pixel - pixelCenter, kPixelGaussianSigma);
    pathData.thp *= pixelWeight / detach(pixelWeight);

    float3 primaryRayOrigin = sceneQuery.loadCameraPosition();
    float3 primaryRayDirection = sceneQuery.computeCameraRayDirection(pixel, gDiffPTData.params.frameDim);
    RayAD ray = RayAD(detach(primaryRayOrigin), primaryRayDirection, sceneQuery.gradInfo.pixel);

    if (!sceneQuery.traceRayInlineAD(ray, isect, SceneQueryAD.DiffIntersectionMode.AttachToRay))
        return pathData.radiance;

    [MaxIters(MAX_BOUNCES + 2)]
    while (!pathData.terminated)
    {
        handleHit(sceneQuery, isect, gDiffPTData.emissiveSampler, pathData, ray, sg, kUseWAR);

        if (pathData.terminated)
            break;

        if (!sceneQuery.traceRayInlineAD(ray, isect, SceneQueryAD.DiffIntersectionMode.AttachToRay))
            break;
    }

    return pathData.radiance;
}

[Differentiable]
float3 tracePaths(SceneQueryAD sceneQuery, uint2 pixel)
{
    float3 result = float3(0.f);

    var params = gDiffPTData.params;
    uint frameSeed = bool(params.useFixedSeed) ? params.fixedSeed : params.frameCount;

    // Antithetic sampling for differentiable rendering (interior term).
    SampleGenerator sg = SampleGenerator(pixel, frameSeed / 2);

    // Throw away some samples.
    sampleNext2D(sg);
    sampleNext2D(sg);
    sampleNext2D(sg);

    // Use Gaussian pixel filter.
    float2 pixelf = pixel + float2(0.5f, 0.5f);

    float2 uniform2D = no_diff sampleNext2D(sg);
    float2 jitter2D = no_diff sampleGaussian(uniform2D, kPixelGaussianSigma);

    // Antithetic sampling for differentiable rendering (interior term).
    if (kUseAntitheticSampling && frameSeed % 2 == 0)
        jitter2D = -jitter2D;

    // Trace an interior path.
    result += tracePath(sceneQuery, pixelf + jitter2D, pixelf, sg);

    return result;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    printSetPixel(pixel);

    DiffSceneIO diffSceneIO;
    SceneQueryAD sceneQuery = SceneQueryAD(diffSceneIO, SceneGradientInfo(SceneGradientFlag.make(GradientMode.Scene), pixel));

    if (kDiffMode == DiffMode::Primal)
    {
        // Primal rendering.
        float3 color = tracePaths(sceneQuery, pixel);
        gOutputColor[pixel] = float4(color, 1.f);
    }
    else if (kDiffMode == DiffMode::BackwardDiff)
    {
        var params = gDiffPTData.params;
        float3 color = float3(0.f);
        if (params.runBackward == 0)
        {
            color = tracePaths(sceneQuery, pixel);
            gOutputColor[pixel] = float4(color, 1.f);
        }
        else
        {
            // Set differential query mode to `Scene` for visualizing gradient images.
            SceneQueryAD.Differential dQuery;
            dQuery.gradInfo = { SceneGradientFlag.make(GradientMode.Scene) };

            DifferentialPair<SceneQueryAD> dpQuery = diffPair(sceneQuery, dQuery);

            uint pixelIdx = pixel.y * params.frameDim.x + pixel.x;
            bwd_diff(tracePaths)(dpQuery, pixel, dLdI[pixelIdx]);
        }
    }
    else if (kDiffMode == DiffMode::ForwardDiffDebug)
    {
        // Forward-mode differentiable rendering for visualizing gradients.

        // Set differential query mode to `ForwardDebug` for visualizing gradient images.
        SceneQueryAD.Differential dQuery;
        dQuery.gradInfo = { SceneGradientFlag.make(GradientMode.ForwardDebug) };

        DifferentialPair<float3> dpColor = fwd_diff(tracePaths)(diffPair(sceneQuery, dQuery), pixel);
        gOutputColor[pixel] = float4(dpColor.p, 1.f);

        // Clamp gradient values.
        float maxValue = gDiffPTData.params.assertThreshold;
        bool resIsFinite = (dpColor.d[0] >= -maxValue && dpColor.d[0] <= maxValue) &&
                           (dpColor.d[1] >= -maxValue && dpColor.d[1] <= maxValue) &&
                           (dpColor.d[2] >= -maxValue && dpColor.d[2] <= maxValue);
        if (resIsFinite)
            gOutputDColor[pixel] = float4(dpColor.d[0], 0.f, 0.f, 1.f); // Only show the red channel.
    }
    else if (kDiffMode == DiffMode::BackwardDiffDebug)
    {
        // Reverse-mode differentiable rendering for visualizing gradients.

        // Set differential query mode to `Scene` for visualizing gradient images.
        SceneQueryAD.Differential dQuery;
        dQuery.gradInfo = { SceneGradientFlag.make(GradientMode.Scene) };

        DifferentialPair<SceneQueryAD> dpQuery = diffPair(sceneQuery, dQuery);
        bwd_diff(tracePaths)(dpQuery, pixel, float3(1.f, 0.f, 0.f));
    }
}
