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

import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import Utils.Math.DiffMathHelpers;
import Utils.Math.MatrixUtils;
import Utils.Debug.PixelDebug;
import DiffRendering.SharedTypes;
import DiffRendering.DiffSceneIO;
import DiffRendering.DiffSceneQuery;
import StaticParams;
import Params;

[PreferRecompute]
float computeBoundaryTerm(float3 normalW, float3 direction)
{
    float dotND = dot(normalW, direction);
    return dotND * dotND;
}

[Differentiable]
[PreferRecompute]
[ForwardDerivative(fwd_computeHarmonicWeight)]
float computeHarmonicWeight(
    no_diff IntersectionAD isect,
    no_diff float3 origin,
    no_diff float3 auxDirection,
    no_diff float auxSampleY,
    no_diff float kappa,
    float3 direction
)
{
    float boundaryTerm = no_diff computeBoundaryTerm(isect.normalW, auxDirection);

    float sy = max(1.f - auxSampleY, 1e-6f);
    float invVMFDensity = 1.f / ((1.f - sy) * exp(-2.f * kappa) + sy);

    float wDenom = invVMFDensity - 1.f + boundaryTerm;
    float wDenomRcp = select(wDenom > 1e-4f, 1.f / wDenom, 0.f);
    float w = wDenomRcp * wDenomRcp * wDenomRcp * invVMFDensity;
    return w;
}

[Differentiable]
[PreferRecompute]
DifferentialPair<float> fwd_computeHarmonicWeight(
    IntersectionAD isect,
    float3 origin,
    float3 auxDirection,
    float auxSampleY,
    float kappa,
    DifferentialPair<float3> direction
)
{
    float boundaryTerm = no_diff computeBoundaryTerm(isect.normalW, auxDirection);

    float sy = max(1.f - auxSampleY, 1e-6f);
    float invVMFDensity = 1.f / ((1.f - sy) * exp(-2.f * kappa) + sy);

    float wDenom = invVMFDensity - 1.f + boundaryTerm;
    float wDenomRcp = select(wDenom > 1e-4f, 1.f / wDenom, 0.f);
    float w = wDenomRcp * wDenomRcp * wDenomRcp * invVMFDensity;

    float tmp1 = invVMFDensity * w * wDenomRcp * kappa * 3;
    // Project `auxDirection` onto the plane of `direction`.
    float3 tmp2 = -cross(direction.p, cross(direction.p, direction.d));
    float dW = tmp1 * dot(tmp2, auxDirection);

    return DifferentialPair<float>(w, clamp(dW, -1e10f, 1e10f));
}

// Create a wrapper for reflect(). This isn't marked with [Differentiable].
// (For some reason, we're ignoring detach when differentiating things.)
[PreferRecompute]
float3 _reflect(float3 dir, float3 normal)
{
    return reflect(dir, normal);
}

// Compute the warped-area sampled intersection for a given input ray.
[Differentiable]
[PreferRecompute]
__generic<let N : int> bool traceAsymptoticWeightedMeanIntersection(
    SceneQueryAD sceneQuery,
    no_diff float kappa,
    float3 origin,
    no_diff float3 baseDirection,
    float3 direction[N],
    out float3 warpedDirection[N],
    inout SampleGenerator sg
)
{
    float totalWeight[N];
    float3 totalWarpedDirection[N];

    // Initialize.
    [ForceUnroll]
    for (int j = 0; j < N; j++)
    {
        totalWeight[j] = 0.f;
        totalWarpedDirection[j] = float3(0.f);
    }

    float rnd[AUX_SAMPLE_COUNT];
    for (uint i = 0; i < kAuxSampleCount; i++)
        rnd[i] = no_diff sampleNext1D(sg);

    // Create a local frame based on `baseDirection`.
    float3 t = no_diff perp_stark(baseDirection);
    ShadingFrame baseDirFrame = ShadingFrame(baseDirection, float4(t, 1.f));

    // Loop over the number of auxiliary samples.
    [MaxIters(AUX_SAMPLE_COUNT)]
    for (uint i = 0; i < kAuxSampleCount; i++)
    {
        float2 sample;
        if (kUseAntitheticSampling)
        {
            uint antitheticIdx = i / 2;
            sample = float2(rnd[antitheticIdx * 2], rnd[antitheticIdx * 2 + 1]);
        }
        else
        {
            sample = no_diff sampleNext2D(sg);
        }

        float3 dirLocal = no_diff sampleVonMisesFisher(sample, kappa);
        float3 auxDirection = detach(normalize(baseDirFrame.fromLocal(dirLocal)));
        if (kUseAntitheticSampling && i % 2 == 1)
        {
            // Every other sample, reflect `auxDirection` about `baseDirection`.
            auxDirection = no_diff _reflect(-auxDirection, baseDirection);
        }

        // Build a new RayAD with `auxDirection`.
        RayAD auxRay = RayAD(origin, auxDirection, sceneQuery.gradInfo.pixel);
        IntersectionAD auxIsect = IntersectionAD();

        float3 dirW = auxDirection;
        // Trace the ray.
        if (sceneQuery.traceRayInlineAD(auxRay, auxIsect, SceneQueryAD.DiffIntersectionMode.AttachToGeometry))
        {
            dirW = normalizeSafe(auxIsect.posW - origin);
        }

        [ForceUnroll]
        for (int j = 0; j < N; j++)
        {
            float weight = computeHarmonicWeight(auxIsect, origin, auxDirection, sample.y, kappa, direction[j]);
            totalWeight[j] += weight;
            totalWarpedDirection[j] += float3(weight) * dirW;
        }
    }

    [ForceUnroll]
    for (int j = 0; j < N; j++)
    {
        // Clamp the total weight to avoid NaNs.
        totalWeight[j] = max(totalWeight[j], 1e-8f);

        // Harmonic weighted mean.
        warpedDirection[j] = totalWarpedDirection[j] / float3(totalWeight[j]);
    }
    return totalWeight[0] > 0.f;
}

// Compute the warped-area reparameterization for (secondary) rays.

[Differentiable]
[PreferRecompute]
void computeWarpedRay(
    SceneQueryAD sceneQuery,
    no_diff float kappa,
    float3 origin,
    float3 baseDirection,
    float3 direction[3],
    out float3 warpedDirection[3],
    inout SampleGenerator sg
)
{
    float3 meanDirection[3];
    if (traceAsymptoticWeightedMeanIntersection<3>(sceneQuery, kappa, origin, baseDirection, direction, meanDirection, sg))
    {
        [ForceUnroll]
        for (int j = 0; j < 3; j++)
            warpedDirection[j] = normalizeSafe(meanDirection[j]);
    }
    else
    {
        [ForceUnroll]
        for (int j = 0; j < 3; j++)
            warpedDirection[j] = direction[j];
    }
}

[Differentiable]
[PreferRecompute]
float reparameterizeRay(SceneQueryAD sceneQuery, inout RayAD ray, inout SampleGenerator sg)
{
    DifferentialPair<float3[3]> dpDirections = DifferentialPair<float3[3]>(
        { ray.direction, ray.direction, ray.direction }, { float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f), float3(0.f, 0.f, 1.f) }
    );

    // Zero out any gradients from/to the scene for this Jacobian computation.
    SceneQueryAD.Differential dQuery = {};
    dQuery.gradInfo = { SceneGradientFlag.make(GradientMode.None) };

    // Use the same SampleGenerator for correlated samples.
    SampleGenerator sgCopy = sg;

    // Compute warped directions and differentials.
    DifferentialPair<float3[3]> dpWarpedDirections;
    float kappa = pow(10.f, kLog10vMFConcentration);
    fwd_diff(computeWarpedRay
    )(diffPair(sceneQuery, dQuery),
      kappa,
      diffPair(ray.origin, float3(0.f)),
      diffPair(ray.direction, float3(0.f)),
      dpDirections,
      dpWarpedDirections,
      sgCopy);

    float3 warpDir = infinitesimal<3>(dpWarpedDirections.p[0]); // Can use any of dpWarpedDirections.p[j].
    float3 warpDiffXDir = infinitesimal<3>(dpWarpedDirections.d[0]);
    float3 warpDiffYDir = infinitesimal<3>(dpWarpedDirections.d[1]);
    float3 warpDiffZDir = infinitesimal<3>(dpWarpedDirections.d[2]);

    // Compute Jacobian determinant.
    float3x3 jacobian = no_diff float3x3::identity() + float3x3(warpDiffXDir, warpDiffYDir, warpDiffZDir);
    float detJ = determinant(jacobian);

    // Update the ray.
    ray.direction += warpDir;
    return detJ;
}

// Compute the warped-area reparameterization for (primary) screen samples.

[Differentiable]
[PreferRecompute]
void computeWarpedPrimarySample(
    SceneQueryAD sceneQuery,
    no_diff float kappa,
    uint2 frameDim,
    float2 basePosS,
    float2 posS[2],
    out float2 warpedPosS[2],
    inout SampleGenerator sg
)
{
    float3 primaryRayOrigin = sceneQuery.loadCameraPosition();
    float3 baseRayDir = sceneQuery.computeCameraRayDirection(basePosS, frameDim);
    float3 rayDir[2] = { sceneQuery.computeCameraRayDirection(posS[0], frameDim), sceneQuery.computeCameraRayDirection(posS[1], frameDim) };

    float3 meanDirection[2];
    if (traceAsymptoticWeightedMeanIntersection<2>(sceneQuery, kappa, primaryRayOrigin, baseRayDir, rayDir, meanDirection, sg))
    {
        // Project the warped direction onto screen space.
        [ForceUnroll]
        for (int j = 0; j < 2; j++)
        {
            float3 warpedDir = normalizeSafe(meanDirection[j]);
            warpedPosS[j] = sceneQuery.computeCameraRayScreenPos(warpedDir, frameDim);
        }
    }
    else
    {
        // If the ray is missed, just return the original sample.
        [ForceUnroll]
        for (int j = 0; j < 2; j++)
            warpedPosS[j] = posS[j];
    }
}

[Differentiable]
[PreferRecompute]
float reparameterizeScreenSample(SceneQueryAD sceneQuery, uint2 frameDim, SampleGenerator sg, inout float2 posS, )
{
    // Forward-mode gradients w.r.t. screen-space position.
    DifferentialPair<float2[2]> dpPosS = DifferentialPair<float2[2]>( { posS, posS }, { float2(1.f, 0.f), float2(0.f, 1.f) });

    // Zero out any gradients from/to the scene for this Jacobian computation.
    SceneQueryAD.Differential dQuery = {};
    dQuery.gradInfo = { SceneGradientFlag.make(GradientMode.None) };

    // Use the same SampleGenerator for correlated samples.
    SampleGenerator sgCopy = sg;

    DifferentialPair<float2[2]> dpWarpedPosS;
    float kappa = pow(10.f, kLog10vMFConcentrationScreen);
    fwd_diff(computeWarpedPrimarySample
    )(diffPair(sceneQuery, dQuery), kappa, frameDim, diffPair(posS, float2(0.f)), dpPosS, dpWarpedPosS, sgCopy);

    float2 warpPosS = infinitesimal<2>(dpWarpedPosS.p[0]); // Can use either dpWarpedPosS.p[0] or dpWarpedPosS.p[1].
    float2 warpDiffXPosS = infinitesimal<2>(dpWarpedPosS.d[0]);
    float2 warpDiffYPosS = infinitesimal<2>(dpWarpedPosS.d[1]);

    // Compute Jacobian determinant.
    float2x2 jacobian = no_diff float2x2::identity() + float2x2(warpDiffXPosS, warpDiffYPosS);
    float detJ = determinant(jacobian);

    posS += warpPosS;
    return detJ;
}
