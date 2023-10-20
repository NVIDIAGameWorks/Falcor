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
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Scene.Raytracing;
import Scene.Intersection;
import Scene.RaytracingInline;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;
import DiffRendering.SharedTypes;
import DiffRendering.DiffSceneIO;
import DiffRendering.DiffSceneQuery;

__exported import StaticParams;
__exported import Params;
import WarpedAreaReparam;

struct PathData : IDifferentiable
{
    float3 radiance;
    float3 thp;
    uint length;
    bool terminated;

    // For MIS.
    no_diff float3 normal;
    no_diff float pdf;

    [Differentiable]
    __init()
    {
        this.radiance = float3(0.f);
        this.thp = float3(1.f);
        this.length = 0;
        this.terminated = false;
        this.normal = float3(0.f);
        this.pdf = 0.f;
    }
};

struct LightSample
{
    float3 Li;      ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float pdf;      ///< Pdf with respect to solid angle at the shading point.
    float3 origin;  ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float distance; ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3 dir;     ///< Ray direction for visibility evaluation (normalized).

    // For differentiable evaluation.
    float3 lightPos;
    float3 lightNormal;

    Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
};

float evalMIS(float n0, float p0, float n1, float p1)
{
    // Power two heuristic
    float q0 = (n0 * p0) * (n0 * p0);
    float q1 = (n1 * p1) * (n1 * p1);
    return q0 / (q0 + q1);
}

[Differentiable]
[PreferRecompute]
ShadingData loadShadingData(
    SceneQueryAD sceneQuery,
    HitInfo hit,
    IntersectionAD isect,
    const float3 rayOrigin,
    const float3 rayDir,
    const ITextureSampler lod
)
{
    VertexData v = {};
    uint materialID = {};

    // We only support triangle hits for now.
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(isect.instanceID);
    }

    ShadingData sd = no_diff gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

    // Overwrite some fields to enable auto-diff.
    sd.V = -rayDir;
    bool valid;
    sd.frame = ShadingFrame::createSafe(isect.normalW, v.tangentW, valid);

    // Set offset value for the material gradient propagation.
    sd.materialGradOffset = materialID * DiffMaterialData::kMaterialParamCount;

    return sd;
}

[Differentiable]
[PreferRecompute]
IMaterialInstance getDiffMaterialInstance(out DiffMaterialData diffData, const ShadingData sd, const ITextureSampler lod)
{
    let material = gScene.materials.getMaterial(sd.materialID);
    let mi = material.setupDiffMaterialInstance(diffData, gScene.materials, sd, lod);
    return mi;
}

[Differentiable]
[PreferRecompute]
bool generateScatterRay(
    SceneQueryAD sceneQuery,
    ShadingData sd,
    IMaterialInstance mi,
    DiffMaterialData diffData,
    IntersectionAD isect,
    inout RayAD ray,
    inout PathData pathData,
    inout SampleGenerator sg,
    bool shouldReparameterize = true
)
{
    // Sample material.
    BSDFSample bsdfSample = {};
    if (mi.sample(sd, sg, bsdfSample, kUseBSDFSampling))
    {
        ray.direction = bsdfSample.wo;
        ray.origin = computeRayOrigin(isect.posW, ((sd.frontFacing) ? sd.faceN : -sd.faceN));

        if (shouldReparameterize)
        {
            float weight = reparameterizeRay(sceneQuery, ray, sg);
            pathData.thp *= weight;
        }

        float3 weight = mi.evalAD(diffData, sd, ray.direction, sg) / bsdfSample.pdf;
        pathData.thp *= weight;

        // Save normal and pdf for MIS.
        pathData.normal = sd.getOrientedFaceNormal();
        pathData.pdf = bsdfSample.pdf;

        return any(weight > 0.f);
    }

    return false;
}

// Generate a light sample on the emissive geometry.
// Assume the area lights are static (we don't allow differentiation of dynamic area lights for now).
[PreferRecompute]
bool generateEmissiveSample(
    IntersectionAD isect,
    EmissiveLightSampler emissiveSampler,
    bool upperHemisphere,
    inout SampleGenerator sg,
    out LightSample ls
)
{
    ls = {}; // Default initialization to avoid divergence at returns.
    if (!kUseEmissiveLights)
        return false;

    TriangleLightSample tls;
    if (!emissiveSampler.sampleLight(isect.posW, isect.normalW, upperHemisphere, sg, tls))
        return false;

    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = computeRayOrigin(isect.posW, isect.normalW);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.dir = normalize(toLight);

    ls.lightPos = tls.posW;
    ls.lightNormal = tls.normalW;

    return any(ls.Li > 0.f);
}

[Differentiable]
[PreferRecompute]
void computeNEE(
    SceneQueryAD sceneQuery,
    ShadingData sd,
    IMaterialInstance mi,
    DiffMaterialData diffData,
    IntersectionAD isect,
    LightSample ls,
    inout PathData pathData,
    inout SampleGenerator sg,
    bool shouldReparameterize = true
)
{
    float3 wo = ls.lightPos - isect.posW;
    float dist = length(wo);
    wo /= dist;

    RayAD ray = RayAD(computeRayOrigin(isect.posW, ((sd.frontFacing) ? sd.faceN : -sd.faceN)), wo, sceneQuery.gradInfo.pixel);

    float3 thp = pathData.thp;

    if (shouldReparameterize)
    {
        float weight = reparameterizeRay(sceneQuery, ray, sg);
        thp *= weight;
    }

    // Note: ls.Li and ls.pdf are already w.r.t. the solid angle.
    float areaToSolidAngle = (dist * dist) / dot(ls.lightNormal, -ray.direction);
    float3 bsdfValue = mi.evalAD(diffData, sd, ray.direction, sg);
    float3 weight = ls.Li * bsdfValue * areaToSolidAngle / detach(areaToSolidAngle);

    if (kUseMIS)
    {
        float scatterPdf = mi.evalPdf(sd, ls.dir, kUseBSDFSampling);
        weight *= no_diff evalMIS(1, ls.pdf, 1, scatterPdf);
    }

    pathData.radiance += thp * weight;
}

[Differentiable]
[PreferRecompute]
void handleHit(
    SceneQueryAD sceneQuery,
    IntersectionAD isect,
    EmissiveLightSampler emissiveSampler,
    inout PathData pathData,
    inout RayAD ray,
    inout SampleGenerator sg,
    bool shouldReparameterize = true
)
{
    let lod = ExplicitLodTextureSampler(0.f);

    TriangleHit triHit = { isect.instanceID, isect.triangleID, isect.barycentrics.y, isect.barycentrics.z };
    HitInfo hit = HitInfo(triHit);

    // Load shading data.
    ShadingData sd = loadShadingData(sceneQuery, hit, isect, ray.origin, ray.direction, lod);

    // Create differentiable material instance.
    DiffMaterialData diffData;
    let mi = getDiffMaterialInstance(diffData, sd, lod);

    // Add emitted light.
    const bool isPrimaryHit = pathData.length == 0;
    if (isPrimaryHit || kUseEmissiveLights && (!kUseNEE || kUseMIS))
    {
        float misWeight = 1.f;

        if (!isPrimaryHit && kUseEmissiveLights && kUseNEE && kUseMIS)
        {
            // Prepare hit point struct with data needed for emissive light PDF evaluation.
            TriangleLightHit lightHit;
            lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID, triHit.primitiveIndex);
            lightHit.posW = sd.posW;
            lightHit.normalW = sd.getOrientedFaceNormal();

            // Evaluate PDF at the hit, had it been generated with light sampling.
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = true;
            float lightPdf = no_diff emissiveSampler.evalPdf(detach(ray.origin), pathData.normal, upperHemisphere, lightHit);

            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = no_diff evalMIS(1, pathData.pdf, 1, lightPdf);
        }

        // TODO: support differentiable for the emission of lights.
        pathData.radiance += pathData.thp * mi.getProperties(sd).emission * misWeight;
    }

    // Check whether to terminate based on max depth.
    if (pathData.length >= kMaxBounces + 1)
    {
        pathData.terminated = true;
        return;
    }

    // Compute NEE. Assume we only have area lights for now.
    if (kUseNEE)
    {
        LightSample ls = {};
        bool validSample = generateEmissiveSample(isect, emissiveSampler, true, sg, ls);
        if (validSample)
        {
            Ray shadowRay = ls.getVisibilityRay();
            bool visible = sceneQuery.traceVisibilityRay(shadowRay);

            if (visible)
            {
                computeNEE(sceneQuery, sd, mi, diffData, isect, ls, pathData, sg, shouldReparameterize);
            }
        }
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin;

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sceneQuery, sd, mi, diffData, isect, ray, pathData, sg, shouldReparameterize))
    {
        pathData.terminated = true;
        return;
    }

    pathData.length++;
}
