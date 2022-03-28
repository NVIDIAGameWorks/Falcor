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

/** Whitted ray tracer.

    The purpose is to use it as a baseline for showing how to implement different
    texture level-of-detail filtering methods.

    The host sets the following defines:

    MAX_BOUNCES                 Maximum number of indirect bounces (0 means no indirect).
    USE_ANALYTIC_LIGHTS         Nonzero if Falcor's analytic lights should be used.
    USE_EMISSIVE_LIGHTS         Nonzero if emissive geometry should be used as lights.
    USE_ENV_LIGHT               Nonzero if env map is available and should be used as light source.
    USE_ENV_BACKGROUND          Nonzero if env map is available and should be used as background.
    TEX_LOD_MODE                Could be either of Mip0, RayCones, RayDiffs.
    RAY_CONE_MODE               Could be either Combo or Unified.
    VISUALIZE_SURFACE_SPREAD    True if we want to visualize the surface spread angle for the ray cone methods.
    USE_ROUGHNESS_TO_VARIANCE   True if we want to grow the size of the ray cones base on the BRDF roughness.
    USE_FRESNEL_AS_BRDF         True if we want to use Fresnel term as BRDF.
    is_valid_<name>             1 if optional I/O buffer with this name should be used.
*/

#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Rendering.Materials.Fresnel;
import WhittedRayTracerTypes;

cbuffer CB
{
    uint        gFrameCount;                    // Frame count since scene was loaded.
    uint        gPRNGDimension;                 // First available PRNG dimension.
    float       gScreenSpacePixelSpreadAngle;   // The angle an "average" pixel spans from camera (texLOD).
}

// Inputs
Texture2D<float4> gWorldPosition;
Texture2D<float4> gWorldShadingNormal;
Texture2D<float4> gWorldShadingTangent;
Texture2D<float4> gWorldFaceNormal;
Texture2D<float2> gTextureCoord;
Texture2D<float4> gTextureGrads;        // Optional
Texture2D<uint4>  gMaterialData;
Texture2D<PackedHitInfo> gVBuffer;      // Optional

// Outputs
RWTexture2D<float4> gOutputColor;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const TexLODMode kTexLODMode = TexLODMode(TEX_LOD_MODE);
static const RayConeMode kRayConeMode = RayConeMode(RAY_CONE_MODE);
static const RayFootprintFilterMode kRayConeFilterMode = RayFootprintFilterMode(RAY_CONE_FILTER_MODE);
static const RayFootprintFilterMode kRayDiffFilterMode = RayFootprintFilterMode(RAY_DIFF_FILTER_MODE);
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0.25f, 0.25f, 0.25f);
static const float kRayTMax = FLT_MAX;
static const bool kUseRoughnessToVariance = USE_ROUGHNESS_TO_VARIANCE;
static const bool kVisualizeSurfaceSpread = VISUALIZE_SURFACE_SPREAD;
static const bool kUseFresnelAsBRDF = USE_FRESNEL_AS_BRDF;

/** Payload for shadow ray.
*/
struct ShadowRayData
{
    bool visible;
};

float roughnessToSpread(float roughness)
{
    const float ggxAlpha = roughness * roughness;
    const float s = ggxAlpha * ggxAlpha;
    const float sigma2 =  (s / max(0.0001f, (1.0f - s))) * 0.5f;
    return 2.0f * sqrt(sigma2);
}

/** Payload for scatter ray (128B when ray cones are used, 164B when ray diffs are used).
*/
struct ScatterRayData
{
    float3  radiance;                   ///< Accumulated outgoing radiance from path.
    bool    terminated;                 ///< Set to true when path is terminated.
    bool    hasHitRefractiveMaterial;   ///> Whether the path has hit some refractive material on the way.
    float3  thp;                        ///< Current path throughput. This is updated at each path vertex.
    uint    pathLength;                 ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3  origin;                     ///< Next path segment origin.
    float3  direction;                  ///< Next path segment direction.
                                        //   only one (at most) of RayCone and RayDiff will be used (none, if we use Mip0).
    RayCone rayCone;                    ///< Ray cone (2 floats) for texture LOD.
    RayDiff rayDiff;                    ///< Ray differential (12 floats) for texture LOD.

    SampleGenerator sg;                 ///< Per-ray state for the sample generator (up to 16B).

    /** Initializes ray payload with default parameters.
    */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.hasHitRefractiveMaterial = false;
        this.pathLength = 0;
        this.radiance = float3(0.0f, 0.0f, 0.0f);
        this.thp = float3(1.0f, 1.0f, 1.0f);
        this.origin = float3(0.0f, 0.0f, 0.0f);
        this.direction = float3(0.0f, 0.0f, 0.0f);
        this.sg = sg;
        this.rayCone = {};
        this.rayDiff = {};
    }
};

/** Helper to create a texture sampler instance.
    The method for computing texture level-of-detail depends on the configuration.
    \param[in] pixel Current pixel coordinates.
    \return Texture sampler instance.
*/
ITextureSampler createTextureSampler(const uint2 pixel)
{
    if (is_valid(gTextureGrads))
    {
        float4 texGrads = gTextureGrads[pixel];
        return ExplicitGradientTextureSampler(texGrads.xy, texGrads.zw);
    }
    else
    {
        return ExplicitLodTextureSampler(0.f);
    }
}

/** Helper to load G-buffer data and prepare shading data.
*/
ShadingData loadShadingData(const uint2 pixel, const float3 posW, const float3 rayDir, const ITextureSampler lod)
{
    // Load G-buffer data.
    float3 normal = gWorldShadingNormal[pixel].xyz;
    float4 tangent = gWorldShadingTangent[pixel];
    float3 faceNormal = gWorldFaceNormal[pixel].xyz;
    float2 texC = gTextureCoord[pixel];
    uint materialID = gMaterialData[pixel].x;

    VertexData v = {};
    v.posW = posW;
    v.normalW = normalize(normal);
    v.tangentW = float4(normalize(tangent.xyz), tangent.w);
    v.faceNormalW = normalize(faceNormal);
    v.texC = texC;
    v.coneTexLODValue = 0.f;

    // Prepare shading data.
    // Set 'useNormalMap' parameter to false as normal mapping has already been applied when generating the G-buffer.
    return gScene.materials.prepareShadingData(v, materialID, -rayDir, lod, false);
}

/** Helper for computing relatice index of refraction (eta) at a hit.
    This is computed based on the IoRs of the media outside and inside the material.
*/
float computeEta(const ShadingData sd)
{
    float insideIoR = gScene.materials.evalIoR(sd.materialID);
    float eta = sd.frontFacing ? (sd.IoR / insideIoR) : (insideIoR / sd.IoR);
    return eta;
}

/** Traces a shadow ray towards a light source.
    \param[in] origin Ray origin for the shadow ray.
    \param[in] dir Direction from shading point towards the light source (normalized).
    \param[in] distance Distance to the light source.
    \return True if light is visible, false otherwise.
*/
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before.
    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff /* instanceInclusionMask */, 1 /* hitIdx */, rayTypeCount, 1 /* missIdx */, ray, rayData);

    return rayData.visible;
}

/** Traces a scatter ray based on ray parameters stored in the ray payload.
    \param[in] rayData Describes the ray parameters. The struct is modified based on the result.
*/
void traceScatterRay(inout ScatterRayData rayData)
{
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0;      // TODO: Set cull mode from the app.
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

/** Evaluates the direct illumination from all analytic lights.
    This function samples Falcor's light list uniformly with one shadow ray per light.
    \param[in] sd Shading data.
    \param[in] bsdf BSDF at the shading point.
    \param[in] rayOrigin Ray origin for the shadow ray.
    \param[in] sg SampleGenerator object.
    \return Outgoing radiance in view direction.
*/
float3 evalDirectAnalytic(const ShadingData sd, const IBSDF bsdf, const float3 rayOrigin, inout SampleGenerator sg)
{
    // Use all analytic light sources.
    const uint lightCount = gScene.getLightCount();
    float invPdf = 1.0f / lightCount; // Light selection pdf = 1.0f / lightCount.

    float3 contribution = float3(0.0f);
    for (uint lightIndex = 0; lightIndex < lightCount; lightIndex++)
    {
        // Sample local light source.
        AnalyticLightSample ls;
        bool valid = sampleLight(rayOrigin, gScene.getLight(lightIndex), sg, ls);

        // Reject sample if lower hemisphere.
        if (valid && dot(ls.dir, sd.N) > kMinCosTheta)
        {
            // Test visibility by tracing a shadow ray.
            bool V = traceShadowRay(rayOrigin, ls.dir, ls.distance);
            contribution +=  V ? bsdf.eval(sd, ls.dir, sg) * ls.Li * invPdf : float3(0);
        }
    }
    return contribution;
}

/** Processes a hit point to generate a reflection ray or terminate.
    \param[in] sd Shading data.
    \param[in] bsdfProperties BSDF properties.
    \param[in] rayOrigin Ray origin for the new ray.
    \param[in] rayDir Ray direction for incident ray.
    \param[in] normal Surface normal to use for reflection (may be different from the shading normal).
    \param[in] tir True if total internal reflection.
    \param[in,out] rayData Ray payload.
*/
void generateReflectionRay(const ShadingData sd, const BSDFProperties bsdfProperties, const float3 rayOrigin, const float3 rayDir, const float3 normal, bool tir, inout ScatterRayData rayData)
{
    float3 wo = reflect(rayDir, normal);
    rayData.origin = rayOrigin;
    rayData.direction = wo;
    float3 specular = bsdfProperties.specularReflectance;

    if (kUseFresnelAsBRDF)
    {
        rayData.thp *= tir ? 1.0f : evalFresnelSchlick(specular, 1.0f, max(0.00001f, dot(normal, wo)));
    }
    else
    {
        // TODO: better proposals here are welcome...
        rayData.thp *= tir ? 1.0f : (specular.x + specular.y + specular.z) / 3.0f;
    }
}

/*********************** Ray index 0: Scatter ray ************************ */
[shader("miss")]
void scatterMiss(inout ScatterRayData rayData : SV_RayPayload)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight)
    {
        float lod = 0.0f; // TexLODMode::Mip0
        if (kTexLODMode == TexLODMode::RayCones)
        {
            lod = computeEnvironmentMapLOD(rayData.rayCone.getSpreadAngle(), gScene.envMap.envMap);
        }
        else if (kTexLODMode == TexLODMode::RayDiffs)
        {
            lod = computeEnvironmentMapLOD(rayData.rayDiff.getdDdx(), rayData.rayDiff.getdDdy(), gScene.envMap.envMap);
        }

        float3 Le = gScene.envMap.eval(WorldRayDirection(), lod);
        rayData.radiance += rayData.thp * Le;
    }
}

[shader("anyhit")]
void scatterAnyHit(inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
}

[shader("closesthit")]
void scatterClosestHit(inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Evaluate Falcor's material parameters at the hit point.
    const float3 rayDir = WorldRayDirection();
    const float3 rayOrg = WorldRayOrigin();
    float3 refractedRayDir = {};
    bool tir = false;

    float hitT = RayTCurrent();
    const GeometryInstanceID instanceID = getGeometryInstanceID();
    const uint materialID = gScene.getMaterialID(instanceID);
    float surfaceSpreadAngle = 0.0f;
    float eta = 1.f;
    ShadingData sd = {};
    VertexData v = {};
    ITextureSampler lod;

    if (kTexLODMode == TexLODMode::RayCones)
    {
        v = getVertexDataRayCones(instanceID, PrimitiveIndex(), attribs);
        float curvature = gScene.computeCurvatureIsotropic(instanceID, PrimitiveIndex());

        // Compute texture LOD for prepareShadingData().
        if (kRayConeMode == RayConeMode::Combo || kRayConeMode == RayConeMode::Unified)
        {
            float rayConeWidth = rayData.rayCone.getSpreadAngle() * hitT + rayData.rayCone.getWidth();
            surfaceSpreadAngle = computeSpreadAngleFromCurvatureIso(curvature, rayConeWidth, rayDir, v.normalW);
        }

        // Propagate (only) to hit point (not setting any surfaceSpreadAngle -- this is cone in the next propagate call, a few lines down).
        rayData.rayCone = rayData.rayCone.propagateDistance(hitT);

        if (kRayConeFilterMode == RayFootprintFilterMode::Isotropic || (kRayConeFilterMode == RayFootprintFilterMode::AnisotropicWhenRefraction && !rayData.hasHitRefractiveMaterial))
        {
            float lambda = rayData.rayCone.computeLOD(v.coneTexLODValue, rayDir, v.normalW);
            lod = ExplicitRayConesLodTextureSampler(lambda);
            sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
        }
        else
        {
            float3 positions[3];
            float2 txcoords[3];
            gScene.getVertexPositionsW(instanceID, PrimitiveIndex(), positions);
            gScene.getVertexTexCoords(instanceID, PrimitiveIndex(), txcoords);

            float coneWidthAtHitPoint = rayData.rayCone.getWidth();
            float2 ddx = float2(0), ddy = float2(0);
            computeAnisotropicEllipseAxes(v.posW, v.faceNormalW, rayDir, coneWidthAtHitPoint, positions, txcoords, v.texC, ddx, ddy);
            lod = ExplicitGradientTextureSampler(ddx, ddy);
        }
        sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
        eta = computeEta(sd);
    }
    else if (kTexLODMode == TexLODMode::RayDiffs)
    {
        float2 dUVdx, dUVdy;    // Ray differential variables for the texture lookup.
        float2 dBarydx, dBarydy;
        float3 geometricNormal = gScene.getFaceNormalW(instanceID, PrimitiveIndex());
        RayDiff newRayDiff = rayData.rayDiff.propagate(rayOrg, rayDir, hitT, geometricNormal);              // Propagate the ray differential to the current hit point.

        float3 barycentrics;
        StaticVertexData triangleVertices[3];
        v = getVertexData(instanceID, PrimitiveIndex(), attribs, barycentrics, triangleVertices);

        float3 unnormalizedN;   // Non-normalized interpolated normal for ray differential scatter.
        float3 normals[3];      // Non-normalized normals for ray differential scatter.

        prepareRayDiffAtHitPoint(v, triangleVertices, barycentrics, rayDir, gScene.getWorldMatrix(instanceID), gScene.getInverseTransposeWorldMatrix(instanceID),
            newRayDiff, unnormalizedN, normals, dBarydx, dBarydy, dUVdx, dUVdy);

        if (kRayDiffFilterMode == RayFootprintFilterMode::Isotropic || (kRayDiffFilterMode == RayFootprintFilterMode::AnisotropicWhenRefraction && !rayData.hasHitRefractiveMaterial))
        {
            // When using ExplicitRayDiffsIsotropicTextureSampler, the texture sampler will compute a single lambda for texture LOD
            // using the dUVdx and dUVdy.
            lod = ExplicitRayDiffsIsotropicTextureSampler(dUVdx, dUVdy);
        }
        else // kRayDiffFilterMode == RayFootprintFilterMode::Anisotropic
        {
            // Use the dUVdx and dUVdy directly --> gives you anisotropic filtering.
            lod = ExplicitGradientTextureSampler(dUVdx, dUVdy);
        }
        sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
        eta = computeEta(sd);

        if (eta == 1.0f)
        {
            reflectRayDifferential(newRayDiff, rayDir, unnormalizedN, v.normalW, dBarydx, dBarydy, normals);
        }
        else
        {
            if (!refractRayDifferential(newRayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, normals, eta, refractedRayDir))
            {
                reflectRayDifferential(newRayDiff, rayDir, unnormalizedN, v.normalW, dBarydx, dBarydy, normals);
                tir = true;
            }
        }
        rayData.rayDiff = newRayDiff;
    }
    else // kTexLODMode == TexLODMode::Mip0
    {
        v = getVertexData(instanceID, PrimitiveIndex(), attribs);
        lod = ExplicitLodTextureSampler(0.f);
        sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
        eta = computeEta(sd);

        if (eta != 1.0f)
        {
            if (!refractWithTIR(rayDir, sd.N, eta, refractedRayDir))
            {
                tir = true;
            }
        }
    }

    // Create BSDF instance and query its properties.
    let bsdf = gScene.materials.getBSDF(sd, lod);
    let bsdfProperties = bsdf.getProperties(sd);

    // Update ray cones.
    // This has to be done after creating the BSDF instance because we need to query it
    // for surface roughness in order to update the ray cone spread angle.
    if (kTexLODMode == TexLODMode::RayCones)
    {
        // Bounce on hit surface.
#if USE_ROUGHNESS_TO_VARIANCE
        surfaceSpreadAngle += roughnessToSpread(bsdfProperties.roughness);
#endif
        if (eta == 1.0f) // Reflection
        {
            // Reflection needs a factor of 2.0 (see RTG1-article on texture LOD).
            rayData.rayCone = rayData.rayCone.addToSpreadAngle(2.0f * surfaceSpreadAngle);
        }
        else // Refraction
        {
            if (!refractRayCone(rayData.rayCone, rayOrg, rayDir, sd.posW, sd.N, surfaceSpreadAngle, eta, refractedRayDir))
            {
                rayData.rayCone = rayData.rayCone.addToSpreadAngle(surfaceSpreadAngle);
                tir = true;
            }
        }
    }

    // Add emitted light.
    if (kUseEmissiveLights)
    {
        rayData.radiance += rayData.thp * bsdfProperties.emission;
    }

    // Check whether to terminate based on max depth.
    if (rayData.pathLength >= kMaxBounces)
    {
        rayData.terminated = true;
        return;
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOriginAtHitPoint = sd.computeNewRayOrigin();

    // Add contribution of direct light from analytic lights.
    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, bsdf, rayOriginAtHitPoint, rayData.sg);
        rayData.radiance += rayData.thp * Lr;
    }

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (eta == 1.0f || tir)
    {
        generateReflectionRay(sd, bsdfProperties, rayOriginAtHitPoint, rayDir, v.normalW, tir, rayData);
    }
    else
    {
        rayData.direction = refractedRayDir;
        rayData.origin = sd.computeNewRayOrigin(false);

        if (kUseFresnelAsBRDF)
        {
            rayData.thp *= 1.0f - evalFresnelSchlick(bsdfProperties.specularReflectance, 1.0f, max(0.00001f, dot(sd.N, -rayDir)));
            // There's convinient 'evalFresnelDielectric' we could use here, but it's somewhat darker (worse for development)
            //rayData.thp *= 1.0f - evalFresnelDielectric(eta, dot(sd.N, -rayDir));
        }
        else
        {
            float roughness = bsdfProperties.roughness;
            rayData.thp *= (1.0f - roughness);
        }
        rayData.hasHitRefractiveMaterial = true;
    }
    rayData.pathLength++;

    if (max(max(rayData.thp.x, rayData.thp.y), rayData.thp.z) < 0.01f)  // TODO: this is a hack to terminate rays
    {
        rayData.terminated = true;
        return;
    }
}

/************************** Ray index 1: Shadow ray ************************ */

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData : SV_RayPayload)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
}

/** ******************************** RayGen ******************************** */

/** This is the entry point for the Whitted ray tracer.

    One path per pixel is generated, which  is traced into the scene.
    The path tracer is written as a for-loop over path segments.

    Built-in light sources (point, directional, ..) are sampled explicitly at each
    path vertex. The contributions from area lights (env map and mesh lights)
    are explicitly added by the scatter ray hit/miss shaders.
*/
[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    float3 outColor = float3(0.0f);

    const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(launchIndex, launchDim);    // Used by ray diffs.
    const float3 rayDir = normalize(nonNormalizedRayDir);
    const float4 worldPos = gWorldPosition[launchIndex];

    if (worldPos.w != 0.f) // Using w to indicate valid geometry for now.
    {
        // Pixel represents a valid primary hit. Compute its contribution.
        let lod = createTextureSampler(launchIndex);
        const ShadingData sd = loadShadingData(launchIndex, worldPos.xyz, rayDir, lod);
        const float eta = computeEta(sd);

        // Create BSDF instance and query its properties.
        let bsdf = gScene.materials.getBSDF(sd, lod);
        let bsdfProperties = bsdf.getProperties(sd);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(launchIndex, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++) sampleNext1D(sg);

        // Compute ray origin for new rays spawned from the G-buffer.
        const float3 rayOriginAtHitPoint = sd.computeNewRayOrigin();

        // Always output directly emitted light, independent of whether emissive materials are treated as light sources or not.
        outColor += bsdfProperties.emission;

        // Add contribution of direct light from analytic lights.
        // Light probe and mesh lights are handled by the scatter ray hit/miss shaders.
        outColor += kUseAnalyticLights ? evalDirectAnalytic(sd, bsdf, rayOriginAtHitPoint, sg) : float3(0.f);

        // Prepare ray payload.
        ScatterRayData rayData = ScatterRayData(sg);

        float3 refractedRayDir = float3(0.0f);
        bool tir = false;

        if (kTexLODMode == TexLODMode::RayCones)
        {
            // Set up ray cone (texLOD) at the hit point: first create ray cone at camera, the propagate to hit point.
            float surfaceSpreadAngle = 0.0f;
            const float3 rayOrg = gScene.camera.getPosition();
            const float hitT = length(rayOrg - worldPos.xyz);

            const HitInfo hit = HitInfo(gVBuffer[launchIndex]);
            if (hit.isValid() && hit.getType() == HitType::Triangle)
            {
                const TriangleHit triangleHit = hit.getTriangleHit();
                const float4x4 worldMat = gScene.getWorldMatrix(triangleHit.instanceID);
                const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(triangleHit.instanceID);
                const uint3 vertexIndices = gScene.getIndices(triangleHit.instanceID, triangleHit.primitiveIndex);
                const float3 barycentrics = triangleHit.getBarycentricWeights();
                float2 txcoords[3], dBarydx, dBarydy, dUVdx, dUVdy;

                StaticVertexData vertices[3] = { gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2]) };

                float curvature = gScene.computeCurvatureIsotropicFirstHit(triangleHit.instanceID, triangleHit.primitiveIndex, rayDir);

                if (kRayConeMode == RayConeMode::Combo)
                {
                    RayDiff rayDiff;
                    float3 dDdx, dDdy, unnormalizedN, normals[3], dNdx, dNdy, edge1, edge2;
                    prepareVerticesForRayDiffs(rayDir, vertices, worldMat, worldInvTransposeMat, barycentrics, edge1, edge2, normals, unnormalizedN, txcoords);
                    computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, launchDim, dDdx, dDdy);
                    RayDiff rd = RayDiff(float3(0.f), float3(0.f), dDdx, dDdy);    // Init ray diff. dOdx = 0 , dOdy = 0, and the directions are from above.
                    rayDiff = rd.propagate(worldPos.xyz, rayDir, hitT, sd.N);               // Propagate the ray differential to the current hit point.
                    computeBarycentricDifferentials(rayDiff, rayDir, edge1, edge2, sd.faceN, dBarydx, dBarydy);
                    computeNormalDifferentials(rayDiff, unnormalizedN, dBarydx, dBarydy, normals, dNdx, dNdy);
                    float3 right = normalize(gScene.camera.computeNonNormalizedRayDirPinhole(launchIndex + uint2(1, 0), launchDim)) - rayDir;
                    float3 up = normalize(gScene.camera.computeNonNormalizedRayDirPinhole(launchIndex + uint2(0, 1), launchDim)) - rayDir;
                    surfaceSpreadAngle = computeScreenSpaceSurfaceSpreadAngle(right, up, dNdx, dNdy);
                }
                else // kRayConeMode == RayConeMode::Unified
                {
                    float curvature = gScene.computeCurvatureIsotropicFirstHit(triangleHit.instanceID, triangleHit.primitiveIndex, rayDir);
                    float rayConeWidth = hitT * gScreenSpacePixelSpreadAngle;
                    surfaceSpreadAngle = computeSpreadAngleFromCurvatureIso(curvature, hitT * gScreenSpacePixelSpreadAngle, rayDir, sd.N);
                }

#if USE_ROUGHNESS_TO_VARIANCE
                surfaceSpreadAngle += roughnessToSpread(bsdfProperties.roughness);
#endif
                if (kVisualizeSurfaceSpread)
                {
                    const float eps = 1.0e-4;
                    float r = surfaceSpreadAngle > eps ? surfaceSpreadAngle : 0.0f;
                    float g = surfaceSpreadAngle <= -eps ? -surfaceSpreadAngle : 0.0f;
                    float b = abs(surfaceSpreadAngle) < eps ? 1.0f : 0.0f;
                    gOutputColor[launchIndex] = float4(r, g, b, 1.0f) * 10.0f;
                    return;
                }

                RayCone rc = RayCone(0.0f, gScreenSpacePixelSpreadAngle);
                rayData.rayCone = rc.propagateDistance(hitT);
                if (eta == 1.0f) // Reflection
                {
                    rayData.rayCone = rayData.rayCone.addToSpreadAngle(2.0f * surfaceSpreadAngle); // Reflection needs a factor of 2.0 (see RTG1 article on texture LOD).
                }
                else // Refraction
                {
                    float3 hitPoint = rayOrg + rayDir * hitT;
                    if (!refractRayCone(rayData.rayCone, rayOrg, rayDir, hitPoint, sd.N, surfaceSpreadAngle, eta, refractedRayDir))
                    {
                        rayData.rayCone = rayData.rayCone.addToSpreadAngle(surfaceSpreadAngle);
                        tir = true;
                    }
                }
            }
        }
        else if (kTexLODMode == TexLODMode::RayDiffs)
        {
            RayDiff rd;
            float3 dDdx, dDdy;
            const float3 rayOrg = gScene.camera.getPosition();
            const float hitT = length(rayOrg - worldPos.xyz);

            computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, launchDim, dDdx, dDdy);

            rayData.rayDiff = RayDiff(float3(0.f), float3(0.f), dDdx, dDdy); // Init ray diff. dOdx = 0 , dOdy = 0, and the directions are from above.

            rayData.rayDiff = rayData.rayDiff.propagate(worldPos.xyz, rayDir, hitT, sd.faceN); // Propagate the ray differential to the current hit point.

            const HitInfo hit = HitInfo(gVBuffer[launchIndex]);
            if (hit.isValid() && hit.getType() == HitType::Triangle)
            {
                const TriangleHit triangleHit = hit.getTriangleHit();
                const float4x4 worldMat = gScene.getWorldMatrix(triangleHit.instanceID);
                const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(triangleHit.instanceID);
                const uint3 vertexIndices = gScene.getIndices(triangleHit.instanceID, triangleHit.primitiveIndex);
                const float3 barycentrics = triangleHit.getBarycentricWeights();
                float3 unnormalizedN, normals[3], dNdx, dNdy, edge1, edge2;
                float2 txcoords[3], dBarydx, dBarydy, dUVdx, dUVdy;

                StaticVertexData vertices[3] = { gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2]) };
                prepareVerticesForRayDiffs(rayDir, vertices, worldMat, worldInvTransposeMat, barycentrics, edge1, edge2, normals, unnormalizedN, txcoords);

                computeBarycentricDifferentials(rayData.rayDiff, rayDir, edge1, edge2, sd.faceN, dBarydx, dBarydy);

                if (eta == 1.0f)
                {
                    reflectRayDifferential(rayData.rayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, normals);
                }
                else
                {
                    if (!refractRayDifferential(rayData.rayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, normals, eta, refractedRayDir))
                    {
                        reflectRayDifferential(rayData.rayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, normals);
                        tir = true;
                    }
                }
            }
        }
        else // Mip0.
        {
            if (!refractWithTIR(rayDir, sd.N, eta, refractedRayDir))
            {
                tir = true;
            }
        }

        // Generate scatter ray.
        if (eta == 1.0f || tir)
        {
            generateReflectionRay(sd, bsdfProperties, rayOriginAtHitPoint, rayDir, sd.N, tir, rayData);
        }
        else
        {
            rayData.direction = refractedRayDir;
            rayData.origin = sd.computeNewRayOrigin(false);

            if (kUseFresnelAsBRDF)
            {
                rayData.thp *= 1.0f - evalFresnelSchlick(bsdfProperties.specularReflectance, 1.0f, max(0.00001f, dot(sd.N, -rayDir)));

                // There's convenient 'evalFresnelDielectric' we could use here, but it's somewhat darker (worse for development)
                //rayData.thp *= 1.0f - evalFresnelDielectric(eta, dot(sd.N, -rayDir));
            }
            else
            {
                rayData.thp *= (1.0f - bsdfProperties.roughness);
            }
        }

        if (bsdfProperties.roughness <= 0.19f) // Only trace reflection rays when material is highly reflective.
        {
            // Follow path into the scene and compute its total contribution.
            for (uint depth = 0; depth <= kMaxBounces && !rayData.terminated; depth++)
            {
                // Trace scatter ray. If it hits geometry, the closest hit shader samples
                // direct illumination and generates the next scatter ray.
                traceScatterRay(rayData);
            }
        }
        // Store contribution from scatter ray.
        outColor += rayData.radiance;
    }
    else
    {
        // Background pixel.
        if (kUseEnvBackground)
        {
            float lod = 0.0f;    // TexLODMode::Mip0.
            if (kTexLODMode == TexLODMode::RayCones)
            {
                lod = computeEnvironmentMapLOD(gScreenSpacePixelSpreadAngle, gScene.envMap.envMap);
            }
            else if (kTexLODMode == TexLODMode::RayDiffs)
            {
                float3 dDdx, dDdy;
                computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, launchDim, dDdx, dDdy);
                lod = computeEnvironmentMapLOD(dDdx, dDdy, gScene.envMap.envMap);
            }
            outColor = gScene.envMap.eval(rayDir, lod);
        }
        else
        {
            outColor = kDefaultBackgroundColor;
        }
    }

    gOutputColor[launchIndex] = float4(outColor, 1.0f);
}
