/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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

    MAX_BOUNCES         Maximum number of indirect bounces (0 means no indirect).
    USE_ANALYTIC_LIGHTS Nonzero if Falcor's analytic lights should be used.
    USE_EMISSIVE_LIGHTS Nonzero if emissive geometry should be used as lights.
    USE_ENV_LIGHT       Nonzero if env map is available and should be used as light source.
    USE_ENV_BACKGROUND  Nonzero if env map is available and should be used as background.
    TEXLOD_MODE         Could be either of Mip0, RayCones, RayDiffsIsotropic, RayDiffsAnisotropic
    USE_RASTERIZED_GBUFFER True if we use rasterized G-buffer.
    is_valid_<name>     1 if optional I/O buffer with this name should be used.
*/

#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.HitInfo;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Experimental.Scene.Material.MaterialShading;
import Experimental.Scene.Lights.EnvProbe;
import Experimental.Scene.Lights.LightHelpers;
import Experimental.Scene.Material.TexLODHelpers;
import Experimental.Scene.Material.TexLODTypes;

cbuffer CB
{
    uint        gFrameCount;                    // Frame count since scene was loaded.
    uint        gPRNGDimension;                 // First available PRNG dimension.
    EnvProbe    gEnvProbe;                      // Environment map sampling functions.
    float       gScreenSpacePixelSpreadAngle;   // The angle an "average" pixel spans from camera (texLOD)
}

// Inputs
Texture2D<float4> gWorldPosition;
Texture2D<float4> gWorldShadingNormal;
Texture2D<float4> gWorldShadingBitangent; // Optional
Texture2D<float4> gWorldFaceNormal;
Texture2D<float4> gWorldView; // Optional
Texture2D<float4> gMaterialDiffuseOpacity;
Texture2D<float4> gMaterialSpecularRoughness;
Texture2D<float4> gMaterialEmissive;
Texture2D<float4> gMaterialExtraParams;
Texture2D<float>  gSurfaceSpreadAngle;
Texture2D<float4> gRayDifferentialX;
Texture2D<float4> gRayDifferentialY;
Texture2D<float4> gRayDifferentialZ;
Texture2D<uint2>  gVBuffer;


// Outputs
RWTexture2D<float4> gOutputColor;

// Static configuration based on defines set from the host.
#define isValid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const uint kTexLODMode = TEXLOD_MODE;
static const bool kUsingRasterizedGBuffer = USE_RASTERIZED_GBUFFER;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

/** Payload for shadow ray.
*/
struct ShadowRayData
{
    bool visible;
};

/** Payload for scatter ray (128B when ray cones are used, 164B when ray diffs are used).
*/
struct ScatterRayData
{
    float3  radiance;       ///< Accumulated outgoing radiance from path.
    bool    terminated;     ///< Set to true when path is terminated.
    float3  thp;            ///< Current path throughput. This is updated at each path vertex.
    uint    pathLength;     ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3  origin;         ///< Next path segment origin.
    float3  direction;      ///< Next path segment direction.
                            //   only one (at most) of RayCone and RayDiff will be used (none, if we use Mip0)
    RayCone rayCone;        ///< Ray cone (2 floats) for texture LOD 
    RayDiff rayDiff;        ///< Ray differential (12 floats) for texture LOD

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Create ray payload with default parameters.
    */
    static ScatterRayData create(SampleGenerator sg)
    {
        ScatterRayData d;
        d.terminated = false;
        d.pathLength = 0;
        d.radiance = float3(0, 0, 0);
        d.thp = float3(1, 1, 1);
        d.origin = float3(0, 0, 0);
        d.direction = float3(0, 0, 0);
        d.sg = sg;
        d.rayCone = {};
        d.rayDiff = {};
        return d;
    }
};

/** Helper to load the material attributes.
*/
MaterialParams loadMaterialParams(uint2 pixelPos)
{
    MaterialParams matParams;
    matParams.diffuseOpacity = gMaterialDiffuseOpacity[pixelPos];
    matParams.specularRoughness = gMaterialSpecularRoughness[pixelPos];
    matParams.emissive = gMaterialEmissive[pixelPos];
    matParams.extraParams = gMaterialExtraParams[pixelPos];

    return matParams;
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
    rayData.visible = false;    // Set to true by miss shader if ray is not terminated before
    TraceRay(gRtScene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff /* instanceInclusionMask */, 1 /* hitIdx */, hitProgramCount, 1 /* missIdx */, ray, rayData);

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

    uint rayFlags = 0;      // TODO: Set cull mode from the app
    TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, hitProgramCount, 0 /* missIdx */, ray, rayData);
}

/** Evaluates the direct illumination from analytic lights.
    This function samples Falcor's light list uniformly with one shadow ray.
    \param[in] sd Shading data.
    \param[in] rayOrigin Ray origin for the shadow ray.
    \param[in] sg SampleGenerator object.
    \return Outgoing radiance in view direction.
*/
float3 evalDirectAnalytic(const ShadingData sd, float3 rayOrigin, inout SampleGenerator sg)
{
    // Use all analytic light sources
    const uint lightCount = gScene.getLightCount();
    float invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.

    float3 contribution = float3(0);
    for (uint lightIndex = 0; lightIndex < lightCount; lightIndex++)
    {
        // Sample local light source.
        AnalyticLightSample ls;
        sampleLight(rayOrigin, gScene.getLight(lightIndex), sg, ls);

        // Reject sample if lower hemisphere.
        if (dot(ls.dir, sd.N) > kMinCosTheta)
        {
            // Test visibility by tracing a shadow ray.
            bool V = traceShadowRay(rayOrigin, ls.dir, ls.distance);
            contribution +=  V ? evalBSDFCosine(sd, ls.dir) * ls.Li * invPdf : float3(0);
        }
    }
    return contribution;
}

/** Processes a hit point to generate a scatter ray or terminate.
    This function generates a cosine-weighted direction over the hemisphere.
    \param[in] sd Shading data.
    \param[in] rayOrigin Ray origin for the new ray.
    \param[in] rayData Ray payload.
*/
void generateReflectionRay(const ShadingData sd, const float3 rayOrigin, const float3 rayDir, const float3 normal, inout ScatterRayData rayData)
{
    float3 wi = reflect(rayDir, normal);
    rayData.origin = rayOrigin;
    rayData.direction = wi;
}

/** ********************* Ray index 0: Scatter ray ************************ */

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData : SV_RayPayload)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight)
    {
        float3 Le = evalEnvProbe(gEnvProbe, WorldRayDirection());
        rayData.radiance += rayData.thp * Le;
    }
}

[shader("anyhit")]
void scatterAnyHit(
    uniform HitShaderParams hitParams,
    inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
}

[shader("closesthit")]
void scatterClosestHit(
    uniform HitShaderParams hitParams,
    inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Evaluate Falcor's material parameters at the hit point.
    // Note we pass hitPos-rayDir as "camera position" to avoid zero-length rays causing NaNs
    // in the view direction. It'd been cleaner if prepareShadingData() took ray dir directly.
    const float3 rayDir = WorldRayDirection();
    const float3 rayOrg = WorldRayOrigin();

    float hitT = RayTCurrent();
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    ShadingData sd;
    VertexData v;

    if (TexLODMode(kTexLODMode) == TexLODMode::Mip0)
    {
        v = getVertexData(hitParams, PrimitiveIndex(), attribs);
        sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], v.posW - rayDir, 0.0);
    }
    else if(TexLODMode(kTexLODMode) == TexLODMode::RayCones)
    {
        v = getVertexDataRayCones(hitParams, PrimitiveIndex(), attribs);

        // Compute texture LOD for prepareShadingData()
        rayData.rayCone = rayData.rayCone.propagate(0.0, hitT);     // Using 0.0 here since we do not have any curvature measure, except when starting with gBuffer (and only for the first hit)
        float lambda = rayData.rayCone.computeLOD(v.coneTexLODValue, hitT, rayDir, v.normalW);
        sd = prepareShadingDataUsingRayConesLOD(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], v.posW - rayDir, lambda);
    }
    else        // RayDiffs
    {
        float2 dUVdx, dUVdy;    // ray differential variables for the texture lookup
        float3 geometricNormal = gScene.getFaceNormalW(hitParams.getGlobalHitID(), PrimitiveIndex());
        RayDiff newRayDiff = rayData.rayDiff.propagate(rayOrg, rayDir, hitT, geometricNormal); // Propagate the ray differential to the current hit point

        v = getVertexDataRayDiff(hitParams, PrimitiveIndex(), attribs, rayDir, hitT, newRayDiff, dUVdx, dUVdy);

        rayData.rayDiff = newRayDiff;
        if (TexLODMode(kTexLODMode) == TexLODMode::RayDiffsIsotropic)
        {
            // When using prepareShadingDataUsingRayDiffsLOD(), the texture samepler will compute a single lambda for texture LOD
            // using the dUVdx and dUVdy
            sd = prepareShadingDataUsingRayDiffsLOD(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], v.posW - rayDir, dUVdx, dUVdy);
        }
        else        // TexLODMode::RayDiffsAnisotropic
        {
            // Use the dUVdx and dUVdy directly --> gives you anisotropic filtering
            sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], v.posW - rayDir, dUVdx, dUVdy);
        }
    }

    // Compute tangent space if it is invalid.
    // TODO: Remove this check when we're sure the tangent space is always valid.
    if (!(dot(sd.B, sd.B) > 0.f))   // Note: Comparison written so that NaNs trigger
    {
        sd.B = perp_stark(sd.N);
        sd.T = cross(sd.B, sd.N);
    }

    // Add emitted light.
    if (kUseEmissiveLights)
    {
        rayData.radiance += rayData.thp * sd.emissive;
    }

    // Check whether to terminate based on max depth.
    if (rayData.pathLength >= kMaxBounces)
    {
        rayData.terminated = true;
        return;
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin = sd.computeNewRayOrigin();

    // Add contribution of direct light from analytic lights.
    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, rayOrigin, rayData.sg);
        rayData.radiance += rayData.thp * Lr;
    }

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.

    generateReflectionRay(sd, rayOrigin, rayDir, v.normalW, rayData);

    rayData.pathLength++;
}

/************************** Ray index 1: Shadow ray ************************ */

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData : SV_RayPayload)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")]
void shadowAnyHit(
    uniform HitShaderParams hitParams,
    inout ShadowRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
}

/** ******************************** RayGen ******************************** */

/** This is the entry point for the minimal path tracer.

    One path per pixel is generated, which  is traced into the scene.
    The path tracer is written as a for-loop over path segments.

    Built-in light sources (point, directional) are sampled explicitly at each
    path vertex. The contributions from area lights (env map and mesh lights)
    are explicitly added by the scatter ray hit/miss shaders.
*/
[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    float3 outColor = float3(0, 0, 0);

    const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(launchIndex, launchDim);    // used by ray diffs
    const float3 rayDir = normalize(nonNormalizedRayDir);
    const float4 worldPos = gWorldPosition[launchIndex];


    if (worldPos.w != 0.f)  // Using w to indicate valid geometry for now
    {
        // Pixel represents a valid primary hit. Compute its contribution.

        // Load geometry parameters from G-buffer.
        float3 normal = gWorldShadingNormal[launchIndex].xyz;
        float3 bitangent = isValid(gWorldShadingBitangent) ? gWorldShadingBitangent[launchIndex].xyz : perp_stark(normal);
        float3 faceNormal = gWorldFaceNormal[launchIndex].xyz;
        GeometryParams geoParams = prepareGeometryParams(worldPos.xyz, -rayDir, normal, bitangent, faceNormal);

        // Load material parameters from G-buffer.
        MaterialParams matParams = loadMaterialParams(launchIndex);

        // Prepare ShadingData struct.
        ShadingData sd = prepareShadingData(geoParams, matParams);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator.create(launchIndex, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++) sampleNext1D(sg);

        // Compute ray origin for new rays spawned from the G-buffer.
        const float3 rayOrigin = sd.computeNewRayOrigin();

        // Always output directly emitted light, independent of whether emissive materials are treated as light sources or not.
        outColor += sd.emissive;

        // Add contribution of direct light from analytic lights.
        // Light probe and mesh lights are handled by the scatter ray hit/miss shaders.
        outColor += kUseAnalyticLights ? evalDirectAnalytic(sd, rayOrigin, sg) : float3(0, 0, 0);

        // Prepare ray payload.
        ScatterRayData rayData = ScatterRayData.create(sg);

        if (!kUsingRasterizedGBuffer)
        {
            rayData.pathLength++;
        }

        // Generate scatter ray.
        generateReflectionRay(sd, rayOrigin, rayDir, normal, rayData);

        if (TexLODMode(kTexLODMode) == TexLODMode::RayCones) 
        {
            // Set up ray cone (texLOD) at the hit point: first create ray cone at camera, the propagate to hit point
            float surfaceSpreadAngle = gSurfaceSpreadAngle[launchIndex];    // load the surface spread angle from G-buffer (for ray cones, texture LOD)
            rayData.rayCone = RayCone.create(0.0, gScreenSpacePixelSpreadAngle);
            const float hitT = length(gScene.camera.getPosition() - worldPos.xyz);
            rayData.rayCone = rayData.rayCone.propagate(surfaceSpreadAngle, hitT);
        }
        else if (TexLODMode(kTexLODMode) == TexLODMode::RayDiffsIsotropic || TexLODMode(kTexLODMode) == TexLODMode::RayDiffsAnisotropic)
        {
            RayDiff rd;
            if (kUsingRasterizedGBuffer)
            {
                // Create ray diff from G-buffer
                // The most expensive method in terms of G-buffer storage -- generate the ray diff in pixel shader and store out in G-buffer
                float4 X = gRayDifferentialX[launchIndex];
                float4 Y = gRayDifferentialY[launchIndex];
                float4 Z = gRayDifferentialZ[launchIndex];
                float3 dOdx = float3(X.x, Y.x, Z.x);
                float3 dOdy = float3(X.y, Y.y, Z.y);
                float3 dDdx = float3(X.z, Y.z, Z.z);
                float3 dDdy = float3(X.w, Y.w, Z.w);
                rayData.rayDiff = RayDiff.create(dOdx, dOdy, dDdx, dDdy);
                rd = RayDiff.create(dOdx, dOdy, dDdx, dDdy);
            }
            else    // Not using a rasterized Gbuffer
            {
                float3 dDdx, dDdy;
                const float3 rayOrg = gScene.camera.getPosition();
                const float hitT = length(rayOrg - worldPos.xyz) * 1;

                getRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, launchDim, dDdx, dDdy);

                rayData.rayDiff = RayDiff.create(float3(0.0), float3(0.0), dDdx, dDdy); // init ray diff. dOdx = 0 , dOdy = 0, and the directions are from above.

                rayData.rayDiff = rayData.rayDiff.propagate(worldPos.xyz, rayDir, hitT, faceNormal); // propagate the ray differential to the current hit point
                HitInfo hit;
           
                if (hit.decode(gVBuffer[launchIndex]))
                {
                    const float4x4 worldMat = gScene.getWorldMatrix(hit.meshInstanceID);
                    const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(hit.meshInstanceID);
                    const uint3 vtxIndices = gScene.getIndices(hit.meshInstanceID, hit.primitiveIndex);
                    const float3 barycentrics = hit.getBarycentricWeights();
                    float3 unnormalizedN;
                    float3 vtxs[3];
                    float3 normals[3];
                    float2 txcoords[3];

                    // Get relevant vertex data and transform to world space
                    StaticVertexData vtx[3] = { gScene.getVertex(vtxIndices[0]), gScene.getVertex(vtxIndices[1]), gScene.getVertex(vtxIndices[2]) };

                    vtxs[0] = mul(float4(vtx[0].position, 1.0), worldMat).xyz;
                    vtxs[1] = mul(float4(vtx[1].position, 1.0), worldMat).xyz;
                    vtxs[2] = mul(float4(vtx[2].position, 1.0), worldMat).xyz;
                    normals[0] = mul(vtx[0].normal, worldInvTransposeMat);
                    normals[1] = mul(vtx[1].normal, worldInvTransposeMat);
                    normals[2] = mul(vtx[2].normal, worldInvTransposeMat);
                    normals[0] = normalize(normals[0]);
                    normals[1] = normalize(normals[1]);
                    normals[2] = normalize(normals[2]);
                    unnormalizedN = normals[0] * barycentrics[0];
                    unnormalizedN += normals[1] * barycentrics[1];
                    unnormalizedN += normals[2] * barycentrics[2];
                    txcoords[0] = vtx[0].texCrd;
                    txcoords[1] = vtx[1].texCrd;
                    txcoords[2] = vtx[2].texCrd;

                    float2 dBarydx, dBarydy;
                    float2 dUVdx, dUVdy;
                    float3 e1 = vtxs[1] - vtxs[0];
                    float3 e2 = vtxs[2] - vtxs[0];
                    float3 normalizedGeometricNormal = normalize(cross(vtxs[1] - vtxs[0], vtxs[2] - vtxs[0]));

                    computeDifferentialsBarysAndUVs(rayData.rayDiff, rayDir, vtxs, txcoords, e1, e2, normalizedGeometricNormal, hitT, dBarydx, dBarydy, dUVdx, dUVdy);
                    reflectRayDifferential(rayData.rayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, e1, e2, normals);
               }
                else
                {
                    // TODO TAM: make it work for environment maps. Chapter 21 in Ray Tracing Gems.
                    rayData.rayDiff = RayDiff.create(float3(0), float3(0), float3(0), float3(0));
                }
            }
        }

        if(sd.linearRoughness <= 0.19)      // Only trace reflection rays when material is highly reflective
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
        outColor = kUseEnvBackground ? evalEnvProbe(gEnvProbe, rayDir) : kDefaultBackgroundColor;
    }

    gOutputColor[launchIndex] = float4(outColor, 1);
}
