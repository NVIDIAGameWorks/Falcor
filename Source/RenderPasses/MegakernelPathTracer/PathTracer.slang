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

/** Path tracer core functions.
    These are called from the raygen programs.

    To use it import PathTracer.slang and instantiate ParameterBlock<PathTracerData>,
    which is passed in as the first argument to all functions that need it.
    Any resources placed in PathTracerData are bound once and shared between all shaders/instances.
*/

#include "Utils/Math/MathConstants.slangh"

__exported import Scene.Scene;
__exported import Scene.Raytracing;
__exported import Scene.HitInfo;
__exported import RenderPasses.Shared.PathTracer.PixelStats;
__exported import RenderPasses.Shared.PathTracer.PathTracerHelpers;
__exported import RenderPasses.Shared.PathTracer.InteriorListHelpers;
__exported import RenderPasses.Shared.PathTracer.RayFootprint;
__exported import Utils.Debug.PixelDebug;

static const bool kForceOpaque = !kUseAlphaTest;

/** Shared path tracer data.
    The ray tracing program instantiates this and passes it to all functions.
*/
struct PathTracerData
{
    PathTracerParams        params;             ///< PathTracer shared parameters.
    EnvMapSampler           envMapSampler;      ///< Environment map sampler.
    EmissiveLightSampler    emissiveSampler;    ///< Emissive light sampler.
};

/** Payload for shadow ray.
*/
struct ShadowRayData
{
    bool visible;
};

/** Payload for scatter ray.
*/
struct ScatterRayData
{
    uint2 packedHitInfo;        ///< Packed HitInfo data, or kInvalidIndex in the first component if ray missed.
};

/** Traces a shadow ray towards a light source.
    \param[in] origin Ray origin for the shadow ray.
    \param[in] dir Direction from ray origin towards the light source (normalized).
    \param[in] distance Distance to the light source.
    \param[in] valid True if ray should be traced, false for dummy rays.
    \return True if light is visible, false otherwise.
*/
bool traceShadowRay(float3 origin, float3 dir, float distance, bool valid = true)
{
    // Setup ray descriptor.
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = valid ? distance : 0.f; // Set tmax = 0.0 for dummy rays.

    ShadowRayData rayData = { false }; // Set to true by miss shader if ray is not terminated before
    uint rayFlags = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    if (kForceOpaque) rayFlags |= RAY_FLAG_FORCE_OPAQUE;
    TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeShadow /* hitIdx */, hitProgramCount, kRayTypeShadow /* missIdx */, ray, rayData);

    // We let TraceRay() execute even when !valid in order to run the miss shader.
    // This is faster on current drivers thanks to reduced divergence.
    if (!valid) return false;

    // Note we're not counting stats for dummy rays. TODO: Make a separate bin in the logger.
    logTraceRay(PixelStatsRayType::Shadow);

    return rayData.visible;
}

/** Traces a scatter ray.
    \param[in] origin Ray origin for the shadow ray.
    \param[in] dir Direction from ray origin towards the light source (normalized).
    \param[in,out] interiorList Interior list for handling nested dielectrics.
    \param[out] hitInfo Hit information
    \return True if scatter ray hit something, false otherwise.
*/
bool traceScatterRay(float3 origin, float3 dir, inout InteriorList interiorList, out HitInfo hitInfo)
{
    // Setup ray based on params passed via payload.
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    ScatterRayData rayData;

    // For nested dielectrics, we potentially have to trace additional rays after false intersections.
    while (true)
    {
        rayData.packedHitInfo = { HitInfo::kInvalidIndex }; // Set the first component to kInvalidIndex

        uint rayFlags = RAY_FLAG_NONE; // TODO: Set cull mode from the app
        if (kForceOpaque) rayFlags |= RAY_FLAG_FORCE_OPAQUE;
        TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, hitProgramCount, kRayTypeScatter /* missIdx */, ray, rayData);

        logTraceRay(PixelStatsRayType::ClosestHit);

        // Check for false intersections.
        if (kUseNestedDielectrics && rayData.packedHitInfo.x != HitInfo::kInvalidIndex)
        {
            HitInfo hitInfo;
            hitInfo.decode(rayData.packedHitInfo);
            uint materialID = gScene.getMaterialID(hitInfo.meshInstanceID);
            uint nestedPriority = gScene.materials[materialID].getNestedPriority();
            if (!interiorList.isTrueIntersection(nestedPriority))
            {
                VertexData v = gScene.getVertexData(hitInfo);
                bool frontFacing = dot(-ray.Direction, v.faceNormalW) >= 0.f;
                interiorList.handleIntersection(materialID, nestedPriority, frontFacing);
                ray.Origin = computeRayOrigin(v.posW, frontFacing ? -v.faceNormalW : v.faceNormalW);
                continue;
            }
        }

        break;
    }

    if (rayData.packedHitInfo.x != HitInfo::kInvalidIndex)
    {
        hitInfo.decode(rayData.packedHitInfo);
        return true;
    }

    return false;
}

// Need to be kept alive until next scatter event for ray footprint to be updated after the ray bounces.
static StaticVertexData triangleVertices[3];
static VertexData v;

/** Handle scatter ray hit.
    \param[in] pt PathTracer data.
    \param[in] sd Shading data.
    \param[in,out] path Path data.
*/
void handleHit(const PathTracerData pt, inout ShadingData sd, inout PathData path)
{
    // Get vertex data for current hit point. VertexData and triangleVertices kept alive out of handleHit() to be used by rayFootprint.bounceOnSurface() if needed.
    v = gScene.getVertexData(path.hit, triangleVertices);

    // Update the ray footprint with current hit surface.
    path.rayFootprint.hitSurface(v, triangleVertices, path.hit, path.origin, path.dir);

    // Evaluate Falcor's material parameters at the hit point using the current ray footprint mode and doing texLOD.
    sd = prepareShadingData(v, path.rayFootprint, triangleVertices, path.hit, path.origin, path.dir);

    // Compute tangent space if it is invalid.
    if (!(dot(sd.T, sd.T) > 0.f))   // Note: Comparison written so that NaNs trigger
    {
        sd.T = perp_stark(sd.N);
        sd.B = cross(sd.N, sd.T);
    }

    if (kUseNestedDielectrics)
    {
        // Compute relative index of refraction at interface.
        sd.eta = computeRelativeIoR(path.interiorList, sd.materialID, sd.frontFacing);

        // Handle volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint interiorMaterialID = path.interiorList.getTopMaterialID();
            float3 sigmaA = gScene.materials[interiorMaterialID].volumeAbsorption;
            float distance = length(path.origin - v.posW);
            path.thp *= exp(-sigmaA * distance);
        }
    }

    // Disable specular lobes if caustics are disabled and path already contains non-specular vertex.
    if (kDisableCaustics && path.nonSpecularBounces > 0 && sd.linearRoughness <= pt.params.specularRoughnessThreshold)
    {
        sd.setActiveLobes((uint)LobeType::Diffuse);
    }

    // Check if the scatter event is samplable by the light sampling technique.
    const bool isLightSamplable = path.isLightSamplable();

    // Determine if we need to compute the emissive based on the current configuration.
    // It's only needed if emissive is enabled, and its full contribution hasn't been sampled elsewhere.
    const bool computeEmissive = kUseEmissiveLights && (kUseLightsInVolumes || !path.isInsideVolume()) && (!kUseNEE || kUseMIS || !isLightSamplable);

    if (computeEmissive && any(sd.emissive > 0.f))
    {
        float misWeight = 1.f;
        if (kUseNEE && kUseMIS && isLightSamplable)
        {
            // If emissive light sampling and MIS is enabled, we've already sampled emissive lights using NEE.
            // We need to evaluate the MIS weight here to account for the remaining contribution.

            // Prepare hit point struct with data needed for emissive light PDF evaluation.
            TriangleHit hit;
            hit.triangleIndex = gScene.lightCollection.getTriangleIndex(path.hit.meshInstanceID, path.hit.primitiveIndex);
            hit.posW = sd.posW;
            hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

            // Evaluate PDF at the hit, had it been generated with light sampling.
            float lightPdf = pt.emissiveSampler.evalPdf(path.origin, path.normal, hit) * getEmissiveLightSelectionPdf();

            // Compute MIS weight by combining this with BRDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = evalMIS(pt.params, 1, path.pdf, kLightSamplesPerVertex, lightPdf);
        }

        // Accumulate emitted radiance weighted by path throughput and MIS weight.
        path.L += path.thp * sd.emissive * misWeight;
    }

    path.length++;

    // Compute the ray origin for the next path segment.
    // The origin is offset slightly from the hit to avoid self-intersections.
    path.origin = sd.computeNewRayOrigin();
}

/** Handle scatter ray miss.
    \param[in] pt PathTracer data.
    \param[in,out] path Path data.
*/
void handleMiss(const PathTracerData pt, inout PathData path)
{
    // Check if the scatter event is samplable by the light sampling technique.
    const bool isLightSamplable = path.isLightSamplable();

    // If we have an environment, add it's weighted contribution here.
    if (kUseEnvLight && (kUseLightsInVolumes || !path.isInsideVolume()) && (!kUseNEE || kUseMIS || !isLightSamplable))
    {
        float misWeight = 1.f;
        if (kUseNEE && kUseMIS && isLightSamplable)
        {
            // We came here through BRDF sampling. The other sampling strategy is
            // env map sampling. Evaluate it's probability for the current ray dir.
            float lightPdf = pt.envMapSampler.evalPdf(path.dir) * getEnvLightSelectionPdf();

            // Compute MIS weighted contribution from the environment map.
            misWeight = evalMIS(pt.params, 1, path.pdf, kLightSamplesPerVertex, lightPdf);
        }

        float3 Le = pt.envMapSampler.eval(path.dir);

        // Accumulate emitted radiance weighted by path throughput.
        path.L += path.thp * Le * misWeight;
    }
}

void evalDirect(const PathTracerData pt, ShadingData sd, inout PathData path)
{
    for (uint i = 0; i < kLightSamplesPerVertex; ++i)
    {
        ShadowRay shadowRay = {};
        bool valid = generateShadowRay(pt.params, pt.envMapSampler, pt.emissiveSampler, sd, i, path, shadowRay);
        bool visible = traceShadowRay(path.origin, shadowRay.rayParams.xyz, shadowRay.rayParams.w, valid);
        path.L += visible ? shadowRay.Lr : float3(0.f);
    }
}

void tracePath(const PathTracerData pt, ShadingData sd, inout PathData path)
{
    // Always output directly emitted light from the primary hit.
    // This is independent of whether emissive materials are treated as light sources or not.
    path.L += sd.emissive;

    // Compute iteration count.
    // We always add one bounce for handling surfaces that cannot be handled using light sampling.
    const uint kIterationCount = kMaxBounces + 1;

    // Note: Do not unroll loop even though iteration count is statically known,
    // as the shader gets very large and runs slower on current generation GPUs.
    [loop]
    for (uint depth = 0; depth < kIterationCount; depth++)
    {
        // Determine if BSDF supports sampling with next-event estimation.
        // The following line generates bad code:
        // bool supportsNEE = (getBSDFLobes(sd) & (uint)LobeType::NonDeltaReflection) != 0;
        uint lobes = getBSDFLobes(sd);
        bool supportsNEE = (lobes & (uint)LobeType::DiffuseReflection) != 0 || (lobes & (uint)LobeType::SpecularReflection) != 0;

        // Compute direct illumination.
        if (kUseNEE && supportsNEE && (kUseLightsInVolumes || !path.isInsideVolume()))
        {
            evalDirect(pt, sd, path);
        }

        // Russian roulette to stochastically terminate the path.
        // We use a fixed absorption probability for now.
        // TODO: Better strategy, e.g., 1-P(absorption) = hemispherical reflectance of BRDF.
        if (kUseRussianRoulette)
        {
            float u = sampleNext1D(path.sg);
            if (u < pt.params.probabilityAbsorption) break;
            path.thp /= (1.f - pt.params.probabilityAbsorption);
        }

        // Save incoming ray direction for ray footprint.
        const float3 rayDirIn = path.dir;

        // Generate next path segment.
        if (!generateScatterRay(pt.params, sd, path)) return;

        // Scatter the ray footprint out of the surface. Primary bounce is handled at footprint creation time.
        if (depth > 0)
        {
            path.rayFootprint.bounceOnSurface(v, triangleVertices, path.hit, sd, v.normalW, rayDirIn, path.dir, path.isSpecular(), path.isTransmission());
        }

        // Update ray origin on transmissive events.
        if (path.isTransmission()) path.origin = path.transmissionOrigin;

        // Check if this is the last path vertex.
        const bool isLastVertex = path.length == kMaxBounces || path.nonSpecularBounces > kMaxNonSpecularBounces;

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (isLastVertex && (kUseNEE && !kUseMIS && path.isLightSamplable())) return;

        // Terminate caustics paths.
        if (kDisableCaustics && path.isSpecular() && path.nonSpecularBounces > 0) return;

        // Trace scatter ray.
        if (traceScatterRay(path.origin, path.dir, path.interiorList, path.hit))
        {
            handleHit(pt, sd, path);
        }
        else
        {
            handleMiss(pt, path);
            break;
        }

        // Terminate after handling the last path vertex.
        if (isLastVertex) return;
    }
}
