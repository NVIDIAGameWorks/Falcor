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
__exported import Utils.Debug.PixelDebug;

/** Shared path tracer data.
    The ray tracing program instantiates this and passes it to all functions.
*/
struct PathTracerData
{
    PathTracerParams        params;             ///< PathTracer shared parameters.
    EnvProbe                envProbe;           ///< Environment map sampling functions.
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
#if USE_NESTED_DIELECTRICS
    NestedRayData nestedData;
#endif
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
    TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeShadow /* hitIdx */, hitProgramCount, kRayTypeShadow /* missIdx */, ray, rayData);

    // We let TraceRay() execute even when !valid in order to run the miss shader.
    // This is faster on current drivers thanks to reduced divergence.
    if (!valid) return false;

    // Note we're not counting stats for dummy rays. TODO: Make a separate bin in the logger.
    logTraceRay();

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
    rayData.packedHitInfo = { kInvalidIndex }; // Set the first component to kInvalidIndex
#if USE_NESTED_DIELECTRICS
    rayData.nestedData.init(interiorList, kRayTMax);
#endif

    uint rayFlags = RAY_FLAG_NONE; // TODO: Set cull mode from the app
    TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, hitProgramCount, kRayTypeScatter /* missIdx */, ray, rayData);

#if USE_NESTED_DIELECTRICS
    rayData.nestedData.commit(interiorList);
#endif

    logTraceRay();

    if (rayData.packedHitInfo.x != kInvalidIndex)
    {
        hitInfo.decode(rayData.packedHitInfo);
        return true;
    }

    return false;
}

/** Handle scatter ray hit.
    \param[in] pt PathTracer data.
    \param[in] sd Shading data.
    \param[in] path Path data.
*/
void handleHit(const PathTracerData pt, inout ShadingData sd, inout PathData path)
{
    // Evaluate Falcor's material parameters at the hit point.
    // TODO: Implement texLOD to enable texture filtering in prepareShadingData().
    VertexData v = gScene.getVertexData(path.hit);
    const uint materialID = gScene.getMaterialID(path.hit.meshInstanceID);
    sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -path.dir, 0.f);

    if (kUseNestedDielectrics)
    {
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

    // Compute tangent space if it is invalid.
    // TODO: Remove this check when we're sure the tangent space is always valid.
    if (!(dot(sd.B, sd.B) > 0.f))   // Note: Comparison written so that NaNs trigger
    {
        sd.B = perp_stark(sd.N);
        sd.T = cross(sd.B, sd.N);
    }

    // Statically determine if we need to compute the emissive based on the current configuration.
    // It's only needed if emissive is enabled, and its full contribution hasn't been sampled elsewhere.
    const bool computeEmissive = kUseEmissiveLights && (!kUseEmissiveSampler || (kUseEmissiveSampler && kUseMIS));

    if (computeEmissive && any(sd.emissive > 0.f))
    {
        float misWeight = 1.f;
        if (kUseEmissiveSampler && kUseMIS)
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
    \param[in] path Path data.
*/
void handleMiss(const PathTracerData pt, inout PathData path)
{
    // If we have an environment map and MIS is enable, add the weighted contribution here.
    if (kUseEnvLight && kUseMIS)
    {
        // We came here through BRDF sampling. The other sampling strategy is
        // env map sampling. Evaluate it's probability for the current ray dir.
        float lightPdf = evalEnvProbePdf(pt.envProbe, path.dir) * getEnvLightSelectionPdf();

        // Compute MIS weighted contribution from the environment map.
        float misWeight = evalMIS(pt.params, path.pdf, lightPdf);
        float3 Le = evalEnvProbe(pt.envProbe, path.dir) * misWeight;

        // Return emitted radiance weighted by path throughput.
        path.L += path.thp * Le;
    }
}

void evalDirect(const PathTracerData pt, ShadingData sd, inout PathData path)
{
    for (uint i = 0; i < kLightSamplesPerVertex; ++i)
    {
        ShadowRay shadowRay = {};
        bool valid = generateShadowRay(pt.params, pt.envProbe, pt.emissiveSampler, sd, i, path, shadowRay);
        bool visible = traceShadowRay(path.origin, shadowRay.rayParams.xyz, shadowRay.rayParams.w, valid);
        path.L += visible ? shadowRay.Lr : float3(0.f);
    }
}

void tracePath(const PathTracerData pt, ShadingData sd, inout PathData path)
{
    // Always output directly emitted light from the primary hit.
    // This is independent of whether emissive materials are treated as light sources or not.
    path.L += sd.emissive;

    // Compute direct illumination.
    evalDirect(pt, sd, path);

    // Compute iteration count.
    // This is statically determined based on the current configuration, is MIS/emissive is needed at the last bounce etc.
    const uint kIterationCount = kTraceScatterRayFromLastPathVertex ? kMaxBounces + 1 : kMaxBounces;

    // Note: Do not unroll loop even though iteration count is statically known,
    // as the shader gets very large and runs slower on current generation GPUs.
    [loop]
    for (uint depth = 0; depth < kIterationCount; depth++)
    {
        // Russian roulette to stochastically terminate the path.
        // We use a fixed absorption probability for now.
        // TODO: Better strategy, e.g., 1-P(absorption) = hemispherical reflectance of BRDF.
        if (kUseRussianRoulette)
        {
            float u = sampleNext1D(path.sg);
            if (u < pt.params.probabilityAbsorption) break;
            path.thp /= (1.f - pt.params.probabilityAbsorption);
        }

        // Generate next path segment.
        if (!generateScatterRay(sd, path)) return;

        if (path.isTransmission())
        {
            path.origin = path.transmissionOrigin;
            if (kUseNestedDielectrics)
            {
                uint nestedPriority = gScene.materials[sd.materialID].getNestedPriority();
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
            }
        }

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

        // Compute direct illumination.
        bool isInsideVolume = kUseNestedDielectrics && path.interiorList.isEmpty() == false;
        if (depth < kMaxBounces && (kUseLightSamplesInVolumes || !isInsideVolume))
        {
            evalDirect(pt, sd, path);
        }
    }
}
