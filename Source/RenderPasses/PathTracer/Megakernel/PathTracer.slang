/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/

/** Path tracer core functions.
    These are called from the raygen, hit, and miss programs.

    To use it import PathTracer.slang and instantiate ParameterBlock<PathTracerData>,
    which is passed in as the first argument to all functions that need it.
    Any resources placed in PathTracerData are bound once and shared between all shaders/instances.
*/

#include "Utils/Math/MathConstants.slang"

__exported import Scene;
__exported import Shading;
__exported import Raytracing;
__exported import RenderPasses.PathTracer.Logging;
__exported import RenderPasses.PathTracer.Megakernel.RayData;
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

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    uint rayFlags = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeShadow /* hitIdx */, hitProgramCount, kRayTypeShadow /* missIdx */, ray, rayData);

    // We let TraceRay() execute even when !valid in order to run the miss shader.
    // This is faster on current drivers thanks to reduced divergence.
    if (!valid) return false;

    // Note we're not counting stats for dummy rays. TODO: Make a separate bin in the logger.
    logTraceRay();

    return rayData.visible;
}

/** Traces a scatter ray based on ray parameters stored in the ray payload.
    \param[in,out] rayData Describes the ray parameters. The struct is modified based on the result.
*/
void traceScatterRay(inout ScatterRayData rayData)
{
    // Setup ray based on params passed via payload.
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    // Pack ray payload.
    ScatterRayDataPacked packedData;
    packedData.pack(rayData);
    uint rayFlags = 0; // TODO: Set cull mode from the app
    TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, hitProgramCount, kRayTypeScatter /* missIdx */, ray, packedData);

    logTraceRay();

    // Unpack ray payload.
    rayData = packedData.unpack();
}

/** Evaluates direct illumination at hit point.
    \param[in] pt PathTracer data.
    \param[in] sd Shading data.
    \param[in] rayOrigin Ray origin for the new ray.
    \param[in,out] sg SampleGenerator object.
    \return Outgoing radiance in view direction due to direct illumination.
*/
float3 evalDirect(const PathTracerData pt, const ShadingData sd, const float3 rayOrigin, inout SampleGenerator sg)
{
    // Draw a light sample. This chooses stochastically between all enabled lights in the scene.
    SceneLightSample ls = sampleSceneLights(pt.params, pt.envProbe, pt.emissiveSampler, sd, rayOrigin, sg);
    const bool valid = any(ls.Li > 0.f);

    // Test visibility by tracing a shadow ray.
    bool V = traceShadowRay(rayOrigin, ls.rayDir, ls.rayDistance, valid);
    return V ? evalBRDFCosine(sd, ls.dir) * ls.Li : float3(0);
}

/** Processes a hit point to generate a shadow ray for sampling the light sources.
    This should be called before generateScatterRay() as the latter updates the path throughput.
    \param[in] pt PathTracer data.
    \param[in] sd Shading data.
    \param[in] rayOrigin Ray origin for the new ray.
    \param[in,out] rayData Ray payload.
*/
void generateShadowRay(const PathTracerData pt, const ShadingData sd, const float3 rayOrigin, inout ScatterRayData rayData)
{
    // Sample the scene lights.
    SceneLightSample ls = sampleSceneLights(pt.params, pt.envProbe, pt.emissiveSampler, sd, rayOrigin, rayData.sg);
    const bool valid = any(ls.Li > 0.f);

    // Return ray parameters and incident radiance, weighted by path throughput.
    rayData.shadowRay = float4(ls.rayDir, ls.rayDistance);
    rayData.Lr = valid ? evalBRDFCosine(sd, ls.dir) * ls.Li * rayData.thp : float3(0);
}

/** Processes a hit point to generate a scatter ray or terminate.
    The function uses BRDF sampling to generate ray parameters for the scatter ray.
    It's called from the raygen shader for primary hits and from the closest hit shader for secondary hits.
    \param[in] pt PathTracer data.
    \param[in] sd Shading data.
    \param[in] rayOrigin Ray origin for the new ray.
    \param[in,out] rayData Ray payload.
*/
void generateScatterRay(const PathTracerData pt, const ShadingData sd, const float3 rayOrigin, inout ScatterRayData rayData)
{
    // Generate next path segment.
    rayData.origin = rayOrigin;
    if (kUseBRDFSampling)
    {
        // Default path that uses BRDF importance sampling.
        BRDFSample result;
        sampleBRDF(sd, rayData.sg, result);

        rayData.direction = result.dir;
        rayData.thp *= result.thp;

        // Note that the pdf can be costly, so we store it in the payload if we need it (MIS enabled),
        // and rely on dead code elimination to remove the computations otherwise.
        rayData.pdf = kUseMIS ? result.pdf : 0.f;
    }
    else
    {
        // Fallback path that uses cosine-weighted hemisphere sampling.
        float pdf;
        float3 dir = sampleHemisphereCosine(sd, sampleNext2D(rayData.sg), pdf); // pdf = cos(theta) / pi

        // Check that L and V are in the positive hemisphere, reset pdf to zero otherwise.
        // This is necessary for consistency with BRDF sampling.
        float NdotL = dot(sd.N, dir);
        if (min(sd.NdotV, NdotL) < kMinCosTheta) pdf = 0.f;

        rayData.direction = dir;
        rayData.thp *= evalBRDF(sd, dir) * M_PI; // dot(N,L) / pdf = pi
        rayData.pdf = kUseMIS ? pdf : 0.f;
    }

    assert(!any(isnan(rayData.thp)));
    assert(!isnan(rayData.pdf));

    // Pass on the shading normal. Only store it if needed (MIS enabled).
    if (kUseMIS) rayData.normal = sd.N;

    // Terminate if ray throughput is zero.
    // This may happen if the sample is invalid, and consequently the BRDF sampling function returns thp == 0.0.
    if (all(rayData.thp == 0.f)) rayData.terminated = true;
}

/** Handles the case when a path misses the scene.
    Depending on the configuration, we add the env map contribution here using MIS.
    \param[in] pt PathTracer data.
    \param[in,out] rayData Ray payload.
*/
void handleMiss(const PathTracerData pt, inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the path as terminated.
    rayData.terminated = true;

    // If we have an environment map and MIS is enable, add the weighted contribution here.
    if (kUseEnvLight && kUseMIS)
    {
        // We came here through BRDF sampling. The other sampling strategy is
        // env map sampling. Evaluate it's probability for the current ray dir.
        const float3 dir = rayData.direction;
        float lightPdf = evalEnvProbePdf(pt.envProbe, dir) * getEnvLightSelectionPdf();

        // Compute MIS weighted contribution from the environment map.
        float misWeight = evalMIS(pt.params, rayData.pdf, lightPdf);
        float3 Le = evalEnvProbe(pt.envProbe, dir) * misWeight;

        // Return emitted radiance weighted by path throughput.
        rayData.Le = rayData.thp * Le;
    }
}

/** Handles the case when a path hits a surface.
    Depending on the configuration, we add the emissive contribution here using MIS.
    Then direct illumination is sampled and the next segment of the path generated.
    \param[in] pt PathTracer data.
    \param[in] sd Shading data.
    \param[in] hit Triangle hit data (for EmissiveLightSampler).
    \param[in] computeIndirect True if indirect illumination is enabled.
    \param[in,out] rayData Ray payload.
*/
void handleHit(const PathTracerData pt, const ShadingData sd, const TriangleHit hit, const bool computeIndirect, inout ScatterRayData rayData)
{
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

            // Evaluate PDF at the hit, had it been generated with light sampling.
            // Note that there is a minor discrepancy compared to BRDF sampling at the previous path vertex:
            // the normal passed through the ray payload is lossily compressed to reduce the ray payload size.
            float lightPdf = pt.emissiveSampler.evalPdf(rayData.origin, rayData.normal, hit) * getEmissiveLightSelectionPdf();

            // Compute MIS weight by combining this with BRDF sampling.
            // Note we can assume rayData.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = evalMIS(pt.params, rayData.pdf, lightPdf);
        }

        // Return emitted radiance weighted by path throughput and MIS weight.
        rayData.Le = rayData.thp * sd.emissive * misWeight;
    }

    // Unless path ends here, sample light and generate next path segment.
    // TODO: We should try running a different shader for the last path segment to see if that's faster than branching out here.
    if (computeIndirect && rayData.pathLength < kMaxBounces)
    {
        // Compute ray origin for new rays spawned from the hit.
        const float3 rayOrigin = sd.computeNewRayOrigin();

        // Generate shadow ray for sampling the scene lights.
        // The raygen shader will trace the shadow ray and accumulate the contribution.
        generateShadowRay(pt, sd, rayOrigin, rayData);

        // Generate the ray parameters for the next path segment.
        // If path is not terminated, the raygen shader will continue the path based on the returned payload.
        generateScatterRay(pt, sd, rayOrigin, rayData);

        rayData.pathLength++;
    }
}
