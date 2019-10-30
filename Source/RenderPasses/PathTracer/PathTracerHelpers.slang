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
#include "Utils/Math/MathConstants.slang"
#include "RenderPasses/PathTracer/PathTracerParams.h"

// TODO: Which ones need __exported
import Scene;
import ShaderCommon;
__exported import Helpers; // For computeRayOrigin()
__exported import Utils.Math.MathHelpers;
__exported import Experimental.Scene.Lights.EnvProbe;
__exported import Experimental.Scene.Material.MaterialShading;
__exported import Experimental.Scene.Lights.EmissiveLightSampler;
__exported import Experimental.Scene.Lights.LightHelpers;
__exported import RenderPasses.PathTracer.PathData;

static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

// Logic for determining if we need to trace a scatter ray from the last path vertex.
// This is needed to account for direct illumination when NEE with MIS is enabled, or NEE is disabled.
// Note that these decisions are all static, no dynamic code is generated.
static const bool kTraceScatterRayFromLastPathVertex =
    (kUseEnvLight && kUseMIS) ||
    (kUseEmissiveLights && (!kUseEmissiveSampler || kUseMIS));


/** A light sample for any of the scene lights (analytic, envmap, and emissive).
    The Li field is nonzero only if the sample is valid (no need to check the pdf).
*/
struct SceneLightSample
{
    // Light sample
    float3  dir;            ///< Direction from the shading point to the light sample in world space (normalized). This is used for BRDF evaluation.
    float   distance;       ///< Distance from the shading point to the light sample.
    float3  Li;             ///< Incident radiance at the shading point (unshadowed). Note: Already divided by the pdf and multiplied by MIS weight.
    float   pdf;            ///< Probability density function with respect to solid angle (pdf == 0 for invalid samples).

    // Shadow ray parameters
    float3  rayDir;         ///< Ray direction for visibility evaluation (normalized). This may differ from 'dir' due to ray offsets.
    float   rayDistance;    ///< Ray distance for visibility evaluation. This may differ from 'distance' due to ray offsets.
};

/** Evaluates the background in a particular direction.
    This function should be called for screen-space samples that missed the scene.
    The result should not be used as lighting, use evalDistantLight() instead.
    \param[in] envProbe EnvProbe data.
    \param[in] dir World-space direction (unnormalized).
    \return Color (rgb).
*/
float3 evalBackground(const EnvProbe envProbe, float3 dir)
{
    return kUseEnvBackground ? evalEnvProbe(envProbe, dir) : kDefaultBackgroundColor;
}

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    This version assumes one sample is taken from each of the sampling strategies.
    \param[in] params PathTracer parameters.
    \param[in] p0 Pdf for the first sampling strategy.
    \param[in] p1 Pdf for the second sampling strategy.
    \return Weight for the contribution from the first strategy (p0).
*/
float evalMIS(const PathTracerParams params, float p0, float p1)
{
    switch (kMISHeuristic)
    {
        case MISHeuristic.BalanceHeuristic:
            return p0 / (p0 + p1);
        case MISHeuristic.PowerTwoHeuristic:
            return p0 * p0 / (p0 * p0 + p1 * p1);
        case MISHeuristic.PowerExpHeuristic:
            float q0 = pow(p0, params.misPowerExponent);
            float q1 = pow(p1, params.misPowerExponent);
            return q0 / (q0 + q1);
        default:
            return 0.f;
    }
}

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    \param[in] params PathTracer parameters.
    \param[in] n0 Number of samples taken from the first sampling strategy.
    \param[in] p0 Pdf for the first sampling strategy.
    \param[in] n1 Number of samples taken from the second sampling strategy.
    \param[in] p1 Pdf for the second sampling strategy.
    \return Weight for the contribution from the first strategy (p0).
*/
float evalMIS(const PathTracerParams params, float n0, float p0, float n1, float p1)
{
    switch (kMISHeuristic)
    {
        case MISHeuristic.BalanceHeuristic:
        {
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic.PowerTwoHeuristic:
        {
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic.PowerExpHeuristic:
        {
            float q0 = pow(n0 * p0, params.misPowerExponent);
            float q1 = pow(n1 * p1, params.misPowerExponent);
            return q0 / (q0 + q1);
        }
        default:
            return 0.f;
    }
}

/** Returns the probability of selecting env map light sampling.
*/
float getEnvLightSelectionPdf()
{
    float p0 = kUseEnvLight ? 1.f : 0.f;
    float p1 = kUseAnalyticLights ? 1.f : 0.f;
    float p2 = kUseEmissiveSampler ? 1.f : 0.f;
    return p0 / (p0 + p1 + p2);
}

/** Returns the probability of selecting emissive light sampling.
*/
float getEmissiveLightSelectionPdf()
{
    float p0 = kUseEnvLight ? 1.f : 0.f;
    float p1 = kUseAnalyticLights ? 1.f : 0.f;
    float p2 = kUseEmissiveSampler ? 1.f : 0.f;
    return p2 / (p0 + p1 + p2);
}

/** Evaluates the probability density function for the BRDF sampling strategy used for the scatter ray.
    \param[in] sd Describes the shading point.
    \param[in] dir The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfScatter(const ShadingData sd, const float3 dir)
{
    if (kUseBRDFSampling)
    {
        return evalPdfBRDF(sd, dir);
    }
    else // Cosine-weighted sampling
    {
        // Check the dot products. The sampling probability for back-facing directions is zero.
        float NdotL = dot(sd.N, dir);
        if (min(sd.NdotV, NdotL) < kMinCosTheta) return 0.f;

        return NdotL * M_1_PI; // pdf = cos(theta) / pi
    }
}

/** Samples a light source in the scene.
    This function first stochastically selects a type of light source to sample,
    and then calls that the sampling function for the chosen light type.
    \param[in] params PathTracer parameters.
    \param[in] envProbe EnvProbe data.
    \param[in] emissiveSampler Emissive light sampler.
    \param[in] sd Shading data.
    \param[in] rayOrigin Ray origin for the shadow ray.
    \param[in,out] sg SampleGenerator object.
    \param[in] numSamples Total number of light samples that will be taken.
    \return Generated light sample (if pdf=0.0 the sample is invalid, but it is sufficient and possibly more performance to test for Li!=(0,0,0)).
*/
SceneLightSample sampleSceneLights(const PathTracerParams params, const EnvProbe envProbe, const EmissiveLightSampler emissiveSampler, const ShadingData sd, const float3 rayOrigin, inout SampleGenerator sg, const uint numSamples = 1)
{
    SceneLightSample ls = {};

    // Set relative probabilities of the different sampling techniques.
    // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
    // TODO: Refactor the selection pdf code the helpers above for evaluating the individual probabilities.
    float p[3];
    p[0] = kUseEnvLight ? 1.f : 0.f;
    p[1] = kUseAnalyticLights ? 1.f : 0.f;
    p[2] = kUseEmissiveSampler ? 1.f : 0.f;

    // Normalize probabilities. Early out if zero.
    float sum = p[0] + p[1] + p[2];
    if (sum == 0.f) return ls;
    float invSum = 1.f / sum;
    p[0] *= invSum;
    p[1] *= invSum;
    p[2] *= invSum;

    // Sample based on uniform random number. Rescale u to [0,1) afterwards.
    float invPdf = 0.f;
    float u = sampleNext1D(sg);

    // We use explicit checks for which light types are enabled so that the compiler
    // can remove the unused code. It won't otherwise since u is unknown at compile time.
    // Note: We're using valid flag to avoid conditional return statements that cause incorrect warnings when compiling for fxc (SM 5.1).
    bool valid = false;

    if (kUseEnvLight)
    {
        if (u < p[0])
        {
            float selectionPdf = p[0];

            // Sample environment map.
            EnvProbeSamplingResult lightSample;
            sampleEnvProbe(envProbe, sampleNext2D(sg), lightSample);

            // Reject sample if lower hemisphere.
            if (dot(sd.N, lightSample.wi) < kMinCosTheta) return ls;

            // Evaluate emitted radiance from the sampled direction.
            // TODO: Move this into EnvProbe.slang
            float3 Le = evalEnvProbe(envProbe, lightSample.wi);

            // Evaluate MIS with BRDF sampling as the other sampling strategy.
            float pdf = selectionPdf * lightSample.pdf;
            float misWeight = 1.f;
            if (kUseMIS && pdf > 0.f)
            {
                float brdfPdf = evalPdfScatter(sd, lightSample.wi);
                misWeight = evalMIS(params, numSamples, pdf, 1, brdfPdf);
            }

            // Setup returned sample.
            ls.rayDir = ls.dir = lightSample.wi;
            ls.rayDistance = ls.distance = kRayTMax;
            ls.pdf = pdf;
            if (pdf > 0.f) ls.Li = Le * misWeight / (pdf * numSamples);

            valid = true;
        }
        u -= p[0];
    }

    if (kUseAnalyticLights)
    {
        if (u < p[1] && !valid)
        {
            // Sample analytic light source selected uniformly from the light list.
            // TODO: Sample based on estimated contributions as pdf.            
            u /= p[1]; // Rescale to [0,1)
            uint lightCount = gScene.getLightCount();
            uint lightIndex = min(uint(u * lightCount), lightCount - 1);
            float selectionPdf = p[1] / lightCount; // TODO: Precompute 1.f/lightCount in cbuffer

            // Sample local light source.
            AnalyticLightSample lightSample;
            sampleLight(rayOrigin, gScene.getLight(lightIndex), sampleNext2D(sg), lightSample);

            // Reject sample if lower hemisphere.
            if (dot(sd.N, lightSample.dir) < kMinCosTheta) return ls;

            // Setup returned sample.
            // Analytic lights do not currently have a geometric representation in the scene.
            // Do not worry about adjusting the ray to avoid self-intersections at the light.
            ls.rayDir = ls.dir = lightSample.dir;
            ls.rayDistance = ls.distance = lightSample.distance;
            ls.pdf = selectionPdf * lightSample.pdf;
            ls.Li = lightSample.Li / (selectionPdf * numSamples);

            valid = true;
        }
        u -= p[1];
    }

    if (kUseEmissiveSampler)
    {
        //if (u < p[2]) // Always true
        if (!valid)
        {
            float selectionPdf = p[2];

            // Sample emissive lights.
            TriangleLightSample lightSample;
            emissiveSampler.sampleLight(rayOrigin, sd.N, sg, lightSample);

            // Reject sample if lower hemisphere.
            if (dot(sd.N, lightSample.dir) < kMinCosTheta) return ls;

            // Evaluate MIS with BRDF sampling as the other sampling strategy.
            float pdf = selectionPdf * lightSample.pdf;
            float misWeight = 1.f;
            if (kUseMIS && pdf > 0.f)
            {
                float brdfPdf = evalPdfScatter(sd, lightSample.dir);
                misWeight = evalMIS(params, numSamples, pdf, 1, brdfPdf);
            }

            // Compute offset light sample position to reduce self-intersections at the light.
            // We compute the shadow ray parameters based on the offset position.
            float3 offsetPos = computeRayOrigin(lightSample.posW, lightSample.normalW);
            float3 toLight = offsetPos - rayOrigin;
            ls.rayDistance = length(toLight);
            ls.rayDir = normalize(toLight);

            // Setup returned sample.
            ls.dir = lightSample.dir;
            ls.distance = lightSample.distance;
            ls.pdf = pdf;
            if (pdf > 0.f) ls.Li = lightSample.Le * misWeight / (pdf * numSamples);

            valid = true;
        }
    }

    return ls;
}

/** Generates a shadow ray for sampling the light sources.
    This should be called before generateScatterRay() as the latter updates the path throughput.
    The function assumes path.origin is the ray origin for the shadow ray, and that the PathData
    flags for the shadow ray have already been cleared.
    \param[in] params Path tracer parameters.
    \param[in] envProbe Environment map sampler.
    \param[in] emissiveSampler Emissive light sampler.
    \param[in] sd Shading data.
    \param[in] i The sample index in the range [0, kLightSamplesPerVertex).
    \param[in,out] pathData Path data. The path flags will be updated to enable the i:th shadow ray if a sample was generated.
    \param[in,out] shadowRay Shadow ray parameters and unoccluded contribution for the generated sample.
    \return True if a sample was generated, false otherwise.
*/
bool generateShadowRay(const PathTracerParams params, const EnvProbe envProbe, const EmissiveLightSampler emissiveSampler, const ShadingData sd, const uint i, inout PathData path, inout ShadowRay shadowRay)
{
    // Sample the scene lights.
    SceneLightSample ls = sampleSceneLights(params, envProbe, emissiveSampler, sd, path.origin, path.sg, kLightSamplesPerVertex);

    if (any(ls.Li > 0.f))
    {
        float3 Lr = evalBRDFCosine(sd, ls.dir) * ls.Li * path.thp;
        if (any(Lr > 0.f))
        {
            // The sample is valid and has a non-zero contribution.
            // Store ray parameters and unoccluded radiance, weighted by path throughput.
            path.flags |= (uint(PathFlags::shadowRay) << i);
            shadowRay.rayParams = float4(ls.rayDir, ls.rayDistance);
            shadowRay.Lr = Lr;
            return true;
        }
    }
    return false;
}

/** Generates a scatter ray or terminates the path.
    The function uses BRDF sampling to generate ray parameters for the scatter ray.
    The function assumes path.origin is the ray origin for the new ray.
    \param[in] sd Shading data.
    \param[in,out] pathData Path data.
*/
void generateScatterRay(const ShadingData sd, inout PathData path)
{
    // Early out if we're at the last path vertex and the ray is not needed.
    if (!kTraceScatterRayFromLastPathVertex && path.length == kMaxBounces) return;

    // Generate next path segment.
    if (kUseBRDFSampling)
    {
        // Default path that uses BRDF importance sampling.
        BRDFSample result;
        sampleBRDF(sd, path.sg, result);

        path.dir = result.dir;
        path.thp *= result.thp;
        path.pdf = result.pdf;
    }
    else
    {
        // Fallback path that uses cosine-weighted hemisphere sampling.
        float pdf;
        float3 dir = sampleHemisphereCosine(sd, sampleNext2D(path.sg), pdf); // pdf = cos(theta) / pi

        // Check that L and V are in the positive hemisphere, mark the sample invalid otherwise.
        // This is necessary for consistency with BRDF sampling.
        float NdotL = dot(sd.N, dir);
        if (min(sd.NdotV, NdotL) < kMinCosTheta)
        {
            path.thp = float3(0, 0, 0); // Just in case, evalBRDF() should already do this.
            pdf = 0.f;
        }

        path.dir = dir;
        path.thp *= evalBRDF(sd, dir) * M_PI; // dot(N,L) / pdf = pi
        path.pdf = pdf;
    }

//    assert(!any(isnan(path.thp)));
//    assert(!isnan(path.pdf));

    // Pass on the shading normal. This is needed for MIS.
    path.normal = sd.N;

    // Mark the ray as active only if the path throughput is nonzero.
    // If we failed to generate a valid sample, the throughput will be zero.
    if (any(path.thp > 0.f)) path.flags |= (uint)PathFlags::scatterRay;
}
