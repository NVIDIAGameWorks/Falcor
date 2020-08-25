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
#include "Utils/Math/MathConstants.slangh"

// TODO: Which ones need __exported
import Scene.Scene;
import Scene.ShadingData;
__exported import Utils.Helpers; // For computeRayOrigin()
__exported import Utils.Math.MathHelpers;
__exported import Experimental.Scene.Lights.EnvMapSampler;
__exported import Experimental.Scene.Material.MaterialShading;
__exported import Experimental.Scene.Lights.EmissiveLightSampler;
__exported import Experimental.Scene.Lights.LightHelpers;
__exported import RenderPasses.Shared.PathTracer.PathData;
__exported import RenderPasses.Shared.PathTracer.PathTracerParams;

static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

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
    \param[in] dir World-space direction (unnormalized).
    \return Color (rgb).
*/
float3 evalBackground(float3 dir)
{
    return kUseEnvBackground ? gScene.envMap.eval(dir) : kDefaultBackgroundColor;
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
    case MISHeuristic::BalanceHeuristic:
        return p0 / (p0 + p1);
    case MISHeuristic::PowerTwoHeuristic:
        return p0 * p0 / (p0 * p0 + p1 * p1);
    case MISHeuristic::PowerExpHeuristic:
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
    case MISHeuristic::BalanceHeuristic:
    {
        float q0 = n0 * p0;
        float q1 = n1 * p1;
        return q0 / (q0 + q1);
    }
    case MISHeuristic::PowerTwoHeuristic:
    {
        float q0 = (n0 * p0) * (n0 * p0);
        float q1 = (n1 * p1) * (n1 * p1);
        return q0 / (q0 + q1);
    }
    case MISHeuristic::PowerExpHeuristic:
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
    float p2 = kUseEmissiveLights ? 1.f : 0.f;
    return p0 / (p0 + p1 + p2);
}

/** Returns the probability of selecting emissive light sampling.
*/
float getEmissiveLightSelectionPdf()
{
    float p0 = kUseEnvLight ? 1.f : 0.f;
    float p1 = kUseAnalyticLights ? 1.f : 0.f;
    float p2 = kUseEmissiveLights ? 1.f : 0.f;
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
        return evalPdfBSDF(sd, dir);
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
    \param[in] envMapSampler Environment map sampler.
    \param[in] emissiveSampler Emissive light sampler.
    \param[in] sd Shading data.
    \param[in] rayOrigin Ray origin for the shadow ray.
    \param[in] numSamples Total number of light samples that will be taken. This is needed for the MIS weighting.
    \param[in,out] sg SampleGenerator object.
    \param[out] ls Generated light sample. Only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool sampleSceneLights(const PathTracerParams params, const EnvMapSampler envMapSampler, const EmissiveLightSampler emissiveSampler, const ShadingData sd, const float3 rayOrigin, const uint numSamples, inout SampleGenerator sg, out SceneLightSample ls)
{
    // Set relative probabilities of the different sampling techniques.
    // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
    // TODO: Refactor the selection pdf code the helpers above for evaluating the individual probabilities.
    float p[3];
    p[0] = kUseEnvLight ? 1.f : 0.f;
    p[1] = kUseAnalyticLights ? 1.f : 0.f;
    p[2] = kUseEmissiveLights ? 1.f : 0.f;

    // Normalize probabilities. Early out if zero.
    float sum = p[0] + p[1] + p[2];
    if (sum == 0.f) return false;

    float invSum = 1.f / sum;
    p[0] *= invSum;
    p[1] *= invSum;
    p[2] *= invSum;

    // Sample based on uniform random number. Rescale u to [0,1) afterwards.
    float invPdf = 0.f;
    float u = sampleNext1D(sg);

    // We use explicit checks for which light types are enabled so that the compiler
    // can remove the unused code. It won't otherwise since u is unknown at compile time.
    if (kUseEnvLight)
    {
        if (u < p[0])
        {
            float selectionPdf = p[0];

            // Sample environment map.
            EnvMapSample lightSample;
            envMapSampler.sample(sampleNext2D(sg), lightSample);

            // Reject sample if lower hemisphere.
            if (dot(sd.N, lightSample.dir) < kMinCosTheta) return false;

            // Evaluate MIS with BRDF sampling as the other sampling strategy.
            float pdf = selectionPdf * lightSample.pdf;
            float misWeight = 1.f;
            if (kUseMIS && pdf > 0.f)
            {
                float brdfPdf = evalPdfScatter(sd, lightSample.dir);
                misWeight = evalMIS(params, numSamples, pdf, 1, brdfPdf);
            }

            // Setup returned sample.
            ls.rayDir = ls.dir = lightSample.dir;
            ls.rayDistance = ls.distance = kRayTMax;
            ls.pdf = pdf;
            ls.Li = pdf > 0.f ? lightSample.Le * misWeight / (pdf * numSamples) : float3(0);

            return true;
        }
        u -= p[0];
    }

    if (kUseAnalyticLights)
    {
        if (u < p[1])
        {
            // Sample analytic light source selected uniformly from the light list.
            // TODO: Sample based on estimated contributions as pdf.
            u /= p[1]; // Rescale to [0,1)
            uint lightCount = gScene.getLightCount();
            uint lightIndex = min(uint(u * lightCount), lightCount - 1);
            float selectionPdf = p[1] / lightCount; // TODO: Precompute 1.f/lightCount in cbuffer

            // Sample local light source.
            AnalyticLightSample lightSample;
            bool valid = sampleLight(rayOrigin, gScene.getLight(lightIndex), sg, lightSample);

            // Reject sample if lower hemisphere.
            if (!valid || dot(sd.N, lightSample.dir) < kMinCosTheta) return false;

            // Setup returned sample.
            // Analytic lights do not currently have a geometric representation in the scene.
            // Do not worry about adjusting the ray to avoid self-intersections at the light.
            ls.rayDir = ls.dir = lightSample.dir;
            ls.rayDistance = ls.distance = lightSample.distance;
            ls.pdf = selectionPdf * lightSample.pdf;
            ls.Li = lightSample.Li / (selectionPdf * numSamples);

            return true;
        }
        u -= p[1];
    }

    if (kUseEmissiveLights)
    {
        //if (u < p[2]) // Always true
        {
            float selectionPdf = p[2];

            // Sample emissive lights.
            TriangleLightSample lightSample;
            bool valid = emissiveSampler.sampleLight(rayOrigin, sd.N, sg, lightSample);

            // Reject sample if lower hemisphere.
            if (!valid || dot(sd.N, lightSample.dir) < kMinCosTheta) return false;

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
            ls.Li = pdf > 0.f ? lightSample.Le * misWeight / (pdf * numSamples) : float3(0);

            return true;
        }
    }

    return false;
}

/** Generates a shadow ray for sampling the light sources.
    This should be called before generateScatterRay() as the latter updates the path throughput.
    The function assumes path.origin is the ray origin for the shadow ray, and that the PathData
    flags for the shadow ray have already been cleared.
    \param[in] params Path tracer parameters.
    \param[in] envMapSampler Environment map sampler.
    \param[in] emissiveSampler Emissive light sampler.
    \param[in] sd Shading data.
    \param[in] i The sample index in the range [0, kLightSamplesPerVertex).
    \param[in,out] path Path data. The path flags will be updated to enable the i:th shadow ray if a sample was generated.
    \param[in,out] shadowRay Shadow ray parameters and unoccluded contribution for the generated sample.
    \return True if a sample was generated, false otherwise.
*/
bool generateShadowRay(const PathTracerParams params, const EnvMapSampler envMapSampler, const EmissiveLightSampler emissiveSampler, const ShadingData sd, const uint i, inout PathData path, inout ShadowRay shadowRay)
{
    // Sample the scene lights.
    SceneLightSample ls;
    bool valid = sampleSceneLights(params, envMapSampler, emissiveSampler, sd, path.origin, kLightSamplesPerVertex, path.sg, ls);

    if (valid && any(ls.Li > 0.f))
    {
        float3 Lr = evalBSDFCosine(sd, ls.dir) * ls.Li * path.thp;
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
    \param[in] params Path tracer parameters.
    \param[in] sd Shading data.
    \param[in,out] path Path data.
    \return True if a sample was generated and path should continue, false otherwise.
*/
bool generateScatterRay(const PathTracerParams params, const ShadingData sd, inout PathData path)
{
    // Generate next path segment.
    BSDFSample result;
    bool valid;
    if (kUseBRDFSampling) valid = sampleBSDF(sd, path.sg, result);
    else valid = sampleBSDF_Reference(sd, path.sg, result);

    if (valid)
    {
        path.dir = result.wi;
        path.thp *= result.weight;
        path.pdf = result.pdf;

        path.clearEventFlags();

        // Handle specular events.
        if (result.isLobe(LobeType::SpecularTransmission) ||
            (result.isLobe(LobeType::SpecularReflection) && sd.linearRoughness <= params.specularRoughnessThreshold) ||
            result.isLobe(LobeType::Delta))
        {
            path.setSpecular();
        }
        else
        {
            // Count non-specular bounces.
            path.nonSpecularBounces++;
        }

        // Handle delta events.
        if (result.isLobe(LobeType::Delta))
        {
            path.setDelta();
        }

        // Handle transmission events.
        if (result.isLobe(LobeType::Transmission))
        {
            path.setTransmission();

            // Compute ray origin for next ray segment.
            path.transmissionOrigin = sd.computeNewRayOrigin(false);

            // Update interior list and inside volume flag.
            if (kUseNestedDielectrics)
            {
                uint nestedPriority = gScene.materials[sd.materialID].getNestedPriority();
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.setInsideVolume(!path.interiorList.isEmpty());
            }
        }

        // assert(!any(isnan(path.thp)));
        // assert(!isnan(path.pdf));

        // Pass on the shading normal. This is needed for MIS.
        path.normal = sd.N;

        valid = any(path.thp > 0.f);
    }

    // Mark the ray as active only if the sample is valid and throughput is non-zero.
    if (valid) path.flags |= uint(PathFlags::scatterRay);
    return valid;
}
