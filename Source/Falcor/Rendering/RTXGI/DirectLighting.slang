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

/** Utility code for computing the direct illumination at a world-space position.

    This is called for the probe update ray hits to compute their radiance.
    The same direct lighting code is used in visualization mode.

    The helpers assume that ray index 0 represents a shadow ray that uses
    the `ShadowRayData` struct declared here as ray payload.
*/
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.ShadingData;
import Utils.Geometry.GeometryHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
__exported import Utils.Sampling.SampleGeneratorInterface;

struct ShadowRayData
{
    bool visible;
};

struct LightSample
{
    float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float   cosTheta;   ///< Cosine term for the incident direction.
    float3  dir;        ///< Ray direction for visibility evaluation (normalized). This may differ from 'dir' due to ray offsets.
    float   distance;   ///< Ray distance for visibility evaluation. This may differ from 'distance' due to ray offsets.
};

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

    ShadowRayData payload;
    payload.visible = false;    // Set to true by miss shader if ray is not terminated before
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        0 /* ray type for shadow ray */,
        rayTypeCount,
        0 /* miss index for shadow ray */,
        ray,
        payload);

    return payload.visible;
}

/** Utility for computing the direct lighting at a hit point.
*/
struct DirectLighting
{
    bool enableEnvMap;
    bool enableAnalyticLights;
    bool enableEmissiveLights;
    uint emissiveSampleCount;
    EnvMapSampler envMapSampler;
    EmissiveLightSampler emissiveSampler;

    /** Evaluates the direct diffuse illumination at a hit point in world space.
    */
    float3 computeDirectDiffuse<S : ISampleGenerator>(const ShadingData sd, const IBSDF bsdf, inout S sg)
    {
        let bsdfProperties = bsdf.getProperties(sd);
        const float3 rayOrigin = sd.computeNewRayOrigin();
        const float3 albedo = bsdfProperties.diffuseReflectionAlbedo;
        float3 L = float3(0);

        if (enableEnvMap)
        {
            // Draw a sample from the environment map.
            EnvMapSample ls;
            bool valid = envMapSampler.sample(sampleNext2D(sg), ls);

            // Trace shadow ray if sample is valid and in the upper hemisphere.
            float cosTheta = dot(ls.dir, sd.N);
            valid = valid && any(ls.Le > 0.f) && cosTheta > 0.f;

            if (valid)
            {
                if (traceShadowRay(rayOrigin, ls.dir, FLT_MAX))
                {
                    // We assume a perfect Lambertian diffuse lobe.
                    // Note that this is an approximation to actual directional diffuse BSDF lobe used elsewhere.
                    L += (float)M_1_PI * albedo * cosTheta * ls.Le;
                }
            }
        }

        if (enableAnalyticLights)
        {
            const uint lightCount = gScene.getLightCount();
            for (uint lightIndex = 0; lightIndex < lightCount; lightIndex++)
            {
                // Draw a sample on the given analytic light.
                AnalyticLightSample ls;
                bool valid = sampleLight(sd.posW, gScene.getLight(lightIndex), sg, ls);

                // Trace shadow ray if sample is valid and in the upper hemisphere.
                float cosTheta = dot(ls.dir, sd.N);
                valid = valid && any(ls.Li > 0.f) && cosTheta > 0.f;

                if (valid)
                {
                    if (traceShadowRay(rayOrigin, ls.dir, ls.distance))
                    {
                        // We assume a perfect Lambertian diffuse lobe.
                        // Note that this is an approximation to actual directional diffuse BSDF lobe used elsewhere.
                        L += (float)M_1_PI * albedo * cosTheta * ls.Li;
                    }
                }
            }
        }

        if (enableEmissiveLights)
        {
            // Add directly emitted radiance.
            L += bsdfProperties.emission;

            // Add stochastically sampled diffuse lighting from emissive geometry.
            for (uint sampleIndex = 0; sampleIndex < emissiveSampleCount; sampleIndex++)
            {
                LightSample ls;
                if (generateEmissiveSample(rayOrigin, sd.N, sg, ls))
                {
                    if (traceShadowRay(rayOrigin, ls.dir, ls.distance))
                    {
                        // We assume a perfect Lambertian diffuse lobe.
                        // Note that this is an approximation to actual directional diffuse BSDF lobe used elsewhere.
                        L += (float)M_1_PI * albedo * ls.cosTheta * ls.Li / (float)emissiveSampleCount;
                    }
                }
            }
        }

        return L;
    }

    /** Generates a sample on the emissive geometry.
        \param[in] rayOrigin Ray origin for the shadow ray.
        \param[in] normal Shading normal at the shading point.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample<S : ISampleGenerator>(const float3 rayOrigin, const float3 normal, inout S sg, out LightSample ls)
    {
        TriangleLightSample lightSample;
        if (!emissiveSampler.sampleLight(rayOrigin, normal, true, sg, lightSample)) return false;

        // Reject sample if lower hemisphere.
        ls.cosTheta = dot(normal, lightSample.dir);
        if (ls.cosTheta <= 0.f) return false;

        // Compute offset light sample position to reduce self-intersections at the light.
        // We compute the shadow ray parameters based on the offset position.
        float3 offsetPos = computeRayOrigin(lightSample.posW, lightSample.normalW);
        float3 toLight = offsetPos - rayOrigin;

        ls.distance = length(toLight);
        ls.dir = normalize(toLight);
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);

        return any(ls.Li > 0.f);
    }
};
