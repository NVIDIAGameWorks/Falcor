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

import Utils.Math.MathHelpers;
__exported import Rendering.Materials.IBSDF;

/**
 * Oren-Nayar reflection model "Generalization of Lambert’s Reflectance Model" for retroreflective surfaces
 * The model will become Lambertian for roughness=0.0
 */
struct OrenNayarBRDF : IBSDF
{
    float3 albedo;   ///< Diffuse albedo.
    float roughness; ///< Rougness

    __init(float3 albedo, float roughness)
    {
        this.albedo = albedo;
        this.roughness = roughness;
    }

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return float3(0.f);

        const float sinThetaI = sqrt(1.0f - wi.z * wi.z);
        const float sinPhiI = (sinThetaI == 0.0f ? 0.0f : clamp(wi.y / sinThetaI, -1.0f, 1.0f));
        const float cosPhiI = (sinThetaI == 0.0f ? 0.0f : clamp(wi.x / sinThetaI, -1.0f, 1.0f));

        float sinThetaO = sqrt(1.0f - wo.z * wo.z);
        float sinPhiO = (sinThetaO == 0.0f ? 0.0f : clamp(wo.y / sinThetaO, -1.0f, 1.0f));
        float cosPhiO = (sinThetaO == 0.0f ? 0.0f : clamp(wo.x / sinThetaO, -1.0f, 1.0f));

        const float cosDiff = max(cosPhiI * cosPhiO + sinPhiI * sinPhiO, 0.0f);

        const float sigma2 = roughness * roughness;
        const float A = 1.0f - 0.5f * (sigma2 / (sigma2 + 0.33f));
        const float B = 0.45f * sigma2 / (sigma2 + 0.09f);

        float sinAlpha, tanBeta;

        if (abs(wi.z) > abs(wo.z))
        {
            sinAlpha = sinThetaO;
            tanBeta = sinThetaI / abs(wi.z);
        }
        else
        {
            sinAlpha = sinThetaI;
            tanBeta = sinThetaO / abs(wo.z);
        }

        return albedo * M_1_PI * (A + B * cosDiff * sinAlpha * tanBeta) * wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        // Note that cosine sampling can become rather inefficient if we have high retro-reflectivity
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        lobeType = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = eval(wi, wo, sg) / evalPdf(wi, wo);
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return 0.f;

        return M_1_PI * wo.z;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        return AlbedoContributions(albedo, 1.0f - albedo, 0.0f, 0.0f);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(0.5f, 0.5f);
        return r;
    }
};
