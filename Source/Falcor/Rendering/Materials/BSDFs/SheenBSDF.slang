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
 * Sheen-only implementation from Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF";
 * all energy which is not reflected will be transmitted as a Dirac impulse
 */
struct SheenBSDF : IBSDF
{
    float3 color;    ///< Sheen reflectivity
    float roughness; ///< Surface roughness (max 1.0)

    __init(float3 color, float roughness)
    {
        this.color = color;
        this.roughness = roughness;
    }

    float SheenD(float NoH)
    {
        float invAlpha = 1.f / roughness;
        float cos2h = NoH * NoH;
        float sin2h = max(1.f - cos2h, 0.0078125f);
        return (2.f + invAlpha) * pow(sin2h, invAlpha * 0.5f) / (2.f * M_PI);
    }

    float L(float x)
    {
        const float r = (1.0f - roughness) * (1.0f - roughness);
        const float oneMinR = 1.0 - r;

        const float a = 25.3245f * r + 21.5473f * oneMinR;
        const float b = 3.32435f * r + 3.82987f * oneMinR;
        const float c = 0.16801f * r + 0.19823f * oneMinR;
        const float d = (-1.27393f) * r + (-1.97760f) * oneMinR;
        const float e = (-4.85967f) * r + (-4.32054f) * oneMinR;

        return a / (1.0f + b * pow(x, c)) + d * x + e;
    }

    float lambda(float cos)
    {
        if (cos < 0.5f)
            return exp(L(cos));
        else
            return exp(2.0f * L(0.5f) - L(1.0f - cos));
    }

    float G(float NdotO, float NdotI)
    {
        // this is the full visibility function; note that we have the questionable terminator softening from equ. 4
        return 1.f / (1.f + pow(lambda(NdotO), 1.f + 2.f * pow(1.f - NdotO, 8)) + lambda(NdotI));
    }

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return float3(0.f);

        float3 wh = normalize(wi + wo);
        float NdotH = wh.z;

        const float3 D = SheenD(NdotH);
        // We multiply by wo.z
        float3 Fr = D * G(wo.z, wi.z) * color / (4.f * wi.z);

        return Fr;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = {};
        lobeType = {};

        // Without a baselayer, this BSDF should be transparent. Until we can overwrite local importance sampling with the lobetypes, we
        // will pretend for sample() that there is no transparency. This is a performance optimisation for the LayeredMaterialInstance.
        /*
        float3 albedoT = evalAlbedo(wi, LobeType::All).transmission;
        const float selectionWeight = luminance(albedoT);
        if (sampleNext1D(sg) < selectionWeight)
        {
            wo = float3(-wi.x, -wi.y, -wi.z);
            pdf = 0.f;
            weight = 1.0;
            lobeType = (uint)LobeType::DeltaTransmission;
            return true;
        }*/

        // The paper recommends cosine sampling.
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        lobeType = (uint)LobeType::DiffuseReflection;

        weight = (eval(wi, wo, sg) / (M_1_PI * wo.z));
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return 0.f;

        float3 albedoT = evalAlbedo(wi, LobeType::All).transmission;
        return M_1_PI * wo.z;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        const float coefficients[11][5] = {
            { 1.01108, 75.2286, -9.4671, 0.000973445, -1.01325 },
            { 8.44447, 0.0135054, 1.11637, 0.479139, -9.30873 },
            { 5.67124, 0.0194668, 1.12489, 0.453551, -6.53104 },
            { 4.50086, 0.0237758, 1.09376, 0.423665, -5.36521 },
            { 3.77781, 0.0273286, 1.0666, 0.400267, -4.65601 },
            { 3.35338, 0.0307392, 0.989189, 0.371074, -4.24089 },
            { 3.39605, 0.0308828, 0.845455, 0.333813, -4.28662 },
            { 3.62583, 0.0303531, 0.79754, 0.325588, -4.51725 },
            { 4.11524, 0.0172797, 0.701166, 0.275242, -5.04394 },
            { 4.4012, 0.0159469, 0.685892, 0.270221, -5.32997 },
            { 5.8912, 0.0171461, 0.66366, 0.288735, -6.78867 }
        };

        // fit was [1, 21] for cosI(0) = 21
        const float x = (wi.z) * 20.f + 1.f;

        // roughness precomputed between [0,1] in 0.1 steps
        const int indexB = max(floor(roughness * 10), 0);
        const int indexT = min(indexB + 1, 10);
        const float indexInt = indexT - (roughness * 10.f);

        // albedo = e + pow(x, d) + a/(1.f + b * pow(x,c))
        const float albedoTB = coefficients[indexB][4] + pow(x, coefficients[indexB][3]) +
                               coefficients[indexB][0] / (1.f + coefficients[indexB][1] * pow(x, coefficients[indexB][2]));
        const float albedoTT = coefficients[indexT][4] + pow(x, coefficients[indexT][3]) +
                               coefficients[indexT][0] / (1.f + coefficients[indexT][1] * pow(x, coefficients[indexT][2]));
        const float albedoR = 1.f - clamp(albedoTB * indexInt + albedoTT * (1.f - indexInt), 0.f, 1.f);

        return AlbedoContributions(color * albedoR, albedoR - (albedoR * color), (1.f - albedoR), 0.f);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(0.5f, 0.5f);
        return r;
    }
};
