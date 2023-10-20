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
#include "Rendering/Materials/BSDFConfig.slangh"

import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;
__exported import Rendering.Materials.IBSDF;

// Enable support for delta reflection/transmission.
#define EnableDeltaBSDF 1

// Enable GGX sampling using the distribution of visible normals (VNDF) instead of classic NDF sampling.
// This should be the default as it has lower variance, disable for testing only.
#define EnableVNDFSampling 1

// Enable explicitly computing sampling weights using eval(wi, wo) / evalPdf(wi, wo).
// This is for testing only, as many terms of the equation cancel out allowing to save on computation.
#define ExplicitSampleWeights 0

/**
 * Specular reflection using microfacets.
 */
struct SpecularMicrofacetBRDF : IBSDF, IDifferentiable
{
    float3 albedo;    ///< Specular albedo.
    float alpha;      ///< GGX width parameter.
    uint activeLobes; ///< BSDF lobes to include for sampling and evaluation. See LobeType.slang.

    bool hasLobe(LobeType lobeType) { return (activeLobes & (uint)lobeType) != 0; }

    [Differentiable]
    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return float3(0.f);

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f)
            return float3(0.f);
#endif

        if (!hasLobe(LobeType::SpecularReflection))
            return float3(0.f);

        float3 h = normalize(wi + wo);
        float wiDotH = dot(wi, h);

        float D = evalNdfGGX(alpha, h.z);
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wi.z, wo.z);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
#endif
        float3 F = evalFresnelSchlick(albedo, 1.f, wiDotH);
        return F * D * G * 0.25f / wi.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = 0.f;
        lobeType = (uint)LobeType::SpecularReflection;

        if (wi.z < kMinCosTheta)
            return false;

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f)
        {
            if (!hasLobe(LobeType::DeltaReflection))
                return false;

            wo = float3(-wi.x, -wi.y, wi.z);
            pdf = 0.f;
            weight = evalFresnelSchlick(albedo, 1.f, wi.z);
            lobeType = (uint)LobeType::DeltaReflection;
            return true;
        }
#endif

        if (!hasLobe(LobeType::SpecularReflection))
            return false;

            // Sample the GGX distribution to find a microfacet normal (half vector).
#if EnableVNDFSampling
        float3 h = sampleGGX_VNDF(alpha, wi, sampleNext2D(sg), pdf); // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
#else
        float3 h = sampleGGX_NDF(alpha, sampleNext2D(sg), pdf); // pdf = D(h) * h.z
#endif

        // Reflect the incident direction to find the outgoing direction.
        float wiDotH = dot(wi, h);
        wo = 2.f * wiDotH * h - wi;
        if (wo.z < kMinCosTheta)
            return false;

#if ExplicitSampleWeights
        // For testing.
        pdf = evalPdf(wi, wo);
        weight = eval(wi, wo, sg) / pdf;
        lobeType = (uint)LobeType::SpecularReflection;
        return true;
#endif

#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wi.z, wo.z);
        float GOverG1wo = evalG1GGX(alpha * alpha, wo.z);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
        float GOverG1wo = G * (1.f + evalLambdaGGX(alpha * alpha, wi.z));
#endif
        float3 F = evalFresnelSchlick(albedo, 1.f, wiDotH);

        pdf /= (4.f * wiDotH); // Jacobian of the reflection operator.
#if EnableVNDFSampling
        weight = F * GOverG1wo;
#else
        weight = F * G * wiDotH / (wi.z * h.z);
#endif
        lobeType = (uint)LobeType::SpecularReflection;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return 0.f;

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f)
            return 0.f;
#endif

        if (!hasLobe(LobeType::SpecularReflection))
            return 0.f;

        float3 h = normalize(wi + wo);
        float wiDotH = dot(wi, h);
#if EnableVNDFSampling
        float pdf = evalPdfGGX_VNDF(alpha, wi, h);
#else
        float pdf = evalPdfGGX_NDF(alpha, h.z);
#endif
        return pdf / (4.f * wiDotH);
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        float3 r = evalFresnelSchlick(albedo, 1.f, wi.z);
        // The energy needs to go somewhere. This is a reflection-only BSDF --> put it into R
        return AlbedoContributions(r, 1.0f - r, 0.0f, 0.0f);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(alpha);
        return r;
    }
};

/**
 * Specular reflection and transmission using microfacets.
 */
struct SpecularMicrofacetBSDF : IBSDF, IDifferentiable
{
    float3 transmissionAlbedo; ///< Transmission albedo.
    float alpha;               ///< GGX width parameter.
    float eta;                 ///< Relative index of refraction (etaI / etaT).
    uint activeLobes;          ///< BSDF lobes to include for sampling and evaluation. See LobeType.slang.

    bool hasLobe(LobeType lobeType) { return (activeLobes & (uint)lobeType) != 0; }

    [Differentiable]
    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, abs(wo.z)) < kMinCosTheta)
            return float3(0.f);

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f)
            return float3(0.f);
#endif

        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        const bool isReflection = wo.z > 0.f;
        if ((isReflection && !hasReflection) || (!isReflection && !hasTransmission))
            return float3(0.f);

        // Compute half-vector and make sure it's in the upper hemisphere.
        float3 h = normalize(wo + wi * (isReflection ? 1.f : eta));
        h *= float(sign(h.z));

        float wiDotH = dot(wi, h);
        float woDotH = dot(wo, h);

        float D = evalNdfGGX(alpha, h.z);
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wi.z, abs(wo.z));
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, abs(wo.z));
#endif
        float F = evalFresnelDielectric(eta, wiDotH);

        if (isReflection)
        {
            return F * D * G * 0.25f / wi.z;
        }
        else
        {
            float sqrtDenom = woDotH + eta * wiDotH;
            float t = eta * eta * wiDotH * woDotH / (wi.z * sqrtDenom * sqrtDenom);
            return transmissionAlbedo * (1.f - F) * D * G * abs(t);
        }
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = 0.f;
        lobeType = (uint)LobeType::SpecularReflection;

        if (wi.z < kMinCosTheta)
            return false;

        // Get a random number to decide what lobe to sample.
        float lobeSample = sampleNext1D(sg);

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f)
        {
            const bool hasReflection = hasLobe(LobeType::DeltaReflection);
            const bool hasTransmission = hasLobe(LobeType::DeltaTransmission);
            if (!(hasReflection || hasTransmission))
                return false;

            float cosThetaT;
            float F = evalFresnelDielectric(eta, wi.z, cosThetaT);

            bool isReflection = hasReflection;
            if (hasReflection && hasTransmission)
            {
                isReflection = lobeSample < F;
            }
            else if (hasTransmission && F == 1.f)
            {
                return false;
            }

            pdf = 0.f;
            weight = isReflection ? float3(1.f) : transmissionAlbedo;
            if (!(hasReflection && hasTransmission))
                weight *= float3(isReflection ? F : 1.f - F);
            wo = isReflection ? float3(-wi.x, -wi.y, wi.z) : float3(-wi.x * eta, -wi.y * eta, -cosThetaT);
            lobeType = isReflection ? (uint)LobeType::DeltaReflection : (uint)LobeType::DeltaTransmission;

            if (abs(wo.z) < kMinCosTheta || (wo.z > 0.f != isReflection))
                return false;

            return true;
        }
#endif

        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        if (!(hasReflection || hasTransmission))
            return false;

            // Sample the GGX distribution of (visible) normals. This is our half vector.
#if EnableVNDFSampling
        float3 h = sampleGGX_VNDF(alpha, wi, sampleNext2D(sg), pdf); // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
#else
        float3 h = sampleGGX_NDF(alpha, sampleNext2D(sg), pdf); // pdf = D(h) * h.z
#endif

        // Reflect/refract the incident direction to find the outgoing direction.
        float wiDotH = dot(wi, h);

        float cosThetaT;
        float F = evalFresnelDielectric(eta, wiDotH, cosThetaT);

        bool isReflection = hasReflection;
        if (hasReflection && hasTransmission)
        {
            isReflection = lobeSample < F;
        }
        else if (hasTransmission && F == 1.f)
        {
            return false;
        }

        wo = isReflection ? (2.f * wiDotH * h - wi) : ((eta * wiDotH - cosThetaT) * h - eta * wi);

        if (abs(wo.z) < kMinCosTheta || (wo.z > 0.f != isReflection))
            return false;

        float woDotH = dot(wo, h);

        lobeType = isReflection ? (uint)LobeType::SpecularReflection : (uint)LobeType::SpecularTransmission;

#if ExplicitSampleWeights
        // For testing.
        pdf = evalPdf(wi, wo);
        weight = pdf > 0.f ? eval(wi, wo, sg) / pdf : float3(0.f);
        return true;
#endif

#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wi.z, abs(wo.z));
        float GOverG1wo = evalG1GGX(alpha * alpha, abs(wo.z));
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, abs(wo.z));
        float GOverG1wo = G * (1.f + evalLambdaGGX(alpha * alpha, wi.z));
#endif

#if EnableVNDFSampling
        weight = GOverG1wo;
#else
        weight = G * wiDotH / (wi.z * h.z);
#endif

        if (isReflection)
        {
            pdf /= 4.f * woDotH; // Jacobian of the reflection operator.
        }
        else
        {
            float sqrtDenom = woDotH + eta * wiDotH;
            float denom = sqrtDenom * sqrtDenom;
            pdf = (denom > 0.f) ? pdf * abs(woDotH) / denom : FLT_MAX; // Jacobian of the refraction operator.
            weight *= transmissionAlbedo * eta * eta;
        }

        if (hasReflection && hasTransmission)
        {
            pdf *= isReflection ? F : 1.f - F;
        }
        else
        {
            weight *= isReflection ? F : 1.f - F;
        }

        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, abs(wo.z)) < kMinCosTheta)
            return 0.f;

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f)
            return 0.f;
#endif

        bool isReflection = wo.z > 0.f;
        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        if ((isReflection && !hasReflection) || (!isReflection && !hasTransmission))
            return 0.f;

        // Compute half-vector and make sure it's in the upper hemisphere.
        float3 h = normalize(wo + wi * (isReflection ? 1.f : eta));
        h *= float(sign(h.z));

        float wiDotH = dot(wi, h);
        float woDotH = dot(wo, h);

        float F = evalFresnelDielectric(eta, wiDotH);

#if EnableVNDFSampling
        float pdf = evalPdfGGX_VNDF(alpha, wi, h);
#else
        float pdf = evalPdfGGX_NDF(alpha, h.z);
#endif
        if (isReflection)
        {
            pdf /= 4.f * woDotH; // Jacobian of the reflection operator.
        }
        else
        {
            if (woDotH > 0.f)
                return 0.f;
            float sqrtDenom = woDotH + eta * wiDotH;
            float denom = sqrtDenom * sqrtDenom;
            pdf = (denom > 0.f) ? pdf * abs(woDotH) / denom : FLT_MAX; // Jacobian of the refraction operator.
        }

        if (hasReflection && hasTransmission)
        {
            pdf *= isReflection ? F : 1.f - F;
        }

        return pdf;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        float3 r = evalFresnelDielectric(eta, wi.z);
        // The energy needs to go somewhere. This is a transmission-only BSDF --> put it into AT
        return AlbedoContributions(0.0, 0.0, 1.0f - r, r);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(alpha);
        return r;
    }
};
