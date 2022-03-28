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
#include "Utils/Math/MathConstants.slangh"
#include "BxDFConfig.slangh"

import Scene.ShadingData;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Rendering.Materials.Fresnel;
import Rendering.Materials.Microfacet;
__exported import Rendering.Materials.IBxDF;

// Enable support for delta reflection/transmission.
#define EnableDeltaBSDF         1

// Enable GGX sampling using the distribution of visible normals (VNDF) instead of classic NDF sampling.
// This should be the default as it has lower variance, disable for testing only.
// TODO: Make default when transmission with VNDF sampling is properly validated
#define EnableVNDFSampling      1

// Enable explicitly computing sampling weights using eval(wi, wo) / evalPdf(wi, wo).
// This is for testing only, as many terms of the equation cancel out allowing to save on computation.
#define ExplicitSampleWeights   0

// We clamp the GGX width parameter to avoid numerical instability.
// In some computations, we can avoid clamps etc. if 1.0 - alpha^2 != 1.0, so the epsilon should be 1.72666361e-4 or larger in fp32.
// The the value below is sufficient to avoid visible artifacts.
// Falcor used to clamp roughness to 0.08 before the clamp was removed for allowing delta events. We continue to use the same threshold.
static const float kMinGGXAlpha = 0.0064f;

/** Lambertian diffuse reflection.
    f_r(wi, wo) = albedo / pi
*/
struct DiffuseReflectionLambert : IBxDF
{
    float3 albedo;  ///< Diffuse albedo.

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0.f);

        return M_1_PI * albedo * wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = albedo;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wo.z;
    }
};

/** Disney's diffuse reflection.
    Based on https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
*/
struct DiffuseReflectionDisney : IBxDF
{
    float3 albedo;          ///< Diffuse albedo.
    float roughness;        ///< Roughness before remapping.

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0.f);

        return evalWeight(wi, wo) * M_1_PI * wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = evalWeight(wi, wo);
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wo.z;
    }

    // private

    // Returns f(wi, wo) * pi.
    float3 evalWeight(float3 wi, float3 wo)
    {
        float3 h = normalize(wi + wo);
        float woDotH = dot(wo, h);
        float fd90 = 0.5f + 2.f * woDotH * woDotH * roughness;
        float fd0 = 1.f;
        float wiScatter = evalFresnelSchlick(fd0, fd90, wi.z);
        float woScatter = evalFresnelSchlick(fd0, fd90, wo.z);
        return albedo * wiScatter * woScatter;
    }
};

/** Frostbites's diffuse reflection.
    This is Disney's diffuse BRDF with an ad-hoc normalization factor to ensure energy conservation.
    Based on https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
*/
struct DiffuseReflectionFrostbite : IBxDF
{
    float3 albedo;          ///< Diffuse albedo.
    float roughness;        ///< Roughness before remapping.

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0.f);

        return evalWeight(wi, wo) * M_1_PI * wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = evalWeight(wi, wo);
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wo.z;
    }

    // private

    // Returns f(wi, wo) * pi.
    float3 evalWeight(float3 wi, float3 wo)
    {
        float3 h = normalize(wi + wo);
        float woDotH = dot(wo, h);
        float energyBias = lerp(0.f, 0.5f, roughness);
        float energyFactor = lerp(1.f, 1.f / 1.51f, roughness);
        float fd90 = energyBias + 2.f * woDotH * woDotH * roughness;
        float fd0 = 1.f;
        float wiScatter = evalFresnelSchlick(fd0, fd90, wi.z);
        float woScatter = evalFresnelSchlick(fd0, fd90, wo.z);
        return albedo * wiScatter * woScatter * energyFactor;
    }
};

/** Lambertian diffuse transmission.
*/
struct DiffuseTransmissionLambert : IBxDF
{
    float3 albedo;  ///< Diffuse albedo.

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, -wo.z) < kMinCosTheta) return float3(0.f);

        return M_1_PI * albedo * -wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        wo.z = -wo.z;
        lobe = (uint)LobeType::DiffuseTransmission;

        if (min(wi.z, -wo.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = albedo;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, -wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * -wo.z;
    }
}

/** Specular reflection using microfacets.
*/
struct SpecularReflectionMicrofacet : IBxDF
{
    float3 albedo;      ///< Specular albedo.
    float alpha;        ///< GGX width parameter.
    uint activeLobes;   ///< BSDF lobes to include for sampling and evaluation. See LobeType.slang.

    bool hasLobe(LobeType lobe) { return (activeLobes & (uint)lobe) != 0; }

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0.f);

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f) return float3(0.f);
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return float3(0.f);

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

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = 0.f;
        lobe = (uint)LobeType::SpecularReflection;

        if (wi.z < kMinCosTheta) return false;

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f)
        {
            if (!hasLobe(LobeType::DeltaReflection)) return false;

            wo = float3(-wi.x, -wi.y, wi.z);
            pdf = 0.f;
            weight = evalFresnelSchlick(albedo, 1.f, wi.z);
            lobe = (uint)LobeType::DeltaReflection;
            return true;
        }
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return false;

        // Sample the GGX distribution to find a microfacet normal (half vector).
#if EnableVNDFSampling
        float3 h = sampleGGX_VNDF(alpha, wi, sampleNext2D(sg), pdf);    // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
#else
        float3 h = sampleGGX_NDF(alpha, sampleNext2D(sg), pdf);         // pdf = D(h) * h.z
#endif

        // Reflect the incident direction to find the outgoing direction.
        float wiDotH = dot(wi, h);
        wo = 2.f * wiDotH * h - wi;
        if (wo.z < kMinCosTheta) return false;

#if ExplicitSampleWeights
        // For testing.
        pdf = evalPdf(wi, wo);
        weight = eval(wi, wo, sg) / pdf;
        lobe = (uint)LobeType::SpecularReflection;
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
        lobe = (uint)LobeType::SpecularReflection;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f) return 0.f;
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return 0.f;

        float3 h = normalize(wi + wo);
        float wiDotH = dot(wi, h);
#if EnableVNDFSampling
        float pdf = evalPdfGGX_VNDF(alpha, wi, h);
#else
        float pdf = evalPdfGGX_NDF(alpha, h.z);
#endif
        return pdf / (4.f * wiDotH);
    }
};

/** Specular reflection and transmission using microfacets.
*/
struct SpecularReflectionTransmissionMicrofacet : IBxDF
{
    float3 transmissionAlbedo;  ///< Transmission albedo.
    float alpha;                ///< GGX width parameter.
    float eta;                  ///< Relative index of refraction (etaI / etaT).
    uint activeLobes;           ///< BSDF lobes to include for sampling and evaluation. See LobeType.slang.

    bool hasLobe(LobeType lobe) { return (activeLobes & (uint)lobe) != 0; }

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, abs(wo.z)) < kMinCosTheta) return float3(0.f);

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f) return float3(0.f);
#endif

        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        const bool isReflection = wo.z > 0.f;
        if ((isReflection && !hasReflection) || (!isReflection && !hasTransmission)) return float3(0.f);

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

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = 0.f;
        lobe = (uint)LobeType::SpecularReflection;

        if (wi.z < kMinCosTheta) return false;

        // Get a random number to decide what lobe to sample.
        float lobeSample = sampleNext1D(sg);

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f)
        {
            const bool hasReflection = hasLobe(LobeType::DeltaReflection);
            const bool hasTransmission = hasLobe(LobeType::DeltaTransmission);
            if (!(hasReflection || hasTransmission)) return false;

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
            if (!(hasReflection && hasTransmission)) weight *= float3(isReflection ? F : 1.f - F);
            wo = isReflection ? float3(-wi.x, -wi.y, wi.z) : float3(-wi.x * eta, -wi.y * eta, -cosThetaT);
            lobe = isReflection ? (uint)LobeType::DeltaReflection : (uint)LobeType::DeltaTransmission;

            if (abs(wo.z) < kMinCosTheta || (wo.z > 0.f != isReflection)) return false;

            return true;
        }
#endif

        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        if (!(hasReflection || hasTransmission)) return false;

        // Sample the GGX distribution of (visible) normals. This is our half vector.
#if EnableVNDFSampling
        float3 h = sampleGGX_VNDF(alpha, wi, sampleNext2D(sg), pdf);    // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
#else
        float3 h = sampleGGX_NDF(alpha, sampleNext2D(sg), pdf);         // pdf = D(h) * h.z
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

        wo = isReflection ?
            (2.f * wiDotH * h - wi) :
            ((eta * wiDotH - cosThetaT) * h - eta * wi);

        if (abs(wo.z) < kMinCosTheta || (wo.z > 0.f != isReflection)) return false;

        float woDotH = dot(wo, h);

        lobe = isReflection ? (uint)LobeType::SpecularReflection : (uint)LobeType::SpecularTransmission;

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
        if (min(wi.z, abs(wo.z)) < kMinCosTheta) return 0.f;

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f) return 0.f;
#endif

        bool isReflection = wo.z > 0.f;
        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        if ((isReflection && !hasReflection) || (!isReflection && !hasTransmission)) return 0.f;

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
            if (woDotH > 0.f) return 0.f;
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
};

// TODO: Reduce to 52B
/** BSDF parameters for the standard BSDF.
    These are needed for initializing a `FalcorBSDF` instance.
*/
struct StandardBSDFData
{
    float3 diffuse;                 ///< Diffuse albedo.
    float3 specular;                ///< Specular albedo.
    float roughness;                ///< This is the original roughness, before remapping.
    float metallic;                 ///< Metallic parameter, blends between dielectric and conducting BSDFs.
    float eta;                      ///< Relative index of refraction (incident IoR / transmissive IoR).
    float3 transmission;            ///< Transmission color.
    float diffuseTransmission;      ///< Diffuse transmission, blends between diffuse reflection and transmission lobes.
    float specularTransmission;     ///< Specular transmission, blends between opaque dielectric BRDF and specular transmissive BSDF.
}

/** Mixed BSDF used for the standard material in Falcor.

    This consists of a diffuse and specular BRDF.
    A specular BSDF is mixed in using the specularTransmission parameter.
*/
struct FalcorBSDF : IBxDF
{
#if DiffuseBrdf == DiffuseBrdfLambert
    DiffuseReflectionLambert diffuseReflection;
#elif DiffuseBrdf == DiffuseBrdfDisney
    DiffuseReflectionDisney diffuseReflection;
#elif DiffuseBrdf == DiffuseBrdfFrostbite
    DiffuseReflectionFrostbite diffuseReflection;
#endif
    DiffuseTransmissionLambert diffuseTransmission;

    SpecularReflectionMicrofacet specularReflection;
    SpecularReflectionTransmissionMicrofacet specularReflectionTransmission;

    float diffTrans;                        ///< Mix between diffuse BRDF and diffuse BTDF.
    float specTrans;                        ///< Mix between dielectric BRDF and specular BSDF.

    float pDiffuseReflection;               ///< Probability for sampling the diffuse BRDF.
    float pDiffuseTransmission;             ///< Probability for sampling the diffuse BTDF.
    float pSpecularReflection;              ///< Probability for sampling the specular BRDF.
    float pSpecularReflectionTransmission;  ///< Probability for sampling the specular BSDF.

    /** Initialize a new instance.
        \param[in] sd Shading data.
        \param[in] data BSDF parameters.
    */
    __init(const ShadingData sd, const StandardBSDFData data)
    {
        // TODO: Currently specular reflection and transmission lobes are not properly separated.
        // This leads to incorrect behaviour if only the specular reflection or transmission lobe is selected.
        // Things work fine as long as both or none are selected.

        // Use square root if we can assume the shaded object is intersected twice.
        float3 transmissionAlbedo = sd.mtl.isThinSurface() ? data.transmission : sqrt(data.transmission);

        // Setup lobes.
        diffuseReflection.albedo = data.diffuse;
#if DiffuseBrdf != DiffuseBrdfLambert
        diffuseReflection.roughness = data.roughness;
#endif
        diffuseTransmission.albedo = transmissionAlbedo;

        // Compute GGX alpha.
        float alpha = data.roughness * data.roughness;
#if EnableDeltaBSDF
        // Alpha below min alpha value means using delta reflection/transmission.
        if (alpha < kMinGGXAlpha) alpha = 0.f;
#else
        alpha = max(alpha, kMinGGXAlpha);
#endif
        const uint activeLobes = sd.mtl.getActiveLobes();

        specularReflection.albedo = data.specular;
        specularReflection.alpha = alpha;
        specularReflection.activeLobes = activeLobes;

        specularReflectionTransmission.transmissionAlbedo = transmissionAlbedo;
        // Transmission through rough interface with same IoR on both sides is not well defined, switch to delta lobe instead.
        specularReflectionTransmission.alpha = data.eta == 1.f ? 0.f : alpha;
        specularReflectionTransmission.eta = data.eta;
        specularReflectionTransmission.activeLobes = activeLobes;

        diffTrans = data.diffuseTransmission;
        specTrans = data.specularTransmission;

        // Compute sampling weights.
        float metallicBRDF = data.metallic * (1.f - specTrans);
        float dielectricBSDF = (1.f - data.metallic) * (1.f - specTrans);
        float specularBSDF = specTrans;

        float diffuseWeight = luminance(data.diffuse);
        float specularWeight = luminance(evalFresnelSchlick(data.specular, 1.f, dot(sd.V, sd.N)));

        pDiffuseReflection = (activeLobes & (uint)LobeType::DiffuseReflection) ? diffuseWeight * dielectricBSDF * (1.f - diffTrans) : 0.f;
        pDiffuseTransmission = (activeLobes & (uint)LobeType::DiffuseTransmission) ? diffuseWeight * dielectricBSDF * diffTrans : 0.f;
        pSpecularReflection = (activeLobes & ((uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection)) ? specularWeight * (metallicBRDF + dielectricBSDF) : 0.f;
        pSpecularReflectionTransmission = (activeLobes & ((uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection | (uint)LobeType::SpecularTransmission | (uint)LobeType::DeltaTransmission)) ? specularBSDF : 0.f;

        float normFactor = pDiffuseReflection + pDiffuseTransmission + pSpecularReflection + pSpecularReflectionTransmission;
        if (normFactor > 0.f)
        {
            normFactor = 1.f / normFactor;
            pDiffuseReflection *= normFactor;
            pDiffuseTransmission *= normFactor;
            pSpecularReflection *= normFactor;
            pSpecularReflectionTransmission *= normFactor;
        }
    }

    /** Returns the set of BSDF lobes.
        \param[in] data BSDF parameters.
        \return Returns a set of lobes (see LobeType.slang).
    */
    static uint getLobes(const StandardBSDFData data)
    {
#if EnableDeltaBSDF
        float alpha = data.roughness * data.roughness;
        bool isDelta = alpha < kMinGGXAlpha;
#else
        bool isDelta = false;
#endif
        float diffTrans = data.diffuseTransmission;
        float specTrans = data.specularTransmission;

        uint lobes = isDelta ? (uint)LobeType::DeltaReflection : (uint)LobeType::SpecularReflection;
        if (any(data.diffuse > 0.f) && specTrans < 1.f)
        {
            if (diffTrans < 1.f) lobes |= (uint)LobeType::DiffuseReflection;
            if (diffTrans > 0.f) lobes |= (uint)LobeType::DiffuseTransmission;
        }
        if (specTrans > 0.f) lobes |= (isDelta ? (uint)LobeType::DeltaTransmission : (uint)LobeType::SpecularTransmission);

        return lobes;
    }

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        float3 result = 0.f;
        if (pDiffuseReflection > 0.f) result += (1.f - specTrans) * (1.f - diffTrans) * diffuseReflection.eval(wi, wo, sg);
        if (pDiffuseTransmission > 0.f) result += (1.f - specTrans) * diffTrans * diffuseTransmission.eval(wi, wo, sg);
        if (pSpecularReflection > 0.f) result += (1.f - specTrans) * specularReflection.eval(wi, wo, sg);
        if (pSpecularReflectionTransmission > 0.f) result += specTrans * (specularReflectionTransmission.eval(wi, wo, sg));
        return result;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = 0.f;
        lobe = (uint)LobeType::DiffuseReflection;

        bool valid = false;
        float uSelect = sampleNext1D(sg);

        // Note: The commented-out pdf contributions below are always zero, so no need to compute them.

        if (uSelect < pDiffuseReflection)
        {
            valid = diffuseReflection.sample(wi, wo, pdf, weight, lobe, sg);
            weight /= pDiffuseReflection;
            weight *= (1.f - specTrans) * (1.f - diffTrans);
            pdf *= pDiffuseReflection;
            // if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
            if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
            if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wi, wo);
        }
        else if (uSelect < pDiffuseReflection + pDiffuseTransmission)
        {
            valid = diffuseTransmission.sample(wi, wo, pdf, weight, lobe, sg);
            weight /= pDiffuseTransmission;
            weight *= (1.f - specTrans) * diffTrans;
            pdf *= pDiffuseTransmission;
            // if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
            // if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
            if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wi, wo);
        }
        else if (uSelect < pDiffuseReflection + pDiffuseTransmission + pSpecularReflection)
        {
            valid = specularReflection.sample(wi, wo, pdf, weight, lobe, sg);
            weight /= pSpecularReflection;
            weight *= (1.f - specTrans);
            pdf *= pSpecularReflection;
            if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
            // if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
            if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wi, wo);
        }
        else if (pSpecularReflectionTransmission > 0.f)
        {
            valid = specularReflectionTransmission.sample(wi, wo, pdf, weight, lobe, sg);
            weight /= pSpecularReflectionTransmission;
            weight *= specTrans;
            pdf *= pSpecularReflectionTransmission;
            if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
            if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
            if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
        }

        return valid;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        float pdf = 0.f;
        if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
        if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
        if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
        if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wi, wo);
        return pdf;
    }
};
