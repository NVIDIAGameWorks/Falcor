/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
import Experimental.Scene.Material.Fresnel;
import Experimental.Scene.Material.Microfacet;
__exported import Experimental.Scene.Material.IBxDF;

// Enable support for delta reflection/transmission.
#define EnableDeltaBSDF         1

// Enable GGX sampling using the distribution of visible normals (VNDF) instead of classic NDF sampling.
// This should be the default as it has lower variance, disable for testing only.
// TODO: Make default when transmission with VNDF sampling is properly validated
#define EnableVNDFSampling      1

// Enable explicitly computing sampling weights using eval(wo, wi) / evalPdf(wo, wi).
// This is for testing only, as many terms of the equation cancel out allowing to save on computation.
#define ExplicitSampleWeights   0

// We clamp the GGX width parameter to avoid numerical instability.
// In some computations, we can avoid clamps etc. if 1.0 - alpha^2 != 1.0, so the epsilon should be 1.72666361e-4 or larger in fp32.
// The the value below is sufficient to avoid visible artifacts.
// Falcor used to clamp roughness to 0.08 before the clamp was removed for allowing delta events. We continue to use the same threshold.
static const float kMinGGXAlpha = 0.0064f;

/** Lambertian diffuse reflection.
    f_r(wo, wi) = albedo / pi
*/
struct DiffuseReflectionLambert : IBxDF
{
    float3 albedo;  ///< Diffuse albedo.

    float3 eval(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return float3(0.f);

        return M_1_PI * albedo * wi.z;
    }

    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        wi = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wo.z, wi.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = albedo;
        return true;
    }

    float evalPdf(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wi.z;
    }
};

/** Disney's diffuse reflection.
    Based on https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
*/
struct DiffuseReflectionDisney : IBxDF
{
    float3 albedo;          ///< Diffuse albedo.
    float linearRoughness;  ///< Roughness before remapping.

    float3 eval(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return float3(0.f);

        return evalWeight(wo, wi) * M_1_PI * wi.z;
    }

    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        wi = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wo.z, wi.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = evalWeight(wo, wi);
        return true;
    }

    float evalPdf(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wi.z;
    }

    // private

    // Returns f(wo, wi) * pi.
    float3 evalWeight(float3 wo, float3 wi)
    {
        float3 h = normalize(wo + wi);
        float wiDotH = dot(wi, h);
        float fd90 = 0.5f + 2.f * wiDotH * wiDotH * linearRoughness;
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
    float linearRoughness;  ///< Roughness before remapping.

    float3 eval(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return float3(0.f);

        return evalWeight(wo, wi) * M_1_PI * wi.z;
    }

    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        wi = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        lobe = (uint)LobeType::DiffuseReflection;

        if (min(wo.z, wi.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = evalWeight(wo, wi);
        return true;
    }

    float evalPdf(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wi.z;
    }

    // private

    // Returns f(wo, wi) * pi.
    float3 evalWeight(float3 wo, float3 wi)
    {
        float3 h = normalize(wo + wi);
        float wiDotH = dot(wi, h);
        float energyBias = lerp(0.f, 0.5f, linearRoughness);
        float energyFactor = lerp(1.f, 1.f / 1.51f, linearRoughness);
        float fd90 = energyBias + 2.f * wiDotH * wiDotH * linearRoughness;
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

    float3 eval(float3 wo, float3 wi)
    {
        if (min(wo.z, -wi.z) < kMinCosTheta) return float3(0.f);

        return M_1_PI * albedo * -wi.z;
    }

    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        wi = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        wi.z = -wi.z;
        lobe = (uint)LobeType::DiffuseTransmission;

        if (min(wo.z, -wi.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = albedo;
        return true;
    }

    float evalPdf(float3 wo, float3 wi)
    {
        if (min(wo.z, -wi.z) < kMinCosTheta) return 0.f;

        return M_1_PI * -wi.z;
    }
}

/** Specular reflection using microfacets.
*/
struct SpecularReflectionMicrofacet : IBxDF
{
    float3 albedo;      ///< Specular albedo.
    float alpha;        ///< GGX width parameter.
    uint activeLobes;   ///< BSDF lobes to include for sampling and evaluation. See LobeType in IBxDF.slang.

    bool hasLobe(LobeType lobe) { return (activeLobes & (uint)lobe) != 0; }

    float3 eval(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return float3(0.f);

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f) return float3(0.f);
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return float3(0.f);

        float3 h = normalize(wo + wi);
        float woDotH = dot(wo, h);

        float D = evalNdfGGX(alpha, h.z);
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wo.z, wi.z);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wo.z, wi.z);
#endif
        float3 F = evalFresnelSchlick(albedo, 1, woDotH);
        return F * D * G * 0.25f / wo.z;
    }

    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wi = {};
        weight = {};
        pdf = 0.f;
        lobe = (uint)LobeType::SpecularReflection;

        if (wo.z < kMinCosTheta) return false;

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f)
        {
            if (!hasLobe(LobeType::DeltaReflection)) return false;

            wi = float3(-wo.x, -wo.y, wo.z);
            pdf = 0.f;
            weight = evalFresnelSchlick(albedo, 1.f, wo.z);
            lobe = (uint)LobeType::DeltaReflection;
            return true;
        }
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return false;

        // Sample the GGX distribution to find a microfacet normal (half vector).
#if EnableVNDFSampling
        float3 h = sampleGGX_VNDF(alpha, wo, sampleNext2D(sg), pdf);    // pdf = G1(wo) * D(h) * max(0,dot(wo,h)) / wo.z
#else
        float3 h = sampleGGX_NDF(alpha, sampleNext2D(sg), pdf);         // pdf = D(h) * h.z
#endif

        // Reflect the outgoing direction to find the incident direction.
        float woDotH = dot(wo, h);
        wi = 2.f * woDotH * h - wo;
        if (wi.z < kMinCosTheta) return false;

#if ExplicitSampleWeights
        // For testing.
        pdf = evalPdf(wo, wi);
        weight = eval(wo, wi) / pdf;
        lobe = (uint)LobeType::SpecularReflection;
        return true;
#endif

#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wo.z, wi.z);
        float GOverG1wo = evalG1GGX(alpha * alpha, wi.z);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wo.z, wi.z);
        float GOverG1wo = G * (1.f + evalLambdaGGX(alpha * alpha, wo.z));
#endif
        float3 F = evalFresnelSchlick(albedo, 1.f, woDotH);

        pdf /= (4.f * woDotH); // Jacobian of the reflection operator.
#if EnableVNDFSampling
        weight = F * GOverG1wo;
#else
        weight = F * G * woDotH / (wo.z * h.z);
#endif
        lobe = (uint)LobeType::SpecularReflection;
        return true;
    }

    float evalPdf(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return 0.f;

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0.f) return 0.f;
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return 0.f;

        float3 h = normalize(wo + wi);
        float woDotH = dot(wo, h);
#if EnableVNDFSampling
        float pdf = evalPdfGGX_VNDF(alpha, wo, h);
#else
        float pdf = evalPdfGGX_NDF(alpha, h.z);
#endif
        return pdf / (4.f * woDotH);
    }
};

/** Specular reflection and transmission using microfacets.
*/
struct SpecularReflectionTransmissionMicrofacet : IBxDF
{
    float3 transmissionAlbedo;  ///< Transmission albedo.
    float alpha;                ///< GGX width parameter.
    float eta;                  ///< Relative index of refraction (e.g. etaI / etaT).
    uint activeLobes;           ///< BSDF lobes to include for sampling and evaluation. See LobeType in IBxDF.slang.

    bool hasLobe(LobeType lobe) { return (activeLobes & (uint)lobe) != 0; }

    float3 eval(float3 wo, float3 wi)
    {
        if (min(wo.z, abs(wi.z)) < kMinCosTheta) return float3(0.f);

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f) return float3(0.f);
#endif

        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        const bool isReflection = wi.z > 0.f;
        if ((isReflection && !hasReflection) || (!isReflection && !hasTransmission)) return float3(0.f);

        // Compute half-vector and make sure it's in the upper hemisphere.
        float3 h = normalize(wi + wo * (isReflection ? 1.f : eta));
        h *= float(sign(h.z));

        float woDotH = dot(wo, h);
        float wiDotH = dot(wi, h);

        float D = evalNdfGGX(alpha, h.z);
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wo.z, abs(wi.z));
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wo.z, abs(wi.z));
#endif
        float F = evalFresnelDielectric(eta, woDotH);

        if (isReflection)
        {
            return F * D * G * 0.25f / wo.z;
        }
        else
        {
            float sqrtDenom = wiDotH + eta * woDotH;
            float t = eta * eta * woDotH * wiDotH / (wo.z * sqrtDenom * sqrtDenom);
            return transmissionAlbedo * (1.f - F) * D * G * abs(t);
        }
    }

    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wi = {};
        weight = {};
        pdf = 0.f;
        lobe = (uint)LobeType::SpecularReflection;

        if (wo.z < kMinCosTheta) return false;

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
            float F = evalFresnelDielectric(eta, wo.z, cosThetaT);

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
            wi = isReflection ? float3(-wo.x, -wo.y, wo.z) : float3(-wo.x * eta, -wo.y * eta, -cosThetaT);
            lobe = isReflection ? (uint)LobeType::DeltaReflection : (uint)LobeType::DeltaTransmission;

            if (abs(wi.z) < kMinCosTheta || (wi.z > 0.f != isReflection)) return false;

            return true;
        }
#endif

        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        if (!(hasReflection || hasTransmission)) return false;

        // Sample the GGX distribution of (visible) normals. This is our half vector.
#if EnableVNDFSampling
        float3 h = sampleGGX_VNDF(alpha, wo, sampleNext2D(sg), pdf);    // pdf = G1(wo) * D(h) * max(0,dot(wo,h)) / wo.z
#else
        float3 h = sampleGGX_NDF(alpha, sampleNext2D(sg), pdf);         // pdf = D(h) * h.z
#endif

        // Reflect/refract the outgoing direction to find the incident direction.
        float woDotH = dot(wo, h);

        float cosThetaT;
        float F = evalFresnelDielectric(eta, woDotH, cosThetaT);

        bool isReflection = hasReflection;
        if (hasReflection && hasTransmission)
        {
            isReflection = lobeSample < F;
        }
        else if (hasTransmission && F == 1.f)
        {
            return false;
        }

        wi = isReflection ?
            (2.f * woDotH * h - wo) :
            ((eta * woDotH - cosThetaT) * h - eta * wo);

        if (abs(wi.z) < kMinCosTheta || (wi.z > 0.f != isReflection)) return false;

        float wiDotH = dot(wi, h);

        lobe = isReflection ? (uint)LobeType::SpecularReflection : (uint)LobeType::SpecularTransmission;

#if ExplicitSampleWeights
        // For testing.
        pdf = evalPdf(wo, wi);
        weight = pdf > 0.f ? eval(wo, wi) / pdf : float3(0.f);
        return true;
#endif

#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wo.z, abs(wi.z));
        float GOverG1wo = evalG1GGX(alpha * alpha, abs(wi.z));
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wo.z, abs(wi.z));
        float GOverG1wo = G * (1.f + evalLambdaGGX(alpha * alpha, wo.z));
#endif

#if EnableVNDFSampling
        weight = GOverG1wo;
#else
        weight = G * woDotH / (wo.z * h.z);
#endif

        if (isReflection)
        {
            pdf /= 4.f * wiDotH; // Jacobian of the reflection operator.
        }
        else
        {
            float sqrtDenom = wiDotH + eta * woDotH;
            float denom = sqrtDenom * sqrtDenom;
            pdf = (denom > 0.f) ? pdf * abs(wiDotH) / denom : FLT_MAX; // Jacobian of the refraction operator.
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

    float evalPdf(float3 wo, float3 wi)
    {
        if (min(wo.z, abs(wi.z)) < kMinCosTheta) return 0.f;

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0.f) return 0.f;
#endif

        bool isReflection = wi.z > 0.f;
        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        if ((isReflection && !hasReflection) || (!isReflection && !hasTransmission)) return 0.f;

        // Compute half-vector and make sure it's in the upper hemisphere.
        float3 h = normalize(wi + wo * (isReflection ? 1.f : eta));
        h *= float(sign(h.z));

        float woDotH = dot(wo, h);
        float wiDotH = dot(wi, h);

        float F = evalFresnelDielectric(eta, woDotH);

#if EnableVNDFSampling
        float pdf = evalPdfGGX_VNDF(alpha, wo, h);
#else
        float pdf = evalPdfGGX_NDF(alpha, h.z);
#endif
        if (isReflection)
        {
            pdf /= 4.f * wiDotH; // Jacobian of the reflection operator.
        }
        else
        {
            if (wiDotH > 0.f) return 0.f;
            float sqrtDenom = wiDotH + eta * woDotH;
            float denom = sqrtDenom * sqrtDenom;
            pdf = (denom > 0.f) ? pdf * abs(wiDotH) / denom : FLT_MAX; // Jacobian of the refraction operator.
        }

        if (hasReflection && hasTransmission)
        {
            pdf *= isReflection ? F : 1.f - F;
        }

        return pdf;
    }
};

/** Layered BSDF used for the standard material in Falcor.

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

    /** Setup the BSDF for sampling and evaluation.
        TODO: Currently specular reflection and transmission lobes are not properly separated.
        This leads to incorrect behaviour if only the specular reflection or transmission lobe is selected.
        Things work fine as long as both or none are selected.
        \param[in] sd Shading data.
    */
    [mutating] void setup(const ShadingData sd)
    {
        // Use square root if we can assume the shaded object is intersected twice.
        float3 transmissionAlbedo = sd.isThinSurface() ? sd.transmission : sqrt(sd.transmission);

        // Setup lobes.
        diffuseReflection.albedo = sd.diffuse;
#if DiffuseBrdf != DiffuseBrdfLambert
        diffuseReflection.linearRoughness = sd.linearRoughness;
#endif
        diffuseTransmission.albedo = transmissionAlbedo;

        // Compute GGX alpha.
        float alpha = sd.linearRoughness * sd.linearRoughness;
#if EnableDeltaBSDF
        // Alpha below min alpha value means using delta reflection/transmission.
        if (alpha < kMinGGXAlpha) alpha = 0.f;
#else
        alpha = max(alpha, kMinGGXAlpha);
#endif
        const uint activeLobes = sd.getActiveLobes();

        specularReflection.albedo = sd.specular;
        specularReflection.alpha = alpha;
        specularReflection.activeLobes = activeLobes;

        specularReflectionTransmission.transmissionAlbedo = transmissionAlbedo;
        // Transmission through rough interface with IoR == 1 is not well defined, switch to delta reflection instead.
        specularReflectionTransmission.alpha = sd.eta == 1.f ? 0.f : alpha;
        specularReflectionTransmission.eta = sd.eta;
        specularReflectionTransmission.activeLobes = activeLobes;

        diffTrans = sd.diffuseTransmission;
        specTrans = sd.specularTransmission;

        // Compute sampling weights.
        float metallicBRDF = sd.metallic;
        float specularBSDF = (1.f - sd.metallic) * specTrans;
        float dielectricBSDF = (1.f - sd.metallic) * (1.f - specTrans);

        float diffuseWeight = luminance(sd.diffuse);
        float specularWeight = luminance(evalFresnelSchlick(sd.specular, 1.f, dot(sd.V, sd.N)));

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

    /** Returns the set of BSDF lobes given some shading data.
        \param[in] sd Shading data.
        \return Returns a set of lobes (see LobeType in IBxDF.slang).
    */
    static uint getLobes(const ShadingData sd)
    {
#if EnableDeltaBSDF
        float alpha = sd.linearRoughness * sd.linearRoughness;
        bool isDelta = alpha < kMinGGXAlpha;
#else
        bool isDelta = false;
#endif
        float diffTrans = sd.diffuseTransmission;
        float specTrans = sd.specularTransmission;

        uint lobes = isDelta ? (uint)LobeType::DeltaReflection : (uint)LobeType::SpecularReflection;
        if (any(sd.diffuse > 0.f) && specTrans < 1.f)
        {
            if (diffTrans < 1.f) lobes |= (uint)LobeType::DiffuseReflection;
            if (diffTrans > 0.f) lobes |= (uint)LobeType::DiffuseTransmission;
        }
        if (specTrans > 0.f) lobes |= (isDelta ? (uint)LobeType::DeltaTransmission : (uint)LobeType::SpecularTransmission);

        return lobes;
    }

    float3 eval(float3 wo, float3 wi)
    {
        float3 result = 0.f;
        if (pDiffuseReflection > 0.f) result += (1.f - specTrans) * (1.f - diffTrans) * diffuseReflection.eval(wo, wi);
        if (pDiffuseTransmission > 0.f) result += (1.f - specTrans) * diffTrans * diffuseTransmission.eval(wo, wi);
        if (pSpecularReflection > 0.f) result += (1.f - specTrans) * specularReflection.eval(wo, wi);
        if (pSpecularReflectionTransmission > 0.f) result += specTrans * (specularReflectionTransmission.eval(wo, wi));
        return result;
    }

    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wi = {};
        weight = {};
        pdf = 0.f;
        lobe = (uint)LobeType::DiffuseReflection;

        bool valid = false;
        float uSelect = sampleNext1D(sg);

        // Note: The commented-out pdf contributions below are always zero, so no need to compute them.

        if (uSelect < pDiffuseReflection)
        {
            valid = diffuseReflection.sample(wo, wi, pdf, weight, lobe, sg);
            weight /= pDiffuseReflection;
            weight *= (1.f - specTrans) * (1.f - diffTrans);
            pdf *= pDiffuseReflection;
            // if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wo, wi);
            if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wo, wi);
            if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wo, wi);
        }
        else if (uSelect < pDiffuseReflection + pDiffuseTransmission)
        {
            valid = diffuseTransmission.sample(wo, wi, pdf, weight, lobe, sg);
            weight /= pDiffuseTransmission;
            weight *= (1.f - specTrans) * diffTrans;
            pdf *= pDiffuseTransmission;
            // if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wo, wi);
            // if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wo, wi);
            if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wo, wi);
        }
        else if (uSelect < pDiffuseReflection + pDiffuseTransmission + pSpecularReflection)
        {
            valid = specularReflection.sample(wo, wi, pdf, weight, lobe, sg);
            weight /= pSpecularReflection;
            weight *= (1.f - specTrans);
            pdf *= pSpecularReflection;
            if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wo, wi);
            // if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wo, wi);
            if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wo, wi);
        }
        else if (pSpecularReflectionTransmission > 0.f)
        {
            valid = specularReflectionTransmission.sample(wo, wi, pdf, weight, lobe, sg);
            weight /= pSpecularReflectionTransmission;
            weight *= specTrans;
            pdf *= pSpecularReflectionTransmission;
            if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wo, wi);
            if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wo, wi);
            if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wo, wi);
        }

        return valid;
    }

    float evalPdf(float3 wo, float3 wi)
    {
        float pdf = 0.f;
        if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wo, wi);
        if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wo, wi);
        if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wo, wi);
        if (pSpecularReflectionTransmission > 0.f) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wo, wi);
        return pdf;
    }
};
