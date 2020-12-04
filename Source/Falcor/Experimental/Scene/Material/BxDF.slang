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
#include "BxDFConfig.slangh"

import Scene.ShadingData;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGeneratorInterface;
import Experimental.Scene.Material.Fresnel;
import Experimental.Scene.Material.Microfacet;
__exported import Experimental.Scene.Material.BxDFTypes;

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

// Minimum cos(theta) for the view and light vectors.
// A few functions are not robust for cos(theta) == 0.0.
// TODO: Derive appropriate bounds
static const float kMinCosTheta = 1e-6f;

/** Interface for BxDFs.
    Conventions:
    - wo is the outgoing or scattering direction and points away from the shading location.
    - wi is the incident or light direction and points away from the shading location.
    - the local shading frame has normal N=(0,0,1), tangent T=(1,0,0) and bitangent B=(0,1,0).
    - the outgoing direction is always in the positive hemisphere.
    - evaluating the BxDF always includes the foreshortening term (dot(wi, n) = wi.z).
*/
interface IBxDF
{
    /** Evaluates the BxDF.
        \param[in] wo Outgoing direction.
        \param[in] wi Incident direction.
        \return Returns f(wo, wi) * dot(wi, n).
    */
    float3 eval(float3 wo, float3 wi);

    /** Samples the BxDF.
        \param[in] wo Outgoing direction.
        \param[out] wi Incident direction.
        \param[out] pdf pdf with respect to solid angle for sampling incident direction wi (0 if a delta event is sampled).
        \param[out] weight Sample weight f(wo, wi) * dot(wi, n) / pdf(wi).
        \param[out] lobe Sampled lobe.
        \param[in,out] sg Sample generator.
        \return Returns true if successful.
    */
    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg);

    /** Evaluates the BxDF directional pdf for sampling incident direction wi.
        \param[in] wo Outgoing direction.
        \param[in] wi Incident direction.
        \return Returns the pdf with respect to solid angle for sampling incident direction wi (0 for delta events).
    */
    float evalPdf(float3 wo, float3 wi);
}

/** Lambertian diffuse reflection.
    f_r(wo, wi) = albedo / pi
*/
struct DiffuseReflectionLambert : IBxDF
{
    float3 albedo;  ///< Diffuse albedo.

    float3 eval(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return float3(0);

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
        if (min(wo.z, wi.z) < kMinCosTheta) return 0;

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
        if (min(wo.z, wi.z) < kMinCosTheta) return float3(0);

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
        if (min(wo.z, wi.z) < kMinCosTheta) return 0;

        return M_1_PI * wi.z;
    }

    // private

    // Returns f(wo, wi) * pi.
    float3 evalWeight(float3 wo, float3 wi)
    {
        float3 h = normalize(wo + wi);
        float wiDotH = dot(wi, h);
        float fd90 = 0.5 + 2 * wiDotH * wiDotH * linearRoughness;
        float fd0 = 1;
        float wiScatter = evalFresnelSchlick(fd0, fd90, wi.z).r;
        float woScatter = evalFresnelSchlick(fd0, fd90, wo.z).r;
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
        if (min(wo.z, wi.z) < kMinCosTheta) return float3(0);

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
        if (min(wo.z, wi.z) < kMinCosTheta) return 0;

        return M_1_PI * wi.z;
    }

    // private

    // Returns f(wo, wi) * pi.
    float3 evalWeight(float3 wo, float3 wi)
    {
        float3 h = normalize(wo + wi);
        float wiDotH = dot(wi, h);
        float energyBias = lerp(0, 0.5, linearRoughness);
        float energyFactor = lerp(1, 1.0 / 1.51, linearRoughness);
        float fd90 = energyBias + 2 * wiDotH * wiDotH * linearRoughness;
        float fd0 = 1;
        float wiScatter = evalFresnelSchlick(fd0, fd90, wi.z).r;
        float woScatter = evalFresnelSchlick(fd0, fd90, wo.z).r;
        return albedo * wiScatter * woScatter * energyFactor;
    }
};

/** Specular reflection using microfacets.
*/
struct SpecularReflectionMicrofacet : IBxDF
{
    float3 albedo;      ///< Specular albedo.
    float alpha;        ///< GGX width parameter.
    uint activeLobes;   ///< BSDF lobes to include for sampling and evaluation. See LobeType in BxDFTypes.slang.

    bool hasLobe(LobeType lobe) { return activeLobes & (uint)lobe; }

    float3 eval(float3 wo, float3 wi)
    {
        if (min(wo.z, wi.z) < kMinCosTheta) return float3(0);

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0) return float3(0);
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return float3(0);

        float3 h = normalize(wo + wi);
        float woDotH = dot(wo, h);

        float D = evalNdfGGX(alpha, h.z);
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wo.z, wi.z);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wo.z, wi.z);
#endif
        float3 F = evalFresnelSchlick(albedo, 1, woDotH);
        return F * D * G * 0.25 / wo.z;
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
        if (alpha == 0)
        {
            if (!hasLobe(LobeType::DeltaReflection)) return false;

            wi = float3(-wo.x, -wo.y, wo.z);
            pdf = 0;
            weight = evalFresnelSchlick(albedo, 1, wo.z);
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
        wi = 2 * woDotH * h - wo;
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
        float3 F = evalFresnelSchlick(albedo, 1, woDotH);

        pdf /= (4 * woDotH); // Jacobian of the reflection operator.
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
        if (min(wo.z, wi.z) < kMinCosTheta) return 0;

#if EnableDeltaBSDF
        // Handle delta reflection.
        if (alpha == 0) return 0;
#endif

        if (!hasLobe(LobeType::SpecularReflection)) return 0;

        float3 h = normalize(wo + wi);
        float woDotH = dot(wo, h);
#if EnableVNDFSampling
        float pdf = evalPdfGGX_VNDF(alpha, wo, h);
#else
        float pdf = evalPdfGGX_NDF(alpha, h.z);
#endif
        return pdf / (4 * woDotH);
    }
};

/** Specular reflection and transmission using microfacets.
*/
struct SpecularReflectionTransmissionMicrofacet : IBxDF
{
    float alpha;        ///< GGX width parameter.
    float eta;          ///< Relative index of refraction (e.g. etaI / etaT).
    uint activeLobes;   ///< BSDF lobes to include for sampling and evaluation. See LobeType in BxDFTypes.slang.

    bool hasLobe(LobeType lobe) { return activeLobes & (uint)lobe; }

    float3 eval(float3 wo, float3 wi)
    {
        if (min(wo.z, abs(wi.z)) < kMinCosTheta) return float3(0);

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0) return float3(0);
#endif

        bool isReflection = wi.z > 0;
        if ((isReflection && !hasLobe(LobeType::SpecularReflection)) ||
            (!isReflection && !hasLobe(LobeType::SpecularTransmission))) return float3(0);

        float3 h =
            isReflection ?
            normalize(wo + wi) :
            normalize(-(wo * eta + wi));

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
            return F * D * G * 0.25 / wo.z;
        }
        else
        {
            float sqrtDenom = woDotH + eta * wiDotH;
            float t = eta * eta * abs(woDotH * wiDotH) / (abs(wo.z * wi.z) * sqrtDenom * sqrtDenom);
            return (1 - F) * D * G * t * abs(wi.z);
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
        if (alpha == 0)
        {
            const bool hasReflection = hasLobe(LobeType::DeltaReflection);
            const bool hasTransmission = hasLobe(LobeType::DeltaTransmission);
            if (!(hasReflection || hasTransmission)) return false;

            float cosThetaT;
            float F = evalFresnelDielectric(eta, wo.z, cosThetaT);

            if (!hasReflection) F = 0.f;
            else if (!hasTransmission) F = 1.f;

            bool isReflection = lobeSample < F;

            pdf = 0;
            weight = float3(1);
            wi = isReflection ? float3(-wo.x, -wo.y, wo.z) : float3(-wo.x * eta, -wo.y * eta, -cosThetaT);
            lobe = isReflection ? (uint)LobeType::DeltaReflection : (uint)LobeType::DeltaTransmission;

            if (abs(wi.z) < kMinCosTheta || (wi.z > 0 != isReflection)) return false;

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

        if (!hasReflection) F = 0.f;
        else if (!hasTransmission) F = 1.f;

        bool isReflection = lobeSample < F;

        wi = isReflection ?
            2 * woDotH * h - wo :
            (eta * woDotH - cosThetaT) * h - eta * wo;

        if (abs(wi.z) < kMinCosTheta || (wi.z > 0 != isReflection)) return false;

        float wiDotH = dot(wi, h);

        lobe = isReflection ? (uint)LobeType::SpecularReflection : (uint)LobeType::SpecularTransmission;

#if ExplicitSampleWeights
        // For testing.
        pdf = evalPdf(wo, wi);
        weight = eval(wo, wi) / pdf;
        return true;
#endif

#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wo.z, abs(wi.z));
        float GOverG1wo = evalG1GGX(alpha * alpha, abs(wi.z));
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wo.z, abs(wi.z));
        float GOverG1wo = G * (1.f + evalLambdaGGX(alpha * alpha, wo.z));
#endif

        if (isReflection)
        {
            pdf *= F / (4 * wiDotH); // Jacobian of the reflection operator.
        }
        else
        {
            float sqrtDenom = woDotH + eta * wiDotH;
            pdf *= (1 - F) * eta * eta * wiDotH / (sqrtDenom * sqrtDenom); // Jacobian of the refraction operator.
        }

#if EnableVNDFSampling
        weight = GOverG1wo;
#else
        weight = G * woDotH / (wo.z * h.z);
#endif

        return true;
    }

    float evalPdf(float3 wo, float3 wi)
    {
        if (min(wo.z, abs(wi.z)) < kMinCosTheta) return 0;

#if EnableDeltaBSDF
        // Handle delta reflection/transmission.
        if (alpha == 0) return 0;
#endif

        bool isReflection = wi.z > 0;
        const bool hasReflection = hasLobe(LobeType::SpecularReflection);
        const bool hasTransmission = hasLobe(LobeType::SpecularTransmission);
        if ((isReflection && !hasReflection) || (!isReflection && !hasTransmission)) return 0;

        float3 h =
            isReflection ?
            normalize(wo + wi) :
            normalize(-(wo * eta + wi));

        float woDotH = dot(wo, h);
        float wiDotH = dot(wi, h);

        float F = evalFresnelDielectric(eta, woDotH);

        if (!hasReflection) F = 0.f;
        else if (!hasTransmission) F = 1.f;

#if EnableVNDFSampling
        float pdf = evalPdfGGX_VNDF(alpha, wo, h);
#else
        float pdf = evalPdfGGX_NDF(alpha, h.z);
#endif
        if (isReflection)
        {
            return pdf * F / (4 * wiDotH); // Jacobian of the reflection operator.
        }
        else
        {
            float sqrtDenom = woDotH + eta * wiDotH;
            return pdf * (1 - F) * eta * eta * wiDotH / (sqrtDenom * sqrtDenom); // Jacobian of the refraction operator.
        }
    }
};

/** Layered BSDF used as primary material in Falcor.

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

    SpecularReflectionMicrofacet specularReflection;
    SpecularReflectionTransmissionMicrofacet specularReflectionTransmission;

    float specularTransmission;             ///< Mix between dielectric BRDF and specular BSDF.

    float pDiffuseReflection;               ///< Probability for sampling the diffuse BRDF.
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
        // Setup lobes.
        diffuseReflection.albedo = sd.diffuse;
#if DiffuseBrdf != DiffuseBrdfLambert
        diffuseReflection.linearRoughness = sd.linearRoughness;
#endif

        // Compute GGX alpha.
        float alpha = sd.linearRoughness * sd.linearRoughness;
#if EnableDeltaBSDF
        // Alpha below min alpha value means using delta reflection/transmission.
        if (alpha < kMinGGXAlpha) alpha = 0;
#else
        alpha = max(alpha, kMinGGXAlpha);
#endif

        specularReflection.albedo = sd.specular;
        specularReflection.alpha = alpha;
        specularReflection.activeLobes = sd.activeLobes;

        specularReflectionTransmission.alpha = alpha;
        specularReflectionTransmission.eta = sd.eta;
        specularReflectionTransmission.activeLobes = sd.activeLobes;

        specularTransmission = sd.specularTransmission;

        // Compute sampling weights.
        float metallicBRDF = sd.metallic;
        float specularBSDF = (1 - sd.metallic) * sd.specularTransmission;
        float dielectricBRDF = (1 - sd.metallic) * (1 - sd.specularTransmission);

        float diffuseWeight = luminance(sd.diffuse);
        float specularWeight = luminance(evalFresnelSchlick(sd.specular, 1.f, dot(sd.V, sd.N)));

        pDiffuseReflection = (sd.activeLobes & (uint)LobeType::DiffuseReflection) ? diffuseWeight * dielectricBRDF : 0;
        pSpecularReflection = (sd.activeLobes & ((uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection)) ? specularWeight * (metallicBRDF + dielectricBRDF) : 0;
        pSpecularReflectionTransmission = (sd.activeLobes & ((uint)LobeType::SpecularTransmission | (uint)LobeType::DeltaTransmission)) ? specularBSDF : 0;

        float normFactor = pDiffuseReflection + pSpecularReflection + pSpecularReflectionTransmission;
        if (normFactor > 0)
        {
            normFactor = 1 / normFactor;
            pDiffuseReflection *= normFactor;
            pSpecularReflection *= normFactor;
            pSpecularReflectionTransmission *= normFactor;
        }
    }

    /** Returns the set of BSDF lobes given some shading data.
        \param[in] sd Shading data.
        \return Returns a set of lobes (see LobeType in BxDFTypes.slang).
    */
    static uint getLobes(const ShadingData sd)
    {
#if EnableDeltaBSDF
        float alpha = sd.linearRoughness * sd.linearRoughness;
        bool isDelta = alpha < kMinGGXAlpha;
#else
        bool isDelta = false;
#endif
        uint lobes = isDelta ? (uint)LobeType::DeltaReflection : (uint)LobeType::SpecularReflection;
        if (any(sd.diffuse > 0) && sd.specularTransmission < 1) lobes |= (uint)LobeType::DiffuseReflection;
        if (sd.specularTransmission > 0) lobes |= (isDelta ? (uint)LobeType::DeltaTransmission : (uint)LobeType::SpecularTransmission);

        return lobes;
    }

    float3 eval(float3 wo, float3 wi)
    {
        float3 result = 0;
        if (pDiffuseReflection > 0) result += (1 - specularTransmission) * diffuseReflection.eval(wo, wi);
        if (pSpecularReflection > 0) result += (1 - specularTransmission) * specularReflection.eval(wo, wi);
        if (pSpecularReflectionTransmission > 0) result += specularTransmission * (specularReflectionTransmission.eval(wo, wi));
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

        if (uSelect < pDiffuseReflection)
        {
            valid = diffuseReflection.sample(wo, wi, pdf, weight, lobe, sg);
            weight /= pDiffuseReflection;
            weight *= (1 - specularTransmission);
            pdf *= pDiffuseReflection;
            if (pSpecularReflection > 0) pdf += pSpecularReflection * specularReflection.evalPdf(wo, wi);
            if (pSpecularReflectionTransmission > 0) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wo, wi);
        }
        else if (uSelect < pDiffuseReflection + pSpecularReflection)
        {
            valid = specularReflection.sample(wo, wi, pdf, weight, lobe, sg);
            weight /= pSpecularReflection;
            weight *= (1 - specularTransmission);
            pdf *= pSpecularReflection;
            if (pDiffuseReflection > 0) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wo, wi);
            if (pSpecularReflectionTransmission > 0) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wo, wi);
        }
        else if (pSpecularReflectionTransmission > 0)
        {
            valid = specularReflectionTransmission.sample(wo, wi, pdf, weight, lobe, sg);
            weight /= pSpecularReflectionTransmission;
            weight *= specularTransmission;
            pdf *= pSpecularReflectionTransmission;
            if (pDiffuseReflection > 0) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wo, wi);
            if (pSpecularReflection > 0) pdf += pSpecularReflection * specularReflection.evalPdf(wo, wi);
        }

        return valid;
    }

    float evalPdf(float3 wo, float3 wi)
    {
        float pdf = 0;
        if (pDiffuseReflection > 0) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wo, wi);
        if (pSpecularReflection > 0) pdf += pSpecularReflection * specularReflection.evalPdf(wo, wi);
        if (pSpecularReflectionTransmission > 0) pdf += pSpecularReflectionTransmission * specularReflectionTransmission.evalPdf(wo, wi);
        return pdf;
    }
};
