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

/** This file contains BSDF evaluation and importance sampling functions.

    There are two versions of the code:
    1) by default it uses the original BSDF code in this file,
    2) if _USE_LEGACY_SHADING_CODE == 0, it calls into the new functions in BxDF.slang.

    The new code in BxDF.slang is under development.
    TODO: Remove the legacy code when the new code is well-tested.

*/

#include "Utils/Math/MathConstants.slangh"
#include "BxDFConfig.slangh"

import Scene.ShadingData;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
__exported import Utils.Sampling.SampleGeneratorInterface;
__exported import Experimental.Scene.Material.BxDF;
__exported import Experimental.Scene.Material.MaterialHelpers;

// Run legacy BSDF code by default. The host can override this by defining _USE_LEGACY_SHADING_CODE=0.
// TODO: Remove this when the new code has been properly validated
#ifndef _USE_LEGACY_SHADING_CODE
#define _USE_LEGACY_SHADING_CODE 1
#endif

/** Describes a BSDF sample.
*/
struct BSDFSample
{
    float3  wi;             ///< Incident direction in world space (normalized).
    float   pdf;            ///< pdf with respect to solid angle for sampling incident direction wi.
    float3  weight;         ///< Sample weight f(wo, wi) * dot(wi, n) / pdf(wi).
    uint    lobe;           ///< Sampled lobe. This is a combination of LobeType flags (see BxDFTypes.slang).

    bool isLobe(LobeType type)
    {
        return (lobe & uint(type)) != 0;
    }
};


#if _USE_LEGACY_SHADING_CODE == 0

// ----------------------------------------------------------------------------
// Legacy interface wrapper
// ----------------------------------------------------------------------------

/** Returns the set of available BSDF lobes given some shading data.
    \param[in] sd Shading data.
    \return Returns a set of lobes (see LobeType in BxDFTypes.slang).
*/
uint getBSDFLobes(const ShadingData sd)
{
    return FalcorBSDF::getLobes(sd);
}

/** Evaluates the BSDF multiplied by NdotL for a given incident direction.

    \param[in] sd Shading point data.
    \param[in] L Normalized incident direction from shading point towards light source.
    \return (f_d + f_r) * saturate(dot(N,L))
*/
float3 evalBSDFCosine(const ShadingData sd, float3 L)
{
    float3 wo = toLocal(sd.V, sd);
    float3 wi = toLocal(L, sd);

    FalcorBSDF bsdf;
    bsdf.setup(sd);
    return bsdf.eval(wo, wi);
}

/** Importance sampling of the BSDF.

    Note: The evaluated pdf for the generated sample is expensive to compute, as the pdf is a weighted
    combination of two sampling strategies. If the caller doesn't explicitly need the probability, they
    should be careful not to touch the value so that the compiler can do dead code elimination.

    \param[in] sd Shading point data.
    \param[in] sg Sample generator.
    \param[out] result Generated sample. Only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool sampleBSDF<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
{
    float3 wo = toLocal(sd.V, sd);
    float3 wi;

    FalcorBSDF bsdf;
    bsdf.setup(sd);
    bool valid = bsdf.sample(wo, wi, result.pdf, result.weight, result.lobe, sg);
    result.wi = fromLocal(wi, sd);
    return valid;
}

/** Evaluates the probability density function for both the diffuse and specular sampling strategy.
    \param[in] sd Describes the shading point.
    \param[in] L The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfBSDF(const ShadingData sd, float3 L)
{
    float3 wo = toLocal(sd.V, sd);
    float3 wi = toLocal(L, sd);

    FalcorBSDF bsdf;
    bsdf.setup(sd);
    return bsdf.evalPdf(wo, wi);
}

/** Reference implementation that uses cosine-weighted hemisphere sampling.
    This is for testing purposes only.
    \param[in] sd Shading point data.
    \param[in] sg Sample generator.
    \param[out] result Generated sample. Only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool sampleBSDF_Reference<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
{
    float3 wo = toLocal(sd.V, sd);
    float3 wi = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi

    if (result.pdf == 0.f) return false;

    FalcorBSDF bsdf;
    bsdf.setup(sd);

    result.wi = fromLocal(wi, sd);
    result.weight = bsdf.eval(wo, wi) / result.pdf;
    result.lobe = (uint)LobeType::DiffuseReflection;

    return true;
}


#else // _USE_LEGACY_SHADING_CODE

// ----------------------------------------------------------------------------
// Legacy BRDF functions
// ----------------------------------------------------------------------------

import Experimental.Scene.Material.Fresnel;
import Experimental.Scene.Material.Microfacet;

/** Returns the set of available BSDF lobes given some shading data.
    \param[in] sd Shading data.
    \return Returns a set of lobes (see LobeType in BxDFTypes.slang).
*/
uint getBSDFLobes(const ShadingData sd)
{
    uint lobes = (uint)LobeType::SpecularReflection;
    if (any(sd.diffuse > 0)) lobes |= (uint)LobeType::DiffuseReflection;
    return lobes;
}

/*******************************************************************
                     BSDF evaluation functions
*******************************************************************/

/** Evaluates the Lambertian BRDF.

    \param[in] sd Shading point data.
    \return f_d
*/
float3 evalDiffuseLambert(const ShadingData sd)
{
    return sd.diffuse.rgb * (1 / M_PI);
}

/** Disney's diffuse term.
    Based on https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf

    \param[in] sd Shading point data.
    \param[in] NdotL Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] NdotV Dot product between shading normal and outgoing direction, in positive hemisphere.
    \param[in] LdotH Dot product between half vector and incident direction, in positive hemisphere.
    \return f_d
*/
float3 evalDiffuseDisney(const ShadingData sd, float NdotL, float NdotV, float LdotH)
{
    float fd90 = 0.5 + 2 * LdotH * LdotH * sd.linearRoughness;
    float fd0 = 1;
    float lightScatter = evalFresnelSchlick(fd0, fd90, NdotL).r;
    float viewScatter = evalFresnelSchlick(fd0, fd90, NdotV).r;
    return sd.diffuse.rgb * (viewScatter * lightScatter * (1 / M_PI));
}

/** Frostbites's diffuse term.
    This is Disney's diffuse BRDF with an ad-hoc normalization factor to ensure energy conservation.
    Based on https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf

    \param[in] sd Shading point data.
    \param[in] NdotL Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] NdotV Dot product between shading normal and outgoing direction, in positive hemisphere.
    \param[in] LdotH Dot product between half vector and incident direction, in positive hemisphere.
    \return f_d
*/
float3 evalDiffuseFrostbite(const ShadingData sd, float NdotL, float NdotV, float LdotH)
{
    float energyBias = lerp(0, 0.5, sd.linearRoughness);
    float energyFactor = lerp(1, 1.0 / 1.51, sd.linearRoughness);
    float fd90 = energyBias + 2 * LdotH * LdotH * sd.linearRoughness;
    float fd0 = 1;
    float lightScatter = evalFresnelSchlick(fd0, fd90, NdotL).r;
    float viewScatter = evalFresnelSchlick(fd0, fd90, NdotV).r;
    return sd.diffuse.rgb * (viewScatter * lightScatter * energyFactor * (1 / M_PI));
}

/** Evaluates the diffuse component (f_d) of the BRDF based on currently configured model.

    \param[in] sd Shading point data.
    \param[in] NdotL Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] NdotV Dot product between shading normal and outgoing direction, in positive hemisphere.
    \param[in] LdotH Dot product between half vector and incident direction, in positive hemisphere.
    \return f_d
*/
float3 evalDiffuse(const ShadingData sd, float NdotL, float NdotV, float LdotH)
{
#if DiffuseBrdf == DiffuseBrdfLambert
    return evalDiffuseLambert(sd);
#elif DiffuseBrdf == DiffuseBrdfDisney
    return evalDiffuseDisney(sd, NdotL, NdotV, LdotH);
#elif DiffuseBrdf == DiffuseBrdfFrostbite
    return evalDiffuseFrostbite(sd, NdotL, NdotV, LdotH);
#endif
}

/** Evaluates the separable form of the masking-shadowing function for the GGX normal distribution, using Smith's approximation.

    This optimized form evaluates Equation 98 in http://jcgt.org/published/0003/02/03/paper.pdf divided by (4 * NdotL * NdotV),
    and is used in UE4 according to http://graphicrants.blogspot.fi/2013/08/specular-brdf-reference.html
    The function is only valid for V and L in the positive hemisphere, and should be clamped to 0 otherwise.

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] NdotL Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] NdotV Dot product between shading normal and outgoing direction, in positive hemisphere.
    \return G(L,V,N) / (4 * NdotL * NdotV)
*/
float evalMaskingSmithGGXSeparable_PreDivided(float alpha, float NdotL, float NdotV)
{
    float a2 = alpha * alpha;
    float lambdaV = NdotV + sqrt((-NdotV * a2 + NdotV) * NdotV + a2);
    float lambdaL = NdotL + sqrt((-NdotL * a2 + NdotL) * NdotL + a2);
    return 1.f / (lambdaV * lambdaL);
}

/** Evaluates the height-correlated form of the masking-shadowing function for the GGX normal distribution, using Smith's approximation.

    This optimized form evaluates Equation 99 in http://jcgt.org/published/0003/02/03/paper.pdf divided by (4 * NdotL * NdotV).
    Eric Heitz recommends using it in favor of the separable form as it is more accurate and of similar complexity.
    The function is only valid for V and L in the positive hemisphere, and should be clamped to 0 otherwise.

    \note The function is +inf if NdotL = NdotV = 0. The dot products should be clamped to small epsilon beforehand.
    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] NdotL Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] NdotV Dot product between shading normal and outgoing direction, in positive hemisphere.
    \return G(L,V,N) / (4 * NdotL * NdotV)
*/
float evalMaskingSmithGGXCorrelated_PreDivided(float alpha, float NdotL, float NdotV)
{
    float a2 = alpha * alpha;
    float lambdaV = NdotL * sqrt((-NdotV * a2 + NdotV) * NdotV + a2);   // V,L should be flipped, it's not a mistake
    float lambdaL = NdotV * sqrt((-NdotL * a2 + NdotL) * NdotL + a2);
    return 0.5f / (lambdaV + lambdaL);
}

/** Evaluates the specular term of the BRDF based on the currently configured model.

    \param[in] sd Shading point data.
    \param[in] NdotL Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] NdotV Dot product between shading normal and outgoing direction, in positive hemisphere.
    \param[in] NdotH Dot product between shading normal and half vector, in positive hemisphere.
    \param[in] LdotH Dot product between half vector and incident direction, in positive hemisphere.
    \return f_r
*/
float3 evalSpecular(const ShadingData sd, float NdotL, float NdotV, float NdotH, float LdotH)
{
    float alpha = max(kMinGGXAlpha, sd.ggxAlpha);    // TODO: Derive appropriate epsilon
    float D = evalNdfGGX(alpha, NdotH);
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
    float G = evalMaskingSmithGGXSeparable_PreDivided(alpha, NdotL, NdotV);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
    float G = evalMaskingSmithGGXCorrelated_PreDivided(alpha, NdotL, NdotV);
#endif
    float3 F = evalFresnelSchlick(sd.specular, 1, LdotH);
    return D * G * F;   // Note: G already includes 1/(4*NdotL*NdotV) factor.
}

/** Evaluates the BRDF for a given incident direction.

    \param[in] sd Shading point data.
    \param[in] L Normalized incident direction from shading point towards light source.
    \return f_d + f_r
*/
float3 evalBSDFLobes(const ShadingData sd, float3 L, uint lobes)
{
    // Check that L and V are in the positive hemisphere.
    // The G term on the correlated form is not robust for NdotL = NdotV = 0.0.
    float NdotL = dot(sd.N, L);
    if (min(sd.NdotV, NdotL) < kMinCosTheta) return float3(0, 0, 0);

    // Pre-compute half vector and dot products.
    // TODO: Using saturate() here to be sure all dot products are within bounds.
    // Some can be replaced by clamps on the upper end only (since we check NdotV and NdotL above) or removed altogether.
    float3 H = normalize(sd.V + L);
    float NdotH = saturate(dot(sd.N, H));
    float LdotH = saturate(dot(L, H));
    float NdotV = saturate(sd.NdotV);
    NdotL = saturate(NdotL);

    // Evaluate diffuse and specular terms to compute total throughput.
    float3 thp = 0;
    if ((lobes & (uint)LobeType::DiffuseReflection) != 0) thp += evalDiffuse(sd, NdotL, NdotV, LdotH);
    if ((lobes & (uint)LobeType::SpecularReflection) != 0) thp += evalSpecular(sd, NdotL, NdotV, NdotH, LdotH);

    return thp;
}

float3 evalBSDF(const ShadingData sd, float3 L)
{
    return evalBSDFLobes(sd, L, sd.activeLobes);
}

/** Evaluates the BRDF multiplied by NdotL for a given incident direction.

    \param[in] sd Shading point data.
    \param[in] L Normalized incident direction from shading point towards light source.
    \return (f_d + f_r) * saturate(dot(N,L))
*/
float3 evalBSDFCosine(const ShadingData sd, float3 L)
{
    return evalBSDF(sd, L) * saturate(dot(sd.N, L));
}


/*******************************************************************
                      BSDF sampling functions
*******************************************************************/

/** Samples the cosine-weighted hemisphere at a hit point.

    \param[in] sd Describes the shading point.
    \param[in] u Uniform random number (2D).
    \param[out] pdf Sampling probability (= cos(theta) / pi). Note that pdf goes to zero at the horizon (relative to the shading normal).
    \return Sampled direction in world space.
*/
float3 sampleHemisphereCosine(ShadingData sd, float2 u, out float pdf)
{
    float3 dir = sample_cosine_hemisphere_concentric(u, pdf);
    return fromLocal(dir, sd);
}

/** Samples the GGX (Trowbridge-Reitz) normal distribution function (D) using Walter et al. 2007's method.
    See Eqn 35 & 36 in https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    See Listing A.1 in https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf

    \param[in] sd Describes the shading point.
    \param[in] u Uniform random number (2D).
    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[out] pdf Sampling probability or 0.0 if sample is invalid (L is below the horizon).
    \param[out] VdotH Byproduct of the sampling.
    \param[out] NdotH Byproduct of the sampling.
    \return Sampled direction in world space.
*/
float3 sampleNdfGGX_Walter(const ShadingData sd, const float2 u, const float alpha, out float pdf, out float VdotH, out float NdotH)
{
    // Draw sample from D(H) * NdotH.
    float a2 = alpha * alpha;
    float cosThetaHSqr = min((1 - u.x) / ((a2 - 1) * u.x + 1), 1.0f); // Clamp to avoid 1.0+epsilon causing NaNs below.
    float cosThetaH = sqrt(cosThetaHSqr);
    float sinThetaH = sqrt(1 - cosThetaHSqr);
    float phiH = u.y * M_2PI;

    // Convert half vector to world space.
    float3 H = float3(sinThetaH * cos(phiH), sinThetaH * sin(phiH), cosThetaH);
    H = fromLocal(H, sd);

    NdotH = cosThetaH;
    VdotH = dot(sd.V, H);

    // Compute incident direction L by reflecting V about H.
    float3 L = normalize(2.f * VdotH * H - sd.V);   // Note: L is already of unit length, but normalize to reduce the risk of round-off errors.

    // Evaluate the pdf.
    // The pdf in half vector space is pdf = D(H) * NdotH, which we multiply by the Jacobian of the half-vector transform.
    float d = (a2 - 1) * cosThetaHSqr + 1;
    pdf = (a2 * NdotH) / (d * d * VdotH * M_4PI);
    //pdf = evalNdfGGX(alpha, NdotH) * NdotH / (4.f * VdotH);   // For reference, identical to the line above

    // Reject sample if L is in the lower hemisphere. Note: We should check dot(N,V) elsewhere.
    float NdotL = dot(sd.N, L);
    if (NdotL < kMinCosTheta)
    {
        pdf = 0.f;
    }

    return L;
}

/** Evaluates the probability density function for the diffuse sampling strategy.
    \param[in] sd Describes the shading point.
    \param[in] L The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfDiffuse(const ShadingData sd, const float3 L)
{
    // We're using cosine-weighted sampling over the hemisphere.
    float NdotL = dot(sd.N, L);
    return NdotL < kMinCosTheta ? 0.f : NdotL * M_1_PI;
}

/** Importance sampling of the diffuse lobe of the BSDF times dot(N,L).

    \param[in] sd Describes the shading point.
    \param[in] u Uniform random number (2D).
    \param[out] result Generated sample with pre-evaluated weight (= f_d * dot(N,L) / pdf).
*/
void sampleDiffuse(const ShadingData sd, const float2 u, out BSDFSample result)
{
    // Sample the diffuse lobe with pdf = NdotL / pi.
    // The Disney diffuse is a Lambert times a Fresnel term to increase grazing retroreflection. The latter is not included in the pdf.
    // TODO: Derive sampling method that better approminates the Disney diffuse lobe.
    result.wi = sampleHemisphereCosine(sd, u, result.pdf);
    result.lobe = (uint)LobeType::DiffuseReflection;

    // Check that L and V are in the positive hemisphere.
    float NdotL = dot(sd.N, result.wi);
    if (min(sd.NdotV, NdotL) < kMinCosTheta)
    {
        result.pdf = 0.f;
        result.weight = float3(0);
        return;
    }

    // Compute weight. Note that NdotL cancels out by the pdf.
    result.weight = evalBSDFLobes(sd, result.wi, (uint)LobeType::DiffuseReflection) * M_PI;
}

/** Evaluates the probability density function for the specular sampling strategy.
    \param[in] sd Describes the shading point.
    \param[in] L The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfSpecular(const ShadingData sd, const float3 L)
{
    // We're never generating samples for back-facing V or L.
    float NdotL = dot(sd.N, L);
    if (min(sd.NdotV, NdotL) < kMinCosTheta)
    {
        return 0.f;
    }

    float3 H = normalize(sd.V + L);
    float NdotH = saturate(dot(sd.N, H));
    float VdotH = saturate(dot(sd.V, H));

    // We're sampling the GGX distribution with pdf = D(H) * NdotH / (4.f * VdotH).
    float alpha = max(kMinGGXAlpha, sd.ggxAlpha);
    return evalNdfGGX(alpha, NdotH) * NdotH / (4.f * VdotH);
}

/** Importance sampling the specular lobe of the BSDF.

    \param[in] sd Describes the shading point.
    \param[in] u Uniform random number (2D).
    \param[out] result Generated sample with pre-evaluated weight (= f_r * dot(N,L) / pdf).
*/
void sampleSpecular(const ShadingData sd, const float2 u, out BSDFSample result)
{
    // Sample the GGX distribution with pdf = D(H) * NdotH / (4.f * VdotH).
    float alpha = max(kMinGGXAlpha, sd.ggxAlpha);
    float VdotH, NdotH;
    result.wi = sampleNdfGGX_Walter(sd, u, alpha, result.pdf, VdotH, NdotH);
    result.lobe = (uint)LobeType::SpecularReflection;

    // Check that L and V are in the positive hemisphere.
    float NdotL = dot(sd.N, result.wi);
    if (min(sd.NdotV, NdotL) < kMinCosTheta)
    {
        result.pdf = 0.f;
        result.weight = float3(0);
        return;
    }

    // Pre-compute half vector and dot products.
    // TODO: Look into necessary conditions for numerical robustness below.
    float NdotV = saturate(sd.NdotV);
    NdotL = saturate(NdotL);
    NdotH = saturate(NdotH);
    VdotH = saturate(VdotH);

    // Compute weight. Note that D cancels out by the pdf.
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
    float G = evalMaskingSmithGGXSeparable_PreDivided(alpha, NdotL, NdotV); // Note: G already includes 1/(4*NdotL*NdotV) factor.
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
    float G = evalMaskingSmithGGXCorrelated_PreDivided(alpha, NdotL, NdotV);
#endif
    float3 F = evalFresnelSchlick(sd.specular, 1, VdotH);
    result.weight = F * (G * NdotL * VdotH * 4.f) / NdotH;
    //result.weight = evalBSDFCosine(sd, result.wi, (uint)LobeType::SpecularReflection) / result.pdf;
}

float getDiffuseProbability(const ShadingData sd)
{
    float diffuseWeight = (sd.activeLobes & (uint)LobeType::DiffuseReflection) ? (1.f - sd.metallic) * luminance(sd.diffuse) : 0.f;
    float specularWeight = (sd.activeLobes & (uint)LobeType::SpecularReflection) ? luminance(evalFresnelSchlick(sd.specular, 1.f, dot(sd.V, sd.N))) : 0.f;
    float pDiffuse = (diffuseWeight + specularWeight) > 0.f ? diffuseWeight / (diffuseWeight + specularWeight) : 1.f;
    return pDiffuse;
}

/** Evaluates the probability density function for both the diffuse and specular sampling strategy.
    \param[in] sd Describes the shading point.
    \param[in] L The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfBSDF(const ShadingData sd, float3 L)
{
    float pDiffuse = getDiffuseProbability(sd);

    // Evaluate the pdf for the sample as a linear combination of the two sampling strategies' pdfs.
    return pDiffuse * evalPdfDiffuse(sd, L) + (1.0f - pDiffuse) * evalPdfSpecular(sd, L);
}

/** Importance sampling of the BSDF.

    Note: The evaluated pdf for the generated sample is expensive to compute, as the pdf is a weighted
    combination of two sampling strategies. If the caller doesn't explicitly need the probability, they
    should be careful not to touch the value so that the compiler can do dead code elimination.

    \param[in] sd Shading point data.
    \param[in] sg Sample generator.
    \param[out] result Generated sample. Only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool sampleBSDF<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
{
    // Draw uniform random numbers for lobe selection (1D) and sampling (2D).
    const float2 u = sampleNext2D(sg);
    const float uSelect = sampleNext1D(sg);

    float pDiffuse = getDiffuseProbability(sd);

    float pmfSelectedLobe;
    float pdfOther;

    // Randomly select which lobe to sample.
    if (uSelect < pDiffuse)
    {
        // Sample diffuse lobe.
        sampleDiffuse(sd, u, result);
        pmfSelectedLobe = pDiffuse;

        // Evaluate the pdf of the other sampling strategy.
        pdfOther = evalPdfSpecular(sd, result.wi);
    }
    else
    {
        // Sample specular lobe.
        sampleSpecular(sd, u, result);
        pmfSelectedLobe = 1.f - pDiffuse;

        // Evaluate the pdf of the other sampling strategy.
        pdfOther = evalPdfDiffuse(sd, result.wi);
    }

    // Evaluate the pdf for the sample as a linear combination of the two sampling strategies' pdfs.
    result.pdf = pmfSelectedLobe * result.pdf + (1.f - pmfSelectedLobe) * pdfOther;

    // Divide weight by the probability of the sampled lobe.
    result.weight /= pmfSelectedLobe;

    return result.pdf > 0.f;
}

/** Reference implementation that uses cosine-weighted hemisphere sampling.
    This is for testing purposes only.
    \param[in] sd Shading point data.
    \param[in] sg Sample generator.
    \param[out] result Generated sample. Only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool sampleBSDF_Reference<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
{
    float pdf;
    float3 dir = sampleHemisphereCosine(sd, sampleNext2D(sg), pdf); // pdf = cos(theta) / pi

    // Check that L and V are in the positive hemisphere, reset pdf to zero otherwise.
    // This is necessary for consistency with BRDF sampling.
    float NdotL = dot(sd.N, dir);
    if (min(sd.NdotV, NdotL) < kMinCosTheta) pdf = 0.f;

    result.wi = dir;
    result.weight = evalBSDF(sd, dir) * M_PI; // dot(N,L) / pdf = pi
    result.pdf = pdf;
    result.lobe = (uint)LobeType::DiffuseReflection;

    return pdf > 0.f;
}

#endif // _USE_LEGACY_SHADING_CODE
