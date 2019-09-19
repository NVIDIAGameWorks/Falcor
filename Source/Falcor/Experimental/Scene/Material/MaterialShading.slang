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

/** This file contains BRDF evaluation and importance sampling functions.

    It is work in progress.
*/

#include "Utils/Math/MathConstants.slang"

__import Shading;
__import Utils.Math.MathHelpers;
__exported import Utils.Sampling.SampleGenerator;
__exported import Experimental.Scene.Material.MaterialHelpers;

// NOTE: These are defined in BRDF.slang, but macro definitions are not exported so not visible here!
#define DiffuseBrdfLambert      0
#define DiffuseBrdfDisney       1
#define DiffuseBrdfFrostbite    2

#define DiffuseBrdf DiffuseBrdfFrostbite

#define SpecularMaskingFunctionSmithGGXSeparable    0       ///< Used by UE4.
#define SpecularMaskingFunctionSmithGGXCorrelated   1       ///< Used by Frostbite. This is the more accurate form (default).

#define SpecularMaskingFunction SpecularMaskingFunctionSmithGGXCorrelated

// Disable sampling of the specular component for debugging
//#define DisableSpecularSampling

// TODO: Remove this, replace by accurately derived epsilons where needed.
#define epsilon 1e-5f

// We clamp the GGX width parameter (alpha_g) to avoid numerical instability. 
// In some computations, we can avoid clamps etc. if 1.0 - alpha^2 != 1.0, so the epsilon should be 1.72666361e-4 or larger in fp32.
#define epsilon_alpha_g 0.001f

// Minimum cos(theta) for the view and light vectors.
// A few functions are not robust for cos(theta) == 0.0.
// TODO: Derive appropriate bounds
static const float kMinCosTheta = 1e-6f;

// --------------------------------------------------------------
// TODO: Most function below exist in two versions, one for Falcor's BRDF (Frostbite) and one for the original Disney BRDF.
// We should refactor this code to use Slang interfaces and share common functions.
// We likely want something similar to C++ inheritance for this, so that we can have shared implementations in the "base class",
// and override the parts that are specific to the different models. Talk to Tim about this.
// --------------------------------------------------------------

/*******************************************************************
                     BSDF evaluation functions
*******************************************************************/

/** Evaluates the Fresnel term using Schlick's approximation.
    Introduced in http://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf

    The Fresnel term equals f0 at normal incidence, and approaches f90=1.0 at 90 degrees.
    The formulation below is generalized to allow both f0 and f90 to be specified.

    \param[in] f0 Specular reflectance at normal incidence (0 degrees).
    \param[in] f90 Reflectance at orthogonal incidence (90 degrees), which should be 1.0 for specular surface reflection.
    \param[in] cosTheta Cosine of angle between microfacet normal and incident direction (LdotH).
    \return Fresnel term.
*/
float3 evalFresnelSchlick(float3 f0, float3 f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow(1 - cosTheta, 5);
}

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

/** Evaluates the GGX (Trowbridge-Reitz) normal distribution function (D).

    Introduced by Trowbridge and Reitz, "Average irregularity representation of a rough surface for ray reflection", Journal of the Optical Society of America, vol. 65(5), 1975.
    See the correct normalization factor in Walter et al. https://dl.acm.org/citation.cfm?id=2383874
    We use the simpler, but equivalent expression in Eqn 19 from http://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf

    For microfacet models, D is evaluated for the direction h to find the density of potentially active microfacets (those for which microfacet normal m = h).
    The 'alpha' parameter is the standard GGX width, e.g., it is the square of the linear roughness parameter in Disney's BRDF.
    Note there is a singularity (0/0 = NaN) at NdotH = 1 and alpha = 0, so alpha should be clamped to some epsilon.

    \param[in] NdotH Dot product between shading normal and half vector, in positive hemisphere.
    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \return D(H)
*/
float evalNdfGGX(float NdotH, float alpha)
{
    float a2 = alpha * alpha;
    float d = ((NdotH * a2 - NdotH) * NdotH + 1);
    return a2 / (d * d * M_PI);
}

/** Evaluates the separable form of the masking-shadowing function for the GGX normal distribution, using Smith's approximation.

    This optimized form evaluates Equation 98 in http://jcgt.org/published/0003/02/03/paper.pdf divided by (4 * NdotL * NdotV),
    and is used in UE4 according to http://graphicrants.blogspot.fi/2013/08/specular-brdf-reference.html
    The function is only valid for V and L in the positive hemisphere, and should be clamped to 0 otherwise.

    \param[in] NdotL Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] NdotV Dot product between shading normal and outgoing direction, in positive hemisphere.
    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \return G(L,V,N) / (4 * NdotL * NdotV)
*/
float evalMaskingSmithGGXSeparable(float NdotL, float NdotV, float alpha)
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
    \param[in] NdotL Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] NdotV Dot product between shading normal and outgoing direction, in positive hemisphere.
    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \return G(L,V,N) / (4 * NdotL * NdotV)
*/
float evalMaskingSmithGGXCorrelated(float NdotL, float NdotV, float alpha)
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
    float alpha = max(epsilon_alpha_g, sd.ggxAlpha); // TODO: Derive appropriate epsilon
    float D = evalNdfGGX(NdotH, alpha);
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
    float G = evalMaskingSmithGGXSeparable(NdotL, NdotV, alpha);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
    float G = evalMaskingSmithGGXCorrelated(NdotL, NdotV, alpha);
#endif
    float3 F = evalFresnelSchlick(sd.specular, 1, LdotH);
    return D * G * F;   // Note: G already includes 1/(4*NdotL*NdotV) factor.
}

/** Evaluates the specular term of the BRDF based on the Disney's original model.

    The implementation matches exactly Disney's BRDF Explorer https://github.com/wdas/brdf,
    (using its default values for all parameters we don't currently support).

    Note: Disney originally used a remapping alpha = pow(0.5 + 0.5*sd.linearRoughness, 2.f) for the masking-shadowing term (G) in their 2012 course notes.
    This was later removed in their addendum and from the BRDF Explorer. The BRDF slice images in their course notes were rendered _with_ the remapping.

    \param[in] sd Shading point data.
    \param[in] NdotL Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] NdotV Dot product between shading normal and outgoing direction, in positive hemisphere.
    \param[in] NdotH Dot product between shading normal and half vector, in positive hemisphere.
    \param[in] LdotH Dot product between half vector and incident direction, in positive hemisphere.
    \return f_r
*/
float3 evalSpecularDisney(const ShadingData sd, float NdotL, float NdotV, float NdotH, float LdotH)
{
    float alpha = max(epsilon_alpha_g, sd.ggxAlpha);
    float D = evalNdfGGX(NdotH, alpha);
    float G = evalMaskingSmithGGXSeparable(NdotL, NdotV, alpha);
    float3 F = evalFresnelSchlick(sd.specular, 1, LdotH);
    return D * G * F;     // Note: G already includes 1/(4*NdotL*NdotV) factor.
}

/** Evaluates the BRDF for a given incident direction.

    \param[in] sd Shading point data.
    \param[in] L Normalized incident direction from shading point towards light source.
    \return f_d + f_r
*/
float3 evalBRDF(const ShadingData sd, float3 L, const bool enableDiffuse = true, const bool enableSpecular = true)
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
    float3 thp = enableDiffuse ? evalDiffuse(sd, NdotL, NdotV, LdotH) : float3(0);
    if (enableSpecular) thp += evalSpecular(sd, NdotL, NdotV, NdotH, LdotH);

    return thp;
}

/** Evaluates the BRDF for a given incident direction.
    This version explicitly uses Disney's BRDF.
    
    \param[in] sd Shading point data.
    \param[in] L Normalized incident direction from shading point towards light source.
    \return f_d + f_r
*/
float3 evalDisneyBRDF(const ShadingData sd, float3 L, const bool enableDiffuse = true, const bool enableSpecular = true)
{
    // Check that L and V are in the positive hemisphere.
    // Everything below is robust for NdotL = NdotV = 0.0, as long as alpha is clamped to an epsilon.
    // But leaving the check here to be consistent across different BRDF versions.
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
    float3 thp = enableDiffuse ? evalDiffuseDisney(sd, NdotL, NdotV, LdotH) : float3(0);
    if (enableSpecular) thp += evalSpecularDisney(sd, NdotL, NdotV, NdotH, LdotH);

    // TODO: pbrt seems to include an extra multiplication by metallic * baseColor (or similar).
    // That doesn't exist in Disney's BRDF explorer, which matches our implementation here exactly.

    return thp;
}

/** Evaluates the BRDF multiplied by NdotL for a given incident direction.

    \param[in] sd Shading point data.
    \param[in] L Normalized incident direction from shading point towards light source.
    \return (f_d + f_r) * saturate(dot(N,L))
*/
float3 evalBRDFCosine(const ShadingData sd, float3 L, const bool enableDiffuse = true, const bool enableSpecular = true)
{
    return evalBRDF(sd, L, enableDiffuse, enableSpecular) * saturate(dot(sd.N, L));
}

/** Evaluates the BRDF multiplied by NdotL for a given incident direction.
    This version explicitly uses Disney's BRDF.

    \param[in] sd Shading point data.
    \param[in] L Normalized incident direction from shading point towards light source.
    \return (f_d + f_r) * saturate(dot(N,L))
*/
float3 evalDisneyBRDFCosine(const ShadingData sd, float3 L, const bool enableDiffuse = true, const bool enableSpecular = true)
{
    return evalDisneyBRDF(sd, L, enableDiffuse, enableSpecular) * saturate(dot(sd.N, L));
}


/*******************************************************************
                      BSDF sampling functions
*******************************************************************/

/** Describes a BRDF sample.

    Be careful not to use the returned pdf value unless necessary, as it can
    be expensive to compute. It is often faster to check if thp != 0 to see
    if a sample is valid, if the pdf is not otherwise needed.
*/
struct BRDFSample
{
    float3  dir;           ///< Sampling direction in world space (normalized).
    float   pdf;           ///< Evaluated pdf with respect to solid angle from the shading point, or 0.0 if sample is invalid.
    float3  thp;           ///< Evaluated weight for the chosen direction (= f(V,L) * dot(N,L) / pdf).
};

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
    //pdf = evalNdfGGX(NdotH, alpha) * NdotH / (4.f * VdotH);   // For reference, identical to the line above

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

/** Evaluates the probability density function for the Disney diffuse sampling strategy.
    \param[in] sd Describes the shading point.
    \param[in] L The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfDiffuseDisney(const ShadingData sd, const float3 L)
{
    return evalPdfDiffuse(sd, L);
}

/** Importance sampling of the diffuse lobe of the BRDF times dot(N,L).

    \param[in] sd Describes the shading point.
    \param[in] u Uniform random number (2D).
    \param[out] result Generated sample with pre-evaluated weight (= f_d * dot(N,L) / pdf).
*/
void sampleDiffuse(const ShadingData sd, const float2 u, out BRDFSample result)
{
    // Sample the diffuse lobe with pdf = NdotL / pi. 
    // The Disney diffuse is a Lambert times a Fresnel term to increase grazing retroreflection. The latter is not included in the pdf.
    // TODO: Derive sampling method that better approminates the Disney diffuse lobe.
    result.dir = sampleHemisphereCosine(sd, u, result.pdf);

    // Check that L and V are in the positive hemisphere.
    float NdotL = dot(sd.N, result.dir);
    if (min(sd.NdotV, NdotL) < kMinCosTheta)
    {
        result.pdf = 0.f;
        result.thp = float3(0);
        return;
    }

    // Compute weight. Note that NdotL cancels out by the pdf.
    result.thp = evalBRDF(sd, result.dir, true, false) * M_PI;
}

/** Importance sampling of the diffuse lobe of Disney's BRDF times dot(N,L).

    \param[in] sd Describes the shading point.
    \param[in] u Uniform random number (2D).
    \param[out] result Generated sample with pre-evaluated weight (= f_d * dot(N,L) / pdf).
*/
void sampleDiffuseDisney(const ShadingData sd, const float2 u, out BRDFSample result)
{
    // Sample the diffuse lobe with pdf = NdotL / pi. 
    // The Disney diffuse is a Lambert times a Fresnel term to increase grazing retroreflection. The latter is not included in the pdf.
    // TODO: Derive sampling method that better approminates the Disney diffuse lobe.
    result.dir = sampleHemisphereCosine(sd, u, result.pdf);

    // Check that L and V are in the positive hemisphere.
    float NdotL = dot(sd.N, result.dir);
    if (min(sd.NdotV, NdotL) < kMinCosTheta)
    {
        result.pdf = 0.f;
        result.thp = float3(0);
        return;
    }

    // Compute weight. Note that NdotL cancels out by the pdf.
    result.thp = evalDisneyBRDF(sd, result.dir, true, false) * M_PI;
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
    float alpha = max(epsilon_alpha_g, sd.ggxAlpha);
    return evalNdfGGX(NdotH, alpha) * NdotH / (4.f * VdotH);
}

/** Evaluates the probability density function for the Disney specular sampling strategy.
    \param[in] sd Describes the shading point.
    \param[in] L The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfSpecularDisney(const ShadingData sd, const float3 L)
{
    return evalPdfSpecular(sd, L);
}

/** Importance sampling the specular lobe of the BRDF.

    \param[in] sd Describes the shading point.
    \param[in] u Uniform random number (2D).
    \param[out] result Generated sample with pre-evaluated weight (= f_r * dot(N,L) / pdf).
*/
void sampleSpecular(const ShadingData sd, const float2 u, out BRDFSample result)
{
    // Sample the GGX distribution with pdf = D(H) * NdotH / (4.f * VdotH).
    float alpha = max(epsilon_alpha_g, sd.ggxAlpha);
    float VdotH, NdotH;
    result.dir = sampleNdfGGX_Walter(sd, u, alpha, result.pdf, VdotH, NdotH);

    // Check that L and V are in the positive hemisphere.
    float NdotL = dot(sd.N, result.dir);
    if (min(sd.NdotV, NdotL) < kMinCosTheta)
    {
        result.pdf = 0.f;
        result.thp = float3(0);
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
    float G = evalMaskingSmithGGXSeparable(NdotL, NdotV, alpha); // Note: G already includes 1/(4*NdotL*NdotV) factor.
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
    float G = evalMaskingSmithGGXCorrelated(NdotL, NdotV, alpha);
#endif
    float3 F = evalFresnelSchlick(sd.specular, 1, VdotH);
    result.thp = F * (G * NdotL * VdotH * 4.f) / NdotH;
    //result.thp = evalBRDFCosine(sd, result.dir, false, true) / result.pdf;
}

/** Importance sampling the specular lobe of Disney's BRDF.

    \param[in] sd Describes the shading point.
    \param[in] u Uniform random number (2D).
    \param[out] result Generated sample with pre-evaluated weight (= f_r * dot(N,L) / pdf).
*/
void sampleSpecularDisney(const ShadingData sd, const float2 u, out BRDFSample result)
{
    // Sample the GGX distribution with pdf = D(H) * NdotH / (4.f * VdotH).
    float alpha = max(epsilon_alpha_g, sd.ggxAlpha);
    float VdotH, NdotH;
    result.dir = sampleNdfGGX_Walter(sd, u, alpha, result.pdf, VdotH, NdotH);

    // Check that L and V are in the positive hemisphere.
    float NdotL = dot(sd.N, result.dir);
    if (min(sd.NdotV, NdotL) < kMinCosTheta)
    {
        result.pdf = 0.f;
        result.thp = float3(0);
        return;
    }

    // Pre-compute half vector and dot products.
    // TODO: Look into necessary conditions for numerical robustness below.
    float NdotV = saturate(sd.NdotV);
    NdotL = saturate(NdotL);
    NdotH = saturate(NdotH);
    VdotH = saturate(VdotH);

    // Compute weight. Note that D cancels out by the pdf.
    // TODO: Make choice of masking-shadowing function configurable.
    float G = evalMaskingSmithGGXSeparable(NdotL, NdotV, alpha);    // Note: G already includes 1/(4*NdotL*NdotV) factor.
    float3 F = evalFresnelSchlick(sd.specular, 1, VdotH);
    result.thp = F * (G * NdotL * VdotH * 4.f) / NdotH;
    //result.thp = evalDisneyBRDFCosine(sd, result.dir, false, true) / result.pdf;
}

/** Evaluates the probability density function for both the diffuse and specular sampling strategy.
    \param[in] sd Describes the shading point.
    \param[in] L The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfBRDF(const ShadingData sd, const float3 L)
{
    float pDiffuse = 0.5f; // TODO: Better probabilities

    // Evaluate the pdf for the sample as a linear combination of the two sampling strategies' pdfs.
    return pDiffuse * evalPdfDiffuse(sd, L) + (1.0f - pDiffuse) * evalPdfSpecular(sd, L);
}

/** Evaluates the probability density function for both the Disney diffuse and specular sampling strategy.
    \param[in] sd Describes the shading point.
    \param[in] L The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfDisneyBRDF(const ShadingData sd, const float3 L)
{
    float pDiffuse = 0.5f;      // TODO: Better probabilities

    // Evaluate the pdf for the sample as a linear combination of the two sampling strategies' pdfs.
    return pDiffuse * evalPdfDiffuseDisney(sd, L) + (1.0f - pDiffuse) * evalPdfSpecularDisney(sd, L);
}

/** Importance sampling of the BRDF.

    Note: The evaluated pdf for the generated sample is expensive to compute, as the pdf is a weighted
    combination of two sampling strategies. If the caller doesn't explicitly need the probability, they
    should be careful not to touch the value so that the compiler can do dead code elimination.

    \param[in] sd Shading point data.
    \param[in] sg SampleGenerator.
    \param[out] result Generated sample.
*/
void sampleBRDF(const ShadingData sd, inout SampleGenerator sg, out BRDFSample result)
{
    // Draw uniform random numbers for lobe selection (1D) and sampling (2D).
    const float2 u = sampleNext2D(sg);
    const float uSelect = sampleNext1D(sg);

    float pDiffuse = 0.5f; // TODO: Better probabilities

    float pmfSelectedLobe;
    float pdfOther;

    // Randomly select which lobe to sample.
    if (uSelect < pDiffuse)
    {
        // Sample diffuse lobe.
        sampleDiffuse(sd, u, result);
        pmfSelectedLobe = pDiffuse;

        // Evaluate the pdf of the other sampling strategy.
        pdfOther = evalPdfSpecular(sd, result.dir);
    }
    else
    {
        // Sample specular lobe.
        sampleSpecular(sd, u, result);
        pmfSelectedLobe = 1.f - pDiffuse;

        // Evaluate the pdf of the other sampling strategy.
        pdfOther = evalPdfDiffuse(sd, result.dir);
    }

    // Evaluate the pdf for the sample as a linear combination of the two sampling strategies' pdfs.
    result.pdf = pmfSelectedLobe * result.pdf + (1.f - pmfSelectedLobe) * pdfOther;

    // Divide weight by the probability of the sampled lobe.
    result.thp /= pmfSelectedLobe;
}

/** Importance sampling of the Disney BRDF.

    Note: The evaluated pdf for the generated sample is expensive to compute, as the pdf is a weighted
    combination of two sampling strategies. If the caller doesn't explicitly need the probability, they
    should be careful not to touch the value so that the compiler can do dead code elimination.

    \param[in] sd Shading point data.
    \param[in] sg SampleGenerator.
    \param[out] result Generated sample.
*/
void sampleDisneyBRDF(const ShadingData sd, inout SampleGenerator sg, out BRDFSample result)
{
    // Draw uniform random numbers for lobe selection (1D) and sampling (2D).
    const float2 u = sampleNext2D(sg);
    const float uSelect = sampleNext1D(sg);

    float pDiffuse = 0.5f;      // TODO: Better probabilities

    float pmfSelectedLobe;
    float pdfOther;

    // Randomly select which lobe to sample.
    if (uSelect < pDiffuse)
    {
        // Sample diffuse lobe.
        sampleDiffuseDisney(sd, u, result);
        pmfSelectedLobe = pDiffuse;

        // Evaluate the pdf of the other sampling strategy.
        pdfOther = evalPdfSpecularDisney(sd, result.dir);
    }
    else
    {
        // Sample specular lobe.
        sampleSpecularDisney(sd, u, result);
        pmfSelectedLobe = 1.f - pDiffuse;

        // Evaluate the pdf of the other sampling strategy.
        pdfOther = evalPdfDiffuseDisney(sd, result.dir);
    }

    // Evaluate the pdf for the sample as a linear combination of the two sampling strategies' pdfs.
    result.pdf = pmfSelectedLobe * result.pdf + (1.f - pmfSelectedLobe) * pdfOther;

    // Divide weight by the probability of the sampled lobe.
    result.thp /= pmfSelectedLobe;
}


// ----------------------------------------------------------------------
// TODO: Validate all functions below!! 
// The definitions of the BRDF evaluation/clamps/etc. above have changed.
// ----------------------------------------------------------------------

/** Result of material sampling.
*/
struct MaterialSamplingResult
{
    float3 wi;                  ///< Incident direction after importance sampling the BRDF.
    float pdf;                  ///< Probability density function for choosing the incident direction.
    float3 thp;                 ///< Current path throughput.
};

/** Samples the diffuse lighting at a hit point.
    \param[in] sd Describes the material properties and geometry at the hit point.
    \param[in] u 2D random number
    \param[out] msr MaterialSamplingResult struct with scattered direction and pdf
*/
void sampleDiffuse(ShadingData sd, float2 u, inout MaterialSamplingResult msr)
{
    float3 L = sample_cosine_hemisphere_polar(u, msr.pdf);
    L = fromLocal(L, sd);
    msr.wi = L;

    // Eval diffuse lighting
    // Compute half vector and dot products
    float3 H = normalize(sd.V + L);          // Note: The half vector H can be back-facing if V lies in the negative hemisphere
    float NdotL = saturate(dot(sd.N, L));
    float NdotV = saturate(sd.NdotV);
    float NdotH = saturate(dot(sd.N, H));
    float LdotH = saturate(dot(L, H));

    msr.thp = evalDiffuse(sd, NdotL, NdotV, LdotH)* M_PI;
}

/** Samples the lighting at a hit point.
    \param[in] sd Describes the material properties and geometry at the hit point.
    \param[in] u 2D random number
    \param[out] msr MaterialSamplingResult struct with scattered direction and pdf
*/
void sampleCosineBruteForce(ShadingData sd, float2 u, inout MaterialSamplingResult msr)
{
    float3 L = sample_cosine_hemisphere_polar(u, msr.pdf);
    L = fromLocal(L, sd);
    msr.wi = L;

    // Eval all lighting
    // Compute half vector and dot products
    float NdotL = saturate(dot(sd.N, L));
    msr.thp = evalBRDF(sd, L) * NdotL / msr.pdf;
}

#if 0
// DEPRECATED: Use evalPdfSpecularDisney() as replacement for now.

/** Evaluates the probability density function for GGX ndf sampling.
    \param[in] sd Describes the material properties and geometry at the hit point.
    \param[in] L World space direction for which to evaluate the pdf.
    \return The probability density for direction L.
*/
float evalNdfGGXPdf(ShadingData sd, float3 L)
{
    // Early out if L or V lies in the negative hemisphere. The probability of such samples is zero.
    float NdotL = dot(sd.N, L);
    float NdotV = dot(sd.N, sd.V);
    if (NdotL < epsilon || NdotV < epsilon)
    {
        return 0.f;
    }

    float3 H = normalize(sd.V + L);
    float NdotH = saturate(dot(sd.N, H));
    float LdotH = saturate(dot(L, H));

    // Compute pdf.
    // The pdf in half vector space is pdf = D(H) * |NdotH|, which is then divided by the Jacobian of the half-vector transform.
    float pdf = evalNdfGGX(NdotH, sd.ggxAlpha) * NdotH;
    pdf /= 4.f * LdotH;

    // Sanity check
    if (LdotH < epsilon)
    {
        pdf = 0.f;
    }

    return pdf;
}
#endif

// /** Smith masking-shadowing function for the GGX normal distribution.
// See, e.g.: Eq 34 in https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
// Only valid for cosTheta > 0
float evalG1(float cosTheta, float ggxAlpha)
{
    if (cosTheta <= 0) return 0;
    float a2 = ggxAlpha * ggxAlpha;
    float cosTheta2 = cosTheta*cosTheta;
    return 2.0 / (1.0f + sqrt(1.0f + a2*(1.0f - cosTheta2) / cosTheta2));
}

float evalG2(float NdotL, float NdotV, float ggxAlpha)
{
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
    float G = evalMaskingSmithGGXSeparable(NdotL, NdotV, ggxAlpha);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
    float G = evalMaskingSmithGGXCorrelated(NdotL, NdotV, ggxAlpha);
#endif
    return 4.0f * NdotL * NdotV * G;
}

/** Samples the GGX lighting at a hit point using distribution of visible normals 
    according to https://hal.inria.fr/hal-00996995v2/document
    Faster routine: https://hal.archives-ouvertes.fr/hal-01509746
    \param[in] sd Describes the material properties and geometry at the hit point.
    \param[in] u 2D random number
    \param[out] msr MaterialSamplingResult struct with scattered direction and pdf
*/
void sampleGGX_Heitz(ShadingData sd, float2 u, float bias, inout MaterialSamplingResult msr)
{
    float alpha = sd.ggxAlpha * bias;

    float3 H = float3(0,0,1);
    if (bias > 0.001f)
    {
        float3 V_local = toLocal(sd.V, sd);

        // Stretch
        float3 V = normalize(float3(alpha * V_local.x, alpha * V_local.y, V_local.z));

        // orthonormal basis
        float3 T1 = (V.z < 0.9999) ? normalize(cross(V, float3(0, 0, 1))) : float3(1, 0, 0);
        float3 T2 = cross(T1, V);

        // sample point with polar coordinates (r, phi)
        float a = 1.0 / (1.0 + V.z);
        float r = sqrt(u.x);
        float phi = (u.y < a) ? u.y / a * M_PI : M_PI + (u.y - a) / (1.0 - a) * M_PI;
        float P1 = r*cos(phi);
        float P2 = r*sin(phi)*((u.y < a) ? 1.0 : V.z);

        // compute normal
        H = P1*T1 + P2*T2 + sqrt(max(0.0, 1.0 - P1*P1 - P2*P2))*V;

        // unstretch
        H = normalize(float3(alpha*H.x, alpha*H.y, max(0.0, H.z)));
    }
    H = fromLocal(H, sd);

    float3 L = normalize(2.f * dot(sd.V, H) * H - sd.V);      // L sampled dir in world space
    float NdotL = saturate(dot(L, sd.N));
    float NdotH = saturate(dot(H, sd.N));
    float LdotH = saturate(dot(H, L));
    msr.wi = L; // scattered direction

    // Sanity checks. Reject sample L if backfacing relative to N or H.
    if (LdotH < epsilon || dot(sd.N, L) < epsilon || sd.NdotV < epsilon)
    {
        msr.pdf = 0.f;
        msr.thp = float3(0, 0, 0);
        return;
    }

    // restore alpha without bias
    alpha = sd.ggxAlpha;

    // Compute D_omega (distribution of visible normals)
    // Equation 2 in https://hal.inria.fr/hal-00996995v2/document
    // In that formula w_i : Direction from which light is incident, in our case: V
    //                 w_o : Direction in which light is scattered,  in our case: L

    float D = evalNdfGGX(NdotH, alpha);
    float G1 = evalG1(sd.NdotV, alpha);
    float G2 = evalG2(NdotL, sd.NdotV, alpha);

    // Compute pdf.
    msr.pdf = G1 * D * 0.25f / sd.NdotV;

    // Compute weight
    float3 F = evalFresnelSchlick(sd.specular, 1, max(0, LdotH));
    msr.thp = F * G2 / G1;
}

/** Samples the GGX lighting at a hit point according to.
    https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    \param[in] sd Describes the material properties and geometry at the hit point.
    \param[in] u 2D random number
    \param[out] msr MaterialSamplingResult struct with scattered direction and pdf
*/
void sampleGGX_Walter(ShadingData sd, float2 u, inout MaterialSamplingResult msr)
{
    // GGX NDF sampling
    float a2 = sd.ggxAlpha * sd.ggxAlpha;
    float cosThetaH = sqrt((1 - u.x) / ((a2 - 1) * u.x + 1));
    float sinThetaH = sqrt(1 - cosThetaH * cosThetaH);
    //float sinThetaH = sqrt(max(0, 1 - cosThetaH * cosThetaH));  // TODO: Check if clamp is ever needed due to numerical imprecision (example code from Frostbite had it). Should not be.
    float phiH = u.y * M_PI * 2;

    // Convert sample from half angle to incident angle
    float3 H = float3(sinThetaH * cos(phiH), sinThetaH * sin(phiH), cosThetaH);     // H in local frame

    H = fromLocal(H, sd);

    float3 L = normalize(2.f * dot(sd.V, H) * H - sd.V);      // L sampled dir in world space

    float NdotH = saturate(dot(H, sd.N));
    float NdotL = saturate(dot(L, sd.N));
    float LdotH = saturate(dot(H, L));

    // Compute pdf.
    // The pdf in half vector space is pdf = D(H) * |NdotH|, which is then divided by the Jacobian of the half-vector transform.
    msr.pdf = evalNdfGGX(NdotH, sd.ggxAlpha) * NdotH * 0.25f / LdotH;
    msr.wi = L; // scattered direction

    // Sanity checks. Reject sample L if backfacing relative to N or H.
    if (LdotH < epsilon || dot(sd.N, L) < epsilon)
    {
        msr.pdf = 0.f;
        msr.thp = float3(0, 0, 0);
        return;
    }

    // Compute weight
    float3 F = evalFresnelSchlick(sd.specular, 1, max(0, LdotH));
    float G2 = evalG2(NdotL, sd.NdotV, sd.ggxAlpha);
    msr.thp = F * LdotH * G2 / (sd.NdotV * NdotH); // sample weight
}


/** Helper function to compute the probability of sampling the diffuse lobe at a hit point. 
*/
float getDiffusePmf(ShadingData sd)
{
    // The specular component can be non-zero even if specular albedo is zero, as the fresnel term goes to 1 at grazing angles.
    // We handle the case of diff+spec < epsilon by always sampling the specular component in that case.
    float diffAlbedo = luminance(sd.diffuse);
    float specAlbedo = luminance(sd.specular);
    float pmf = (diffAlbedo + specAlbedo) < epsilon ? 0.f : diffAlbedo / (diffAlbedo + specAlbedo);
#ifdef DisableSpecularSampling
    pmf = 1.f;
#endif
    return pmf;
}

/** Evaluates the probability density function used for material sampling.
    This is useful for MIS, where we need to evaluate the probability for samples generated using other techniques.
    \param[in] sd Describes the material properties and geometry at the hit point.
    \param[in] L World space direction for which to evaluate the pdf.
    \return The probability density for direction L.
*/
float evalMaterialPdf(ShadingData sd, float3 L)
{
    // Samples in the negative hemisphere have zero probability.
    float NdotL = dot(sd.N, L);
    if (NdotL < epsilon)
    {
        return 0.f;
    }

    float pdfDiffuse = NdotL * M_1_PI;

    //float pdfSpecular = evalNdfGGXPdf(sd, L);         // Old function
    float pdfSpecular = evalPdfSpecularDisney(sd, L);

    // The final pdf is a blend between the diffuse and specular pdf:s as the sample function stochastically chooses between them.
    float pmf = getDiffusePmf(sd);
    return pmf * pdfDiffuse + (1.f - pmf) * pdfSpecular;
}


/** Importance sampling of the material.
    This takes both the diffuse and specular parts into account.
*/
void sampleMaterial(ShadingData sd, inout SampleGenerator sg, inout MaterialSamplingResult result, float specularLobeBias=0.0f)
{
    float3 sample = sampleNext3D(sg);

 //   sampleCosineBruteForce(sd, sample.xy, result); // For verification purposes

#if 1
    float pmf = getDiffusePmf(sd);   // Probability of sampling the diffuse component
    if (sample.z < pmf)
    {
        sampleDiffuse(sd, sample.xy, result); // Sample diffuse
        result.pdf /= pmf;  // re-weigh the contributions
        result.thp /= pmf;
    }
    else
    {
        // Sample specular
        float2 u = sample.xy;
        u.x = lerp(0.0, sample.x, saturate(1.0 - specularLobeBias)); // Hack to avoid sampling the tail of the GGX distrib

        // Old sampling functions.
        //sampleGGX_Heitz(sd, sample, saturate(1.0 - specularLobeBias), result); // Sample specular
        //sampleGGX_Walter(sd, u, result); // Sample specular

        // Use new sampling function.
        // TODO: This clamps dot products etc. differently, so we should update all code. This is just a temporary fix.
        BRDFSample s;
        sampleSpecularDisney(sd, u, s);
        result.wi = s.dir;
        result.thp = s.thp;
        result.pdf = s.pdf;


        float pmf_spec = saturate(1.0 - pmf); // Probability of sampling the specular component
        result.pdf /= pmf_spec;  // re-weigh the contributions
        result.thp /= pmf_spec;
    }
#endif
}

#undef epsilon
