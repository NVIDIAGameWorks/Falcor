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
#include "Rendering/Materials/BSDFConfig.slangh"

import Scene.Material.MaterialData;
import Utils.Color.ColorHelpers;
import Rendering.Materials.Fresnel;
import Rendering.Materials.IMaterialInstance;
#if DiffuseBrdf == DiffuseBrdfLambert
import Rendering.Materials.BSDFs.LambertDiffuseBRDF;
#elif DiffuseBrdf == DiffuseBrdfDisney
import Rendering.Materials.BSDFs.DisneyDiffuseBRDF;
#elif DiffuseBrdf == DiffuseBrdfFrostbite
import Rendering.Materials.BSDFs.FrostbiteDiffuseBRDF;
#endif
import Rendering.Materials.BSDFs.LambertDiffuseBTDF;
import Rendering.Materials.BSDFs.SpecularMicrofacet;
import DiffRendering.DiffMaterialData;
__exported import Rendering.Materials.IBSDF;

// Enable support for delta reflection/transmission.
#define EnableDeltaBSDF 1

// We clamp the GGX width parameter to avoid numerical instability.
// In some computations, we can avoid clamps etc. if 1.0 - alpha^2 != 1.0, so the epsilon should be 1.72666361e-4 or
// larger in fp32. The the value below is sufficient to avoid visible artifacts. Falcor used to clamp roughness to 0.08
// before the clamp was removed for allowing delta events. We continue to use the same threshold.
static const float kMinGGXAlpha = 0.0064f;

/**
 * BSDF parameters for the standard MaterialInstance.
 * These are needed for initializing a `StandardBSDF` instance.
 */
struct StandardBSDFData : IDifferentiable
{
    float3 diffuse;                     ///< Diffuse albedo.
    float3 specular;                    ///< Specular albedo.
    float roughness;                    ///< This is the original roughness, before remapping.
    float metallic;                     ///< Metallic parameter, blends between dielectric and conducting BSDFs.
    float eta;                          ///< Relative index of refraction (incident IoR / transmissive IoR).
    float3 transmission;                ///< Transmission color.
    float diffuseTransmission;          ///< Diffuse transmission, blends between diffuse reflection and transmission lobes.
    float specularTransmission;         ///< Specular transmission, blends between opaque dielectric BRDF and specular transmissive BSDF.
    float3 volumeScattering;            ///< Volume scattering parameters from the volume.
    float volumeAnsiotropy;             ///< Mean cosine of the volumetric phase function.
    bool hasEntryPointVolumeProperties; ///< Flag that specifies from where sigmaA is taken from (volume or surface).
    bool hasSigmaSGreaterZero;          ///< Flag that indicates if we have to do albedo conversion

    [Differentiable]
    __init(const DiffMaterialData diffData)
    {
        uint offset = 0;
        diffData.read(diffuse, offset);
        diffData.read(specular, offset);
        roughness = diffData.read(offset);
        metallic = diffData.read(offset);
        eta = diffData.read(offset);
        diffData.read(transmission, offset);
        diffuseTransmission = diffData.read(offset);
        specularTransmission = diffData.read(offset);
    }
}

/**
 * Mixed BSDF used for the standard material in Falcor.
 *
 * This consists of a diffuse and specular BRDF.
 * A specular BSDF is mixed in using the specularTransmission parameter.
 */
struct StandardBSDF : IBSDF, IDifferentiable
{
#if DiffuseBrdf == DiffuseBrdfLambert
    LambertDiffuseBRDF diffuseReflection;
#elif DiffuseBrdf == DiffuseBrdfDisney
    DisneyDiffuseBRDF diffuseReflection;
#elif DiffuseBrdf == DiffuseBrdfFrostbite
    FrostbiteDiffuseBRDF diffuseReflection;
#endif
    LambertDiffuseBTDF diffuseTransmission;

    SpecularMicrofacetBRDF specularReflection;
    SpecularMicrofacetBSDF specularTransmission;

    float diffTrans; ///< Mix between diffuse BRDF and diffuse BTDF.
    float specTrans; ///< Mix between dielectric BRDF and specular BSDF.

    // No need to differentiate probabilities.
    no_diff float pDiffuseReflection;    ///< Probability for sampling the diffuse BRDF.
    no_diff float pDiffuseTransmission;  ///< Probability for sampling the diffuse BTDF.
    no_diff float pSpecularReflection;   ///< Probability for sampling the specular BRDF.
    no_diff float pSpecularTransmission; ///< Probability for sampling the specular BSDF.

    /**
     * Initialize a new instance.
     * @param[in] wi Incident direction in the local frame.
     * @param[in] mtl Material header.
     * @param[in] data BSDF parameters.
     */
    [Differentiable]
    __init(const float3 wi, const MaterialHeader mtl, const StandardBSDFData data)
    {
        // TODO: Currently specular reflection and transmission lobes are not properly separated.
        // This leads to incorrect behaviour if only the specular reflection or transmission lobe is selected.
        // Things work fine as long as both or none are selected.

        // Use square root if we can assume the shaded object is intersected twice.
        float3 transmissionAlbedo = mtl.isThinSurface() ? data.transmission : sqrt(data.transmission);

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
        if (alpha < kMinGGXAlpha)
            alpha = 0.f;
#else
        alpha = max(alpha, kMinGGXAlpha);
#endif
        const uint activeLobes = mtl.getActiveLobes();

        specularReflection.albedo = data.specular;
        specularReflection.alpha = alpha;
        specularReflection.activeLobes = activeLobes;

        specularTransmission.transmissionAlbedo = transmissionAlbedo;
        // Transmission through rough interface with same IoR on both sides is not well defined, switch to delta lobe
        // instead.
        specularTransmission.alpha = data.eta == 1.f ? 0.f : alpha;
        specularTransmission.eta = data.eta;
        specularTransmission.activeLobes = activeLobes;

        diffTrans = data.diffuseTransmission;
        specTrans = data.specularTransmission;

        // Compute sampling weights.
        float metallicBRDF = data.metallic * (1.f - specTrans);
        float dielectricBSDF = (1.f - data.metallic) * (1.f - specTrans);
        float specularBSDF = specTrans;

        float diffuseWeight = luminance(data.diffuse);
        float specularWeight = luminance(evalFresnelSchlick(data.specular, 1.f, wi.z));

        pDiffuseReflection =
            (activeLobes & (uint)LobeType::DiffuseReflection) ? detach(diffuseWeight * dielectricBSDF * (1.f - diffTrans)) : 0.f;
        pDiffuseTransmission =
            (activeLobes & (uint)LobeType::DiffuseTransmission) ? detach(diffuseWeight * dielectricBSDF * diffTrans) : 0.f;
        pSpecularReflection = (activeLobes & ((uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection))
                                  ? detach(specularWeight * (metallicBRDF + dielectricBSDF))
                                  : 0.f;
        pSpecularTransmission = (activeLobes & ((uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection |
                                                (uint)LobeType::SpecularTransmission | (uint)LobeType::DeltaTransmission))
                                    ? detach(specularBSDF)
                                    : 0.f;

        float normFactor = pDiffuseReflection + pDiffuseTransmission + pSpecularReflection + pSpecularTransmission;
        if (normFactor > 0.f)
        {
            normFactor = 1.f / normFactor;
            pDiffuseReflection *= normFactor;
            pDiffuseTransmission *= normFactor;
            pSpecularReflection *= normFactor;
            pSpecularTransmission *= normFactor;
        }
    }

    /**
     * Returns the set of BSDF lobes.
     * @param[in] data BSDF parameters.
     * @return Returns a set of lobes (see LobeType.slang).
     */
    static uint getLobeTypes(const StandardBSDFData data)
    {
#if EnableDeltaBSDF
        float alpha = data.roughness * data.roughness;
        bool isDelta = alpha < kMinGGXAlpha;
#else
        bool isDelta = false;
#endif
        float diffTrans = data.diffuseTransmission;
        float specTrans = data.specularTransmission;

        uint lobeTypes = isDelta ? (uint)LobeType::DeltaReflection : (uint)LobeType::SpecularReflection;
        if (any(data.diffuse > 0.f) && specTrans < 1.f)
        {
            if (diffTrans < 1.f)
                lobeTypes |= (uint)LobeType::DiffuseReflection;
            if (diffTrans > 0.f)
                lobeTypes |= (uint)LobeType::DiffuseTransmission;
        }
        if (specTrans > 0.f)
            lobeTypes |= (isDelta ? (uint)LobeType::DeltaTransmission : (uint)LobeType::SpecularTransmission);

        return lobeTypes;
    }

    [Differentiable]
    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        float3 result = 0.f;
        if (pDiffuseReflection > 0.f)
            result += (1.f - specTrans) * (1.f - diffTrans) * diffuseReflection.eval(wi, wo, sg);
        if (pDiffuseTransmission > 0.f)
            result += (1.f - specTrans) * diffTrans * diffuseTransmission.eval(wi, wo, sg);
        if (pSpecularReflection > 0.f)
            result += (1.f - specTrans) * specularReflection.eval(wi, wo, sg);
        if (pSpecularTransmission > 0.f)
            result += specTrans * (specularTransmission.eval(wi, wo, sg));
        return result;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = 0.f;
        lobeType = (uint)LobeType::DiffuseReflection;

        bool valid = false;
        float uSelect = sampleNext1D(sg);

        // Note: The commented-out pdf contributions below are always zero, so no need to compute them.

        if (uSelect < pDiffuseReflection)
        {
            valid = diffuseReflection.sample(wi, wo, pdf, weight, lobeType, sg);
            weight /= pDiffuseReflection;
            weight *= (1.f - specTrans) * (1.f - diffTrans);
            pdf *= pDiffuseReflection;
            // if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
            if (pSpecularReflection > 0.f)
                pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
            if (pSpecularTransmission > 0.f)
                pdf += pSpecularTransmission * specularTransmission.evalPdf(wi, wo);
        }
        else if (uSelect < pDiffuseReflection + pDiffuseTransmission)
        {
            valid = diffuseTransmission.sample(wi, wo, pdf, weight, lobeType, sg);
            weight /= pDiffuseTransmission;
            weight *= (1.f - specTrans) * diffTrans;
            pdf *= pDiffuseTransmission;
            // if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
            // if (pSpecularReflection > 0.f) pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
            if (pSpecularTransmission > 0.f)
                pdf += pSpecularTransmission * specularTransmission.evalPdf(wi, wo);
        }
        else if (uSelect < pDiffuseReflection + pDiffuseTransmission + pSpecularReflection)
        {
            valid = specularReflection.sample(wi, wo, pdf, weight, lobeType, sg);
            weight /= pSpecularReflection;
            weight *= (1.f - specTrans);
            pdf *= pSpecularReflection;
            if (pDiffuseReflection > 0.f)
                pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
            // if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
            if (pSpecularTransmission > 0.f)
                pdf += pSpecularTransmission * specularTransmission.evalPdf(wi, wo);
        }
        else if (pSpecularTransmission > 0.f)
        {
            valid = specularTransmission.sample(wi, wo, pdf, weight, lobeType, sg);
            weight /= pSpecularTransmission;
            weight *= specTrans;
            pdf *= pSpecularTransmission;
            if (pDiffuseReflection > 0.f)
                pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
            if (pDiffuseTransmission > 0.f)
                pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
            if (pSpecularReflection > 0.f)
                pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
        }

        return valid;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        float pdf = 0.f;
        if (pDiffuseReflection > 0.f)
            pdf += pDiffuseReflection * diffuseReflection.evalPdf(wi, wo);
        if (pDiffuseTransmission > 0.f)
            pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wi, wo);
        if (pSpecularReflection > 0.f)
            pdf += pSpecularReflection * specularReflection.evalPdf(wi, wo);
        if (pSpecularTransmission > 0.f)
            pdf += pSpecularTransmission * specularTransmission.evalPdf(wi, wo);
        return pdf;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        AlbedoContributions a = { 0.0f, 0.0f, 0.0f, 0.0f };

        if (pDiffuseReflection > 0.f)
            a += (1.f - specTrans) * (1.f - diffTrans) * diffuseReflection.evalAlbedo(wi, lobetype);
        if (pDiffuseTransmission > 0.f)
            a += (1.f - specTrans) * diffTrans * diffuseTransmission.evalAlbedo(wi, lobetype);
        if (pSpecularReflection > 0.f)
            a += (1.f - specTrans) * specularReflection.evalAlbedo(wi, lobetype);
        if (pSpecularTransmission > 0.f)
            a += specTrans * specularTransmission.evalAlbedo(wi, lobetype);

        return a;
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        float alpha = specularReflection.alpha;
        r.roughnessBSDFNotation = float2(alpha < kMinGGXAlpha ? 0.f : alpha);
        return r;
    }
};
