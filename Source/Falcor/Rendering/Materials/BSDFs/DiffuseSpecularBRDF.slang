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
import Utils.Color.ColorHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;
__exported import Rendering.Materials.IBSDF;
__exported import Scene.Material.DiffuseSpecularData;

/**
 * Diffuse/specular BRDF using a simple Disney diffuse lobe plus an isotropic GGX specular lobe.
 * This is useful as target for approximate BRDF fits for sampling purposes.
 */
struct DiffuseSpecularBRDF : IBSDF
{
    float3 diffuse;  ///< Diffuse albedo.
    float3 specular; ///< Specular albedo.
    float roughness; ///< Roughness (linear).

    /**
     * Initialize using parameters from the Disney parameterization.
     */
    __init(const DiffuseSpecularData params)
    {
        this.diffuse = params.baseColor * (1.f - params.metallic);
        this.specular = lerp(float3(params.specular * 0.08f), params.baseColor, params.metallic);
        this.roughness = params.roughness;
    }

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return float3(0.f);

        float3 fd = diffuseDisney(wi, wo, diffuse, roughness);
        float3 fs = specularMicrofacet(wi, wo, specular, roughness);
        return (fd + fs) * wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        wo = {};
        pdf = {};
        weight = {};
        lobeType = {};

        float diffuseWeight = max(0.f, luminance(diffuse));
        float specularWeight = max(0.001f, luminance(specular));
        float norm = 1.f / (diffuseWeight + specularWeight);
        diffuseWeight *= norm;
        specularWeight *= norm;

        float alpha = roughness * roughness;
        float uSelect = sampleNext1D(sg);
        float2 u = sampleNext2D(sg);

        if (uSelect < diffuseWeight)
        {
            wo = sample_cosine_hemisphere_concentric(u, pdf);
            pdf = pdf * diffuseWeight + specularWeight * specularMicrofacetPdf(wi, wo, alpha);
            lobeType = (uint)LobeType::DiffuseReflection;
        }
        else
        {
            if (!specularMicrofacetSample(wi, alpha, u, wo, pdf))
                return false;
            pdf = pdf * specularWeight + diffuseWeight * diffusePdf(wo);
            lobeType = (uint)LobeType::SpecularReflection;
        }

        if (min(wi.z, wo.z) < kMinCosTheta || pdf == 0.f)
            return false;

        weight = eval(wi, wo, sg) / pdf;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return 0.f;

        float diffuseWeight = max(0.f, luminance(diffuse));
        float specularWeight = max(0.001f, luminance(specular));
        float norm = 1.f / (diffuseWeight + specularWeight);
        diffuseWeight *= norm;
        specularWeight *= norm;

        float alpha = roughness * roughness;
        float pdf = 0.f;

        if (diffuseWeight > 0.f)
            pdf += diffuseWeight * diffusePdf(wo);
        if (specularWeight > 0.f)
            pdf += specularWeight * specularMicrofacetPdf(wi, wo, alpha);

        return pdf;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        // TODO: Return more accurate albedos.
        return AlbedoContributions(diffuse, 1.0f - diffuse, 0.0f, 0.0f);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(roughness);
        return r;
    }

    // Internal functions

    // Based on https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
    float3 diffuseDisney(const float3 wi, const float3 wo, const float3 albedo, const float roughness)
    {
        float3 h = normalize(wi + wo);
        float woDotH = max(0.f, dot(wo, h));
        float fd90 = 0.5f + 2.f * woDotH * woDotH * roughness;
        float fd0 = 1.f;
        float wiScatter = evalFresnelSchlick(fd0, fd90, wi.z);
        float woScatter = evalFresnelSchlick(fd0, fd90, wo.z);
        return albedo * (wiScatter * woScatter * M_1_PI);
    }

    float diffusePdf(const float3 wo) { return max(0.f, wo.z) * M_1_PI; }

    float3 specularMicrofacet(const float3 wi, const float3 wo, const float3 albedo, const float roughness)
    {
        float alpha = roughness * roughness;
        float3 h = normalize(wi + wo);
        float wiDotH = dot(wi, h);
        float D = evalNdfGGX(alpha, h.z);
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
        float3 F = evalFresnelSchlick(albedo, float3(1.f), wiDotH);
        return F * (D * G * 0.25f / max(1e-6f, wi.z * wo.z));
    }

    bool specularMicrofacetSample(const float3 wi, const float alpha, const float2 u, out float3 wo, out float pdf)
    {
        float3 h = sampleGGX_VNDF(alpha, wi, u, pdf); // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
        float wiDotH = dot(wi, h);
        wo = 2.f * wiDotH * h - wi;
        if (wo.z <= 0.f)
            return false;
        pdf /= (4.f * wiDotH); // Jacobian of the reflection operator.
        return true;
    }

    float specularMicrofacetPdf(const float3 wi, const float3 wo, const float alpha)
    {
        float3 h = normalize(wi + wo);
        float wiDotH = max(0.f, dot(wi, h));
        float pdf = evalPdfGGX_VNDF(alpha, wi, h);
        return pdf / (4.f * wiDotH);
    }
};
