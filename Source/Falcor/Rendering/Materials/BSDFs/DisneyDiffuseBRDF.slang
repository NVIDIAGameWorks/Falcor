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
import Rendering.Materials.Fresnel;
__exported import Rendering.Materials.IBSDF;

/**
 * Disney's diffuse reflection.
 * Based on https ://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
 */
struct DisneyDiffuseBRDF : IBSDF, IDifferentiable
{
    float3 albedo;   ///< Diffuse albedo.
    float roughness; ///< Roughness before remapping.

    [Differentiable]
    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return float3(0.f);

        return evalWeight(wi, wo) * M_1_PI * wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        lobeType = (uint)LobeType::DiffuseReflection;

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
        if (min(wi.z, wo.z) < kMinCosTheta)
            return 0.f;

        return M_1_PI * wo.z;
    }

    // private

    // Returns f(wi, wo) * pi.
    [Differentiable]
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
