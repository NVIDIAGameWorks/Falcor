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

__exported import Rendering.Materials.IBSDF;
__exported import Rendering.Materials.BxDF;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;

struct PBRTDiffuseTransmissionBSDF : IBSDF
{
    float3 albedoR;
    float3 albedoT;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);
        
        float3 albedo = (wiLocal.z * woLocal.z > 0.0f) ? albedoR : albedoT;
        return albedo * abs(woLocal.z) * M_1_PI;
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf);
        if (wiLocal.z < 0.0f) woLocal = -woLocal;
        
        float PR = luminance(albedoR);
        float PT = luminance(albedoT);
        if (PR + PT == 0.0f) return false;
        if (sampleNext1D(sg) < PT / (PR + PT))
        {
            woLocal = -woLocal;
            result.weight = albedoT / PT * (PR + PT);
            result.lobe = (uint)LobeType::DiffuseTransmission;
            result.pdf *= PT / (PR + PT);
        }
        else
        {
            result.weight = albedoR / PR * (PR + PT);
            result.lobe = (uint)LobeType::DiffuseReflection;
            result.pdf *= PR / (PR + PT);
        }
        result.wo = sd.fromLocal(woLocal);

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);
        
        float PR = luminance(albedoR);
        float PT = luminance(albedoT);
        if (PR + PT == 0.0f) return 0.0f;
        float P = (wiLocal.z * woLocal.z > 0.0f) ? PR / (PR + PT) : PT / (PR + PT);
        
        return P * abs(woLocal.z) * M_1_PI;
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.roughness = 1.0f;
        p.diffuseReflectionAlbedo = albedoR;
        p.diffuseTransmissionAlbedo = albedoT;
        p.flags |= (uint)BSDFProperties::Flags::IsTransmissive;

        return p;
    }

    uint getLobes(const ShadingData sd)
    {
        return (uint)LobeType::DiffuseReflection | (uint)LobeType::DiffuseTransmission;
    }
};
