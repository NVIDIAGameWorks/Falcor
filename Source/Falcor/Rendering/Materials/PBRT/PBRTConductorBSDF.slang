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
__exported import Rendering.Materials.AnisotropicGGX;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import Rendering.Materials.Fresnel;

struct PBRTConductorBxDF : IBxDF
{
    AnisotropicGGX D;
    float3 eta;
    float3 k;

    float3 eval<S : ISampleGenerator>(const float3 wiLocal, const float3 woLocal, inout S sg)
    {
        if (D.isSingular() || min(wiLocal.z, woLocal.z) < 0.0f) return float3(0.0f);

        float3 h = normalize(wiLocal + woLocal);
        float wiDotH = abs(dot(wiLocal, h));
        float3 F = evalFresnelConductor(eta, k, wiDotH);

        return D.G2(woLocal, wiLocal) * D.evalNDF(h) * F / (4.0f * abs(wiLocal.z));
    }

    bool sample<S : ISampleGenerator>(float3 wiLocal, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
        if (wiLocal.z < 0.0f) return false;
        
        if (D.isSingular())
        {
            wo = float3(-wiLocal.x, -wiLocal.y, wiLocal.z);
            weight = evalFresnelConductor(eta, k, wiLocal.z);
            lobe = (uint)LobeType::DeltaReflection;
            pdf = 1.0f;
        }
        else
        {
            float3 h = D.sample(wiLocal, sg);
            
            float3 woLocal = reflect(-wiLocal, h);
            if (woLocal.z < 0.0f) return false;
            float3 F = evalFresnelConductor(eta, k, dot(h, wiLocal));
            
            wo = woLocal;
            weight = F * D.G2(woLocal, wiLocal) / D.G1(wiLocal);
            lobe = (uint)LobeType::SpecularReflection;
            pdf = D.evalPDF(wiLocal, h) / (4.0f * abs(dot(wiLocal, h)));
        }

        return true;
    }

    float evalPdf(const float3 wiLocal, const float3 woLocal)
    {
        if (D.isSingular() || min(wiLocal.z, woLocal.z) < 0.0f) return 0.0f;
        float3 h = normalize(wiLocal + woLocal);
        return D.evalPDF(wiLocal, h) / (4.0f * abs(dot(wiLocal, h)));
    }
};

struct PBRTConductorBSDF : IBSDF
{
    PBRTConductorBxDF bxdf;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        return bxdf.eval(sd.toLocal(sd.V), sd.toLocal(wo), sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        bool valid = bxdf.sample(sd.toLocal(sd.V), result.wo, result.pdf, result.weight, result.lobe, sg);
        result.wo = sd.fromLocal(result.wo);
        return valid;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        return bxdf.evalPdf(sd.toLocal(sd.V), sd.toLocal(wo));
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.roughness = bxdf.D.getIsotropicRoughness();
        p.specularReflectionAlbedo = float3(1.0f); // TODO: Compute approximate albedo from eta/k

        return p;
    }

    uint getLobes(const ShadingData sd)
    {
        return (uint)(bxdf.D.isSingular() ? LobeType::DeltaReflection : LobeType::SpecularReflection);
    }
};
