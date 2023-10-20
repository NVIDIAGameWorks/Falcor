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
__exported import Rendering.Materials.IMaterialInstance;
__exported import Rendering.Materials.AnisotropicGGX;
import Utils.Math.MathHelpers;
import Rendering.Materials.Fresnel;

struct PBRTDielectricBSDF : IBSDF
{
    AnisotropicGGX D;
    float eta;

    float3 eval<S : ISampleGenerator>(const float3 wiLocal, const float3 woLocal, inout S sg)
    {
        if (D.isSingular() || eta == 1.0f || wiLocal.z * woLocal.z == 0) return float3(0.0f);

        bool reflect = wiLocal.z * woLocal.z > 0.0f;
        float relEta = 1.0 / eta;
        float3 h = reflect ? normalize(wiLocal + woLocal) : normalize(wiLocal / relEta + woLocal);
        h *= sign(h.z);
        float wiDotH = dot(wiLocal, h);
        float woDotH = dot(woLocal, h);
        if (wiDotH * wiLocal.z < 0.0f || woDotH * woLocal.z < 0.0f) return float3(0.0f);
        float F = evalFresnelDielectric(eta, abs(wiDotH));

        return reflect ? D.G2(woLocal, wiLocal) * D.evalNDF(h) * F / (4.0f * abs(wiLocal.z))
                       : D.G2(woLocal, wiLocal) * D.evalNDF(h) * (1.0f - F) * abs(wiDotH * woDotH / (sqr(woDotH + wiDotH / relEta) * abs(wiLocal.z)));
    }

    bool sample<S : ISampleGenerator>(float3 wiLocal, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        wo = {};
        pdf = {};
        weight = {};
        lobeType = {};

        float relEta = 1.0 / eta;

        if (eta == 1 || D.isSingular())
        {
            float F = evalFresnelDielectric(eta, abs(wiLocal.z));

            if (sampleNext1D(sg) < F)
            {
                wo = float3(-wiLocal.x, -wiLocal.y, wiLocal.z);
                weight = float3(1.0f);
                lobeType = (uint)LobeType::DeltaReflection;
                pdf = 1.0f;
            }
            else
            {
                wo = refract(-wiLocal, float3(0.0f, 0.0f, sign(wiLocal.z)), 1.0f / relEta);
                weight = float3(1.0f);
                lobeType = (uint)LobeType::DeltaTransmission;
                pdf = 1.0f;
            }
        }
        else
        {
            float3 h = D.sample(wiLocal, sg);
            float wiDotH = dot(h, wiLocal);
            float F = evalFresnelDielectric(eta, abs(wiDotH));

            if (sampleNext1D(sg) < F)
            {
                float3 woLocal = reflect(-wiLocal, h);
                if (wiLocal.z * woLocal.z < 0.0f) return false;

                wo = woLocal;
                weight = D.G2(woLocal, wiLocal)/D.G1(wiLocal);
                lobeType = (uint)LobeType::SpecularReflection;
                pdf = F * D.evalPDF(wiLocal, h) / (4.0f * abs(wiDotH));
            }
            else
            {
                float3 woLocal = refract(-wiLocal, sign(wiLocal.z) * h, 1.0f/relEta);
                float woDotH = dot(h, woLocal);
                if (wiLocal.z * woLocal.z > 0.0f) return false;

                wo = woLocal;
                weight = D.G2(woLocal, wiLocal)/D.G1(wiLocal);
                lobeType = (uint)LobeType::SpecularTransmission;
                pdf = (1.0f - F) * D.evalPDF(wiLocal, h) * abs(woDotH) / sqr(woDotH + wiDotH / relEta);
            }
        }
        return true;
    }

    float evalPdf(const float3 wiLocal, const float3 woLocal)
    {
        if (D.isSingular() || eta == 1.0f || wiLocal.z * woLocal.z == 0) return 0.0f;

        bool reflect = wiLocal.z * woLocal.z > 0.0f;
        float relEta = (1.0f / eta);
        float3 h = reflect ? normalize(wiLocal + woLocal) : normalize(wiLocal / relEta + woLocal);
        h *= sign(h.z);
        float wiDotH = dot(wiLocal, h);
        float woDotH = dot(woLocal, h);
        if (wiDotH * wiLocal.z < 0.0f || woDotH * woLocal.z < 0.0f) return 0.0f;
        float F = evalFresnelDielectric(eta, abs(wiDotH));

        return reflect ? F * D.evalPDF(wiLocal, h) / (4.0f * abs(wiDotH))
                       : (1.0f - F) * D.evalPDF(wiLocal, h) * abs(woDotH) / sqr(woDotH + wiDotH / relEta);
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        float3 r = evalFresnelDielectric(eta, abs(wi.z));
        return AlbedoContributions(r, 0.0f, 1.0f - r, 0.0f);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = D.alpha;
        return r;
    }
};

/** PBRT dielectric material instance.
    Note that the BSDF is implemented to handle wi/wo over the full sphere.
    The material is treated as single-sided and the shading normal should not be flipped.
*/
struct PBRTDielectricMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;            ///< Shading frame in world space.
    PBRTDielectricBSDF bsdf;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, wo, true)) return float3(0.f);

        return bsdf.eval(wiLocal, woLocal, sg);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = {};

        if (!bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg))
            return false;
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, result.wo, true)) return false;

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, wo, true)) return 0.f;

        return bsdf.evalPdf(wiLocal, woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;
        p.roughness = bsdf.D.getIsotropicRoughness();

        p.specularReflectionAlbedo = float3(1.0f);
        p.specularTransmissionAlbedo = float3(1.0f);
        p.flags |= (uint)BSDFProperties::Flags::IsTransmissive;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return bsdf.D.isSingular() ? ((uint)LobeType::DeltaReflection | (uint)LobeType::DeltaTransmission)
                                   : ((uint)LobeType::SpecularReflection | (uint)LobeType::SpecularTransmission);
    }

};
