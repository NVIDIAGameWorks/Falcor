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
import Rendering.Materials.AnisotropicGGX;
__exported import Rendering.Materials.IBSDF;

/**
 * Thin (architectural) dielectric to simulate plane parallel materials on a single polygon.
 * Relfection is classic microfacet theory, transmission roughness is calculated from the reflection
 * roughness and the IOR when creating the BSDF. The Frensel is standard glass theory (perfectly specular)
 * and should be replaced by a proper formula
 */
struct DielectricPlateBSDF : IBSDF
{
    AnisotropicGGX D;       ///< Reflection roughness
    AnisotropicGGX DInside; ///< Tranmission roughness
    float eta;              ///< IOR

    float3 eval<S : ISampleGenerator>(const float3 wiLocal, const float3 woLocal, inout S sg)
    {
        if (D.isSingular() || eta == 1.0f || wiLocal.z * woLocal.z == 0)
            return float3(0.0f);

        bool reflect = wiLocal.z * woLocal.z > 0.0f;
        float relEta = eta;
        const float etaRefract = 1.005f;
        float3 h = reflect ? normalize(wiLocal + woLocal) : normalize(wiLocal / etaRefract + woLocal);
        h *= sign(h.z);
        float wiDotH = dot(wiLocal, h);
        float woDotH = dot(woLocal, h);
        if (wiDotH * wiLocal.z < 0.0f || woDotH * woLocal.z < 0.0f)
            return float3(0.0f);

        float sinThetaTSq = eta * eta * (1 - wiDotH * wiDotH);
        float cosThetaT = sqrt(1 - sinThetaTSq); // No clamp needed

        float r1 = evalFresnelDielectric(eta, wiDotH);
        float r2 = evalFresnelDielectric(eta, -cosThetaT);
        float t1 = 1.f - r2;
        float F = r1 + t1 * t1 * r2 / (1.f - r1 * r2);

        return reflect ? D.G2(woLocal, wiLocal) * D.evalNDF(h) * F / (4.f * abs(wiLocal.z))
                       : DInside.G2(woLocal, wiLocal) * DInside.evalNDF(h) * (1.0f - F) *
                             abs(wiDotH * woDotH / (sqr(woDotH + wiDotH / etaRefract) * abs(wiLocal.z)));
    }

    bool sample<S : ISampleGenerator>(float3 wiLocal, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        wo = {};
        pdf = {};
        weight = {};
        lobeType = {};

        float relEta = eta;
        const float etaRefract = 1.005f;

        if (eta == 1 || D.isSingular())
        {
            float sinThetaTSq = eta * eta * (1 - wiLocal.z * wiLocal.z);
            float cosThetaT = sqrt(1 - sinThetaTSq); // No clamp needed

            float r1 = evalFresnelDielectric(eta, wiLocal.z);
            float r2 = evalFresnelDielectric(eta, -cosThetaT);
            float t1 = 1.f - r2;
            float F = r1 + t1 * t1 * r2 / (1.f - r1 * r2);

            if (sampleNext1D(sg) < F)
            {
                wo = float3(-wiLocal.x, -wiLocal.y, wiLocal.z);
                weight = float3(1.0f);
                lobeType = (uint)LobeType::DeltaReflection;
                pdf = 1.0f;
            }
            else
            {
                wo = refract(-wiLocal, float3(0.0f, 0.0f, sign(wiLocal.z)), 1.0f / etaRefract);
                weight = float3(1.0f);
                lobeType = (uint)LobeType::DeltaTransmission;
                pdf = 1.0f;
            }
        }
        else
        {
            float3 h = D.sample(wiLocal, sg);
            float wiDotH = dot(h, wiLocal);

            float sinThetaTSq = eta * eta * (1 - wiDotH * wiDotH);
            float cosThetaT = sqrt(1 - sinThetaTSq); // No clamp needed

            float r1 = evalFresnelDielectric(eta, wiDotH);
            float r2 = evalFresnelDielectric(eta, -cosThetaT);
            float t1 = 1.f - r2;
            float F = r1 + t1 * t1 * r2 / (1.f - r1 * r2);

            if (sampleNext1D(sg) < F)
            {
                float3 woLocal = reflect(-wiLocal, h);
                if (wiLocal.z * woLocal.z < 0.0f)
                    return false;

                wo = woLocal;
                weight = D.G2(woLocal, wiLocal) / D.G1(wiLocal);
                lobeType = (uint)LobeType::SpecularReflection;
                pdf = F * D.evalPDF(wiLocal, h) / (4.0f * abs(wiDotH));
            }
            else
            {
                float3 h = DInside.sample(wiLocal, sg);
                float wiDotH = dot(h, wiLocal);

                float3 woLocal = refract(-wiLocal, sign(wiLocal.z) * h, 1.0f / etaRefract);
                float woDotH = dot(h, woLocal);
                if (wiLocal.z * woLocal.z > 0.0f)
                    return false;

                wo = woLocal;
                weight = DInside.G2(woLocal, wiLocal) / DInside.G1(wiLocal);
                lobeType = (uint)LobeType::SpecularTransmission;
                pdf = (1.0f - F) * DInside.evalPDF(wiLocal, h) * abs(woDotH) / sqr(woDotH + wiDotH / etaRefract);
            }
        }
        return true;
    }

    float evalPdf(const float3 wiLocal, const float3 woLocal)
    {
        if (D.isSingular() || eta == 1.0f || wiLocal.z * woLocal.z == 0)
            return 0.0f;

        bool reflect = wiLocal.z * woLocal.z > 0.0f;
        float relEta = eta;
        const float etaRefract = 1.005f;
        float3 h = reflect ? normalize(wiLocal + woLocal) : normalize(wiLocal / etaRefract + woLocal);
        h *= sign(h.z);
        float wiDotH = dot(wiLocal, h);
        float woDotH = dot(woLocal, h);
        if (wiDotH * wiLocal.z < 0.0f || woDotH * woLocal.z < 0.0f)
            return 0.0f;

        float sinThetaTSq = eta * eta * (1 - wiDotH * wiDotH);
        float cosThetaT = sqrt(1 - sinThetaTSq); // No clamp needed

        float r1 = evalFresnelDielectric(eta, wiDotH);
        float r2 = evalFresnelDielectric(eta, -cosThetaT);
        float t1 = 1.f - r2;
        float F = r1 + t1 * t1 * r2 / (1.f - r1 * r2);

        return reflect ? F * D.evalPDF(wiLocal, h) / (4.f * abs(wiDotH))
                       : (1.0f - F) * DInside.evalPDF(wiLocal, h) * abs(woDotH) / sqr(woDotH + wiDotH / etaRefract);
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        float sinThetaTSq = eta * eta * (1 - wi.z * wi.z);
        float cosThetaT = sqrt(1 - sinThetaTSq); // No clamp needed

        float r1 = evalFresnelDielectric(eta, wi.z);
        float r2 = evalFresnelDielectric(eta, -cosThetaT);
        float t1 = 1.f - r2;
        float F = r1 + t1 * t1 * r2 / (1.f - r1 * r2);

        return AlbedoContributions(F, 0.f, 1.f - F, 0.f);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = D.alpha;
        return r;
    }
};
