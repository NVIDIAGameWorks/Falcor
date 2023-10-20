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
__exported import DiffRendering.SceneGradients;
__exported import DiffRendering.GradientIOWrapper;
import Rendering.Materials.Fresnel;

struct PBRTConductorBSDF : IBSDF, IDifferentiable
{
    AnisotropicGGX D;
    float3 eta;
    float3 k;

    [Differentiable]
    __init(const DiffMaterialData diffData)
    {
        uint offset = 0;
        diffData.read(eta, offset);
        diffData.read(k, offset);
        float2 roughness = diffData.read<2>(offset);
        D = AnisotropicGGX(roughness);
    }

    [Differentiable]
    float3 eval<S : ISampleGenerator>(const float3 wiLocal, const float3 woLocal, inout S sg)
    {
        if (D.isSingular() || min(wiLocal.z, woLocal.z) < 0.0f) return float3(0.0f);

        float3 h = normalize(wiLocal + woLocal);
        float wiDotH = abs(dot(wiLocal, h));
        float3 F = evalFresnelConductor(eta, k, wiDotH);

        return D.G2(woLocal, wiLocal) * D.evalNDF(h) * F / (4.0f * abs(wiLocal.z));
    }

    bool sample<S : ISampleGenerator>(float3 wiLocal, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        wo = {};
        pdf = {};
        weight = {};
        lobeType = {};

        if (wiLocal.z < 0.0f) return false;

        if (D.isSingular())
        {
            wo = float3(-wiLocal.x, -wiLocal.y, wiLocal.z);
            weight = evalFresnelConductor(eta, k, wiLocal.z);
            lobeType = (uint)LobeType::DeltaReflection;
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
            lobeType = (uint)LobeType::SpecularReflection;
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

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        float3 r = evalFresnelConductor(eta, k, wi.z);
        return AlbedoContributions(r, 1.0f - r, 0.0f, 0.0f);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = D.alpha;
        return r;
    }
};

struct PBRTConductorMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;            ///< Shading frame in world space.
    PBRTConductorBSDF bsdf;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        return bsdf.eval(wiLocal, woLocal, sg);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = diffData.sf.toLocal(sd.V);
        float3 woLocal = diffData.sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        PBRTConductorBSDF bsdfAD = PBRTConductorBSDF(diffData);
        return bsdfAD.eval(wiLocal, woLocal, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = {};

        bool valid = bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg);
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

        return valid;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;

        return bsdf.evalPdf(wiLocal, woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;
        p.roughness = bsdf.D.getIsotropicRoughness();
        p.specularReflectionAlbedo = float3(1.0f); // TODO: Compute approximate albedo from eta/k

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return (uint)(bsdf.D.isSingular() ? LobeType::DeltaReflection : LobeType::SpecularReflection);
    }

};
