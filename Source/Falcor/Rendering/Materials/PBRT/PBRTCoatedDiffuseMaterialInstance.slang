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
import Rendering.Materials.LayeredBSDF;
import Rendering.Materials.PBRT.PBRTDielectricMaterialInstance;
import Rendering.Materials.PBRT.PBRTDiffuseMaterialInstance;

struct PBRTCoatedDiffuseMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;            ///< Shading frame in world space.
    AnisotropicGGX interfaceD;
    float interfaceEta;
    float3 diffuseAlbedo;
    int maxDepth;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        PBRTDielectricBSDF top = {interfaceD, interfaceEta};
        PBRTDiffuseBSDF bottom = {diffuseAlbedo};

        return LayeredBSDF(top, bottom).eval(wiLocal, woLocal, sg);
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

        PBRTDielectricBSDF top = {interfaceD, interfaceEta};
        PBRTDiffuseBSDF bottom = {diffuseAlbedo};

        bool valid = LayeredBSDF(top, bottom).sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg);
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

        return valid;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;

        PBRTDielectricBSDF top = {interfaceD, interfaceEta};
        PBRTDiffuseBSDF bottom = {diffuseAlbedo};

        return LayeredBSDF(top, bottom).evalPdf(wiLocal, woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;
        p.roughness = interfaceD.getIsotropicRoughness();
        p.specularReflectionAlbedo = float3(1.0f);

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return (uint)(LobeType::DiffuseReflection) | (uint)(interfaceD.isSingular() ? LobeType::DeltaReflection : LobeType::SpecularReflection);
    }

};
