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
import Rendering.Materials.LayeredBxDF;
import Rendering.Materials.PBRT.PBRTDielectricBSDF;
import Rendering.Materials.PBRT.PBRTDiffuseBSDF;
import Utils.Sampling.UniformSampleGenerator;

struct PBRTCoatedDiffuseBSDF : IBSDF
{
    AnisotropicGGX interfaceD;
    float interfaceEta;
    float3 diffuseAlbedo;
    int maxDepth;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);
        
        PBRTDielectricBxDF top = {interfaceD, interfaceEta};
        PBRTDiffuseBxDF bottom = {diffuseAlbedo};

        return LayeredBxDF(top, bottom).eval(wiLocal, woLocal, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        
        PBRTDielectricBxDF top = {interfaceD, interfaceEta};
        PBRTDiffuseBxDF bottom = {diffuseAlbedo};
        
        bool valid = LayeredBxDF(top, bottom).sample(wiLocal, result.wo, result.pdf, result.weight, result.lobe, sg);
        result.wo = sd.fromLocal(result.wo);
        return valid;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        PBRTDielectricBxDF top = {interfaceD, interfaceEta};
        PBRTDiffuseBxDF bottom = {diffuseAlbedo};
        
        return LayeredBxDF(top, bottom).evalPdf(sd.toLocal(sd.V), sd.toLocal(wo));
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.roughness = interfaceD.getIsotropicRoughness();
        p.specularReflectionAlbedo = float3(1.0f);

        return p;
    }

    uint getLobes(const ShadingData sd)
    {
        return (uint)(LobeType::DiffuseReflection) | (uint)(interfaceD.isSingular() ? LobeType::DeltaReflection : LobeType::SpecularReflection);
    }
};
