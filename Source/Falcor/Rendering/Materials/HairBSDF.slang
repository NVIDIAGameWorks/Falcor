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
#include "BCSDFConfig.slangh"

__exported import Rendering.Materials.IBSDF;
__exported import Rendering.Materials.HairChiang16;
import Utils.Math.MathHelpers;

/** Implementation of Falcor's standard hair/fur BSDF.

    The implementation uses the scattering function by Chiang et al. 2016,
    "A Practical and Controllable Hair and Fur Model for Production Path Tracing".

    By overriding the 'BCSDF' define it is possible to select a single-sided
    Lambertian diffuse BRDF for debugging purposes (see BCSDFConfig.slangh).
*/
struct HairBSDF : IBSDF
{
    HairBSDFData data;

    __init(const HairBSDFData data)
    {
        this.data = data;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
#if BCSDF == HairChiang
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        HairChiang16 bcsdf = HairChiang16(sd, data);

        return bcsdf.eval(wiLocal, woLocal, sg);
#else
        // Diffuse BRDF
        return data.baseColor * (float)M_1_PI * saturate(dot(sd.N, wo));
#endif
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
#if BCSDF == HairChiang
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = {};

        HairChiang16 bcsdf = HairChiang16(sd, data);

        bool valid = bcsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobe, sg);
        result.wo = sd.fromLocal(woLocal);

        return valid;
#else
        // Diffuse BRDF
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf);
        result.wo = sd.fromLocal(woLocal);
        result.weight = data.baseColor * (float)M_1_PI;
        result.lobe = (uint)LobeType::DiffuseReflection;
        return true;
#endif
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
#if BCSDF == HairChiang
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        HairChiang16 bcsdf = HairChiang16(sd, data);

        return bcsdf.evalPdf(wiLocal, woLocal);
#else
        // Diffuse BRDF
        return saturate(dot(sd.N, wo));
#endif
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        // Compute approximation of the perceived roughness.
        // It's an open problem how to best do this. For now assume a medium roughness.
        p.roughness = 0.5f;

        // Compute approximation of the albedos.
        // We don't currently have a good way to do this, so divide the energy equally for now.
        // The BSDF reports only diffuse reflection and transmission lobes, so leaving specular albedos as zero.
        // Note the case of black base color is assigned a zero albedo, but presumably there is still some scattering.
        p.diffuseReflectionAlbedo = data.baseColor * 0.5f;
        p.diffuseTransmissionAlbedo = p.diffuseReflectionAlbedo;

        p.flags |= (uint)BSDFProperties::Flags::IsTransmissive;

        return p;
    }

    uint getLobes(const ShadingData sd)
    {
        // The BSDF currently assumes all diffuse lobes (for animal fur).
        // TODO: Update this when lobes are classified as diffuse or specular according to roughness.
        return (uint)LobeType::DiffuseReflection | (uint)LobeType::DiffuseTransmission;
    }
};
