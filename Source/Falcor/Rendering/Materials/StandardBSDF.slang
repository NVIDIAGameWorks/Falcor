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
import Utils.Math.MathHelpers;

/** Implementation of Falcor's standard surface BSDF.

    The BSDF has the following lobes:
    - Delta reflection (ideal specular reflection).
    - Specular reflection using a GGX microfacet model.
    - Diffuse reflection using Disney's diffuse BRDF.
    - Delta transmission (ideal specular transmission).
    - Specular transmission using a GGX microfacet model.
    - Diffuse transmission.

    The BSDF is a linear combination of the above lobes.
*/
struct StandardBSDF : IBSDF
{
    StandardBSDFData data;      ///< BSDF parameters.
    float3 emission;            ///< Radiance emitted in the incident direction (wi).

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        FalcorBSDF bsdf = FalcorBSDF(sd, data);

        return bsdf.eval(wiLocal, woLocal, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        if (!useImportanceSampling) return sampleReference(sd, sg, result);

        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = {};

        FalcorBSDF bsdf = FalcorBSDF(sd, data);

        bool valid = bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobe, sg);
        result.wo = sd.fromLocal(woLocal);

        return valid;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        if (!useImportanceSampling) return evalPdfReference(sd, wo);

        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        FalcorBSDF bsdf = FalcorBSDF(sd, data);

        return bsdf.evalPdf(wiLocal, woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.emission = emission;

        // Clamp roughness so it's representable of what is actually used in FalcorBSDF.
        // Roughness^2 below kMinGGXAlpha is used to indicate perfectly smooth surfaces.
        float alpha = data.roughness * data.roughness;
        p.roughness = alpha < kMinGGXAlpha ? 0.f : data.roughness;

        // Compute approximation of the albedos.
        // For now use the blend weights and colors, but this should be improved to better numerically approximate the integrals.
        p.diffuseReflectionAlbedo = (1.f - data.diffuseTransmission) * (1.f - data.specularTransmission) * data.diffuse;
        p.diffuseTransmissionAlbedo = data.diffuseTransmission * (1.f - data.specularTransmission) * data.transmission;
        p.specularReflectionAlbedo = (1.f - data.specularTransmission) * data.specular;
        p.specularTransmissionAlbedo = data.specularTransmission * data.specular;

        // Pass on our specular reflectance field unmodified.
        p.specularReflectance = data.specular;

        if (data.diffuseTransmission > 0.f || data.specularTransmission > 0.f) p.flags |= (uint)BSDFProperties::Flags::IsTransmissive;

        return p;
    }

    uint getLobes(const ShadingData sd)
    {
        return FalcorBSDF::getLobes(data);
    }


    // Additional functions

    /** Reference implementation that uses cosine-weighted hemisphere sampling.
        This is for testing purposes only.
        \param[in] sd Shading data.
        \param[in] sg Sample generator.
        \param[out] result Generated sample. Only valid if true is returned.
        \return True if a sample was generated, false otherwise.
    */
    bool sampleReference<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
    {
        const bool isTransmissive = (getLobes(sd) & (uint)LobeType::Transmission) != 0;

        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi

        if (isTransmissive)
        {
            if (sampleNext1D(sg) < 0.5f)
            {
                woLocal.z = -woLocal.z;
            }
            result.pdf *= 0.5f;
            if (min(abs(wiLocal.z), abs(woLocal.z)) < kMinCosTheta || result.pdf == 0.f) return false;
        }
        else
        {
            if (min(wiLocal.z, woLocal.z) < kMinCosTheta || result.pdf == 0.f) return false;
        }

        FalcorBSDF bsdf = FalcorBSDF(sd, data);

        result.wo = sd.fromLocal(woLocal);
        result.weight = bsdf.eval(wiLocal, woLocal, sg) / result.pdf;
        result.lobe = (uint)(woLocal.z > 0.f ? LobeType::DiffuseReflection : LobeType::DiffuseTransmission);

        return true;
    }

    /** Evaluates the directional pdf for sampling the given direction using the reference implementation.
        \param[in] sd Shading data.
        \param[in] wo Outgoing direction.
        \return PDF with respect to solid angle for sampling direction wo.
    */
    float evalPdfReference(const ShadingData sd, const float3 wo)
    {
        const bool isTransmissive = (getLobes(sd) & (uint)LobeType::Transmission) != 0;

        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        if (isTransmissive)
        {
            if (min(abs(wiLocal.z), abs(woLocal.z)) < kMinCosTheta) return 0.f;
            return 0.5f * woLocal.z * M_1_PI; // pdf = 0.5 * cos(theta) / pi
        }
        else
        {
            if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;
            return woLocal.z * M_1_PI; // pdf = cos(theta) / pi
        }
    }
};
