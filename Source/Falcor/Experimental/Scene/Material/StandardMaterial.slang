/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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

__exported import Experimental.Scene.Material.IMaterial;
__exported import Experimental.Scene.Material.BxDF;
import Utils.Math.MathHelpers;

/** Implementation of Falcor's standard surface material.

    The material has the following BxDF lobes:
    - Delta reflection (ideal specular reflection).
    - Specular reflection using a GGX microfacet model.
    - Diffuse reflection using Disney's diffuse BRDF.
    - Delta transmission (ideal specular transmission).
    - Specular transmission using a GGX microfacet model.
    - Diffuse transmission.

    The material is a linear combination of the above BSDF lobes.
    By default, the material is single-sided and the back side black.
    Double-sided materials are supported by mirroring the BSDF around the
    shading normal. This is possible because the material is not layered.
*/
struct StandardMaterial : IMaterial
{
    // Implementation of IMaterial interface

    float3 eval(const ShadingData sd, const float3 wo)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        FalcorBSDF bsdf;
        bsdf.setup(sd);

        return bsdf.eval(wiLocal, woLocal);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        if (!useImportanceSampling) return sampleReference(sd, sg, result);

        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = {};

        FalcorBSDF bsdf;
        bsdf.setup(sd);

        bool valid = bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobe, sg);
        result.wo = sd.fromLocal(woLocal);

        return valid;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        if (!useImportanceSampling) return evalPdfReference(sd, wo);

        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        FalcorBSDF bsdf;
        bsdf.setup(sd);

        return bsdf.evalPdf(wiLocal, woLocal);
    }


    // Additional functions

    /** Return the set of available BSDF lobes for the given shading data.
        \param[in] sd Shading data.
        \return A combination of LobeType flags (see IBxDF.slang).
    */
    uint getBSDFLobes(const ShadingData sd)
    {
        return FalcorBSDF::getLobes(sd);
    }

    /** Reference implementation that uses cosine-weighted hemisphere sampling.
        This is for testing purposes only. Note that transmission is not sampled.
        \param[in] sd Shading data.
        \param[in] sg Sample generator.
        \param[out] result Generated sample. Only valid if true is returned.
        \return True if a sample was generated, false otherwise.
    */
    bool sampleReference<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
    {
        // TODO: Sample whole sphere if transmissive
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta || result.pdf == 0.f) return false;

        FalcorBSDF bsdf;
        bsdf.setup(sd);

        result.wo = sd.fromLocal(woLocal);
        result.weight = bsdf.eval(wiLocal, woLocal) / result.pdf;
        result.lobe = (uint)LobeType::DiffuseReflection;

        return true;
    }

    /** Evaluates the directional pdf for sampling the given direction using the reference implementation.
        \param[in] sd Shading data.
        \param[in] wo Outgoing direction.
        \return PDF with respect to solid angle for sampling direction wo (0 for delta events).
    */
    float evalPdfReference(const ShadingData sd, const float3 wo)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;

        return woLocal.z * M_1_PI; // pdf = cos(theta) / pi
    }
};
