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
__exported import Rendering.Materials.ClothBRDF;
import Utils.Math.MathHelpers;

/** Implementation of a simple cloth BSDF.
*/
struct ClothBSDF : IBSDF
{
    ClothBRDF brdf;

    __init(const ClothBRDF brdf)
    {
        this.brdf = brdf;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        return brdf.eval(wiLocal, woLocal, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        return sampleReference(sd, sg, result);
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        return evalPdfReference(sd, wo);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        // Compute approximation of the perceived roughness.
        // It's an open problem how to best do this. For now approximate based on the sheen roughness.
        p.roughness = lerp(0.5f, 1.f, brdf.roughness);

        // Compute approximation of the albedos.
        // For now assume all diffuse reflection, this should be improved as there is some directionality.
        p.diffuseReflectionAlbedo = brdf.diffuseColor;

        return p;
    }

    uint getLobes(const ShadingData sd)
    {
        return (uint)LobeType::DiffuseReflection;
    }


    // Additional functions

    /** Reference implementation that uses cosine-weighted hemisphere sampling.
        This is for testing purposes only. Note that transmission is not sampled.
        \param[in] sd Shading data.
        \param[in] sg Sample generator.
        \param[out] result Generated sample. Only valid if true is returned.
        \return True if a sample was generated, false otherwise.
    */
    bool sampleReference<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta || result.pdf == 0.f) return false;

        result.wo = sd.fromLocal(woLocal);
        result.weight = brdf.eval(wiLocal, woLocal, sg) / result.pdf;
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
