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
import Experimental.Scene.Material.ClothBRDF;
import Utils.Math.MathHelpers;

/** Simple cloth material.
*/
struct ClothMaterial : IMaterial
{
    // Implementation of IMaterial interface

    float3 eval(const ShadingData sd, const float3 wo)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        ClothBRDF brdf;
        brdf.setup(sd);

        return brdf.eval(wiLocal, woLocal);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        return sampleReference(sd, sg, result);
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        return evalPdfReference(sd, wo);
    }


    // Additional functions

    /** Return the set of available BSDF lobes for the given shading data.
        \param[in] sd Shading data.
        \return A combination of LobeType flags (see IBxDF.slang).
    */
    uint getBSDFLobes(const ShadingData sd)
    {
        return (uint)LobeType::DiffuseReflection;
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
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta || result.pdf == 0.f) return false;

        ClothBRDF brdf;
        brdf.setup(sd);

        result.wo = sd.fromLocal(woLocal);
        result.weight = brdf.eval(wiLocal, woLocal) / result.pdf;
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
