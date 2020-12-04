/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import Scene.ShadingData;
import Utils.Math.MathHelpers;
import Experimental.Scene.Material.HairChiang16;
import Experimental.Scene.Material.MaterialHelpers;
import Experimental.Scene.Material.MaterialShading;

/*******************************************************************
                           BCSDF functions
*******************************************************************/

/** Evaluates the BCSDF multiplied by NdotL for a given incident direction.

    \param[in] sd Shading point data.
    \param[in] L Normalized incident direction from shading point towards light source.
    \return f * saturate(dot(N,L))
*/
float3 evalBCSDFCosine(const ShadingData sd, float3 L)
{
#if BCSDF == HairChiang
    return evalHairChiang16(sd, L);
#else
    // Diffuse BRDF
    return sd.diffuse * M_1_PI * saturate(dot(sd.N, L));
#endif
}

/** Importance sampling of the BCSDF.

    Note: The evaluated pdf for the generated sample is expensive to compute. If the caller
    doesn't explicitly need the probability, they should be careful not to touch the value
    so that the compiler can do dead code elimination.

    \param[in] sd Shading point data.
    \param[in] sg Sample generator.
    \param[out] result Generated sample. Only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool sampleBCSDF<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
{
#if BCSDF == HairChiang
    return sampleHairChiang16(sd, sg, result);
#else
    // Diffuse BRDF
    float3 wi = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf);
    result.wi = fromLocal(wi, sd);
    result.weight = sd.diffuse * M_1_PI;
    result.lobe = (uint)LobeType::DiffuseReflection;
    return true;
#endif
}

/** Evaluates the probability density function.
    \param[in] sd Describes the shading point.
    \param[in] L The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfBCSDF(const ShadingData sd, float3 L)
{
#if BCSDF == HairChiang
    return evalPdfHairChiang16(sd, L);
#else
    // Diffuse BRDF
    return saturate(dot(sd.N, L));
#endif
}

/*******************************************************************
                         Interface wrappers
*******************************************************************/

float3 evalHairChiang16(const ShadingData sd, float3 L)
{
    float3 wo = toLocal(sd.V, sd);
    float3 wi = toLocal(L, sd);

    HairChiang16 bcsdf;
    bcsdf.setup(sd);
    return bcsdf.eval(wo, wi);
}

bool sampleHairChiang16<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
{
    float3 wo = toLocal(sd.V, sd);
    float3 wi;

    HairChiang16 bcsdf;
    bcsdf.setup(sd);
    bool valid = bcsdf.sample(wo, wi, result.pdf, result.weight, result.lobe, sg);
    result.wi = fromLocal(wi, sd);
    return valid;
}

float evalPdfHairChiang16(const ShadingData sd, float3 L)
{
    float3 wo = toLocal(sd.V, sd);
    float3 wi = toLocal(L, sd);

    HairChiang16 bcsdf;
    bcsdf.setup(sd);
    return bcsdf.evalPdf(wo, wi);
}
