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
__exported import Scene.ShadingData;
__exported import Experimental.Scene.Material.IBxDF;
__exported import Utils.Sampling.SampleGeneratorInterface;

/** Describes a BSDF sample.
*/
struct BSDFSample
{
    float3  wo;             ///< Sampled direction in world space (normalized).
    float   pdf;            ///< pdf with respect to solid angle for the sampled direction (wo).
    float3  weight;         ///< Sample weight f(wi, wo) * dot(wo, n) / pdf(wo).
    uint    lobe;           ///< Sampled lobe. This is a combination of LobeType flags (see IBxDF.slang).

    bool isLobe(LobeType type)
    {
        return (lobe & uint(type)) != 0;
    }
};

/** Interface for material evaluation and sampling.

    The shading location and its attributes are described by a ShadingData struct.
    The ShadingData view direction field specifies the direction from which the
    shading location is seen. By convention we call this the incident direction (wi).
    The outgoing direction (wo) is the direction in which the transported quantity
    (radiance or importance) is scattered in.

    Conventions:
     - The incident and outgoing direction point away from the shading location.
     - The incident direction (wi) is given by ShadingData.
     - The outgoing direction (wo) is sampled.
     - The foreshortening term with respect to the sampled direction is always included.
*/
interface IMaterial
{
    /** Evaluates the material's BxDF.
        \param[in] sd Shading data.
        \param[in] wo Outgoing direction.
        \return Returns f(wi, wo) * dot(wo, n).
    */
    float3 eval(const ShadingData sd, const float3 wo);

    /** Samples the material's BxDF.
        \param[in] sd Shading data.
        \param[in] sg Sample generator.
        \param[out] result Generated sample. Only valid if true is returned.
        \param[in] useImportanceSampling Hint to use importance sampling, else default to reference implementation if available.
        \return True if a sample was generated, false otherwise.
    */
    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true);

    /** Evaluates the directional pdf for sampling the given direction.
        \param[in] sd Shading data.
        \param[in] wo Outgoing direction.
        \param[in] useImportanceSampling Hint to use importance sampling, else default to reference implementation if available.
        \return PDF with respect to solid angle for sampling direction wo (0 for delta events).
    */
    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true);
}
