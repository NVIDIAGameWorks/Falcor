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
__exported import Rendering.Materials.LobeType;
__exported import Utils.Sampling.SampleGeneratorInterface;

// Minimum cos(theta) for the incident and outgoing vectors.
// Some BSDF functions are not robust for cos(theta) == 0.0,
// so using a small epsilon for consistency.
// TODO: Move into IBxDF if possible
static const float kMinCosTheta = 1e-6f;

/** Low-level interface for BxDF functions.

    Conventions:
    - All operations are done in a local coordinate frame.
    - The local frame has normal N=(0,0,1), tangent T=(1,0,0) and bitangent B=(0,1,0).
    - The incident and outgoing direction point away from the shading location.
    - The incident direction (wi) is always in the positive hemisphere.
    - The outgoing direction (wo) is sampled.
    - Evaluating the BxDF always includes the foreshortening term (dot(wo, n) = wo.z).
*/
[anyValueSize(40)]
interface IBxDF
{
    /** Evaluates the BxDF.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \param[in,out] sg Sample generator.
        \return Returns f(wi, wo) * dot(wo, n).
    */
    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg);

    /** Samples the BxDF.
        \param[in] wi Incident direction.
        \param[out] wo Outgoing direction.
        \param[out] pdf pdf with respect to solid angle for sampling outgoing direction wo (0 if a delta event is sampled).
        \param[out] weight Sample weight f(wi, wo) * dot(wo, n) / pdf(wo).
        \param[out] lobe Sampled lobe (see LobeType).
        \param[in,out] sg Sample generator.
        \return Returns true if successful.
    */
    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout S sg);

    /** Evaluates the BxDF directional pdf for sampling outgoing direction wo.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \return Returns the pdf with respect to solid angle for sampling outgoing direction wo (0 for delta events).
    */
    float evalPdf(const float3 wi, const float3 wo);
}
