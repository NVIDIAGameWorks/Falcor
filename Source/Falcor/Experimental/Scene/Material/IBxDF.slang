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
__exported import Utils.Sampling.SampleGeneratorInterface;

static const uint kLobeTypeBits = 8;    ///< Maximum number of bits to store LobeType.

/** Flags representing the various lobes of a BxDF.
*/
enum class LobeType
{
    DiffuseReflection       = 0x01,
    SpecularReflection      = 0x02,
    DeltaReflection         = 0x04,

    DiffuseTransmission     = 0x10,
    SpecularTransmission    = 0x20,
    DeltaTransmission       = 0x40,

    Diffuse                 = 0x11,
    Specular                = 0x22,
    Delta                   = 0x44,
    NonDelta                = 0x33,

    Reflection              = 0x0f,
    Transmission            = 0xf0,

    NonDeltaReflection      = 0x03,
    NonDeltaTransmission    = 0x30,

    All                     = 0xff,
};

// Minimum cos(theta) for the incident and outgoing vectors.
// Some BSDF functions are not robust for cos(theta) == 0.0,
// so using a small epsilon for consistency.
// TODO: Move into IBxDF if possible
static const float kMinCosTheta = 1e-6f;

/** Low-level interface for BxDFs.

    Conventions:
    - all operations are done in a local coordinate frame.
    - the local frame has normal N=(0,0,1), tangent T=(1,0,0) and bitangent B=(0,1,0).
    - wo is the outgoing or scattering direction and points away from the shading location.
    - wi is the incident or light direction and points away from the shading location.
    - the outgoing direction is always in the positive hemisphere.
    - evaluating the BxDF always includes the foreshortening term (dot(wi, n) = wi.z).
*/
interface IBxDF
{
    /** Evaluates the BxDF.
        \param[in] wo Outgoing direction.
        \param[in] wi Incident direction.
        \return Returns f(wo, wi) * dot(wi, n).
    */
    float3 eval(float3 wo, float3 wi);

    /** Samples the BxDF.
        \param[in] wo Outgoing direction.
        \param[out] wi Incident direction.
        \param[out] pdf pdf with respect to solid angle for sampling incident direction wi (0 if a delta event is sampled).
        \param[out] weight Sample weight f(wo, wi) * dot(wi, n) / pdf(wi).
        \param[out] lobe Sampled lobe (see LobeType).
        \param[in,out] sg Sample generator.
        \return Returns true if successful.
    */
    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg);

    /** Evaluates the BxDF directional pdf for sampling incident direction wi.
        \param[in] wo Outgoing direction.
        \param[in] wi Incident direction.
        \return Returns the pdf with respect to solid angle for sampling incident direction wi (0 for delta events).
    */
    float evalPdf(float3 wo, float3 wi);
}
