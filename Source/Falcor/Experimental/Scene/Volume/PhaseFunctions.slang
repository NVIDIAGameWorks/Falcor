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

import Utils.Math.MathHelpers;

/** Evaluates the isotropic (uniform) phase function.
*/
float evalPhaseIsotropic() { return M_1_4PI; }

/** Evaluates the anisotropic Henyey-Greenstein phase function.
    Note: This function reduces to isotropic phase at g = 0 and has singularities at g = -1 and g = 1.
    \param[in] cos_t Cosine between unscattered and scattered direction.
    \param[in] g Anisotropy parameter in (-1, 1), where positive values promote forward scattering.
*/
float evalPhaseHenyeyGreenstein(const float cos_t, const float g)
{
    const float denom = 1 + g * g + 2 * g * cos_t;
    return M_1_4PI * (1 - g * g) / (denom * sqrt(denom));
}

/** Samples a direction according to the isotropic phase function, uniformly distributed over the sphere.
    \param[in] u Uniform random samples in [0, 1).
*/
float3 samplePhaseIsotropic(const float2 u) { return sample_sphere(u); }

/** Samples a direction according to the anisotropic Henyey-Greenstein phase function, distributed around a given direction.
    Note: This function reduces to isotropic phase at g = 0 and has singularities at g = -1 and g = 1.
    \param[in] dir Unscattered ray direction, pointing towards the scatter location.
    \param[in] g Anisotropy parameter in (-1, 1), where positive values promote forward scattering.
    \param[in] u Uniform random samples in [0, 1).
*/
float3 samplePhaseHenyeyGreenstein(const float3 dir, const float g, const float2 u)
{
    // Sample phase in tangent space.
    const float phi = M_2PI * u.y;
    const float sqr = (1.f - g * g) / (1.f - g + 2.f * g * u.x);
    const float cosTheta = abs(g) < 1e-3f ? 1.f - 2.f * u.x : (1.f + g * g - sqr * sqr) / (2.f * g);
    const float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
    const float3 phase = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    // Build tangent frame.
    const float3 tangent = perp_stark(dir);
    const float3 bitangent = cross(dir, tangent);
    // Tangent to world transformation.
    return normalize(phase.x * tangent + phase.y * bitangent + phase.z * dir);
}
