/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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

__exported import IPhaseFunction;
import Utils.Math.MathHelpers;

/** Null collision phase function
*/
struct NullPhaseFunction : IPhaseFunction
{
    __init()
    {
    }

    float3 eval(const float3 wi, const float3 wo)
    {
        return 0.f;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout S sg)
    {
        wo = -wi;
        pdf = 1.f;
        weight = 1.f;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        return 0.f;
    }
};

/** Isotropic phase function.
    f_p(wi, wo) = 1 / 4PI
*/
struct IsotropicPhaseFunction : IPhaseFunction
{
    __init()
    {
    }

    float3 eval(const float3 wi, const float3 wo)
    {
        return M_1_4PI;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout S sg)
    {
        wo = sample_sphere(sampleNext2D(sg));
        pdf = M_1_4PI;
        weight = 1.f;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        return M_1_4PI;
    }
};

/** Henyey-Greenstein phase function.
    Note: The incident and outgoing directions are assumed to be pointing away from the scattering position.
    Note: This function reduces to isotropic phase function at g = 0 and has singularities at g = -1 and g = 1.
*/
struct HenyeyGreensteinPhaseFunction : IPhaseFunction
{
    float g;

    __init(float g)
    {
        this.g = g;
    }

    float3 eval(const float3 wi, const float3 wo)
    {
        return evalHenyeyGreenstein(dot(wi, wo), g);
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout S sg)
    {
        const float2 u = sampleNext2D(sg);

        float cosTheta;
        if (abs(g) < 1e-3f)
        {
            cosTheta = 1.f - 2.f * u.x;
        }
        else
        {
            const float sqr = (1.f - g * g) / (1.f - g + 2.f * g * u.x);
            cosTheta = (1.f + g * g - sqr * sqr) / (2.f * g);
        }

        const float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        const float phi = M_2PI * u.y;
        const float3 w = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

        const float3 dir = -wi;
        const float3 T = perp_stark(dir);
        const float3 B = cross(dir, T);

        wo = normalize(w.x * T + w.y * B + w.z * dir);
        pdf = evalHenyeyGreenstein(-cosTheta, g);
        weight = 1.f;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        return evalHenyeyGreenstein(dot(wi, wo), g);
    }
};

/** Evaluates the anisotropic Henyey-Greenstein phase function.
    Note: This function reduces to isotropic phase function at g = 0 and has singularities at g = -1 and g = 1.
    Convention: both the incident direction and the outgoing direction point away from the scatter point, hence the "+"
    \param[in] cosTheta Cosine between incoming and scattered direction.
    \param[in] g Anisotropy parameter in (-1, 1), where positive values promote forward scattering.
*/
float evalHenyeyGreenstein(const float cosTheta, const float g)
{
    const float denom = 1 + g * g + 2 * g * cosTheta;
    return M_1_4PI * (1 - g * g) / (denom * sqrt(denom));
}
/** Evaluates the anisotropic Henyey-Greenstein phase function.
    Note: This function reduces to isotropic phase function at g = 0 and has singularities at g = -1 and g = 1.
    Convention: both the incident direction and the outgoing direction point away from the scatter point, hence the "+"
    \param[in] cosTheta Cosine between incoming and scattered direction.
    \param[in] g Anisotropy parameter in (-1, 1), where positive values promote forward scattering.
*/
float3 evalHenyeyGreenstein(const float cosTheta, const float3 g)
{
    const float3 denom = 1 + g * g + 2 * g * cosTheta;
    return M_1_4PI * (1 - g * g) / (denom * sqrt(denom));
}

struct DualHenyeyGreensteinPhaseFunction : IPhaseFunction
{
    float3 meancosine[2];
    float3 w0;

    __init(float3 meancosine0, float3 meancosine1, float3 w0)
    {
        this.meancosine[0] = meancosine0;
        this.meancosine[1] = meancosine1;
        this.w0 = w0;
    }

    float3 eval(const float3 wi, const float3 wo)
    {
        return evalHenyeyGreenstein(dot(wi, wo), meancosine[0]) * w0 + evalHenyeyGreenstein(dot(wi, wo), meancosine[1]) * (1.f - w0);
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout S sg)
    {
        const float3 u = sampleNext3D(sg);

        const float avgW0 = (w0[0]+w0[1]+w0[2]) / 3.f;
        const float3 meancosineSelect = u.z <= avgW0 ? meancosine[0] : meancosine[1];
        // Taking the average is really not the smartest thing.
        const float meancosineSelectAvg = (meancosineSelect[0] + meancosineSelect[1] + meancosineSelect[2]) / 3.f;

        float cosTheta;
        if (abs(meancosineSelectAvg) < 1e-3f)
        {
            cosTheta = 1.f - 2.f * u.x;
        }
        else
        {
            const float sqr = (1.f - meancosineSelectAvg * meancosineSelectAvg) / (1.f - meancosineSelectAvg + 2.f * meancosineSelectAvg * u.x);
            cosTheta = (1.f + meancosineSelectAvg * meancosineSelectAvg - sqr * sqr) / (2.f * meancosineSelectAvg);
        }

        float selectionWeight = {};
        float3 pfWeight = {};
        if (u.z <= avgW0)
        {
            selectionWeight = avgW0;
            pfWeight = w0;
        }
        else
        {
            selectionWeight = (1.0 - avgW0);
            pfWeight = 1.f - w0;
        }

        const float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        const float phi = M_2PI * u.y;
        const float3 w = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

        const float3 dir = -wi;
        const float3 T = perp_stark(dir);
        const float3 B = cross(dir, T);

        wo = normalize(w.x * T + w.y * B + w.z * dir);
        pdf = evalHenyeyGreenstein(-cosTheta, meancosineSelectAvg);
        weight = (evalHenyeyGreenstein(-cosTheta, meancosineSelect) * pfWeight / evalHenyeyGreenstein(-cosTheta, meancosineSelectAvg)) / selectionWeight;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        const float mc0 = (meancosine[0][0]+meancosine[0][1]+meancosine[0][2]) / 3.f;
        const float mc1 = (meancosine[1][0]+meancosine[1][1]+meancosine[1][2]) / 3.f;
        const float avgW0 = (w0[0]+w0[1]+w0[2]) / 3.f;
        return evalHenyeyGreenstein(dot(wi, wo), mc0) * avgW0 + evalHenyeyGreenstein(dot(wi, wo), mc1) * (1.f - avgW0);
    }
}
