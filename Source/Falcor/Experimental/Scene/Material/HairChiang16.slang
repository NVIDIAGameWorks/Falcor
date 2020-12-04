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

import Scene.ShadingData;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGeneratorInterface;
import Experimental.Scene.Material.Fresnel;
import Experimental.Scene.Material.BxDF;
__exported import Experimental.Scene.Material.BxDFTypes;

// Enable explicitly computing sampling weights using eval(wo, wi) / evalPdf(wo, wi).
// This is for testing only, as many terms of the equation cancel out allowing to save on computation.
#define USE_BCSDF_IMPORTANCE_SAMPLING   1

/** Hair BCSDF from "A Practical and Controllable Hair and Fur Model for Production Path Tracing", Chiang et al. 2016.
    Implementation is adapted from pbrt-v3.
*/
struct HairChiang16 : IBxDF
{
    // Max number of scattering events that are explicitly computed.
    // All higher-order scattering terms (>= kMaxScatterEvents) will be represented by a single one.
    static const uint kMaxScatterEvents = 3;

    static const float kSqrtPiOver8 = 0.626657069f;

    float betaM, betaN, alpha, IoR;
    float3 sigmaA;
    float h;
    float eta;

    float gammaO;
    float v[kMaxScatterEvents + 1];
    float s;
    float sin2kAlpha[3], cos2kAlpha[3];

    [mutating] void precompute()
    {
        gammaO = asin(clamp(h, -1.f, 1.f));

        float tmp = 0.726f * betaM + 0.812f * betaM * betaM + 3.7f * pow(betaM, 20.f);
        v[0] = tmp * tmp;
        v[1] = 0.25f * v[0];
        v[2] = 4 * v[0];
        [unroll]
        for (uint p = 3; p <= kMaxScatterEvents; p++) v[p] = v[2];

        // Compute azimuthal logistic scale factor
        s = kSqrtPiOver8 * (0.265f * betaN + 1.194f * betaN * betaN + 5.372f * pow(betaN, 22.f));

        // Compute alpha terms for hair scales
        sin2kAlpha[0] = sin(alpha / 180.f * M_PI);
        cos2kAlpha[0] = sqrt(max(0.f, 1.f - sin2kAlpha[0] * sin2kAlpha[0]));
        [unroll]
        for (uint i = 1; i < 3; i++)
        {
            sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
            cos2kAlpha[i] = cos2kAlpha[i - 1] * cos2kAlpha[i - 1] - sin2kAlpha[i - 1] * sin2kAlpha[i - 1];
        }
    }

    [mutating] void setup(const ShadingData sd)
    {
        betaM = sd.specular.x;
        betaN = sd.specular.y;
        sigmaA = sigmaAFromColor(sd.diffuse, betaN);
        alpha = sd.specular.z;
        IoR = sd.IoR;
        eta = sd.eta;

        // Compute offset h azimuthally with the unit circle cross section.
        float3 woProj = normalize(sd.V - dot(sd.V, sd.T) * sd.T);   // Project wo to the (B, N) plane.
        float3 woProjPerp = cross(woProj, sd.T);
        h = dot(sd.N, woProjPerp);

        precompute();
    }

    float3 eval(float3 wo, float3 wi)
    {
        float sinThetaO = wo.x;
        float cosThetaO = sqrt(max(0.f, 1.f - sinThetaO * sinThetaO));
        float phiO = atan2(wo.z, wo.y);

        float sinThetaI = wi.x;
        float cosThetaI = sqrt(max(0.f, 1.f - sinThetaI * sinThetaI));
        float phiI = atan2(wi.z, wi.y);

        // Compute refracted ray.
        float sinThetaT = sinThetaO / IoR;
        float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));

        float etap = sqrt(IoR * IoR - sinThetaO * sinThetaO) / cosThetaO;
        float sinGammaT = h / etap;
        float cosGammaT = sqrt(max(0.f, 1.f - sinGammaT * sinGammaT));
        float gammaT = asin(clamp(sinGammaT, -1.f, 1.f));

        // Compute the transmittance T of a single path through the cylinder.
        float tmp = -2.f * cosGammaT / cosThetaT;
        float3 T = exp(sigmaA * tmp);

        // Evaluate hair BCSDF for each lobe.
        float phi = phiI - phiO;
        float3 ap[kMaxScatterEvents + 1];
        Ap(cosThetaO, T, ap);
        float3 result = float3(0.f);

        [unroll]
        for (int p = 0; p < kMaxScatterEvents; p++)
        {
            float sinThetaOp, cosThetaOp;
            if (p == 0)
            {
                sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
                cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
            }
            else if (p == 1)
            {
                sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
                cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
            }
            else if (p == 2)
            {
                sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
                cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
            }
            else
            {
                sinThetaOp = sinThetaO;
                cosThetaOp = cosThetaO;
            }

            cosThetaOp = abs(cosThetaOp);
            result += ap[p] * Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[p]) * Np(phi, p, s, gammaO, gammaT);
        }

        // Compute contribution of remaining terms after kMaxScatterEvents.
        result += ap[kMaxScatterEvents] * Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[kMaxScatterEvents]) * M_1_2PI;

        result = isnan(luminance(result)) ? float3(0.f) : result;
        return result;
    }

    bool sample<S : ISampleGenerator>(float3 wo, out float3 wi, out float pdf, out float3 weight, out uint lobe, inout S sg)
    {
#if !USE_BCSDF_IMPORTANCE_SAMPLING
        wi = sample_sphere(sampleNext2D(sg));
        pdf = M_1_4PI;
        weight = eval(wo, wi) / pdf;
        lobe = wi.z > 0 ? (uint)LobeType::SpecularReflection : (uint)LobeType::SpecularTransmission;
        return true;
#endif

        float sinThetaO = wo.x;
        float cosThetaO = sqrt(max(0.f, 1.f - sinThetaO * sinThetaO));
        float phiO = atan2(wo.z, wo.y);

        float2 u[2] = { sampleNext2D(sg), sampleNext2D(sg) };

        // Determine which term p to sample for hair scattering.
        float apPdf[kMaxScatterEvents + 1];
        computeApPdf(cosThetaO, apPdf);
        uint p = 0;
        while (p < kMaxScatterEvents && u[0].x >= apPdf[p])
        {
            u[0].x -= apPdf[p];
            p++;
        }

        float sinThetaOp, cosThetaOp;
        if (p == 0)
        {
            sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
            cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
        }
        else if (p == 1)
        {
            sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
            cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
        }
        else if (p == 2)
        {
            sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
            cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
        }
        else
        {
            sinThetaOp = sinThetaO;
            cosThetaOp = cosThetaO;
        }

        // Sample Mp to compute thetaI.
        u[1].x = max(u[1].x, 1e-5f);
        float cosTheta = 1.f + v[p] * log(u[1].x + (1.f - u[1].x) * exp(-2.f / v[p]));
        float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        float cosPhi = cos(u[1].y * M_2PI);
        float sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
        float cosThetaI = sqrt(max(0.f, 1.f - sinThetaI * sinThetaI));

        // Sample Np to compute dphi.
        float etap = sqrt(IoR * IoR - sinThetaO * sinThetaO) / cosThetaO;
        float sinGammaT = h / etap;
        float gammaT = asin(clamp(sinGammaT, -1.f, 1.f));
        float dphi;
        if (p < kMaxScatterEvents)
        {
            dphi = phiFunction(p, gammaO, gammaT) + sampleTrimmedLogistic(u[0].y, s, -M_PI, M_PI);
        }
        else
        {
            dphi = u[0].y * M_2PI;
        }

        float phiI = phiO + dphi;
        wi = float3(sinThetaI, cosThetaI * cos(phiI), cosThetaI * sin(phiI));
        lobe = wi.z > 0 ? (uint)LobeType::SpecularReflection : (uint)LobeType::SpecularTransmission;

        // Compute pdf.
        pdf = 0;
        [unroll]
        for (uint i = 0; i < kMaxScatterEvents; i++)
        {
            float sinThetaOp, cosThetaOp;
            if (i == 0)
            {
                sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
                cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
            }
            else if (i == 1)
            {
                sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
                cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
            }
            else if (i == 2)
            {
                sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
                cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
            }
            else
            {
                sinThetaOp = sinThetaO;
                cosThetaOp = cosThetaO;
            }

            cosThetaOp = abs(cosThetaOp);
            pdf += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[i]) * apPdf[i] * Np(dphi, i, s, gammaO, gammaT);
        }
        pdf += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[kMaxScatterEvents]) * apPdf[kMaxScatterEvents] * M_1_2PI;

        if (!isnan(pdf))
        {
            weight = eval(wo, wi) / pdf;
        }
        else
        {
            pdf = 0.f;
            weight = float3(0.f);
        }
        return (pdf > 0.f);
    }

    float evalPdf(float3 wo, float3 wi)
    {
#if !USE_BCSDF_IMPORTANCE_SAMPLING
        return M_1_4PI;
#endif

        float sinThetaO = wo.x;
        float cosThetaO = sqrt(max(0.f, 1.f - sinThetaO * sinThetaO));
        float phiO = atan2(wo.z, wo.y);

        float sinThetaI = wi.x;
        float cosThetaI = sqrt(max(0.f, 1.f - sinThetaI * sinThetaI));
        float phiI = atan2(wi.z, wi.y);

        // Compute refracted ray.
        float etap = sqrt(IoR * IoR - sinThetaO * sinThetaO) / cosThetaO;
        float sinGammaT = h / etap;
        float gammaT = asin(clamp(sinGammaT, -1.f, 1.f));

        float apPdf[kMaxScatterEvents + 1];
        computeApPdf(cosThetaO, apPdf);

        // Compute pdf.
        float phi = phiI - phiO;
        float pdf = 0;

        [unroll]
        for (int p = 0; p < kMaxScatterEvents; p++)
        {
            float sinThetaOp, cosThetaOp;
            if (p == 0)
            {
                sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
                cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
            }
            else if (p == 1)
            {
                sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
                cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
            }
            else if (p == 2)
            {
                sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
                cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
            }
            else
            {
                sinThetaOp = sinThetaO;
                cosThetaOp = cosThetaO;
            }

            cosThetaOp = abs(cosThetaOp);
            pdf += apPdf[p] * Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[p]) * Np(phi, p, s, gammaO, gammaT);
        }

        // Compute contribution of remaining terms after kMaxScatterEvents.
        pdf += apPdf[kMaxScatterEvents] * Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[kMaxScatterEvents]) * M_1_2PI;

        pdf = isnan(pdf) ? 0.f : pdf;
        return pdf;
    }

    // private

    /** Attenuation function Ap.
    */
    void Ap(float cosThetaO, float3 T, out float3 ap[kMaxScatterEvents + 1])
    {
        float cosGammaO = sqrt(max(0.f, 1.f - h * h));
        float cosTheta = cosThetaO * cosGammaO;
        float f = evalFresnelDielectric(eta, cosTheta);

        ap[0] = float3(f);
        ap[1] = T * (1 - f) * (1 - f);
        [unroll]
        for (uint p = 2; p < kMaxScatterEvents; p++) ap[p] = ap[p - 1] * T * f;

        // Compute attenuation term accounting for remaining orders of scattering.
        ap[kMaxScatterEvents] = ap[kMaxScatterEvents - 1] * T * f / (float3(1.f) - T * f);
    }

    /** Compute a discrete pdf for sampling Ap (which BCSDF lobe).
    */
    void computeApPdf(float cosThetaO, out float apPdf[kMaxScatterEvents + 1])
    {
        float sinThetaO = sqrt(max(0.f, 1.f - cosThetaO * cosThetaO));

        // Compute refracted ray.
        float sinThetaT = sinThetaO / IoR;
        float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));

        float etap = sqrt(IoR * IoR - sinThetaO * sinThetaO) / cosThetaO;
        float sinGammaT = h / etap;
        float cosGammaT = sqrt(max(0.f, 1.f - sinGammaT * sinGammaT));

        // Compute the transmittance T of a single path through the cylinder.
        float tmp = -2.f * cosGammaT / cosThetaT;
        float3 T = exp(sigmaA * tmp);

        float3 ap[kMaxScatterEvents + 1];
        Ap(cosThetaO, T, ap);

        // Compute apPdf from individal ap terms.
        float sumY = 0.f;
        [unroll]
        for (uint p = 0; p <= kMaxScatterEvents; p++)
        {
            apPdf[p] = luminance(ap[p]);
            sumY += apPdf[p];
        }

        float invSumY = 1.f / sumY;
        [unroll]
        for (uint p = 0; p <= kMaxScatterEvents; p++) apPdf[p] *= invSumY;
    }
};

/*******************************************************************
                          Helper functions
*******************************************************************/

/** Longitudinal scattering function Mp.
*/
float Mp(float cosThetaI, float cosThetaO, float sinThetaI, float sinThetaO, float v)
{
    float a = cosThetaI * cosThetaO / v;
    float b = sinThetaI * sinThetaO / v;
    float mp = (v <= 0.1f) ? exp(logI0(a) - b - 1.f / v + 0.6931f + log(0.5f / v)) : (exp(-b) * I0(a)) / (sinh(1.f / v) * 2.f * v);
    return mp;
}

float I0(float x)
{
    float val = 0.f;
    float x2i = 1.f;
    float ifact = 1.f;
    uint i4 = 1;

    [unroll]
    for (uint i = 0; i < 10; i++)
    {
        if (i > 1) ifact *= i;
        val += x2i / (ifact * ifact * i4);
        x2i *= x * x;
        i4 *= 4;
    }
    return val;
}

float logI0(float x)
{
    if (x > 12)
    {
        return x + 0.5f * (-log(M_2PI) + log(1.f / x) + 0.125f / x);
    }
    else
    {
        return log(I0(x));
    }
}

/** Azimuthal scattering function Np.
*/
float Np(float phi, int p, float s, float gammaO, float gammaT)
{
    float dphi = phi - phiFunction(p, gammaO, gammaT);

    // Remap dphi to [-pi, pi].
    dphi = fmod(dphi, M_2PI);
    if (dphi > M_PI) dphi -= M_2PI;
    if (dphi < -M_PI) dphi += M_2PI;

    return trimmedLogistic(dphi, s, -M_PI, M_PI);
}

float phiFunction(int p, float gammaO, float gammaT)
{
    return 2.f * p * gammaT - 2.f * gammaO + p * M_PI;
}

float logistic(float x, float s)
{
    x = abs(x);
    float tmp = exp(-x / s);
    return tmp / (s * (1.f + tmp) * (1.f + tmp));
}

float logisticCDF(float x, float s)
{
    return 1.f / (1.f + exp(-x / s));
}

float trimmedLogistic(float x, float s, float a, float b)
{
    return logistic(x, s) / (logisticCDF(b, s) - logisticCDF(a, s));
}

float sampleTrimmedLogistic(float u, float s, float a, float b)
{
    float k = logisticCDF(b, s) - logisticCDF(a, s);
    float x = -s * log(1.f / (u * k + logisticCDF(a, s)) - 1.f);
    return clamp(x, a, b);
}

/** Mapping from color to sigmaA.
*/
float3 sigmaAFromColor(float3 color, float betaN)
{
    float tmp = 5.969f - 0.215f * betaN + 2.532f * betaN * betaN - 10.73f * pow(betaN, 3) + 5.574f * pow(betaN, 4) + 0.245f * pow(betaN, 5);
    float3 sqrtSigmaA = log(max(color, 1e-4f)) / tmp;
    return sqrtSigmaA * sqrtSigmaA;
}
