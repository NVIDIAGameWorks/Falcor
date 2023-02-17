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
#include "BSDFConfig.slangh"

import Utils.Math.MathHelpers;
__exported import Rendering.Materials.NDF;

/** General microfacet normal distribution function, including support for
    elliptical anisotropy.

    Should be specialized by a specific (shape invariant) NDF.
*/
struct Microfacet
{
    INDF ndf;                      ///< Underlying normal distribution function.
    float2 alpha;                  ///< Anisotropic roughness parameter.
    float rho, sqrtOneMinusRhoSqr; ///< Encodes elliptical anisotropy axis in tangent space.

    /** Initialize an isotropic microfacet distribution.

        \param[in] ndf Underlying NDF to be used.
        \param[in] alpha Isotropic surface roughness. Should be clamped to small epsilon.
    */
    __init(INDF ndf, float alpha)
    {
        // See "Photorealistic Surface Rendering with Microfacet Theory"
        // by Jonathan Dupuy 2015, Eq. 2.29 - 2.31, simplified for the isotropic case.
        this.ndf = ndf;
        this.alpha = float2(alpha);
        this.rho = 0.f;
        this.sqrtOneMinusRhoSqr = 1.f;
    }

    /** Initialize an anisotropic microfacet distribution.

        \param[in] ndf Underlying NDF to be used.
        \param[in] alpha Anisotropic surface roughness. Should be clamped to small epsilon.
        \param[in] phi Anisotropy rotation in radians. Measured counterclockwise from the x-axis in the local frame. (Default: 0)
    */
    __init(INDF ndf, float2 alpha, float phi = 0.0f)
    {
        // See "Photorealistic Surface Rendering with Microfacet Theory"
        // by Jonathan Dupuy 2015, Eq. 2.29 - 2.31.
        this.ndf = ndf;
        float2 alpha2 = alpha * alpha;
        float tmp = (alpha2.x - alpha2.y) * cos(2.0f * phi);
        this.alpha = { sqrt(0.5f * (alpha2.x + alpha2.y + tmp)), sqrt(0.5f * (alpha2.x + alpha2.y - tmp)) };
        this.rho = (alpha2.x - alpha2.y) / (this.alpha.x * this.alpha.y) * cos(phi) * sin(phi);
        this.sqrtOneMinusRhoSqr = sqrt(1 - this.rho * this.rho);
    }

    /** A 'singular' distribution is one which would be more ideally treated as an
        ideal mirror. It can't be sampled reliably.
    */
    bool isSingular()
    {
        return all(alpha < 1e-3f);
    }

    /** Returns the isotropic roughness, i.e. the maximum along one of the tangent directions.
        */
    float getIsotropicRoughness()
    {
        return max(alpha.x, alpha.y);
    }

    /** Evaluates the NDF (D) for a given microfacet normal.
        This returns the density of potentially active microfacets (those for which the microfacet normal matches
        the half-vector).

        \param[in] wm Microfacet normal in local space, in the positive hemisphere.
        \return D(wm).
    */
    float evalD(float3 wm)
    {
        if (wm.z <= 0.0f)
        {
            return 0.0f;
        }

        // Transform microfacet normal to slope space.
        float2 slope = { -wm.x / wm.z, -wm.y / wm.z };

        // Stretch slope into the standard configuration, see
        // "Photorealistic Surface Rendering with Microfacet Theory"
        // by Jonathan Dupuy 2015, Eq. 2.27.
        float nrm = 1.0f / (alpha.x * alpha.y * sqrtOneMinusRhoSqr);
        float2 slopeStd = { slope.x / alpha.x, (alpha.x * slope.y - rho * alpha.y * slope.x) * nrm };

        // Eval slope distribution of standard configuration.
        float p22Std = ndf.evalP22Std(slopeStd);

        // Transform back from slope to normal distribution.
        float p22 = p22Std * nrm;
        float cosTheta2 = wm.z * wm.z;
        float cosTheta4 = cosTheta2 * cosTheta2;
        return p22 / cosTheta4;
    }

    /** Evaluates the PDF of sampling the NDF, see also `sampleFull`.

        \param[in] wm Microfacet normal in local space, in the positive hemisphere.
        \return D(wm) * dot(wm, n).
    */
    float evalPdfFull(float3 wm)
    {
        return evalD(wm) * wm.z;
    }

    /** Evaluates the PDF of sampling the visible NDF, see also `sampleVisible`.

        \param[in] wi Incident direction in local space.
        \param[in] wm Microfacet normal in local space, in the positive hemisphere.
        \return D_wi(wm) = dot(wm, wi) * D(wm) / sigma(-wi).
    */
    float evalPdfVisible(float3 wi, float3 wm)
    {
        return max(0.0f, dot(wm, wi)) * evalD(wm) / evalSigma(-wi);
    }

    /** Samples a microfacet normal proportionally to D(wm) * dot(wm, n).

        \param[in,out] sg Sample generator.
        \param[out] pdf Sampling density.
        \return Sampled microfacet normal in local space, in the positive hemisphere.
    */
    float3 sampleFull<S : ISampleGenerator>(inout S sg, out float pdf)
    {
        // Sample a slope in the standard configuration.
        float2 slopeStd = ndf.sampleP22Std(sg);

        // Stretch the slope based on the surface roughness, see
        // "Photorealistic Surface Rendering with Microfacet Theory"
        // by Jonathan Dupuy 2015, Eq. 2.35.
        float2 slope = { alpha.x * slopeStd.x, alpha.y * (rho * slopeStd.x + sqrtOneMinusRhoSqr * slopeStd.y) };

        // Transform slope to normal.
        float3 wm = normalize(float3(-slope.x, -slope.y, 1.0f));

        pdf = evalPdfFull(wm);
        return wm;
    }

    /** Samples a microfacet normal proportionally to D_wi(wm).
        This yields significant variance reduction compared to sampling of the full distribution.

        \param[in] wi Incident direction in local space.
        \param[in,out] sg Sample generator.
        \param[out] pdf Sampling density.
        \return Sampled microfacet normal in local space, in the positive hemisphere.
    */
    float3 sampleVisible<S : ISampleGenerator>(float3 wi, inout S sg, out float pdf)
    {
        // Stretch view direction into the standard configuration, see
        // "Photorealistic Surface Rendering with Microfacet Theory"
        // by Jonathan Dupuy 2015, page 28.
        float3 wiStd = normalize(float3(alpha.x * wi.x + rho * alpha.y * wi.y, alpha.y * wi.y * sqrtOneMinusRhoSqr, wi.z));

        // Sample visible slope in standard configuration.
        float2 slopeStd = ndf.sampleVisibleP22Std(wiStd, sg);

        // Stretch the slope based on the surface roughness, see
        // "Photorealistic Surface Rendering with Microfacet Theory"
        // by Jonathan Dupuy 2015, Eq. 2.35.
        float2 slope = { alpha.x * slopeStd.x, alpha.y * (rho * slopeStd.x + sqrtOneMinusRhoSqr * slopeStd.y) };

        // Transform slope to normal.
        float3 wm = normalize(float3(-slope.x, -slope.y, 1.0f));

        pdf = evalPdfVisible(wi, wm);
        return wm;
    }

    /** Evaluates the Smith masking function (G1).

        The function returns the same value for incident directions that lie in either hemisphere.

        \param[in] w Evaluated direction, in local space.
    */
    float evalG1(float3 w)
    {
        return abs(w.z) / evalSigma(-sign(w.z) * w);
    }

    /** Evaluates the separable form of the Smith shadowing-masking function.

        The function returns the same value for incident/outgoing directions that lie in either hemisphere.

        \param[in] wi Incident direction in local space.
        \param[in] wo Outgoing direction in local space.
        \return G2_separable(wi, wo).
    */
    float evalG2Separable(float3 wi, float3 wo)
    {
        return evalG1(wi) * evalG1(wo);
    }

    /** Evaluates the height-correlated form of the Smith shadowing-masking function.

        Correctly accounts for the different cases of reflection (both directions lie in the same hemisphere)
        and transmission (both directions lie in opposite hemispheres).

        \param[in] wi Incident direction in local space.
        \param[in] wo Outgoing direction in local space.
        \return G2_correlated(wi, wo).
    */
    float evalG2Correlated(float3 wi, float3 wo)
    {
        float gi = evalG1(wi);
        float go = evalG1(wo);
        if (wi.z * wo.z > 0.0f)
        {
            // Reflection case, see
            // See "Photorealistic Surface Rendering with Microfacet Theory"
            // by Jonathan Dupuy 2015, Eq. 2.13.
            return gi * go / (gi + go - gi * go);
        }
        else
        {
            // Transmission case, see
            // "Multiple-Scattering Microfacet BSDFs with the Smith Model"
            // by Heitz et al. 2016, Eq. 8.
            float LambdaI = (1.0f - gi) / gi;
            float LambdaO = (1.0f - go) / go;
            return beta(1.0f + LambdaI, 1.0f + LambdaO);
        }
    }

    /** Evaluates the projected area (sigma) of the heighfield along a given direction.

        Note that it is non-symmetric (sigma(dir) != sigma(-dir)).
        It holds that:
            sigma([0, 0, -1]) = 1, i.e. the full surface is visible when going straight down.
            sigma([0, 0, 1])  = 0, i.e. nothing is in the way when escaping the heightfield along the surface
        normal.

        \param[in] dir Evaluated direction in local space.
        \return sigma(dir).
    */
    float evalSigma(float3 dir)
    {
        // Stretch direction into the standard configuration, see
        // "Photorealistic Surface Rendering with Microfacet Theory"
        // by Jonathan Dupuy 2015, page 28.
        float3 dirStd = { alpha.x * dir.x + rho * alpha.y * dir.y, alpha.y * dir.y * sqrtOneMinusRhoSqr, dir.z };
        float nrm = length(dirStd);
        dirStd /= nrm;

        // Eval projected area of standard configuration.
        float sigmaStd = ndf.evalSigmaStd(dirStd);

        // Transform back based on surface roughness.
        return nrm * sigmaStd;
    }

    /** Evaluates the Smith Lambda function.

        Like `evalSigma`, this function is non-symmetric and it holds that
        Lambda(-dir) = 1 + Lambda(dir) for upward travelling directions (i.e. dir.z > 0).

        \param[in] dir Evaluated direction in local space.
        \return Lambda(dir).
    */
    float evalLambda(float3 dir)
    {
        if (dir.z > 0.0f)
        {
            return evalSigma(dir) / abs(dir.z);
        }
        else
        {
            return 1.0f + evalSigma(-dir) / abs(dir.z);
        }
    }
}
