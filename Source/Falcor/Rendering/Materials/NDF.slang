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

 import Utils.Math.MathHelpers;
 __exported import Rendering.Materials.INDF;
 
 /** Implementation of the Trowbridge-Reitz normal distribution function.
 
     Originally from "Average irregularity representation of a rough surface for ray reflection"
     by Trowbridge and Reitz 1975.
 
     Also known as GGX distribution from "Microfacet Models for Refraction through Rough Surfaces"
     by Walter et al. 2007.
 */
 struct TrowbridgeReitzNDF : INDF
 {
     __init() {}
 
     float evalP22Std(float2 slope)
     {
         // See "Photorealistic Surface Rendering with Microfacet Theory"
         // by Jonathan Dupuy 2015, Table 2.II.
         float tmp = (1.0f + slope.x * slope.x + slope.y * slope.y);
         return M_1_PI / (tmp * tmp);
     }
 
     float evalSigmaStd(float3 dir)
     {
         // See "Photorealistic Surface Rendering with Microfacet Theory"
         // by Jonathan Dupuy 2015, Table 2.III (rearranged).
         return 0.5f * (1.0f - dir.z);
     }   
 
     float2 sampleP22Std<S : ISampleGenerator>(inout S sg)
     {
         // See "Importance Sampling Microfacet-Based BSDFs with the Distribution of Visible Normals"
         // (Supplemental Material 2/2) by Heitz and d'Eon 2014.
         float2 u = sampleNext2D(sg);
         float phi = M_2PI * u.y;
         return float2(cos(phi), sin(phi)) * sqrt(u.x) / sqrt(1.0f - u.x);
     }
     
     float2 sampleVisibleP22Std<S : ISampleGenerator>(float3 wi, inout S sg)
     {
         // From "Sampling the GGX Distribution of Visible Normals"
         // by Eric Heitz 2018.
         
         // Orthonormal basis (with special case if cross product is zero).
         float lensqr = wi.x * wi.x + wi.y * wi.y;
         float3 T1 = lensqr > 0.0f ? float3(-wi.y, wi.x, 0.0f) / sqrt(lensqr) : float3(1.0f, 0.0f, 0.0f);
         float3 T2 = cross(wi, T1);
 
         // Parameterization of the projected area.
         float2 p = sample_disk_concentric(sampleNext2D(sg));
         float s = 0.5f * (1.0f + wi.z);
         p.y = (1.0f - s) * sqrt(1.0f - p.x * p.x) + s * p.y;
 
         // Reprojection to the hemisphere.
         float3 mStd = p.x * T1 + p.y * T2 + sqrt(max(0.0f, 1.0f - p.x * p.x - p.y * p.y)) * wi;
 
         // Transform to a slope.
         return float2(-mStd.x / mStd.z, -mStd.y / mStd.z);
     }
 }
 
 /** Implementation of the Beckmann-Spizzichino normal distribution function.
 
     Supports elliptical anisotropy and visible normals sampling.
 
     Originally from "The scattering of electromagnetic waves from rough surfaces"
     by Beckmann and Spizzichino, 1963.
 */
 struct BeckmannSpizzichinoNDF : INDF
 {
     __init() {}
     
     float evalP22Std(float2 slope)
     {
         // See "Photorealistic Surface Rendering with Microfacet Theory"
         // by Jonathan Dupuy 2015, Table 2.II.
         return M_1_PI * exp(-(slope.x * slope.x + slope.y * slope.y));
     }
 
     float evalSigmaStd(float3 dir)
     {
         // See "Photorealistic Surface Rendering with Microfacet Theory"
         // by Jonathan Dupuy 2015, Table 2.III (rearranged).
         if (dir.z >= 1.f)
         {
             return 0.f;
         }
         else if (dir.z <= -1.f)
         {
             return 1.f;
         }
         
         float cosTheta = -dir.z;
         
         if (cosTheta < -0.9f)
         {
             // For rays that are very steep in the up-welling direction, the analytical expression
             // becomes extremely inaccurate---and this can lead to unnormalized (or even negative!)
             // PDFs for the visible slopes.
 
             // The following expression (derived by Eugene d'Eon) is based on a series expansion of
             // the erf() expression and stabilizes the function evaluation.
             float u = cosTheta;
             float u2 = u * u;
             float u4 = u2 * u2;
             float u6 = u2 * u4;
             float u8 = u2 * u6;
             float u10 = u2 * u8;
             return (0.004407731121466846f * exp(u2 / (-1.0f + u2)) * pow(1.0f - u2, 1.5f) * (945.0f - 4200.0f * u2 + 7050.0f * u4 - 5328.0f * u6 + 1565.0f * u8)) / u10;
         }
 
         float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
         float nu = cosTheta / sinTheta;
         float tmp = exp(-nu * nu) / M_SQRTPI;
         return 0.5f * (cosTheta * (1.0f + erf(nu)) + sinTheta * tmp);
     }   
 
     float2 sampleP22Std<S : ISampleGenerator>(inout S sg)
     {
         // See "Importance Sampling Microfacet-Based BSDFs with the Distribution of Visible Normals"
         // (Supplemental Material 2/2) by Heitz and d'Eon 2014.
         float2 u = sampleNext2D(sg);
         float phi = M_2PI * u.y;
         return float2(cos(phi), sin(phi)) * sqrt(-log(1.0f - u.x));
     }
     
     float2 sampleVisibleP22Std<S : ISampleGenerator>(float3 wi, inout S sg)
     {
         float thetaI = 0.0f;
         float phiI = 0.0f;
         if (wi.z < 0.99999f)
         {
             // Avoid a singularity of atan2 at theta == 0.
             thetaI = acos(wi.z);
             phiI = atan2(wi.y, wi.x);
         }
 
         float2 u = sampleNext2D(sg);
         
         if (thetaI < 1e-4f)
         {
             // Trivial sampling for normal incidence.
             float tmp = sqrt(-log(1.0f - u.x));
             float phi = M_2PI * u.y;
             return tmp * float2(cos(phi), sin(phi));
         }
 
         float slopeX;
 #if 0
         // The following implements a numerical inversion scheme for the CDF of visible slopes from
         // "An Improved Visible Normal Sampling Routine for the Beckmann Distribution"
         // by Wenzel Jakob 2014.
         //
         // While this is very robust for the common case of down-welling rays, it turns out to suffer
         // from numerical instability for up-welling rays that are common, e.g., when evaluating
         // multiple-scattering from microfacets via a random walk through a volumetric heightfield.
         // Its accuracy can be pushed a bit further by allowing more Newton solver iterations and
         // setting a stricter termination criterion (at the cost of longer computation times) but is
         // ultimately limited by the numerical precision of the `erf` and `erfinv` functions.
         
         float tanThetaI = tan(thetaI);
         float cotThetaI = 1 / tanThetaI;
 
         // Search interval -- everything is parameterized in the erf() domain.
         float a = -1;
         float c = erf(cotThetaI);
 
         // Start with a good initial guess (inverse of an approximation computed in Mathematica).
         float fit = 1.0f + thetaI * (-0.876f + thetaI * (0.4265f - 0.0594f * thetaI));
         float b = c - (1.0f + c) * pow(1.0f - u.x, fit);
 
         // Normalization factor for the CDF.
         float normalization = 1.0f / (1.0f + c + 1.0f / M_SQRTPI * tanThetaI * exp(-cotThetaI * cotThetaI));
 
         int it = 0;
         while (++it < 10)
         {
             // Bisection criterion.
             if (b <= a || b >= c || isnan(b))
             {
                 b = 0.5f * (a + c);
             }
 
             // Evaluate the CDF and its derivative (i.e. the density function).
             float invErf = erfinv(b);
             float value = normalization * (1.0f + b + 1.0f / M_SQRTPI * tanThetaI * exp(-invErf * invErf)) - u.x;
             float derivative = normalization * (1.0f - invErf * tanThetaI);
 
             // Are we close enough to the solution?
             if (abs(value) < 1e-6f)
             {
                 break;
             }
 
             // Update bisection intervals.
             if (value > 0)
             {
                 c = b;
             }
             else
             {
                 a = b;
             }
 
             // Perform the Newton step.
             b -= value / derivative;
         }
 
         slopeX = erfinv(b);
 #else
         // This alternative scheme is an *analytical* sampling routine for the desired PDF that was
         // derived based on the "triangle-cut parameterization" that is an alternative to the commonly
         // used CDF inversion technique. Details about the theoretical background are described in:
         // "Can't Invert the CDF? The Triangle-Cut Parameterization of the Region under the Curve"
         // by Eric Heitz 2020.
         //
         // It works by first, sampling from an approximate proxy PDF which has an exact sampling scheme
         // available, followed by a correction scheme that generates samples according the the desired
         // target PDF.
         // Our proxy PDF is a Gaussian that is truncated to the domain (-inf, cotThetaI] with its mean
         // aligned with the maximum of the target PDF and with a standard deviation that (approximately)
         // matches the tail of the target PDF.
 
         float cosThetaI = wi.z;
         float sinThetaI = sqrt(1.0f - cosThetaI * cosThetaI);
         float tanThetaI = sinThetaI / cosThetaI;
         float cotThetaI = 1.0f / tanThetaI;
 
         if (cosThetaI > -0.9f)
         {
             // The mean of the proxy distribution (i.e. the maximum of the target PDF) has a closed-form expression.
             float m = (cosThetaI - sqrt(cosThetaI * cosThetaI + 2.f * sinThetaI * sinThetaI)) / (2.f * sinThetaI);
             // The standard deviation was matched empirically and is well approximated using a degree-4 polynomial.
             float s = -0.0141259425f;
             s =  0.0435127509f + s * thetaI;
             s = -0.0464141286f + s * thetaI;
             s = -0.0153781569f + s * thetaI;
             s =  0.6710745099f + s * thetaI;
             // The proxy distribution is clamped to zero where the target PDF evaluates to zero.
             float xMax = cotThetaI;
 
             // Follow the steps of the triangle-cut inversion scheme.
 
             // Sample from the proxy PDF, here done via the analytic inverse of its CDF.
             float erfTmp = erf((xMax - m) / (M_SQRT2 * s));
             float xa = m - M_SQRT2 * s * erfinv(1.f - u.x * (1.f + erfTmp));
 
             // Evaluate the target PDF f(ax), CDF F(ax), and PDF derivative f'(ax) for the sampled point.
             float sigma = 0.5f * (cosThetaI * (1.f + erf(cotThetaI)) + sinThetaI / M_SQRTPI * exp(-cotThetaI * cotThetaI));
             float invSigma = 1.f / sigma;
             float expXa = exp(-xa * xa) / M_SQRTPI;
 
             float fxa = (-xa * sinThetaI + cosThetaI) * expXa;
             fxa *= invSigma;
 
             float Fxa = 0.5f * (cosThetaI + cosThetaI * erf(xa) + expXa * sinThetaI);
             Fxa *= invSigma;
 
             float dfxa = expXa * (-2.f * xa * cosThetaI + (-1.f + 2.f * xa * xa) * sinThetaI);
             dfxa *= invSigma;
 
             // Compute the other coordinates of the cut triangle.
             float xb = xa + 2.f * (u.x - Fxa) / fxa;
             float ya = fxa;
             float yb = 0.f;
 
             // Compute the infinitesimal thicknesses, via an evaluation of the proxy PDF.
             float wa = fxa * fxa + 2.f * (u.x - Fxa) * dfxa;
             float pProxy = exp(-(m - xa) * (m - xa) / (2.f * s * s)) * sqrt(M_2_PI) / ( s * (1.f + erfTmp));
             float wb = 2.f * fxa * pProxy - wa;
 
             // Sample thickness density.
             float t = u.y * (wa + wb) / (wb + sqrt((1.f - u.y) * wb * wb + u.y * wa * wa));
 
             // Interpolate (xa, ya) and (xb, yb).
             float x = xa * t + xb * (1.f - t);
             float y = ya * t + yb * (1.f - t);
 
             // The x-coordinate is now distribution according to the target PDF.
             slopeX = x;
 
             // The y-coordinate can be re-mapped to a (third) uniform variate by
             // dividing by the target PDF evaluated at x and can subsequently be
             // reused to sample the y-coordinate of the slope.
             u.y = y / (invSigma * (-x * sinThetaI + cosThetaI) / M_SQRTPI * exp(-x * x));
         }
         else
         {   
             // All hope for numerical stability is lost in the case of very steep up-welling rays.
             // Instead, fall back on an expression derived by Eugene d'Eon that becomes exact in
             // the limit of mu -> -1.
             // This tends to be an extremely good approximation in these cases, but note that this
             // is likely never going to pass any statistical (e.g. Chi^2) test comparing it to the
             // PDF.
             float t;
             float split = pow(-cosThetaI, 1.3f);
             if (u.x < split)
             {
                 u.x = u.x / split;
                 // Note that an additional (third) random number needs to be created for this
                 // scheme. This technically breaks sample stratification---but based on the
                 // up-welling ray direction we are likely already in a case where many random
                 // numbers are being used to evaluate a stochastic BSDF.
                 t = u.x * sampleNext1D(sg);
             }
             else
             {
                 u.x = (u.x - split) / (1.0f - split);
                 t = 1.0f - erf(sqrt(-log(u.x)));
             }
 
 #if 0
             slopeX = erfinv(-1.f + t + t * erf(cotThetaI));
 #else
             // Because the `erf` and `erfinv` functions also become numerically unstable in these
             // cases, fall back to a series expansion of these terms.
             float cotThetaI2 = cotThetaI * cotThetaI;
             float cotThetaI3 = cotThetaI * cotThetaI2;
 
             float tmp = 2.0f * cotThetaI2 + log(8.0f) - 2.0f * log((t - 2.0f * cotThetaI2 * t) / cotThetaI3);
             slopeX = -sqrt(tmp - log(tmp)) * M_1_SQRT2;
 #endif
         }
 #endif
         float slopeY = erfinv(2.0f * u.y - 1.0f);
 
         // Rotate.
         float sinPhiI = sin(phiI);
         float cosPhiI = cos(phiI);
         return float2(cosPhiI * slopeX - sinPhiI * slopeY,
                       sinPhiI * slopeX + cosPhiI * slopeY);
     }
 }
 