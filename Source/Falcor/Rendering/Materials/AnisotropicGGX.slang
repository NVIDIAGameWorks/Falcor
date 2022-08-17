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

import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGeneratorInterface;

/** This class implements sampling and evaluation of an anisotropic GGX.
    The class uses the visible distribution of normals for sampling.
*/
struct AnisotropicGGX
{
    float2 alpha;
    
    __init(float2 alpha)
    {
        this.alpha = alpha;
    }
    
    /** A 'singular' distribution is one which would be more ideally treated as an
        ideal mirror. It can't be sampled reliably.
    */
    bool isSingular()
    {
        return all(alpha < 1e-3f);
    }
    
    float getIsotropicRoughness()
    {
        return max(alpha.x, alpha.y);
    }

    /** Computes the Smith lambda function. This function is symmetric with respect to direction,
        i.e. smithLambda(v) = smithLambda(-v). The result returned is equivalent to assuming v.z > 0.
    */
    float smithLambda(float3 v)
    {
        float cosThetaSq = v.z * v.z;
        float sinThetaSq = max(1.0f - cosThetaSq, 0.0f);
        float tanThetaSq = sinThetaSq / cosThetaSq;
        if (isinf(tanThetaSq)) return 0.0f;
        
        float2 cosSinPhiSq = sinThetaSq == 0.0f ? float2(1.0f, 0.0f) : float2(v.x * v.x, v.y * v.y) / sinThetaSq;
        float alphaSq = dot(cosSinPhiSq, alpha * alpha);
        return (sqrt(1.0f + alphaSq * tanThetaSq) - 1.0f) * 0.5f;
    }
    
    /** Compute the monostatic Smith shadowing function. This function returns the result
        assuming w.z > 0. This is what you want for single scattering, but you need to be careful
        for multiple scattering on microfacets.
    */
    float G1(float3 w)
    {
        return 1.0f / (1.0f + smithLambda(w));
    }
    
    /** Compute the bistatic Smith shadowing function for reflection.
        This will be off for refraction, where we would have to use the beta
        function instead. This is not worth the trouble unless we do
        multiple scattering on microfacet dielectrics as well.
    */
    float G2(float3 wo, float3 wi)
    {
        return 1.0f / (1.0f + smithLambda(wo) + smithLambda(wi));
    }

    /** Computes the normal distribution function, i.e. D(h)
    */
    float evalNDF(float3 h)
    {
        float cosThetaSq = h.z * h.z;
        float sinThetaSq = max(1.0f - cosThetaSq, 0.0f);
        float tanThetaSq = sinThetaSq / cosThetaSq;
        if (isinf(tanThetaSq)) return 0.0f;
        
        float cosThetaQu = cosThetaSq * cosThetaSq;
        if (cosThetaQu < 1e-16f) return 0.0f;
        
        float2 cosSinPhiSq = sinThetaSq == 0.0f ? float2(1.0f, 0.0f) : float2(h.x * h.x, h.y * h.y) / sinThetaSq;
        float2 alphaSq = cosSinPhiSq / (alpha * alpha);
        float e = tanThetaSq * (alphaSq.x + alphaSq.y);
        return 1.0f / (M_PI * alpha.x * alpha.y * cosThetaQu * sqr(1.0f + e));
    }
    
    /** Computes the distribution of visible normals, i.e. D_wi(h)
    */
    float evalVNDF(float3 w, float3 h)
    {
        return w.z == 0.0f ? 0.0f : (G1(w) * abs(dot(w, h)) * evalNDF(h)) / abs(w.z);
    }

    /** Samples the distribution of normals visible from direction w.
    */
    float3 sample<S : ISampleGenerator>(float3 w, inout S sg)
    {
        float3 wh = normalize(float3(alpha.x * w.x, alpha.y * w.y, w.z));
        if (wh.z < 0.0f) wh = -wh;
        
        float3 T1 = dot(wh.xy, wh.xy) > 0.0f ? normalize(float3(-wh.y, wh.x, 0.0f)) : float3(1.0f, 0.0f, 0.0f);
        float3 T2 = cross(wh, T1);
        
        float r = sqrt(sampleNext1D(sg));
        float phi = 2.0f * M_PI * sampleNext1D(sg);
        float t1 = r * cos(phi);
        float t2 = r * sin(phi);
        float s = 0.5 * (1.0 + wh.z);
        t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
        
        float3 nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * wh;
        return normalize(float3(alpha.x * nh.x, alpha.y * nh.y, max(0.0f, nh.z)));
    }

    /** Returns the PDF of sampling normal h from VNDF(w)
    */
    float evalPDF(float3 w, float3 h)
    {
        return evalVNDF(w, h);
    }
};