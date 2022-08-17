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

__exported import Rendering.Materials.IBSDF;
__exported import Rendering.Materials.IBxDF;
import Rendering.Volumes.PhaseFunction;
import Utils.Color.ColorHelpers;

/** This class implements a layered material based on
    "Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs" by Guo et al.
    
    The model allows for a layerying of two arbitrary BxDFs, where the positive hemisphere
    of the top BxDF is exposed if wi.z>0, and the negative hemisphere of the bottom
    hemisphere is exposed for wi.z<0. The space inbetween the BxDFs can be filled
    with a medium with arbitrary extinction, albedo and mean scattering cosine.
    The thickness of the medium is fixed to be 1.
    
    Evaluation and sampling are done via a Monte Carlo random walk between the top
    and bottom interface. Hence eval is probabilistic, and the PDF of the sampling
    routine is not known exactly (currently we only return a crude approximation,
    for sampling purposes).
*/
struct LayeredBxDF : IBxDF
{
    IBxDF top;
    IBxDF bottom;
    float sigmaT;
    float g;
    float3 albedo;
    int maxDepth;

    __init(IBxDF top, IBxDF bottom, float sigmaT = 0.0f, float g = 0.0f, float3 albedo = float3(0.0f), int maxDepth = 64)
    {
        this.top = top;
        this.bottom = bottom;
        this.sigmaT = sigmaT;
        this.g = g;
        this.albedo = albedo;
        this.maxDepth = maxDepth;
    }

    float3 evalTopOrBottom<S : ISampleGenerator>(bool evalTop, float3 wi, float3 wo, inout S sg)
    {
        if (evalTop)
        {
            return top.eval(wi, wo, sg);
        }
        else
        {
            return bottom.eval(wi, wo, sg);
        }
    }

    bool sampleTopOrBottom<S : ISampleGenerator>(bool sampleTop, float3 wi, out BSDFSample bs, inout S sg)
    {
        if (sampleTop)
        {
            return top.sample(wi, bs.wo, bs.pdf, bs.weight, bs.lobe, sg);
        }
        else
        {
            return bottom.sample(wi, bs.wo, bs.pdf, bs.weight, bs.lobe, sg);
        }
    }

    float evalPdfTopOrBottom(bool evalTop, float3 wi, float3 wo)
    {
        if (evalTop)
        {
            return top.evalPdf(wi, wo);
        }
        else
        {
            return bottom.evalPdf(wi, wo);
        }
    }

    float3 eval<S : ISampleGenerator>(const float3 wiLocal, const float3 woLocal, inout S sg)
    {
        bool enterTop = wiLocal.z > 0.0f;
        bool exitTop = woLocal.z > 0.0f;

        float3 result = float3(0.0f);
        float z = enterTop ? 1.0f : 0.0f;
        float exitZ = exitTop ? 1.0f : 0.0f;
    
        if (z == exitZ) result += evalTopOrBottom(exitTop, wiLocal, woLocal, sg);
    
        BSDFSample bs;
        if (!sampleTopOrBottom(enterTop, wiLocal, bs, sg)) return result;
        if (bs.isLobe(LobeType::Reflection)) return result;

        float3 w = bs.wo;
        float3 weight = bs.weight;
        HenyeyGreensteinPhaseFunction phase = {g};

        for (int depth = 0; depth < maxDepth; ++depth)
        {
            float sigma = sigmaT / abs(w.z);
            if (any(albedo > 0.0f))
            {
                float dz = -log(1.0f - sampleNext1D(sg)) / sigma;
                float zp = w.z > 0.0f ? (z + dz) : (z - dz);
                if (0.0f < zp && zp < 1.0f)
                {
                    float3 phaseW;
                    float phasePDF, phaseWeight;
                    if (!phase.sample(-w, phaseW, phasePDF, phaseWeight, sg)) return false;
                    weight *= albedo * phaseWeight;
                    w = phaseW;
                    z = zp;

                    continue;
                }
                z = clamp(zp, 0.0f, 1.0f);
            }
            else
            {
                z = (z == 1.0f) ? 0.0f : 1.0f;
                weight *= exp(-sigma);
            }
            bool scatterTop = z == 1.0f;

            if (z == exitZ) result += weight * evalTopOrBottom(exitTop, -w, woLocal, sg);
            if (!sampleTopOrBottom(scatterTop, -w, bs, sg)) return result;
            weight *= bs.weight;
            w = bs.wo;

            if (bs.isLobe(LobeType::Transmission)) return result;
        
            float rr = min(luminance(weight), 1.0f);
            if (depth > 3 && rr < 0.25f)
            {
                if (sampleNext1D(sg) >= rr) return false;
                weight /= rr;
            }
        }
        return result;
    }

    bool sample<S : ISampleGenerator>(const float3 wiLocal, out float3 woLocal, out float pdfOut, out float3 weightOut, out uint lobe, inout S sg)
    {
        bool enterTop = wiLocal.z > 0.0f;
    
        BSDFSample bs;
        if (!sampleTopOrBottom(enterTop, wiLocal, bs, sg)) return false;
        if (bs.isLobe(LobeType::Reflection))
        {
            woLocal = bs.wo;
            pdfOut = bs.pdf;
            weightOut = bs.weight;
            lobe = bs.lobe;
            return true;
        }

        float3 w = bs.wo;
        float3 weight = bs.weight;
        float pdf = bs.pdf;
        float z = enterTop ? 1.0f : 0.0f;
        HenyeyGreensteinPhaseFunction phase = {g};

        for (int depth = 0; depth < maxDepth; ++depth)
        {
            float sigma = sigmaT / abs(w.z);
            if (any(albedo > 0.0f))
            {
                float dz = -log(1.0f - sampleNext1D(sg)) / sigma;
                float zp = w.z > 0.0f ? (z + dz) : (z - dz);
                if (0.0f < zp && zp < 1.0f)
                {
                    float3 phaseW;
                    float phasePDF, phaseWeight;
                    if (!phase.sample(-w, phaseW, phasePDF, phaseWeight, sg)) return false;
                    pdf *= phasePDF;
                    weight *= albedo * phaseWeight;
                    w = phaseW;
                    z = zp;

                    continue;
                }
                z = clamp(zp, 0.0f, 1.0f);
            }
            else
            {
                z = (z == 1.0f) ? 0.0f : 1.0f;
                weight *= exp(-sigma);
            }
            bool scatterTop = z == 1.0f;
            if (!sampleTopOrBottom(scatterTop, -w, bs, sg)) return false;
            weight *= bs.weight;
            w = bs.wo;
            pdf *= bs.pdf;

            if (bs.isLobe(LobeType::Transmission))
            {
                woLocal = w;
                pdfOut = evalPdf(wiLocal, w);
                weightOut = weight;
                lobe =
                    (uint)(w.z*wiLocal.z > 0.0f ? LobeType::Reflection : LobeType::Transmission) &
                    (uint)(bs.isLobe(LobeType::Delta) ? LobeType::Delta : LobeType::Specular);
            
                return true;
            }
        
            float rr = min(luminance(weight), 1.0f);
            if (depth > 3 && rr < 0.25f)
            {
                if (sampleNext1D(sg) >= rr) return false;
                weight /= rr;
                pdf *= rr;
            }
        }
        return false;
    }

    /** TODO: This is a relatively crude approximation to the sampling PDF.
        It only evaluates the reflection lobe from the interface facing the ray.
        A better strategy would be to evaluate also the TT/TRT lobes. However, this
        would require transmitting through the closest interface, which we can't
        do without invoking the sampling routine. We don't have sg available here,
        and a stochastic PDF could cause problems with MIS.
    */
    float evalPdf(const float3 wiLocal, const float3 woLocal)
    {
        bool enterTop = wiLocal.z > 0.0f;
        bool exitTop = woLocal.z > 0.0f;
    
        float pdfCos = abs(wiLocal.z) * M_1_PI;
        float pdfSingle = 0.0f;
        if (enterTop == exitTop) pdfSingle = evalPdfTopOrBottom(exitTop, wiLocal, woLocal);
    
        return lerp(pdfSingle, pdfCos, 0.5f);
    }
};
