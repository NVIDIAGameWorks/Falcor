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

__exported import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.BSDFs.DiffuseSpecularBRDF;
import Rendering.Materials.MERLCommon;
import Utils.Math.MathHelpers;
import Scene.Scene;

// Enable to use fitted BRDF approximation for sampling.
#define SAMPLE_FITTED_BRDF 1

// Enable to directly visualize the fitted BRDF instead of the MERL BRDF.
// This is for validation purposes only and should be disabled by default.
#define VISUALIZE_FITTED_BRDF 0

/** Implementation of the BSDF for the measured MERL material.
*/
struct MERLMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;    ///< Shading frame in world space.
    uint bufferID;      ///< Buffer ID in material system where BRDF data is stored.
    float3 albedo;      ///< Approximate albedo.
    DiffuseSpecularBRDF fittedBrdf;

    __init(const ShadingFrame sf, const uint bufferID, const float3 albedo, const DiffuseSpecularData extraData)
    {
        this.sf = sf;
        this.bufferID = bufferID;
        this.albedo = albedo;

        // Setup BRDF approximation.
        this.fittedBrdf = DiffuseSpecularBRDF(extraData);
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

#if VISUALIZE_FITTED_BRDF
        return fittedBrdf.eval(wiLocal, woLocal, sg);
#else
        return evalLocal(wiLocal, woLocal);
#endif
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);

#if SAMPLE_FITTED_BRDF
        float3 woLocal = {};
        if (!fittedBrdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg)) return false;
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

#if !VISUALIZE_FITTED_BRDF
        result.weight = evalLocal(wiLocal, woLocal) / result.pdf;
#endif

#else // !SAMPLE_FITTED_BRDF
        // Draw cosine-weighted sample over the hemisphere.
        // TODO: Implement better importance sampling.
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

#if VISUALIZE_FITTED_BRDF
        result.weight = fittedBrdf.eval(wiLocal, woLocal, sg) / result.pdf;
#else
        result.weight = evalLocal(wiLocal, woLocal) / result.pdf;
#endif
        result.lobeType = (uint)LobeType::DiffuseReflection;
#endif

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;

#if SAMPLE_FITTED_BRDF
        return fittedBrdf.evalPdf(wiLocal, woLocal);
#else
        return woLocal.z * M_1_PI; // pdf = cos(theta) / pi
#endif
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;

        // Compute approximation of the perceived roughness. For now assume a medium roughness.
        p.roughness = fittedBrdf.roughness;

        // Return approximation of the albedos.
        // The BSDF reports only a diffuse reflection lobe, so leaving the other albedos as zero.
        // TODO: Return more accurate albedos.
        p.diffuseReflectionAlbedo = albedo;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return (uint)LobeType::DiffuseReflection | (uint)LobeType::SpecularReflection;
    }


    // Internal helpers

    /** Evaluate the MERL BRDF in the local frame.
        The vectors are assumed to be in the upper hemisphere.
        \param[in] wi Incident direction in the local frame.
        \param[in] wo Outgoing direction in the local frame.
        \return f(wi, wo) * wo.z
    */
    float3 evalLocal(const float3 wi, const float3 wo)
    {
        ByteAddressBuffer brdfData = gScene.materials.getBuffer(bufferID);
        return MERLCommon::eval(wi, wo, brdfData);
    }

};
