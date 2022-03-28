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
import Utils.Math.MathHelpers;
import Scene.Scene;

/** Implementation of the BSDF for the measured MERL material.
*/
struct MERLBSDF : IBSDF
{
    uint bufferID;      ///< Buffer ID in material system where BRDF data is stored.
    float3 albedo;      ///< Approximate albedo.

    static const uint kBRDFSamplingResThetaH = 90;
    static const uint kBRDFSamplingResThetaD = 90;
    static const uint kBRDFSamplingResPhiD = 360;

    __init(uint bufferID, float3 albedo)
    {
        this.bufferID = bufferID;
        this.albedo = albedo;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return float3(0.f);

        return evalLocal(wiLocal, woLocal);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        // Draw cosine-weighted sample over the hemisphere.
        // TODO: Implement better importance sampling.
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta || result.pdf == 0.f) return false;

        result.wo = sd.fromLocal(woLocal);
        result.weight = evalLocal(wiLocal, woLocal) / result.pdf;
        result.lobe = (uint)LobeType::DiffuseReflection;

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;
        return woLocal.z * M_1_PI; // pdf = cos(theta) / pi
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        // Compute approximation of the perceived roughness. For now assume a medium roughness.
        p.roughness = 0.5f;

        // Return approximation of the albedos.
        // The BSDF reports only a diffuse reflection lobe, so leaving the other albedos as zero.
        p.diffuseReflectionAlbedo = albedo;

        return p;
    }

    uint getLobes(const ShadingData sd)
    {
        // Report a single diffuse reflection lobe since we're using cosine-weighted sampling,
        // but in practice the measured data may represent an arbitrary BRDF (non-delta).
        return (uint)LobeType::DiffuseReflection;
    }


    // Internal helpers

    /** Evaluate the BRDF in the local frame.
        The vectors are assumed to be in the upper hemisphere.
        \param[in] wi Incident direction in the local frame.
        \param[in] wo Outgoing direction in the local frame.
        \return f(wi, wo) * wo.z
    */
    float3 evalLocal(const float3 wi, const float3 wo)
    {
        float3 v = computeHalfDiffCoords(wi, wo); // v = (thetaH, thetaD, phiD)
        uint idx = (getThetaDIndex(v.y) + getThetaHIndex(v.x) * kBRDFSamplingResThetaD) * (kBRDFSamplingResPhiD / 2) + getPhiDIndex(v.z);

        // Load BRDF data by bindless buffer ID and index computed above.
        ByteAddressBuffer brdfData = gScene.materials.getBuffer(bufferID);
        float3 f = asfloat(brdfData.Load3(idx * 12));

        return f * wo.z;
    }

    /** Returns half vector/difference vector coordinates.
        \param[in] wi Incident direction in the local frame.
        \param[in] wo Outgoing direction in the local frame.
        \return (thetaH, thetaH, phiD).
    */
    float3 computeHalfDiffCoords(const float3 wi, const float3 wo)
    {
        float3 h = normalize(wi + wo);

        float thetaH = acos(h.z);
        float phiH = atan2(h.y, h.x);

        // Compute diff vector.
        float3 temp = rotateVector(wi, float3(0.f, 0.f, 1.f), -phiH);
        float3 diff = rotateVector(temp, float3(0.f, 1.f, 0.f), -thetaH);

        float thetaD = acos(diff.z);
        float phiD = atan2(diff.y, diff.x);

        return float3(thetaH, thetaD, phiD);
    }

    /** Rotate vector along axis.
    */
    float3 rotateVector(const float3 v, const float3 axis, const float angle)
    {
        float c = cos(angle);
        float s = sin(angle);
        float tmp = dot(v, axis) * (1.f - c);
        float3 w = cross(axis, v);

        float3 result = v * c;
        result += axis * tmp;
        result += w * s;

        return result;
    }

    /** Map thetaH to index. This is a non-linear mapping.
    */
    uint getThetaHIndex(float thetaH)
    {
        if (thetaH <= 0.f) return 0;

        // Map thetaH in [0, pi/2] to idx in [0, 89].
        int idx = (int)(sqrt(thetaH * (float)M_2_PI) * kBRDFSamplingResThetaH);
        return (uint)min(idx, kBRDFSamplingResThetaH - 1);
    }

    /** Map thetaD to index. This is a linear mapping.
    */
    uint getThetaDIndex(float thetaD)
    {
        // Map thetaD in [0, pi/2] to idx in [0, 89].
        int idx = (int)(thetaD * (float)M_2_PI * kBRDFSamplingResThetaD);
        return (uint)clamp(idx, 0, kBRDFSamplingResThetaD - 1);
    }

    /** Map phiD to index. This is a linear mapping.
    */
    uint getPhiDIndex(float phiD)
    {
        // Because of reciprocity, the BRDF is unchanged under phiD -> phiD + M_PI.
        if (phiD < 0.f) phiD += (float)M_PI;

        // Map phiD in [0, pi] to idx in [0, 179].
        int idx = (int)(phiD * (float)M_1_PI * (kBRDFSamplingResPhiD / 2));
        return (uint)clamp(idx, 0, kBRDFSamplingResPhiD / 2 - 1);
    }
};
