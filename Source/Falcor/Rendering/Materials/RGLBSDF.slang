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
import Scene.Material.RGLMaterialData;
import Rendering.Materials.RGLCommon;
import Utils.Math.MathHelpers;
import Scene.Scene;

/** Implementation of the BSDF for the measured RGL material.
*/
struct RGLBSDF : IBSDF
{
    float2 slice;
    float sigma;
    uint2 ndfVndfSize;          ///< Packed size of NDF and last two dimensions of VNDF LUT (first two match lumiSize).
    uint2 lumiSize;             ///< Packed size of luminance LUT. Dimensions identical to RGB LUT.
    uint ndfBufID;
    uint vndfBufID;
    uint lumiBufID;
    uint rgbBufID;
    uint vndfMarginalBufID;
    uint lumiMarginalBufID;
    uint vndfConditionalBufID;
    uint lumiConditionalBufID;
    float3 albedo;              ///< Approximate albedo, only to satisfy getProperties.

    __init(float2 slice, float sigma, float3 albedo, RGLMaterialData data)
    {
        this.slice = slice;
        this.sigma = sigma;
        this.albedo = albedo;
        ndfVndfSize = packSize(uint4(data.ndfSize, data.vndfSize));
        lumiSize = packSize(uint4(data.phiSize, data.thetaSize, data.lumiSize));
        ndfBufID = data.ndfBufID;
        vndfBufID = data.vndfBufID;
        lumiBufID = data.lumiBufID;
        rgbBufID = data.rgbBufID;
        vndfMarginalBufID = data.vndfMarginalBufID;
        lumiMarginalBufID = data.lumiMarginalBufID;
        vndfConditionalBufID = data.vndfConditionalBufID;
        lumiConditionalBufID = data.lumiConditionalBufID;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta)
        {
            return float3(0.0f);
        }

        uint4 ndfsSize = unpackSize(ndfVndfSize);
        uint4 lumiSize = unpackSize(this.lumiSize);
        uint4 vndfSize = uint4(lumiSize.xy, ndfsSize.zw);
        let ndf = Brick2D.fromManagedBuffers(ndfsSize.xy, ndfBufID);
        let rgb = Brick4D.fromManagedBuffers(lumiSize   , rgbBufID);
        let vndf = InterpolatedDistribution2D.fromManagedBuffers(vndfSize, vndfMarginalBufID, vndfConditionalBufID, vndfBufID);
        let lumi = InterpolatedDistribution2D.fromManagedBuffers(lumiSize, lumiMarginalBufID, lumiConditionalBufID, lumiBufID);

        float3 h = normalize(wiLocal + woLocal);
        float2 sphericalI = toSpherical(wiLocal);
        float2 sphericalH = toSpherical(h);
        if (lumiSize.x == 1) sphericalH.y -= sphericalI.y; // Isotropic case
        float2 unitH = sphericalToUnit(sphericalH);

        unitH.y = frac(unitH.y);

        float2 unwarped = vndf.invert(slice, unitH).xy;
        float3 fr = max(rgb.eval(slice, unwarped), float3(0.0f));
        float3 f = sigma == 0.0f ? 0.0f : fr * ndf.eval(unitH) / (4.0f * sigma);

        return f;
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        if (wiLocal.z < kMinCosTheta)
        {
            return false;
        }

        uint4 ndfsSize = unpackSize(ndfVndfSize);
        uint4 lumiSize = unpackSize(this.lumiSize);
        uint4 vndfSize = uint4(lumiSize.xy, ndfsSize.zw);
        let ndf = Brick2D.fromManagedBuffers(ndfsSize.xy, ndfBufID);
        let rgb = Brick4D.fromManagedBuffers(lumiSize   , rgbBufID);
        let vndf = InterpolatedDistribution2D.fromManagedBuffers(vndfSize, vndfMarginalBufID, vndfConditionalBufID, vndfBufID);
        let lumi = InterpolatedDistribution2D.fromManagedBuffers(lumiSize, lumiMarginalBufID, lumiConditionalBufID, lumiBufID);

        float2 sphericalI = toSpherical(wiLocal);
        float2 xi = sampleNext2D(sg);

        float3 lumiSamplePdf = lumi.sample(slice, xi);
        float3 nvdfSamplePdf = vndf.sample(slice, lumiSamplePdf.xy);
        float2 unitH = nvdfSamplePdf.xy;

        float2 sphericalH = unitToSpherical(nvdfSamplePdf.xy);
        if (lumiSize.x == 1) // Isotropic case
        {
            sphericalH.y += sphericalI.y;
        }

        float3 h = fromSpherical(sphericalH);
        float3 woLocal = h * 2.0f * dot(h, wiLocal) - wiLocal;
        if (woLocal.z < kMinCosTheta)
        {
            return false;
        }

        float3 fr = max(rgb.eval(slice, lumiSamplePdf.xy), float3(0.0f));
        float jacobian = max(2.0f * M_PI * M_PI * unitH.x * length(h.xy), 1e-6f) * 4.0f * dot(wiLocal, h);
        float3 f = sigma == 0.0f ? 0.0f : fr * ndf.eval(unitH) / (4.0f * sigma);
        float pdf = nvdfSamplePdf.z * lumiSamplePdf.z / jacobian;

        if (jacobian == 0.0f || pdf == 0.0f) return false;

        result.wo = sd.fromLocal(woLocal);
        result.pdf = pdf;
        result.weight = f / pdf;
        result.lobe = (uint)LobeType::DiffuseReflection;

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = sd.toLocal(wo);

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta)
        {
            return 0.0f;
        }

        uint4 ndfsSize = unpackSize(ndfVndfSize);
        uint4 lumiSize = unpackSize(this.lumiSize);
        uint4 vndfSize = uint4(lumiSize.xy, ndfsSize.zw);
        let vndf = InterpolatedDistribution2D.fromManagedBuffers(vndfSize, vndfMarginalBufID, vndfConditionalBufID, vndfBufID);
        let lumi = InterpolatedDistribution2D.fromManagedBuffers(lumiSize, lumiMarginalBufID, lumiConditionalBufID, lumiBufID);

        float3 h = normalize(wiLocal + woLocal);
        float2 sphericalI = toSpherical(wiLocal);
        float2 sphericalH = toSpherical(h);
        if (lumiSize.x == 1) // Isotropic case
        {
            sphericalH.y -= sphericalI.y;
        }
        float2 unitI = sphericalToUnit(sphericalI);
        float2 unitH = sphericalToUnit(sphericalH);

        unitH.y = unitH.y - floor(unitH.y);

        float3 unwarpedPdf = vndf.invert(slice, unitH);
        float2 unwarped = unwarpedPdf.xy;

        float vndfPdf = unwarpedPdf.z;
        float lumiPdf = lumi.evalPdf(slice, unwarped);
        float jacobian = max(2.0f * M_PI * M_PI * unitH.x * length(h.xy), 1e-6f) * 4.0f * dot(wiLocal, h);
        float pdf = vndfPdf * lumiPdf / jacobian;

        if (jacobian == 0.0f) return 0.0f;

        return pdf;
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
        // Report a single diffuse reflection lobe. We don't really know what the
        // actual lobes in the measured material are like.
        return (uint)LobeType::DiffuseReflection;
    }
};
