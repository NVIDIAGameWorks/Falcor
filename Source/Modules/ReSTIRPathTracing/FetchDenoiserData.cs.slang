/***************************************************************************
 # Copyright (c) 2015-26, NVIDIA CORPORATION. All rights reserved.
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
import Shift;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import LoadShadingDataDynamic;
import HybridShiftTraceRays;
import ResamplingCommon;
import Rendering.Materials.IsotropicGGX;
import RenderPasses.Shared.Denoising.NRDConstants;

#define is_valid(name) (is_valid_##name != 0)
struct FetchDenoiserData
{
    ReSTIRPathTracingParams params; 
    ReSTIRPT restirpt;             
    Texture2D<PackedHitInfo> vbuffer; 
    StructuredBuffer<PathReservoir> reservoirs;
    RWStructuredBuffer<GuideData> sampleGuideData; 
    Texture2D<float4> outputColor;

    RWTexture2D<float4> outputEmission;
    RWTexture2D<float4> outputDiffuseRadianceHitDist;
    RWTexture2D<float4> outputSpecularRadianceHitDist;

    RWTexture2D<float4> outputDeltaReflectionRadianceHitDist;
    RWTexture2D<float4> outputDeltaTransmissionRadianceHitDist;
    RWTexture2D<float4> outputResidualRadianceHitDist; 

    void writeZeroValuesToBuffers(int2 pixel)
    {
        outputDeltaReflectionRadianceHitDist[pixel] = 0.f;
        outputDeltaTransmissionRadianceHitDist[pixel] = 0.f;
        outputResidualRadianceHitDist[pixel] = 0.f;
        outputDiffuseRadianceHitDist[pixel] = 0.f;
        outputSpecularRadianceHitDist[pixel] = 0.f;
    }

    void DoPass(const uint2 pixel)
    {
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = reservoirs[centralOffset];

        Ray centralPrimaryRay;
        HitInfo centralPrimaryHit;
        ShadingData centralPrimarySd;
        if (restirpt.kUseAreaReservoirs)
        {
            centralPrimaryRay = gScene.camera.computeRayThinlensSubPixelRandom(pixel, params.frameDim, centralReservoir.filmSample, centralReservoir.lensSample);
            centralPrimaryHit = retracePrimaryRay(pixel, params.frameDim, centralReservoir.filmSample, centralReservoir.lensSample, false);
            if (centralPrimaryHit.isValid()) centralPrimarySd = loadShadingData(centralPrimaryHit, centralPrimaryRay.origin, centralPrimaryRay.dir, false);
        }
        else
        {
            centralPrimaryRay = computeRayPredefinedThinLens(pixel, params.frameDim, params.frameCount);
            centralPrimarySd = getPixelShadingData(params, vbuffer, pixel, centralPrimaryHit);
        }
        if (!centralPrimaryHit.isValid())
        {
            if (FETCH_NRD_DATA_PASS)
            {
                writeZeroValuesToBuffers(pixel);
                outputEmission[pixel] = restirpt.kUseAreaReservoirs ? float4(centralReservoir.weight * centralReservoir.integrand,0.f) : outputColor[pixel];
            }
            return;
        }
        // Create material instance and query its properties.
        let lod = ExplicitLodTextureSampler(0.f);
        let hints = getMaterialInstanceHints(centralPrimaryHit, true);
        let mi = gScene.materials.getMaterialInstance(centralPrimarySd, lod, hints);
        let bsdfProperties = mi.getProperties(centralPrimarySd);

        // guide data should agree with the guide primary hit sent to the denoiser, if area reservoirs are used 
        if (restirpt.kUseAreaReservoirs && OUTPUT_GUIDE_DATA)
        {
            // Use sum of reflection/transmission albedo as they are denoised together.
            sampleGuideData[centralOffset].setAlbedo(bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
            const float NdotV = saturate(dot(bsdfProperties.guideNormal, centralPrimarySd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
            if (centralPrimarySd.mtl.getMaterialType() == MaterialType::Hair)
            {
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(float3(0.04f), ggxAlpha, NdotV);
                sampleGuideData[centralOffset].setSpecularAlbedo(preintegratedSpecular);
            }
            else
            {
                // Use pre-integrated FG term for specular
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(bsdfProperties.specularReflectance, ggxAlpha, NdotV);
                sampleGuideData[centralOffset].setSpecularAlbedo(preintegratedSpecular);
            }
        }

        if (centralReservoir.weight == 0.f)
        {
            if (FETCH_NRD_DATA_PASS)
            {
                writeZeroValuesToBuffers(pixel);
                outputEmission[pixel] = 0.f;
            }
            return;
        }

        float3 sampleReflectance = 0.f;
        float3 deltaSampleEmission = 0.f;
        float hitDist = 0.f;

        bool isPrimaryBounceDelta = false;
        bool isPrimaryBounceTransmission = false;

        float3 deltaReflectance = 0.f;
        float3 emissionAlongDeltaPath = 0.f;
        if (centralReservoir.pathFlags.rcVertexLength() == 1)
        {
            // this also writes hitDist sampled by diffuse BRDF on a smooth surface, but does not reduce the quality
            const IMaterialInstance centralMi = gScene.materials.getMaterialInstance(centralPrimarySd, ExplicitLodTextureSampler(0.f));
            if (centralMi.getProperties(centralPrimarySd).roughness < params.specularRoughnessThreshold)
            {
                hitDist = length(loadVertexPosition(unpackReservoirHitInfo(centralReservoir.reconnectionHit)) - centralPrimarySd.posW);
            }
        }
        else
        {
            hitDist = TraceDenoiserReplayPath( centralPrimaryHit,
                                                    centralPrimarySd,
                                                    centralReservoir.pathFlags,
                                                    centralReservoir.initRandomSeed,
                                                    isPrimaryBounceDelta,
                                                    isPrimaryBounceTransmission,
                                                    deltaReflectance,
                                                    emissionAlongDeltaPath);
        }

        // NRD Data
        if (FETCH_NRD_DATA_PASS)
        {
            if (isPrimaryBounceDelta)
            {
                float3 color = centralReservoir.integrand * centralReservoir.weight - emissionAlongDeltaPath;
                float3 deltaReflectionIllumination = isPrimaryBounceTransmission ? 0 : color / deltaReflectance;
                float3 deltaTransmissionIllumination = isPrimaryBounceTransmission ? color / deltaReflectance : 0;
                float3 residual = isPrimaryBounceTransmission && all(deltaReflectance == 0.f) ? color : 0.f;
                
                outputResidualRadianceHitDist[pixel] = float4(residual, 0.f);
                outputDeltaReflectionRadianceHitDist[pixel] = float4(deltaReflectionIllumination, 0.f); // hitDist in provided by nrdDeltaReflectionHitDist
                outputDeltaTransmissionRadianceHitDist[pixel] = float4(deltaTransmissionIllumination, 0.f);
                outputDiffuseRadianceHitDist[pixel] = 0.f;
                outputSpecularRadianceHitDist[pixel] = 0.f;
                outputEmission[pixel] = 0.f;
            }
            else
            {
                // Demodulate diffuse reflectance (albedo) from diffuse color.
                float3 diffuseReflectance = bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo;
                float3 diffuseFactor = select(diffuseReflectance <= 0.f, 0.f, 1.f / diffuseReflectance);

                // Demodulate preintegrated specular reflectance from specular color.
                // We use an approximate shading model with a single specular lobe.
                float NdotV = saturate(dot(bsdfProperties.guideNormal, centralPrimarySd.V));
                float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
                float3 specularReflectance = max(kNRDMinReflectance, approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV));

                float3 diffuseBSDF = diffuseReflectance;
                float3 specularBSDF = specularReflectance;

                float3 totalBSDF = diffuseBSDF + specularBSDF;

                float3 reflectingRadiance = restirpt.kUseAreaReservoirs ? 
                                            max(0.f, centralReservoir.integrand * centralReservoir.weight - bsdfProperties.emission) : 
                                            centralReservoir.integrand * centralReservoir.weight;

                float3 diffuse = diffuseBSDF / totalBSDF * reflectingRadiance;
                float3 specular = specularBSDF / totalBSDF * reflectingRadiance;
                if (totalBSDF.x == 0.f)
                {
                    diffuse.x = 0.f;
                    specular.x = 0.f;
                }
                if (totalBSDF.y == 0.f)
                {
                    diffuse.y = 0.f;
                    specular.y = 0.f;
                }
                if (totalBSDF.z == 0.f)
                {
                    diffuse.z = 0.f;
                    specular.z = 0.f;
                }

                float3 diffuseIllumination = diffuse * diffuseFactor;

                float3 specularFactor = select(specularReflectance <= 0.f, 0.f, 1.f / specularReflectance);
                float3 specularIllumination = specular * specularFactor;

                outputEmission[pixel] = float4(bsdfProperties.emission, 1.f);
                outputDiffuseRadianceHitDist[pixel] = float4(diffuseIllumination, hitDist);
                outputSpecularRadianceHitDist[pixel] = float4(specularIllumination, hitDist);

                outputDeltaReflectionRadianceHitDist[pixel] = 0.f;
                outputDeltaTransmissionRadianceHitDist[pixel] = 0.f;
                outputResidualRadianceHitDist[pixel] = 0.f;
            }
        }

    }
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        DoPass(pixel);
    }
};

cbuffer CB
{
    FetchDenoiserData gFetchDenoiserData;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gFetchDenoiserData.execute(dispatchThreadId.xy);
}
