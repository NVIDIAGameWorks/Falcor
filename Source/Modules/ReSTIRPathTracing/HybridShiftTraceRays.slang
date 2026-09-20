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
#include "Utils/Math/MathConstants.slangh"

import PathReservoir;
import Utils.Sampling.SampleGenerator;
import LoadShadingDataDynamic;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Scene.RaytracingInline;
import Scene.Scene;
import PathTracer;
import RenderPasses.Shared.Denoising.NRDData;
import Rendering.Lights.EmissiveLightSampler;

struct ClosestHitQuery : IClosestHitQueryDynamic
{
    HitInfo traceRay(inout PathState path, const Ray ray, out float hitT, bool isPrevFrame)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceRay(ray, hitT, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

struct VisibilityQuery : IVisibilityQueryDynamic
{
    bool traceVisibilityRay(const Ray ray, bool isPrevFrame)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

float3 TraceReplayedPath(HitInfo dstPrimaryHit, ShadingData dstPrimarySd,
                         ReSTIRPathFlags pathFlags, uint initRandomSeed, inout HitInfo rcPrevHit, inout float3 rcPrevHitWo, inout float rcFootprintFactor, bool isPrevFrame)
{
    if (!dstPrimaryHit.isValid()) return 0.f;

    bool isPrevFrameForCamera = isPrevFrame;
    if (!gPathTracer.restirpt.kUsePrevFrameSceneData) isPrevFrame = false;

    PathState path;
    generatePathForRandomReplay(dstPrimaryHit, dstPrimarySd, path, initRandomSeed, pathFlags);

    path.restirPTData.primaryRayFootprint = gPathTracer.restirpt.computePrimaryRayFootprint( dstPrimarySd.faceN, isPrevFrameForCamera ? gScene.camera.data.prevPosW : gScene.camera.data.posW, dstPrimarySd.V, dstPrimarySd.posW);

    const IMaterialInstance mi = gScene.materials.getMaterialInstance(dstPrimarySd, ExplicitLodTextureSampler(0), getMaterialInstanceHints(dstPrimaryHit, true));

    // skipping random numbers and setting the thresholds
    sampleNext1D(path.sg); // for BSDF RIS
    sampleNext1D(path.sg); // for NEE RIS
    gPathTracer.skipLightSampleRandomNumbers(path.sg);
    if (kUseRussianRoulette) sampleNext1D(path.sg);

    bool valid = generateScatterRayRandomReplay(dstPrimarySd, mi, path);

    path.setPrevFrame(isPrevFrame);

    if (!valid || !path.isActive())
    {
        path.terminate();
        return 0.f;
    }

    bool terminateWithMiss = false;

    while (path.isActive())
    {
        ClosestHitQuery chq;
        nextHit(path, chq, isPrevFrame);

        if (path.isHit())
        {
            VisibilityQuery vq;
            handleHitReplayPath(path, vq);
        }
        else
        {
            terminateWithMiss = true;
            handleMissReplayPath(path);
        }
    }

    bool shouldConnect = pathFlags.rcVertexLength() <= pathFlags.pathLength() + 1;
    bool readyToConnect = path.getVertexIndex() - 1 == pathFlags.rcVertexLength() - 1;

    // Check if we need to terminate due to non-invertible shift
    bool nonInvertable = (shouldConnect && (!readyToConnect || terminateWithMiss)) || path.getVertexIndex() == 0;

    float3 L = nonInvertable ? 0.f : path.L;

    if (shouldConnect && !nonInvertable)
    {
        rcPrevHit = path.restirPTData.pathRcInfo.reconnectionHit;
        rcPrevHitWo = path.restirPTData.pathRcInfo.reconnectionWi;
        rcFootprintFactor = path.restirPTData.pathRcInfo.footprintFactor;
        L = path.thp;
    }

    return L;
};


// return the hit dist
// and the first non-delta surface hit reflectance of a delta path (if any)
float TraceDenoiserReplayPath(
    HitInfo dstPrimaryHit, ShadingData dstPrimarySd,
    ReSTIRPathFlags pathFlags, uint initRandomSeed, inout bool isPrimaryBounceDelta, inout bool isPrimaryBounceTransmission, inout float3 deltaReflectance, inout float3 emissionAlongDeltaPath)
{
    // if we need to do random replay of any kind, do it
    PathState path;
    generatePathForRandomReplay(dstPrimaryHit, dstPrimarySd, path, initRandomSeed, pathFlags);

    const IMaterialInstance mi = gScene.materials.getMaterialInstance(dstPrimarySd, ExplicitLodTextureSampler(0), getMaterialInstanceHints(dstPrimaryHit, true));
    // skipping random numbers and setting the thresholds
    if (kUseRussianRoulette) sampleNext1D(path.sg);
    sampleNext1D(path.sg); // for BSDF RIS
    sampleNext1D(path.sg); // for NEE RIS
    gPathTracer.skipLightSampleRandomNumbers(path.sg);

    bool valid = generateScatterRayRandomReplay(dstPrimarySd, mi, path);

    isPrimaryBounceDelta = path.isDelta();
    isPrimaryBounceTransmission = path.isTransmission();

    if (!valid || !path.isActive() || !(path.isSpecular() || path.isDelta()))
    {
        return 0.f;
    }

    float hitDist = 0.f;

    if (path.isActive())
    {
        ClosestHitQuery chq;
        nextHit(path, chq, false);

        if (path.isHit())
        {
            hitDist = path.sceneLength;
        }
        else hitDist = HLF_MAX;
    }

    if (isPrimaryBounceDelta && kFetchNRDDataPass)
    {
        path.L = 0;

        if (path.isHit() && path.isActive()) 
        {
            VisibilityQuery vq;

            handleHitReplayPath(path, vq);

            while (path.isActive())
            {
                ClosestHitQuery chq;
                nextHit(path, chq, false);

                if (path.isHit())
                {
                    handleHitReplayPath(path, vq);
                }
                else
                {
                    break;
                }
            }

            deltaReflectance = path.thp;
            emissionAlongDeltaPath = path.L;
        }
    }

    return hitDist;
}


[ForceInline]
void traceTemporalUpdateNeeAtRc(inout PathReservoir reservoir, inout PathState path, ShadingData rcVertexSd, const IMaterialInstance mi, bool isPrevFrame)
{
    VisibilityQuery vq;
    updateDirectLightingForRcVertex(path, rcVertexSd, mi, vq, true, isPrevFrame);
    reservoir.reconnectionIncidentRadiance = path.restirPTData.pathRcInfo.rcThroughput;
    reservoir.lightPdf = path.restirPTData.pathRcInfo.reconnectionJacobian;
    reservoir.reconnectionWi = path.dir;
    reservoir.pathFlags.insertLightType(path.restirPTData.pathRcInfo.pathFlags.lightType());
}

[ForceInline]
void traceTemporalUpdateEmissiveTriangleAtRc(inout PathReservoir reservoir, ShadingData rcVertexSd, ShadingData rcPrevSd, const IMaterialInstance mi, PathState path)
{
    BSDFProperties bsdfProperties = mi.getProperties(rcVertexSd);
    reservoir.reconnectionIncidentRadiance = bsdfProperties.emission;

    TriangleHit triangleHit = path.hit.getTriangleHit();
    TriangleLightHit hit;
    hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
    hit.posW = rcVertexSd.posW;
    // backfacing emission is 0 for standard material so this doesn't matter (but will matter for other material model)
    hit.normalW = rcVertexSd.frontFacing ? rcVertexSd.faceN : -rcVertexSd.faceN;

    const IMaterialInstance rcPrevMI = gScene.materials.getMaterialInstance( rcPrevSd,
            ExplicitLodTextureSampler(gPathTracer.params.lodBias), getMaterialInstanceHints(rcVertexSd.curveRadius == 0.f, false) );
    uint rcPrevLobes = rcPrevMI.getLobeTypes(rcPrevSd);

    const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
    const bool isCurveHit = gPathTracer.kUseCurves && (path.hit.getType() == HitType::Curve);
    const bool isHairMaterial = gPathTracer.kUseHairMaterial && (rcVertexSd.mtl.getMaterialType() == MaterialType::Hair);
    const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

    bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || ((rcPrevLobes & (uint)LobeType::NonDeltaReflection) != 0);
    // we assume that the emissive surface is outside any dielectric volumes. so, simply having a transmission type of last bounce we know that
    // the previous vertex is inside a dielectric volume
    if (reservoir.pathFlags.decodeIsTransmissionEvent(true) && !kUseLightsInDielectricVolumes) sampleUpperHemisphere = false;
    bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((rcPrevLobes & (uint)LobeType::NonDeltaTransmission) != 0);
    bool upperHemisphere = sampleUpperHemisphere && !sampleUpperHemisphere;
    reservoir.lightPdf = gPathTracer.getEmissiveSelectionProbability() *
                         gPathTracer.emissiveSampler.evalPdf(rcPrevSd.posW, rcPrevSd.frame.N, upperHemisphere, hit);
    // convert to area measure
    float3 disp = rcPrevSd.posW - hit.posW;
    float dist2 = dot(disp, disp);
    float3 dir = disp / sqrt(dist2);
    float G = abs(dot(dir, hit.normalW)) / dist2;
    reservoir.lightPdf *= G;
}

[ForceInline]
void traceTemporalUpdateScatterReplayAtRc(inout PathReservoir reservoir, ShadingData rcVertexSd, const IMaterialInstance mi, inout PathState path, bool isPrevFrame)
{
    bool valid = generateScatterRayGivenDirection(rcVertexSd, mi, path);
    path.restirPTData.pathRcInfo.rcThroughput = float3(1.f); // reset throughput

    // trace stuff to update stuff
    if (!valid) path.terminate();

    while (path.isActive())
    {
        ClosestHitQuery chq;
        nextHit(path, chq, isPrevFrame);

        if (path.isHit())
        {
            VisibilityQuery vq;
            handleHitReplayPath(path, vq);
        }
        else
        {
            handleMissReplayPath(path);
        }
    }

    // Now we don't have NEE in handleHitReplayPath, so we have to do it here
    if (reservoir.pathFlags.lastVertexNEE())
    {
        sampleNext1D(path.sg); // random number for RIS
        ShadingData NEESd = loadShadingData(path.hit, path.origin, path.dir, isPrevFrame);
        const IMaterialInstance NEEmi = gScene.materials.getMaterialInstance( NEESd, ExplicitLodTextureSampler(gPathTracer.params.lodBias), getMaterialInstanceHints(path.hit, false));
        VisibilityQuery vq;
        updateDirectLightingForRcVertex(path, NEESd, NEEmi, vq, false, isPrevFrame);
    }

    reservoir.reconnectionIncidentRadiance = any(path.L > 0) ? path.restirPTData.pathRcInfo.rcThroughput : 0.f;
    reservoir.lightPdf = path.restirPTData.pathRcInfo.reconnectionJacobian;
    reservoir.pathFlags.insertLightType(path.restirPTData.pathRcInfo.pathFlags.lightType());
}

void TraceTemporalUpdate(ShadingData rcPrevSd, inout PathReservoir reservoir, bool isPrevFrame)
{
    if (!gPathTracer.restirpt.kUsePrevFrameSceneData) isPrevFrame = false;

    HitInfo reconnectionHit = unpackReservoirHitInfo(reservoir.reconnectionHit);

    int rcVertexLength = reservoir.pathFlags.rcVertexLength();

    // Branch 1: rc-vertex is the environment light. No path setup needed; just rotate + re-eval.
    if (!reconnectionHit.isValid() && reservoir.pathFlags.lightType() == (uint)LightType::EnvMap)
    {
        assert(reservoir.pathFlags.pathLength() + 1 == rcVertexLength);
        reservoir.reconnectionWi = gScene.envMap.toWorld(gScene.envMap.toLocal(reservoir.reconnectionWi, !isPrevFrame), isPrevFrame);
        reservoir.reconnectionIncidentRadiance = gPathTracer.envMapSampler.eval(reservoir.reconnectionWi, isPrevFrame);
        reservoir.lightPdf = gPathTracer.getEnvMapSelectionProbability() * gPathTracer.envMapSampler.evalPdf(reservoir.reconnectionWi, isPrevFrame);
        return;
    }

    ShadingData rcVertexSd = loadShadingData(reconnectionHit, rcPrevSd.posW, float3(0.f), isPrevFrame);
    const IMaterialInstance mi = gScene.materials.getMaterialInstance( rcVertexSd, ExplicitLodTextureSampler(gPathTracer.params.lodBias), getMaterialInstanceHints(reconnectionHit, false) );

    PathState path;
    generatePathForTemporalUpdate( reconnectionHit, rcVertexSd, path, reservoir.reconnectionWi, reservoir.rcNextVertexBounceSeed, reservoir.initRandomSeed, reservoir.pathFlags);
    path.setPrevFrame(isPrevFrame);

    // NEE replay not yet supported if light tile RIS was used to generate the NEE light sample.
    if (reservoir.pathFlags.lastVertexNEE() && reservoir.pathFlags.pathLength() == rcVertexLength && !kUseLightTileRIS)
    {
        // Branch 2: NEE at rc-vertex.
        traceTemporalUpdateNeeAtRc(reservoir, path, rcVertexSd, mi, isPrevFrame);
    }
    else if (reservoir.pathFlags.pathLength() + 1 == rcVertexLength)
    {
        // Branch 3: rc-vertex is emissive (env-map and emissive-triangle sub-cases).
        if (reservoir.pathFlags.lightType() == (uint)LightType::EnvMap)
        {
            reservoir.reconnectionWi = gScene.envMap.toWorld(gScene.envMap.toLocal(reservoir.reconnectionWi, !isPrevFrame), isPrevFrame);
            reservoir.reconnectionIncidentRadiance = gPathTracer.envMapSampler.eval(reservoir.reconnectionWi, 0.f, isPrevFrame);
            reservoir.lightPdf = gPathTracer.envMapSampler.evalPdf(reservoir.reconnectionWi, isPrevFrame);
        }
        else if (reservoir.pathFlags.lightType() == (uint)LightType::Emissive)
        {
            traceTemporalUpdateEmissiveTriangleAtRc(reservoir, rcVertexSd, rcPrevSd, mi, path);
        }
        // analytic light update not supported!
    }
    else if (!kUseLightTileRIS)
    {
        // Branch 4: general case, re-trace the scatter ray and the remaining path.
        traceTemporalUpdateScatterReplayAtRc(reservoir, rcVertexSd, mi, path, isPrevFrame);
    }
}
