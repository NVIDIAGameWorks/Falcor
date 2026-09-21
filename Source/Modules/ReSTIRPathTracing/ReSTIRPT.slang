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
#pragma once
#include "RetraceScheduleDefinition.slangh"
#include "Utils/HostDeviceShared.slangh"
#include "Utils/Sampling/SampleGeneratorType.slangh"
BEGIN_NAMESPACE_FALCOR

#ifndef HOST_CODE
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Math.PackedFormats;
import RenderPasses.PathTracer.PathState;
import RenderPasses.PathTracer.PathTracerUtils;
__exported import RenderPasses.PathTracer.Params;
__exported import ReSTIRPathTracingParams;
import Rendering.Lights.LightHelpers;
import Scene.Material.MaterialData;
import Scene.Material.MaterialTypes;
import PathReservoir;
import Scene.Scene;
import Scene.ShadingData;
import Utils.Debug.PixelDebug;
import Rendering.Materials.IMaterialInstance;
#include "Utils/Math/MathConstants.slangh"

#ifndef RESTIRPT_SHIFT_MAPPING
#define RESTIRPT_SHIFT_MAPPING 0
#endif

#ifndef TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
#define TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE 0
#endif

#ifndef USE_RESERVOIR_COMPRESSION
#define USE_RESERVOIR_COMPRESSION 0
#endif

#ifndef USE_AREA_RESERVOIRS
#define USE_AREA_RESERVOIRS 0
#endif

#ifndef USE_DECOUPLED_SHADING
#define USE_DECOUPLED_SHADING 0
#endif

#ifndef NUM_SPATIAL_NEIGHBORS
#define NUM_SPATIAL_NEIGHBORS 3
#endif

#ifndef RETRACE_SCHEDULE_TYPE
#define RETRACE_SCHEDULE_TYPE RETRACE_SCHEDULE_NAIVE
#endif

#ifndef USE_PREV_FRAME_SCENE_DATA
#define USE_PREV_FRAME_SCENE_DATA 0
#endif

#endif

struct ReSTIRPT
{
    enum class LocalStrategy
    {
        None = 0x0,
        RoughnessCondition = 0x1,
        DistanceCondition = 0x2,
    };

    enum class RetraceScheduleType
    {
        Naive = RETRACE_SCHEDULE_NAIVE,
        Compact = RETRACE_SCHEDULE_COMPACT
    };

    enum class TemporalAreaReuseType
    {
        Point = 0,
        Fast = 1,
        Robust = 2,
        Splat = 3
    };

    FALCOR_ENUM_INFO(TemporalAreaReuseType, {
        { TemporalAreaReuseType::Point, "Point" },
        { TemporalAreaReuseType::Fast, "Fast" },
        { TemporalAreaReuseType::Robust, "Robust" },
        { TemporalAreaReuseType::Splat, "Splat" }
    });

    enum class PointReprojectionMode
    {
        Nearest = 0,
        Stochastic = 1,
    };

    FALCOR_ENUM_INFO(PointReprojectionMode, {
        { PointReprojectionMode::Nearest, "Nearest" },
        { PointReprojectionMode::Stochastic, "Stochastic" },
    });

    enum class ShiftMapping
    {
        Reconnection = 0,
        Hybrid = 1,
    };

    FALCOR_ENUM_INFO(ShiftMapping, {
        { ShiftMapping::Reconnection, "Reconnection" },
        { ShiftMapping::Hybrid, "Hybrid" },
    });

    struct ShiftMappingSettings
    {
        uint localStrategyType = 0x3;
        float specularRoughnessThreshold = 0.2f;
        float nearFieldDistanceThreshold = 0.02f;
        float specularRoughnessSigma = 0.f; // gaussian stdev for soft threshold
        float nearFieldDistanceSigma = 0.2f;
        bool additionalShadowRayOffset = false; // for reconnection shadow ray

#ifdef HOST_CODE
        template<typename Archive>
        void serialize(Archive& ar)
        {
            ar("localStrategyType", localStrategyType);
            ar("specularRoughnessThreshold", specularRoughnessThreshold);
            ar("nearFieldDistanceThreshold", nearFieldDistanceThreshold);
            ar("specularRoughnessSigma", specularRoughnessSigma);
            ar("nearFieldDistanceSigma", nearFieldDistanceSigma);
            ar("additionalShadowRayOffset", additionalShadowRayOffset);
        }
#endif
    };

    ShiftMappingSettings settings;
    uint numSpatialRounds = 1;
    uint initialLightSampleCount = 1;
    bool sampleDI = true;
    int samplesPerPixel = 1;
    bool useAreaReSTIRBackprojection = true;
    bool useDoFReconnectionShift = true;
    float doFReconnectionShiftProbability = 0.3f;
    bool deriveDoFReconnectionShiftProbabilityFromCoC = true;
    bool temporalRobustShift = false;
    bool useDoFReconnectionForTemporalReuse = true;
    float specularMotionVectorRoughnessThreshold = 0.f;
    bool enableSpecularMotionVectors = false;
    bool enableDisocclusionMotionVectors = false;
    bool useReservoirSplatting = false;

#ifndef HOST_CODE

    uint sgCount()
    {
        return samplesPerPixel + 2 * int(enableSpecularMotionVectors) + int(useAreaReSTIRBackprojection) + (useDoFReconnectionForTemporalReuse ? 2 : 1) + numSpatialRounds;
    }

    uint spatialSgOffset(int spatialRoundId)
    {
        return samplesPerPixel + 2 * int(enableSpecularMotionVectors) + int(useAreaReSTIRBackprojection) + (useDoFReconnectionForTemporalReuse ? 2 : 1) + spatialRoundId;
    }

    uint temporalPopulateFloatingSgOffset()
    {
        return samplesPerPixel + 2 * int(enableSpecularMotionVectors);
    }

    uint specMVecSgOffset()
    {
        return samplesPerPixel + 1;
    }

    uint temporalSgOffset(int temporalPassId)
    {
        return samplesPerPixel + 2 * int(enableSpecularMotionVectors) + int(useAreaReSTIRBackprojection) + temporalPassId;
    }

    uint spatialOrTemporalSgOffset(bool isSpatialPass, int spatialRoundId, int temporalPassId)
    {
        return samplesPerPixel + 2 * int(enableSpecularMotionVectors) + int(useAreaReSTIRBackprojection) + (isSpatialPass ? (useDoFReconnectionForTemporalReuse ? 2 : 1) + spatialRoundId : temporalPassId);
    }

    int computeInitialLightSampleCount(int initialLightSampleCount, int vertexIndex)
    {
        return max(1, initialLightSampleCount / (vertexIndex * vertexIndex));
    }
    // static params goes here
    static const uint kShiftMapping = RESTIRPT_SHIFT_MAPPING;
    static const bool kUseReservoirCompression = USE_RESERVOIR_COMPRESSION;
    static const bool kUseAreaReservoirs = USE_AREA_RESERVOIRS;
    static const bool kTemporalUpdateForDynamicScene = TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE;
    static const uint kNumSpatialNeighbors = NUM_SPATIAL_NEIGHBORS;
    static const uint kRetraceScheduleType = RETRACE_SCHEDULE_TYPE;
    static const uint kMaximumRcLength = 255;
    static const bool kUsePrevFrameSceneData = USE_PREV_FRAME_SCENE_DATA;
    RWStructuredBuffer<PathReservoir> pathReservoirs;

    float computePrimaryRayFootprint(float3 faceN, float3 rayOrg, float3 rayDir, float3 hitPos)
    {
        float3 disp = rayOrg - hitPos;
        return dot(disp, disp) / abs(M_1_4PI * dot(rayDir, faceN)); // M_4PI should actually be M_1_4PI
    }

    float getDiffusePseudoPdf()
    {
        return M_1_2PI;
    }

    uint getCurrentSeed(SampleGenerator sg)
    {
#if defined(SAMPLE_GENERATOR_TYPE) && SAMPLE_GENERATOR_TYPE == SAMPLE_GENERATOR_TINY_UNIFORM
        return sg.rng.state;
#else
        return 0;
#endif
    }
    // kReSTIRPT

    struct PathSegmentData
    {
        LastVertexState lastVertexState = {};
        float geometryFactor = 0;
        float lastScatterPdf = 0;
        float rayFootprint = 0;
        float inverseGeometryFactor = 0;
    }

    void PopulatePathSegmentData(inout PathSegmentData data, inout PathState path, ShadingData sd)
    {
        const bool isPrimaryHit = path.getVertexIndex() == 1;
        if (isPrimaryHit) path.restirPTData.primaryRayFootprint = computePrimaryRayFootprint(sd.faceN, path.origin, path.dir, sd.posW);
        float distanceToPrevVertex = length(sd.posW - path.origin);
        data.geometryFactor = abs(dot(sd.faceN, sd.V)) / (distanceToPrevVertex * distanceToPrevVertex);
        data.lastScatterPdf = path.restirPTData.lastScatterPdf;
        data.inverseGeometryFactor = distanceToPrevVertex * distanceToPrevVertex / abs(dot(path.normal, sd.V));
        data.rayFootprint = 1.f / (data.geometryFactor * path.pdf);
        data.lastVertexState.Init( path.isDelta(), path.isTransmission(), path.getSampledBSDFComponent(), path.supportBSDFIndexing(), path.isRough(),
            data.rayFootprint >= path.restirPTData.primaryRayFootprint * path.restirPTData.distanceThreshold );
    }

    void setRcVertex(inout PathState path, PathSegmentData ps)
    {
        // test connectability
        bool canConnect = kShiftMapping == (uint)ReSTIRPT::ShiftMapping::Reconnection && path.getVertexIndex() == 2 ||
            kShiftMapping == (uint)ReSTIRPT::ShiftMapping::Hybrid &&
                path.restirPTData.pathRcInfo.pathFlags.rcVertexLength() == kMaximumRcLength &&
                (ps.lastVertexState.isLastVertexRough() && min(ps.rayFootprint,
                    ps.inverseGeometryFactor / path.pdf)
                                     >= path.restirPTData.primaryRayFootprint * path.restirPTData.distanceThreshold);

        path.restirPTData.pathRcInfo.rcNextVertexBounceSeed = canConnect ? getCurrentSeed(path.sg) : path.restirPTData.pathRcInfo.rcNextVertexBounceSeed;
        path.restirPTData.pathRcInfo.reconnectionWi = canConnect ? path.dir : path.restirPTData.pathRcInfo.reconnectionWi;
        path.restirPTData.pathRcInfo.pathFlags.insertRcVertexLength( canConnect ? path.getVertexIndex() - 1 : path.restirPTData.pathRcInfo.pathFlags.rcVertexLength() );
        path.restirPTData.pathRcInfo.reconnectionHit.data = canConnect ? path.hit.data : path.restirPTData.pathRcInfo.reconnectionHit.data;

        path.restirPTData.pathRcInfo.pathFlags.tryInsertBounceTypesBeforeRcVertex(ps.lastVertexState.data & 0x3f, canConnect);

        path.restirPTData.pathRcInfo.reconnectionJacobian = canConnect ? ps.geometryFactor : path.restirPTData.pathRcInfo.reconnectionJacobian;

        path.restirPTData.sourcePdf *= canConnect ? ps.lastScatterPdf * path.restirPTData.lastScatterPdf : 1.f;

        path.restirPTData.pathRcInfo.pathFlags.tryInsertBounceTypesAfterRcVertex( (uint(path.isDelta()) | uint(path.isTransmission()) << 1 | uint(path.getSampledBSDFComponent()) << 2), canConnect );

        path.restirPTData.pathRcInfo.rcThroughput = canConnect ? float3(1.f) : path.restirPTData.pathRcInfo.rcThroughput;
    }


    [ForceInline]
    void convertNeeSampledVertexToRcVertex(inout PathState path, ShadingData sd, LightSample ls,
                                                   float3 Le, bool supportBSDFIndexing, bool isPunctualLight)
    {
#if USE_RESTIRPT        
        path.restirPTData.pathRis.reconnectionJacobian = 1.f;
        path.restirPTData.pathRis.integrand *= ls.pdf;
        path.restirPTData.pathRis.reconnectionIncidentRadiance = Le;
        path.restirPTData.pathRis.reconnectionWi = ls.dir;
        path.restirPTData.pathRis.pathFlags.insertRcVertexLength(path.restirPTData.pathRis.pathFlags.pathLength() + 1);

        bool isTransmission = dot(sd.faceN, sd.V) * dot(sd.faceN, ls.dir) < 0.f;
        path.restirPTData.pathRis.pathFlags.insertIsTransmissionEvent(isTransmission, true);
        path.restirPTData.pathRis.pathFlags.insertSupportBSDFIndexingBeforeRcVertex(supportBSDFIndexing);

        // not directional light
        if (ls.lightType == (uint)LightType::Analytic)
        {
            if (ls.distance == kMaxLightDistance) // dir
            {
                path.restirPTData.pathRis.reconnectionHit.data.x = 0;
            }
            else
            {
                path.restirPTData.pathRis.reconnectionWi = ls.origin + ls.dir * ls.distance; // store position
                if (isPunctualLight || kUseLightTileRIS)
                {
                    path.restirPTData.pathRis.reconnectionIncidentRadiance /= ls.geometryTerm; // ls.geometry term is 1/d^2
                    path.restirPTData.pathRis.reconnectionHit.data.x = 1;           // use this as type identifier
                }
                else // area, sphere
                {
                    path.restirPTData.pathRis.reconnectionHit.data.x = 2;
#if !HIT_INFO_USE_COMPRESSION
                    path.restirPTData.pathRis.reconnectionHit.data.yzw = asuint(ls.normal);
#endif
                    path.restirPTData.pathRis.reconnectionJacobian *= ls.geometryTerm;
                    path.restirPTData.pathRis.lightPdf *= ls.geometryTerm; // stored in area measure
                }
            }
        }
        else if (ls.lightType == (uint)LightType::Emissive)
        {
            TriangleHit neeTriangleHit;
            // convert light collection triangle id to scene mesh triangle id
            uint2 meshPrimID = gScene.lightCollection.getSceneMeshPrimID(ls.triangleId);
            GeometryInstanceID neeGeoInstanceId;
            neeGeoInstanceId.index = meshPrimID.x;
            neeTriangleHit.instanceID = neeGeoInstanceId;
            neeTriangleHit.primitiveIndex = meshPrimID.y;
            neeTriangleHit.barycentrics = ls.barycentrics;
            HitInfo neeHitInfo = HitInfo(neeTriangleHit);
            path.restirPTData.pathRis.reconnectionHit = neeHitInfo;
            path.restirPTData.pathRis.reconnectionJacobian *= ls.geometryTerm;
            path.restirPTData.pathRis.lightPdf *= ls.geometryTerm; // stored in area measure
        }
#endif        
    }

    void HandleNEECase( inout PathState path, ShadingData sd, LightSample ls, float uNEERIS, float scatterPdf, float pdfWarpFactor, uint neeSeed, PathSegmentData ps,
                                                float3 Le, float3 Lr, bool supportBSDFIndexing, bool isPunctualLight)
    {
        const bool isPrimaryHit = path.getVertexIndex() == 1;
        bool isPrimaryHitAndNoReSTIRPTSample = isPrimaryHit && !sampleDI;
        if (isPrimaryHitAndNoReSTIRPTSample)
        {
            path.restirPTData.pathL0 += path.thp * Lr; // residual radiance not integrated by ReSTIR PT
            return;
        }

        bool canConnect =
            kShiftMapping == (uint)ReSTIRPT::ShiftMapping::Reconnection && path.getVertexIndex() == 2 ||
                          kShiftMapping == (uint)ReSTIRPT::ShiftMapping::Hybrid &&
                              path.restirPTData.pathRcInfo.pathFlags.rcVertexLength() == kMaximumRcLength &&
                                path.getVertexIndex() >= 2 &&
                              (min(ps.rayFootprint, ps.inverseGeometryFactor / scatterPdf) >= path.restirPTData.primaryRayFootprint * path.restirPTData.distanceThreshold
                    && ps.lastVertexState.isLastVertexRough());

        bool selected = path.restirPTData.pathRis.addNEECandidate( uNEERIS, path.restirPTData.pathRcInfo, canConnect, path.getVertexIndex() - 1,
            path.thp * Lr / pdfWarpFactor, ls.dir, Le, Lr, path.restirPTData.sourcePdf * pdfWarpFactor, ls.pdf, ls.lightType );

        bool shouldOverwrite = selected && canConnect;

        path.restirPTData.pathRis.reconnectionHit.data = shouldOverwrite ? path.hit.data : path.restirPTData.pathRis.reconnectionHit.data;
        path.restirPTData.pathRis.reconnectionJacobian = shouldOverwrite ? ps.geometryFactor : path.restirPTData.pathRis.reconnectionJacobian;
        if (shouldOverwrite) path.restirPTData.pathRis.integrand *= ps.lastScatterPdf * ls.pdf;

        path.restirPTData.pathRis.rcNextVertexBounceSeed = shouldOverwrite ? neeSeed : path.restirPTData.pathRis.rcNextVertexBounceSeed;
        path.restirPTData.pathRis.pathFlags.tryInsertBounceTypesBeforeRcVertex(ps.lastVertexState.data & 0x3f, shouldOverwrite);
        // bouncetypes after rcVertex in the shouldOverwrite case doesn't matter

        // NEE was selected but couldn't use the regular reconnection shift -> store the light sample
        // directly as the reservoir's reconnection vertex.
        if (selected && !canConnect && path.restirPTData.pathRcInfo.pathFlags.rcVertexLength() == kMaximumRcLength)
        {
            convertNeeSampledVertexToRcVertex(path, sd, ls, Le, supportBSDFIndexing, isPunctualLight);
        }
    }

    void populateReservoir(inout PathState path, Texture2D<uint> sampleCountTex, uint2 pixel, uint reservoirIdx, uint frameSeed)
    {
        uint totalSamples = kSamplesPerPixel;
        if (kSamplesPerPixel == 0)
        {
            totalSamples = sampleCountTex[pixel];
        }

        SampleGenerator sgForFilmLens = SampleGenerator(pixel, frameSeed * sgCount()); // all samples must use the same uv

        PathReservoir pathReservoir;
        if (path.getSampleIdx() == totalSamples - 1) // first sample (sampleIdx is actually the remaining samples)
        {
            pathReservoir = PathReservoir::createFromRisState( path.restirPTData.pathRis, path.restirPTData.pathRcInfo.initRandomSeed
                , float4(sampleNext2D(sgForFilmLens), sampleNext2D(sgForFilmLens))
            );
        }
        else
        {
            pathReservoir = pathReservoirs[reservoirIdx];
            pathReservoir.add( path.restirPTData.pathRis, path.restirPTData.pathRcInfo.initRandomSeed,
                float4(sampleNext2D(sgForFilmLens), sampleNext2D(sgForFilmLens)),
                path.sg );
        }

        // converting from w_sum to W
        if (path.getSampleIdx() == 0) // last sample 
        {
            float pHat = PathRISState::toScalar(pathReservoir.integrand);
            pathReservoir.weight = pHat == 0.f ? 0.f : pathReservoir.weight / (pathReservoir.M * pHat);
            pathReservoir.M = 1;
        }
        pathReservoirs[reservoirIdx] = pathReservoir;
    }

    void HandleBSDFCase( inout PathState path, float uBRDFRIS, PathSegmentData ps, float lightPdf,
        float3 attenuatedEmission, float misWeight, float3 emission)
    {
        bool lightHitHandledByReSTIRPT = path.getVertexIndex() > (sampleDI ? 1 : 2);
        const bool isPrimaryHit = path.getVertexIndex() == 1;

        if (kUseAreaReservoirs && isPrimaryHit)
        {
            path.restirPTData.pathRis.addPrimaryVisibleEmitterCandidate(uBRDFRIS, attenuatedEmission, path.restirPTData.sourcePdf);
        }

        if (lightHitHandledByReSTIRPT)
        {
            bool selected = path.restirPTData.pathRis.addBSDFHitCandidate( uBRDFRIS, path.restirPTData.pathRcInfo, (path.getVertexIndex() - 1) - 1, attenuatedEmission,
                path.dir, emission, misWeight * emission, path.restirPTData.sourcePdf, lightPdf, (uint)LightType::Emissive );

            // daqi: if current vertex has path.length >= 2, then it can be used as a rcVertex
            if (selected && (kShiftMapping == (uint)ReSTIRPT::ShiftMapping::Hybrid || path.getVertexIndex() == 2)
                    && path.restirPTData.pathRcInfo.pathFlags.rcVertexLength() == kMaximumRcLength) // we don't yet have an rcVertex
            {
                // emissive vertex is always seen as Lambertian
                if (kShiftMapping == (uint)ReSTIRPT::ShiftMapping::Reconnection ||
                    (min(ps.rayFootprint, ps.inverseGeometryFactor / getDiffusePseudoPdf()) >= path.restirPTData.primaryRayFootprint * path.restirPTData.distanceThreshold
                     // we use mi instead of last mi because all material must be the same in terms of supporting BSDF indexing
                     && (!path.supportSimpleRoughnessThreshold() || ps.lastVertexState.isLastVertexRough())))
                {
                    // we found an RC vertex!
                    // set rcVertexLength to current length (this will make rcVertexLength = reseroivr.pathLength + 1)
                    path.restirPTData.pathRis.integrand *= ps.lastScatterPdf;
                    path.restirPTData.pathRis.markAsRcVertex( path.getVertexIndex() - 1, path.hit, ps.lastVertexState,
                        lightPdf * ps.geometryFactor, (uint)LightType::Emissive, emission, float3(0.f),
                        ps.geometryFactor );
                }
            }
        }

        path.restirPTData.pathL0 += (!lightHitHandledByReSTIRPT && !(kUseAreaReservoirs && isPrimaryHit)) ? attenuatedEmission : 0.f;
    }

    void HandleBSDFCaseHitEnv(inout PathState path, float uBRDFRIS, float lightPdf, float3 emitterRadiance, float3 Le)
    {
        bool lightHitHandledByReSTIRPT = path.getVertexIndex() > (sampleDI ? 1 : 2);
        bool isPrimaryHit = path.getVertexIndex() == 1;

        path.restirPTData.pathL0 += (!lightHitHandledByReSTIRPT && !(kUseAreaReservoirs && isPrimaryHit)) ? path.thp * emitterRadiance : 0.f;

        if (kUseAreaReservoirs && isPrimaryHit)
        {
            path.restirPTData.pathRis.addPrimaryVisibleEmitterCandidate(uBRDFRIS, emitterRadiance, path.restirPTData.sourcePdf);
        }

        if (lightHitHandledByReSTIRPT)
        {
            LastVertexState lastVertexState;
            lastVertexState.Init( path.isDelta(), path.isTransmission(), path.getSampledBSDFComponent(), path.supportBSDFIndexing(), path.isRough(), true );

            bool selected = path.restirPTData.pathRis.addBSDFHitCandidate( uBRDFRIS, path.restirPTData.pathRcInfo, (path.getVertexIndex() - 1) - 1,
                path.thp * emitterRadiance, path.dir, Le, emitterRadiance, path.restirPTData.sourcePdf, lightPdf, (uint)LightType::EnvMap );

            if (selected && (kShiftMapping == (uint)ReSTIRPT::ShiftMapping::Hybrid || path.getVertexIndex() == 2)
                    && path.restirPTData.pathRcInfo.pathFlags.rcVertexLength() == kMaximumRcLength)
            {
                if (path.isRough() || kShiftMapping == (uint)ReSTIRPT::ShiftMapping::Reconnection)
                {
                    path.restirPTData.pathRis.integrand *= path.restirPTData.lastScatterPdf;
                    // we found an RC vertex!
                    // set rcVertexLength to current length (this will make rcVertexLength = reseroivr.pathLength + 1)
                    HitInfo dummyHit = HitInfo();
                    path.restirPTData.pathRis.markAsRcVertex( path.getVertexIndex() - 1, dummyHit, lastVertexState,
                                                lightPdf, (uint)LightType::EnvMap, Le, path.dir, 1.f );
                }
            }
        }
    }

    void ProcessScatterData(inout PathState path, inout bool valid, inout BSDFSample result, ShadingData sd, const IMaterialInstance mi)
    {
        // set result.pdf to a large value for delta
        if (result.isLobe(LobeType::Delta)) result.pdf = 1e20f;


        valid &= !isinf(result.pdf) && all(!isinf(result.weight));
        // result.pdf is the single component PDF

        uint componentType = result.BSDFIndex;
        path.setSampledBSDFComponent(componentType);

        if (mi.supportBSDFIndexing())
        {
            sd.mtl.setBSDFIndex(componentType);
            path.restirPTData.lastScatterPdf = mi.evalPdf(sd, result.wo);
        }
        else path.restirPTData.lastScatterPdf = result.pdf;

        // set lastScatterPdf to a non-zero value for delta
        if (result.isLobe(LobeType::Delta)) path.restirPTData.lastScatterPdf = 1.f;

        // compute augmented path footprint
        path.setRough(mi.isSampleRough(sd, componentType, result.pdf, path.restirPTData.roughnessThreshold));
        path.setSupportBSDFIndexing(mi.supportBSDFIndexing());
        path.setSupportSimpleRoughnessThreshold(mi.supportSimpleRoughnessThreshold());
    }

    void InitializePathData(inout PathState path)
    {
        path.restirPTData.pathL0 = 0;
        path.restirPTData.sourcePdf = 1.f;
        path.restirPTData.pathRis.init();

        path.restirPTData.pathRcInfo.initRandomSeed = getCurrentSeed(path.sg);
        path.restirPTData.pathRcInfo.rcThroughput = 1.f;

        if (settings.nearFieldDistanceSigma == 0.f) path.restirPTData.distanceThreshold = settings.nearFieldDistanceThreshold;
        else path.restirPTData.distanceThreshold = PathRISState::gaussrand(settings.nearFieldDistanceThreshold, settings.nearFieldDistanceSigma, path.sg);

        if (settings.specularRoughnessSigma == 0.f) path.restirPTData.roughnessThreshold = settings.specularRoughnessThreshold;
        else path.restirPTData.roughnessThreshold = PathRISState::gaussrand(settings.specularRoughnessThreshold, settings.specularRoughnessSigma, path.sg);

        path.restirPTData.pathRcInfo.reconnectionHit = HitInfo();                                         // initialize invalid reconnectionHit
        path.restirPTData.pathRcInfo.pathFlags.insertRcVertexLength(kMaximumRcLength); // insert an invalid rcvertexlength
    }

    void WriteReconnectionData(inout PathState path, float3 reconnectionWi, LastVertexState lastVertexState, float rayFootprint, float inverseGeometryFactor)
    {
        path.restirPTData.pathRcInfo.reconnectionHit = path.hit;
        path.restirPTData.pathRcInfo.reconnectionWi = reconnectionWi; 
        // if last vertex isn't rough, current vertex can't be a rcVertex, so we are safe
        bool currentVertexCannotBeReconnected = !lastVertexState.isLastVertexRough() || rayFootprint < path.restirPTData.primaryRayFootprint * path.restirPTData.distanceThreshold;
        path.restirPTData.pathRcInfo.footprintFactor = currentVertexCannotBeReconnected ? -1.f : inverseGeometryFactor;
    }

    void WriteTemporalRetraceData(inout PathState path, uint lightType, float lightPdf, float3 reconnectionIncidentRadiance)
    {
        path.restirPTData.pathRcInfo.pathFlags.insertLightType(lightType);
        path.restirPTData.pathRcInfo.reconnectionJacobian = lightPdf;
        path.restirPTData.pathRcInfo.rcThroughput = reconnectionIncidentRadiance;
    }

#endif
};

FALCOR_ENUM_REGISTER(ReSTIRPT::PointReprojectionMode);
FALCOR_ENUM_REGISTER(ReSTIRPT::ShiftMapping);
FALCOR_ENUM_REGISTER(ReSTIRPT::TemporalAreaReuseType);

END_NAMESPACE_FALCOR
