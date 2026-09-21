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
import RetraceWorkQueue;
import ResamplingCommon;
import PairedNeighborTransform;
#include "RetraceScheduleDefinition.slangh"


struct SpatialPathRetrace
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracingParams params; 
    ReSTIRPT restirpt;             

    Texture2D<PackedHitInfo> vbuffer;                     
    Texture1D<float2> neighborOffsets;

    StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;


    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    RWStructuredBuffer<uint> reconnectionDataOffsets;
    StructuredBuffer<float4> doFReconnectionShiftBuffer;

    StructuredBuffer<uint2> keyValueBuffer;
    StructuredBuffer<uint> numKeysBuffer;
    RWStructuredBuffer<NeighborValidMask> neighborValidMask;

    int gSpatialRoundId;
    int gNeighborCount;
    float gGatherRadius;
    bool resetTemporalReservoirs;

    bool gUsePairedNeighbors;

    bool areaReservoirUseNeighborRejection;

    RetraceWorkQueue queues; 

    int getNeighborCount()
    {
        return gNeighborCount; 
    }

    // compaction
#if RETRACE_SCHEDULE_TYPE == RETRACE_SCHEDULE_COMPACT
    void DoPass(const uint linearIndex)
    {
        // read number of paths
        uint totalPathCount = queues.counters.Load(0);
        if (linearIndex >= totalPathCount) return;
        uint encodedRetraceWorkItem = queues.retraceWorkItems.Load(linearIndex * 4);
        int2 pixel = int2(encodedRetraceWorkItem & 0xfff, encodedRetraceWorkItem >> 12 & 0xfff);
        uint retraceIndex = encodedRetraceWorkItem >> 24 & 0xff;

        printSetPixel(pixel);
        logSetPixel(pixel);

        
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId));
        const uint centralOffset = params.getReservoirOffset(pixel);

        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        float spatialRadius = gGatherRadius;

        int2 neighborPixel;
        var sharedSg = TinyUniformSampleGenerator(params.frameDim + 1, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId));
        PairedNeighborTransform transform = PairedNeighborTransform(sharedSg, params.frameDim.x, params.frameDim.y);

        if (gUsePairedNeighbors)
        {
            int dx, dy;
            transform.readTransformedDelta(retraceIndex / 2, pixel.x, pixel.y, dx, dy);
            neighborPixel = pixel + int2(dx, dy);
        }
        else
        {
            neighborPixel = getNextNeighborPixel(startIndex, neighborOffsets, pixel, retraceIndex / 2, spatialRadius);
        }
        const uint neighborOffset = params.getReservoirOffset(neighborPixel);

        PathReservoir reservoir = prevReservoirs[retraceIndex%2 ? neighborOffset : centralOffset];

        uint neighborCount = getNeighborCount();

        float doFReconnectionShiftProbability = restirpt.doFReconnectionShiftProbability;
        if (restirpt.deriveDoFReconnectionShiftProbabilityFromCoC)
        {
            float3 centralPrimaryWo;
            HitInfo centralPrimaryHit = getPixelHitInfo(params, vbuffer, pixel, centralPrimaryWo);
            VertexData centralPrimaryVertexData = getVertexData(centralPrimaryHit, gScene.camera.data.posW, -centralPrimaryWo);
            doFReconnectionShiftProbability =
                1.f - computeConfidenceWeightForTwoShifts(params.frameDim, centralPrimaryVertexData.posW, !centralPrimaryHit.isValid());
        }

        uint address = gUsePairedNeighbors ? neighborCount * centralOffset + (retraceIndex / 2) : 2 * neighborCount * centralOffset + retraceIndex;

        int doFReconnectionShiftCount = restirpt.useDoFReconnectionShift ? round(doFReconnectionShiftProbability * neighborCount) : 0;
        if (restirpt.kUseAreaReservoirs && int(retraceIndex / 2) < doFReconnectionShiftCount) reservoir.setLensSample(doFReconnectionShiftBuffer[address].xy);

        HitInfo primaryHit;
        ShadingData primarySd;
        if (restirpt.kUseAreaReservoirs)
            primarySd = getPixelShadingData( params, shiftedPrimaryHitBuffer, retraceIndex % 2 ? pixel : neighborPixel,
                address, reservoir.filmSample, reservoir.lensSample, primaryHit);
        else
            primarySd = getPixelShadingData(params, vbuffer, retraceIndex % 2 ? pixel : neighborPixel, primaryHit);

        HitInfo rcPrevHit = HitInfo();
        float3 rcPrevWo = float3(0);
        float rcFootprintFactor = 0.f;
        float3 pathThroughput = TraceReplayedPath( primaryHit, primarySd, reservoir.pathFlags,
                                       reservoir.initRandomSeed, rcPrevHit, rcPrevWo, rcFootprintFactor, false );

        reconnectionDataOffsets[address] = linearIndex;
        reconnectionDataBuffer[linearIndex] = ReconnectionData(rcPrevHit, rcPrevWo, pathThroughput, rcFootprintFactor);
    }
#else
    void DoPass(const uint2 pixel)
    {
        
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId));
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = prevReservoirs[centralOffset];

        HitInfo centralPrimaryHit;
        ShadingData centralPrimarySd = getPixelShadingData(params, vbuffer, pixel, centralPrimaryHit);
        if (!centralPrimaryHit.isValid() && !restirpt.kUseAreaReservoirs) return;

        int neighborCount = getNeighborCount();

        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        float spatialRadius = gGatherRadius;

        float doFReconnectionShiftProbability = restirpt.doFReconnectionShiftProbability;
        if (restirpt.deriveDoFReconnectionShiftProbabilityFromCoC)
        {
            doFReconnectionShiftProbability =
                1.f - computeConfidenceWeightForTwoShifts(params.frameDim, centralPrimarySd.posW, !centralPrimaryHit.isValid());
        }

        int doFReconnectionShiftCount = restirpt.useDoFReconnectionShift ? round(doFReconnectionShiftProbability * neighborCount) : 0;
        var sharedSg = TinyUniformSampleGenerator(params.frameDim + 1, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId));
        PairedNeighborTransform transform = PairedNeighborTransform(sharedSg, params.frameDim.x, params.frameDim.y);

        float cSum = centralReservoir.M;

        for (int i = 0; i < neighborCount; ++i)
        {
            int2 neighborPixel;

            if (gUsePairedNeighbors)
            {
                int dx, dy;
                transform.readTransformedDelta(i, pixel.x, pixel.y, dx, dy);
                neighborPixel = pixel + int2(dx, dy);
            }
            else
            {
                neighborPixel = getNextNeighborPixel(startIndex, neighborOffsets, pixel, i, spatialRadius);
            }

            if (!isValidScreenRegion(params, neighborPixel))
            {
                neighborValidMask[centralOffset].setValid(i, false);
                continue;
            }

            PathReservoir neighborReservoir = prevReservoirs[params.getReservoirOffset(neighborPixel)];

            HitInfo neighborPrimaryHit;
            ShadingData neighborPrimarySd = getPixelShadingData(params, vbuffer, neighborPixel, neighborPrimaryHit);

            if (restirpt.kUseAreaReservoirs && areaReservoirUseNeighborRejection)
            {
                if (centralPrimaryHit.isValid() != neighborPrimaryHit.isValid() ||
                    centralPrimaryHit.isValid() && !isValidGeometry(centralPrimarySd, neighborPrimarySd, gUsePairedNeighbors))
                    neighborValidMask[centralOffset].setValid(i, false);
                continue;
            }

            if (!restirpt.kUseAreaReservoirs &&
                            (!neighborPrimaryHit.isValid() || !isValidGeometry(centralPrimarySd, neighborPrimarySd, gUsePairedNeighbors)))
            {
                neighborValidMask[centralOffset].setValid(i, false);
                continue;
            }

            if (restirpt.kUseAreaReservoirs && restirpt.useDoFReconnectionShift && restirpt.deriveDoFReconnectionShiftProbabilityFromCoC && gUsePairedNeighbors)
            {
                // check if neighbor also uses the same shift
                float neighborDoFReconnectionShiftProbability =
                    1.f - computeConfidenceWeightForTwoShifts(params.frameDim, neighborPrimarySd.posW, !neighborPrimaryHit.isValid());
                int neighborDoFReconnectionShiftCount = round(neighborDoFReconnectionShiftProbability * neighborCount);
                if ((i < doFReconnectionShiftCount) != (i < neighborDoFReconnectionShiftCount))
                {
                    neighborValidMask[centralOffset].setValid(i, false);
                    continue;
                }
            }

            cSum += neighborReservoir.M;

            neighborValidMask[centralOffset].setValid(i, true);
            neighborValidMask[centralOffset].setCSum(cSum);

            if (ReSTIRPT::ShiftMapping(restirpt.kShiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {
                if (!gUsePairedNeighbors && centralReservoir.pathFlags.rcVertexLength() > 1)
                {
                    HitInfo rcPrevHit = HitInfo();
                    float3 rcPrevWo = float3(0.f);
                    float3 pathThroughput = 1.f;
                    float tempFootprint = 0.f;

                    if (restirpt.kUseAreaReservoirs)
                    {
                        float2 lensSample = centralReservoir.lensSample;
                        if (restirpt.useDoFReconnectionShift && i < doFReconnectionShiftCount)
                        {
                            lensSample = doFReconnectionShiftBuffer[2 * neighborCount * centralOffset + 2 * i].xy;
                        }

                        // this is neighborShiftedPrimarySd
                        neighborPrimarySd = getPixelShadingData( params, shiftedPrimaryHitBuffer, neighborPixel, 2 * neighborCount * centralOffset + 2 * i,
                            centralReservoir.filmSample, lensSample, neighborPrimaryHit);
                    }

                    pathThroughput = TraceReplayedPath( neighborPrimaryHit, neighborPrimarySd, centralReservoir.pathFlags,
                                            centralReservoir.initRandomSeed, rcPrevHit, rcPrevWo, tempFootprint, false );
                    reconnectionDataOffsets[2 * neighborCount * centralOffset + 2 * i] = 2 * neighborCount * centralOffset + 2 * i;
                    reconnectionDataBuffer[2 * neighborCount * centralOffset + 2 * i] = ReconnectionData(rcPrevHit, rcPrevWo, pathThroughput, tempFootprint);
                }

                if (neighborReservoir.pathFlags.rcVertexLength() > 1)
                {
                    uint address = gUsePairedNeighbors ? neighborCount * centralOffset + i : 2 * neighborCount * centralOffset + 2 * i + 1;

                    HitInfo rcPrevHit = HitInfo();
                    float3 rcPrevWo = float3(0.f);
                    float3 pathThroughput = 1.f;
                    float tempFootprint = 0.f;

                    if (restirpt.kUseAreaReservoirs)
                    {
                        float2 lensSample = neighborReservoir.lensSample;
                        if (restirpt.useDoFReconnectionShift && i < doFReconnectionShiftCount)
                        {
                            lensSample = doFReconnectionShiftBuffer[address].xy;
                        }

                        HitInfo centralShiftedPrimaryHit;
                        ShadingData centralShiftedPrimarySd = getPixelShadingData( params, shiftedPrimaryHitBuffer, pixel, address,
                            neighborReservoir.filmSample, lensSample, centralShiftedPrimaryHit );
                        pathThroughput = TraceReplayedPath( centralShiftedPrimaryHit, centralShiftedPrimarySd, neighborReservoir.pathFlags,
                            neighborReservoir.initRandomSeed, rcPrevHit, rcPrevWo, tempFootprint, false );
                    }
                    else
                    {
                        pathThroughput = TraceReplayedPath( centralPrimaryHit, centralPrimarySd, neighborReservoir.pathFlags,
                            neighborReservoir.initRandomSeed, rcPrevHit, rcPrevWo, tempFootprint, false );
                    }

                    reconnectionDataOffsets[address] = address;
                    reconnectionDataBuffer[address] = ReconnectionData(rcPrevHit, rcPrevWo, pathThroughput, tempFootprint);
                }
            }
        }
    }
#endif
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    void execute(const uint index)
    {
        if (index >= 2 * gNeighborCount * params.frameDim.x * params.frameDim.y) return;
        DoPass(index);
    }
#else
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        DoPass(pixel);
    }
#endif
};

cbuffer CB
{
    SpatialPathRetrace gSpatialPathRetrace;
}

#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
[numthreads(256, 1, 1)]
#else
[numthreads(16, 16, 1)]
#endif
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    gSpatialPathRetrace.execute(dispatchThreadId.x);
#else
    gSpatialPathRetrace.execute(dispatchThreadId.xy);
#endif
}
