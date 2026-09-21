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
import LoadShadingDataDynamic;
import HybridShiftTraceRays;
import RetraceWorkQueue;
import ResamplingCommon;
#include "RetraceScheduleDefinition.slangh"
struct TemporalPathRetrace
{
    ReSTIRPathTracingParams params; 
    ReSTIRPT restirpt;             

    Texture2D<PackedHitInfo> vbuffer;                     
    Texture2D<PackedHitInfo> temporalVbuffer;
    Texture2D<float2> motionVectors;

    StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;

    StructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint2> keyValueBuffer;
    StructuredBuffer<uint> numKeysBuffer;
    RWStructuredBuffer<NeighborValidMask> neighborValidMask;
    RWStructuredBuffer<uint> reconnectionDataOffsets;
    StructuredBuffer<float4> doFReconnectionShiftBuffer;
    StructuredBuffer<float4> splatSampleInfo;
    StructuredBuffer<uint2> splatTargetPixel;

    bool gEnableTemporalReprojection;
    uint gTemporalHistoryLength;
    bool gStochasticReprojection;
    int temporalPassId;

    RetraceWorkQueue queues; 

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

        
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.temporalSgOffset(temporalPassId));
        const uint centralOffset = params.getReservoirOffset(pixel);
        float2 motionVector = gEnableTemporalReprojection ? motionVectors[pixel] : 0.f;
        float2 prevPixel = reprojectPixelForReuse(pixel, motionVector, params.frameDim, restirpt.useAreaReSTIRBackprojection, gStochasticReprojection, sg);
        const uint temporalOffset = params.getReservoirOffset(restirpt.useAreaReSTIRBackprojection ? pixel : int2(prevPixel));



        PathReservoir reservoir;
        if (retraceIndex % 2)
        {
            reservoir = prevReservoirs[temporalOffset];
            if (restirpt.useReservoirSplatting)
            {
                reservoir.setFilmSample(splatSampleInfo[2 * temporalOffset + 1].xy);
                pixel = splatTargetPixel[2 * temporalOffset + 1];
            }
        }
        else
        {
            reservoir = reservoirs[centralOffset];
            if (restirpt.useReservoirSplatting)
            {
                reservoir.setFilmSample(splatSampleInfo[2 * centralOffset].xy);
                prevPixel = splatTargetPixel[2 * centralOffset];
            }
        }

        if (restirpt.kUseAreaReservoirs && temporalPassId == 1) reservoir.setLensSample(doFReconnectionShiftBuffer[2 * centralOffset + retraceIndex].xy);

        HitInfo primaryHit;
        ShadingData primarySd;
        if (restirpt.kUseAreaReservoirs)
            primarySd = getPixelShadingDataGeneral( params, shiftedPrimaryHitBuffer, retraceIndex % 2 ? pixel : prevPixel, 2 * centralOffset + retraceIndex,
                reservoir.filmSample, reservoir.lensSample, primaryHit, retraceIndex % 2 == 0 );
        else
            primarySd = getPixelShadingDataGeneral(params, vbuffer, temporalVbuffer, retraceIndex % 2 ? pixel : prevPixel, primaryHit, retraceIndex % 2 == 0);

        HitInfo rcPrevHit = HitInfo();
        float3 rcPrevWo = float3(0);
        float tempFootprint = 0.f;
        float3 pathThroughput = TraceReplayedPath( primaryHit, primarySd, reservoir.pathFlags,
                                       reservoir.initRandomSeed, rcPrevHit, rcPrevWo, tempFootprint, retraceIndex % 2 == 0 );

        reconnectionDataOffsets[2 * centralOffset + retraceIndex] = linearIndex;
        reconnectionDataBuffer[linearIndex] = ReconnectionData(rcPrevHit, rcPrevWo, pathThroughput, tempFootprint);
    }
#else
    bool DoPass(uint2 pixel)
    {
        
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.temporalSgOffset(temporalPassId));
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = reservoirs[centralOffset];
        if (restirpt.useReservoirSplatting) centralReservoir.setFilmSample(splatSampleInfo[2 * centralOffset].xy);
        float2 motionVector = gEnableTemporalReprojection ? motionVectors[pixel] : 0.f;
        float2 prevPixel = reprojectPixelForReuse(pixel, motionVector, params.frameDim, restirpt.useAreaReSTIRBackprojection, gStochasticReprojection, sg);

        bool shouldReturnEarly =  !restirpt.useReservoirSplatting && (
            !restirpt.useAreaReSTIRBackprojection && !isValidScreenRegion(params, int2(prevPixel)) ||
             restirpt.useAreaReSTIRBackprojection && !isValidScreenRegionFloat(params, prevPixel));
        if (shouldReturnEarly) return false;

        int2 fetchPixel = restirpt.useAreaReSTIRBackprojection ? pixel : int2(prevPixel);
        uint prevOffset = params.getReservoirOffset(fetchPixel);
        PathReservoir temporalReservoir = prevReservoirs[prevOffset];
        if (restirpt.useReservoirSplatting) temporalReservoir.setFilmSample(splatSampleInfo[2 * prevOffset + 1].xy);

        if (restirpt.useReservoirSplatting)
        {
            pixel = splatTargetPixel[2 * prevOffset + 1];
            prevPixel = splatTargetPixel[2 * centralOffset];
        }

        HitInfo centralShiftedPrimaryHit;
        ShadingData centralShiftedPrimarySd;
        if (restirpt.kUseAreaReservoirs)
            centralShiftedPrimarySd = getPixelShadingData( params, shiftedPrimaryHitBuffer, pixel, 2 * centralOffset + 1, temporalReservoir.filmSample, temporalReservoir.lensSample, centralShiftedPrimaryHit );
        else
            centralShiftedPrimarySd = getPixelShadingData(params, vbuffer, pixel, centralShiftedPrimaryHit);
        if (!centralShiftedPrimaryHit.isValid() && !restirpt.kUseAreaReservoirs) return false;

        HitInfo temporalShiftedPrimaryHit;
        ShadingData temporalShiftedPrimarySd;
        if (restirpt.kUseAreaReservoirs)
            temporalShiftedPrimarySd = getPixelTemporalShadingData( params, shiftedPrimaryHitBuffer, prevPixel, 2 * centralOffset, centralReservoir.filmSample, centralReservoir.lensSample, temporalShiftedPrimaryHit);
        else
            temporalShiftedPrimarySd = getPixelTemporalShadingData(params, temporalVbuffer, prevPixel, temporalShiftedPrimaryHit);
        if (!restirpt.useAreaReSTIRBackprojection && !temporalShiftedPrimaryHit.isValid() && !restirpt.kUseAreaReservoirs) return false;

        if (ReSTIRPT::ShiftMapping(restirpt.kShiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
        {
            // talbot MIS
            // compute mis weight for current pixel

            // write to memory
            if (centralReservoir.pathFlags.rcVertexLength() > 1 && temporalShiftedPrimaryHit.isValid())
            {
                HitInfo rcPrevHit = HitInfo();
                float3 rcPrevWo = float3(0.f);
                float3 pathThroughput = 1.f;
                float tempFootprint = 0.f;
                pathThroughput = TraceReplayedPath( temporalShiftedPrimaryHit, temporalShiftedPrimarySd, centralReservoir.pathFlags,
                                        centralReservoir.initRandomSeed, rcPrevHit, rcPrevWo, tempFootprint, true );
                reconnectionDataOffsets[2 * centralOffset] = 2 * centralOffset;
                reconnectionDataBuffer[2 *centralOffset] = ReconnectionData(rcPrevHit, rcPrevWo, pathThroughput, tempFootprint);
            }

            if (temporalReservoir.pathFlags.rcVertexLength() > 1 && centralShiftedPrimaryHit.isValid())
            {
                HitInfo rcPrevHit = HitInfo();
                float3 rcPrevWo = float3(0.f);
                float3 pathThroughput = 1.f;
                float tempFootprint = 0.f;
                pathThroughput = TraceReplayedPath( centralShiftedPrimaryHit, centralShiftedPrimarySd, temporalReservoir.pathFlags,
                                        temporalReservoir.initRandomSeed, rcPrevHit, rcPrevWo, tempFootprint, false );
                reconnectionDataOffsets[2 * centralOffset + 1] = 2 * centralOffset + 1;
                reconnectionDataBuffer[2 * centralOffset + 1] = ReconnectionData(rcPrevHit, rcPrevWo, pathThroughput, tempFootprint);
            }
        }
        return true;
    }
#endif
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    void execute(const uint index)
    {
        if (index >= 2 * params.frameDim.x * params.frameDim.y) return;
        DoPass(index);
    }
#else
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool isValid = DoPass(pixel);
        const uint centralOffset = params.getReservoirOffset(pixel);
        neighborValidMask[centralOffset].setValid(0, isValid);
    }
#endif
};

cbuffer CB
{
    TemporalPathRetrace gTemporalPathRetrace;
}

#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
[numthreads(256, 1, 1)]
#else
[numthreads(16, 16, 1)]
#endif
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    gTemporalPathRetrace.execute(dispatchThreadId.x);
#else
    gTemporalPathRetrace.execute(dispatchThreadId.xy);
#endif
}
