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
#include "RetraceScheduleDefinition.slangh"


struct Pass
{
    ReSTIRPathTracingParams params; 
    ReSTIRPT restirpt;             

    StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;

    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    RWStructuredBuffer<uint> reconnectionDataOffsets;

    RetraceWorkQueue queues; 

    static const int2 kPrevNeighborOffsets[8] = {
        int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0), int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1)
    };


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

        
        const uint centralOffset = params.getReservoirOffset(pixel);

        int2 neighborPixel = pixel + kPrevNeighborOffsets[retraceIndex];
        const uint neighborOffset = params.getReservoirOffset(neighborPixel);

        PathReservoir reservoir = prevReservoirs[centralOffset];


        HitInfo primaryHit;
        ShadingData primarySd = getPixelTemporalShadingData( params, shiftedPrimaryHitBuffer, neighborPixel,
            8 * centralOffset + retraceIndex, reservoir.filmSample, reservoir.lensSample, primaryHit );


        HitInfo rcPrevHit = HitInfo();
        float3 rcPrevWo = float3(0);
        float tempFootprint = 0.f;
        float3 pathThroughput = TraceReplayedPath( primaryHit, primarySd, reservoir.pathFlags,
                                       reservoir.initRandomSeed, rcPrevHit, rcPrevWo, tempFootprint, true );

        reconnectionDataOffsets[8 * centralOffset + retraceIndex] = linearIndex;
        reconnectionDataBuffer[linearIndex] = ReconnectionData(rcPrevHit, rcPrevWo, pathThroughput, tempFootprint);
    }
    void execute(const uint index)
    {
        if (index >= 8 * params.frameDim.x * params.frameDim.y) return;
        DoPass(index);
    }
};

cbuffer CB
{
    Pass gPass;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadId.x);
}
