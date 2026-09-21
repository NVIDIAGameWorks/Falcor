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

struct Pass
{
    ReSTIRPathTracingParams params;
    ReSTIRPT restirpt; 
    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;
    // for default pattern

    static const int2 kPrevNeighborOffsets[8] = {
        int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0), int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1)
    };

    void DoPass(uint2 pixel, uint shiftID)
    {
        if (any(pixel >= params.frameDim)) return;
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = prevReservoirs[centralOffset];
        int2 offset2D = kPrevNeighborOffsets[shiftID];
        // get primary hit
        int2 neighborPixel = int2(pixel) + offset2D;
        if (!isValidScreenRegion(params, neighborPixel)) return;


        // shift center to neighbor
        HitInfo hit = retracePrimaryRay(neighborPixel, params.frameDim, centralReservoir.filmSample, centralReservoir.lensSample, true);
        shiftedPrimaryHitBuffer[8 * centralOffset + shiftID] = hit.data;
    }
    void execute(const uint2 pixel, uint shiftID)
    {
        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        DoPass(pixel, shiftID);
    }
};

cbuffer CB
{
    Pass gPass;
}

[numthreads(8,8,8)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadId.xy, dispatchThreadId.z);
}
