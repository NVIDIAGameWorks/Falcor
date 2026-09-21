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
import Utils.Attributes;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import ReSTIRPathTracing.ReSTIRPT;
import LoadShadingDataDynamic;
import RetraceWorkQueue;
import Scene.HitInfo;
import PathReservoir;
import Utils.Sampling.TinyUniformSampleGenerator;
import ResamplingCommon;

static const uint kWarpCount = (kScreenTileDim.x * kScreenTileDim.y) / 32;
groupshared uint gRetraceWorkOffset[8 /* kWarpCount */];

struct Pass
{
    ReSTIRPathTracingParams params;
    ReSTIRPT restirpt; 

    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;

    RetraceWorkQueue queues;

    static const int2 kPrevNeighborOffsets[8] = {
        int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0), int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1)
    };

    void execute(const uint2 tileID, const uint threadIdx)
    {
        // Map thread to pixel based on Morton order within tile.
        // The tiles themselves are enumerated in scanline order on screen.
        const uint2 tileOffset = tileID << kScreenTileBits;            // Tile offset in pixels.
        const uint2 pixel = deinterleave_8bit(threadIdx) + tileOffset; // Assumes 16x16 tile or smaller. A host-side assert checks this assumption.

        uint retraceWorkItemCount = 0;
        uint retraceCandidateMask = 0;

        // Note: Do not terminate threads for out-of-bounds pixels because we need all threads active for the prefix sum pass below.
        if (all(pixel < params.frameDim))
        {
            
            const uint centralOffset = params.getReservoirOffset(pixel);

            {
                PathReservoir centralReservoir = prevReservoirs[centralOffset];

                for (int i = 0; i < 8; ++i)
                {
                    if (ReSTIRPT::ShiftMapping(restirpt.kShiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
                    {
                        int2 neighborPixel = int2(pixel) + kPrevNeighborOffsets[i];
                        if (!isValidScreenRegion(params, neighborPixel))
                        {
                            continue;
                        }
                        int p1 = int(centralReservoir.pathFlags.rcVertexLength() > 1);
                        retraceWorkItemCount += p1;
                        retraceCandidateMask |= p1 << i;
                    }
                }
            }
        }

        // Perform a reduction over the tile to determine the number of paths
        // to append to each queue. This is done via wave ops and shared memory.
        // The write offsets are given by prefix sums over the threads.
        const uint warpIdx = threadIdx >> 5;

        // Calculate the path and sample counts over the warp.
        // The first thread in each warp writes the results to shared memory.
        uint retraceWorkItems = WaveActiveSum(retraceWorkItemCount);

        if (WaveIsFirstLane())
        {
            gRetraceWorkOffset[warpIdx] = retraceWorkItems;
        }
        GroupMemoryBarrierWithGroupSync();

        // Compute the prefix sum over the warp totals in shared memory.
        // The first N threads in the thread group perform this computation.
        if (threadIdx < kWarpCount)
        {
            uint retraceWorkItems = gRetraceWorkOffset[threadIdx];
            gRetraceWorkOffset[threadIdx] = queues.addCounter(retraceWorkItems);
        }
        GroupMemoryBarrierWithGroupSync();

        if (all(pixel < params.frameDim))
        {
            uint encodedRetraceWorkItem = (pixel.y << 12) | pixel.x;
            uint dstIdx = gRetraceWorkOffset[warpIdx] + WavePrefixSum(retraceWorkItemCount);
            int j = 0;
            for (uint i = 0; i < 8; i++)
            {
                if (retraceCandidateMask >> i & 1)
                {
                    queues.retraceWorkItems.Store((dstIdx + j) * 4, encodedRetraceWorkItem + (i << 24));
                    j++;
                }
            }
        }
    }
}

cbuffer CB
{
    Pass gPass;
}

[numthreads(256 /* kScreenTileDim.x * kScreenTileDim.y */, 1, 1)]
void main(
    uint3 groupID: SV_GroupID,
    uint3 groupThreadID: SV_GroupThreadID)
{
    gPass.execute(groupID.xy, groupThreadID.x);
}
