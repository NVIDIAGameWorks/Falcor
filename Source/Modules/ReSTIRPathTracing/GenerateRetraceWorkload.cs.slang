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
import PairedNeighborTransform;
import Utils.Debug.PixelDebug;

static const uint kWarpCount = (kScreenTileDim.x * kScreenTileDim.y) / 32;
groupshared uint gRetraceWorkOffset[8 /* kWarpCount */];

struct RetraceWorkloadGenerator
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracingParams params;
    ReSTIRPT restirpt; 

    Texture2D<PackedHitInfo> vbuffer; 
    Texture2D<PackedHitInfo> temporalVbuffer;
    Texture1D<float2> neighborOffsets;
    Texture2D<float2> motionVectors;

    StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;

    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<PathReservoir> reservoirs;

    RWStructuredBuffer<NeighborValidMask> neighborValidMask;
    StructuredBuffer<float4> splatSampleInfo;
    bool gEnableTemporalReprojection;
    bool gNoResamplingForTemporalReuse;
    uint gTemporalHistoryLength;

    int gSpatialRoundId;
    int gNeighborCount;
    float gGatherRadius;

    bool isSpatialPass;
    bool gStochasticReprojection;
    bool resetTemporalReservoirs;
    bool areaReservoirUseNeighborRejection;
    bool gUsePairedNeighbors;

    int temporalPassId;

    RetraceWorkQueue queues;

    int getNeighborCount()
    {
        return isSpatialPass ? gNeighborCount : 1; 
    }

    void execute(const uint2 tileID, const uint threadIdx)
    {
        // Map thread to pixel based on Morton order within tile.
        // The tiles themselves are enumerated in scanline order on screen.
        const uint2 tileOffset = tileID << kScreenTileBits;            // Tile offset in pixels.
        const uint2 pixel = deinterleave_8bit(threadIdx) + tileOffset; // Assumes 16x16 tile or smaller. A host-side assert checks this assumption.

        uint retraceWorkItemCount = 0;
        uint retraceCandidateMask = 0;

        int neighborCount = getNeighborCount();

        // Note: Do not terminate threads for out-of-bounds pixels because we need all threads active for the prefix sum pass below.
        if (all(pixel < params.frameDim))
        {
            var sg = TinyUniformSampleGenerator( pixel, restirpt.sgCount() * params.seed + restirpt.spatialOrTemporalSgOffset(isSpatialPass, gSpatialRoundId, temporalPassId));
            const uint centralOffset = params.getReservoirOffset(pixel);

            HitInfo centralPrimaryHit;
            ShadingData centralPrimarySd = getPixelShadingData(params, vbuffer, pixel, centralPrimaryHit);

            if (centralPrimaryHit.isValid() || restirpt.kUseAreaReservoirs)
            {
                if (isSpatialPass)
                {
                    float spatialRadius = gGatherRadius;

                    PathReservoir centralReservoir = prevReservoirs[centralOffset];

                    float cSum = centralReservoir.M;

                    float doFReconnectionShiftProbability = restirpt.doFReconnectionShiftProbability;
                    if (restirpt.kUseAreaReservoirs && restirpt.useDoFReconnectionShift && restirpt.deriveDoFReconnectionShiftProbabilityFromCoC && gUsePairedNeighbors)
                    {
                        doFReconnectionShiftProbability =
                            1.f - computeConfidenceWeightForTwoShifts(params.frameDim, centralPrimarySd.posW, !centralPrimaryHit.isValid() );
                    }

                    int doFReconnectionShiftCount = restirpt.useDoFReconnectionShift ? round(doFReconnectionShiftProbability * neighborCount) : 0;

                    const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
                    var sharedSg = TinyUniformSampleGenerator( params.frameDim + 1, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId) );
                    PairedNeighborTransform transform = PairedNeighborTransform(sharedSg, params.frameDim.x, params.frameDim.y);

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
                            {
                                neighborValidMask[centralOffset].setValid(i, false);
                                continue;
                            }
                        }

                        if (!restirpt.kUseAreaReservoirs &&
                            (!neighborPrimaryHit.isValid() || !isValidGeometry(centralPrimarySd, neighborPrimarySd, gUsePairedNeighbors)))
                        {
                            neighborValidMask[centralOffset].setValid(i, false);
                            continue;
                        }

                        if (restirpt.kUseAreaReservoirs && restirpt.useDoFReconnectionShift && restirpt.deriveDoFReconnectionShiftProbabilityFromCoC && gUsePairedNeighbors)
                        {
                            float neighborDoFReconnectionShiftProbability =
                                1.f - computeConfidenceWeightForTwoShifts(params.frameDim, neighborPrimarySd.posW, !neighborPrimaryHit.isValid() );
                            int neighborDoFReconnectionShiftCount = round(neighborDoFReconnectionShiftProbability * neighborCount);
                            if ((i < doFReconnectionShiftCount) != (i < neighborDoFReconnectionShiftCount))
                            {
                                neighborValidMask[centralOffset].setValid(i, false);
                                continue;
                            }
                        }

                        neighborValidMask[centralOffset].setValid(i, true);

                        cSum += neighborReservoir.M;

                        if (ReSTIRPT::ShiftMapping(restirpt.kShiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
                        {
                            if (!gUsePairedNeighbors)
                            {
                                int p1 = int(centralReservoir.pathFlags.rcVertexLength() > 1);
                                if (restirpt.kUseAreaReservoirs) p1 = HitInfo(shiftedPrimaryHitBuffer[2 * neighborCount * centralOffset + 2 * i]).isValid() ? p1 : 0;
                                retraceWorkItemCount += p1;
                                retraceCandidateMask |= p1 << (2 * i);
                            }
                            int p2 = int(neighborReservoir.pathFlags.rcVertexLength() > 1);
                            if (restirpt.kUseAreaReservoirs) p2 = HitInfo(shiftedPrimaryHitBuffer[(gUsePairedNeighbors ? 1 : 2) * neighborCount * centralOffset +
                                                                                            (gUsePairedNeighbors ? i : 2 * i + 1)]).isValid() ? p2 : 0;
                            retraceWorkItemCount += p2;
                            retraceCandidateMask |= p2 << (2 * i + 1);
                        }
                    }

                    neighborValidMask[centralOffset].setCSum(cSum);
                }
                else
                {

                    PathReservoir centralReservoir = reservoirs[centralOffset];

                    float2 motionVector = gEnableTemporalReprojection ? motionVectors[pixel] : 0.f;
                    float2 prevPixel = reprojectPixelForReuse(pixel, motionVector, params.frameDim,
                        restirpt.useAreaReSTIRBackprojection, gStochasticReprojection, sg);
                    bool isValid = true;

                    if (!restirpt.useReservoirSplatting)
                    {
                        if (!restirpt.kUseAreaReservoirs)
                        {
                            HitInfo temporalPrimaryHit = getPixelTemporalHitInfo(params, temporalVbuffer, prevPixel);
                            isValid = temporalPrimaryHit.isValid();
                            neighborValidMask[centralOffset].setValid(0, isValid);
                        }
                        else
                        {
                            if ((!restirpt.useAreaReSTIRBackprojection && !isValidScreenRegion(params, int2(prevPixel)) ||
                                restirpt.useAreaReSTIRBackprojection && !isValidScreenRegionFloat(params, prevPixel))) isValid = false;
                        }
                    }

                    if (ReSTIRPT::ShiftMapping(restirpt.kShiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
                    {
                        PathReservoir temporalReservoir = prevReservoirs[params.getReservoirOffset(restirpt.useAreaReSTIRBackprojection ? pixel : int2(prevPixel))];

                        bool isValid1 = isValid;
                        bool isValid2 = isValid;
                        if (restirpt.useReservoirSplatting)
                        {
                            isValid1 = splatSampleInfo[2 * centralOffset].z > 0.f; // Jacobian > 0
                            isValid2 = splatSampleInfo[2 * centralOffset + 1].z > 0.f;
                        }

                        int p1 = int(centralReservoir.pathFlags.rcVertexLength() > 1) && isValid1;
                        int p2 = int(temporalReservoir.pathFlags.rcVertexLength() > 1) && isValid2;

                        if (temporalPassId == 0)
                        {
                            retraceWorkItemCount += p1;
                            retraceCandidateMask |= p1;
                            retraceWorkItemCount += p2;
                            retraceCandidateMask |= p2 << 1;
                        }
                        else
                        {
                            float doFReconnectionShiftProbability = restirpt.deriveDoFReconnectionShiftProbabilityFromCoC ?
                            1.f - computeConfidenceWeightForTwoShifts( params.frameDim, centralPrimarySd.posW, !centralPrimaryHit.isValid() ) :
                            restirpt.doFReconnectionShiftProbability;
                            doFReconnectionShiftProbability = restirpt.useDoFReconnectionForTemporalReuse ? doFReconnectionShiftProbability : 0.f;
                            bool cameraChanged = hasCameraChanged();
                            if (isBorderPixel(params, pixel) && !cameraChanged) doFReconnectionShiftProbability = 0.f;

                            if (restirpt.useDoFReconnectionShift && doFReconnectionShiftProbability > 0.f)
                            {
                                retraceWorkItemCount += p1;
                                retraceCandidateMask |= p1;

                                retraceWorkItemCount += p2;
                                retraceCandidateMask |= p2 << 1;
                            }
                        }
                    }
                }
            }
        }

        const uint warpIdx = threadIdx >> 5;

        uint retraceWorkItems = WaveActiveSum(retraceWorkItemCount);

        if (WaveIsFirstLane())
        {
            gRetraceWorkOffset[warpIdx] = retraceWorkItems;
        }
        GroupMemoryBarrierWithGroupSync();

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
            for (uint i = 0; i < 2 * neighborCount; i++)
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
    RetraceWorkloadGenerator gRetraceWorkloadGenerator;
}

[numthreads(256, 1, 1)]
void main(
    uint3 groupID: SV_GroupID,
    uint3 groupThreadID: SV_GroupThreadID)
{
    gRetraceWorkloadGenerator.execute(groupID.xy, groupThreadID.x);
}
