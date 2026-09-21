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
import PairedNeighborTransform;

struct Pass
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracingParams params;
    ReSTIRPT restirpt; 

    Texture2D<PackedHitInfo> vbuffer; 
    RWTexture2D<PackedHitInfo> sampleVBuffer;
    Texture1D<float2> neighborOffsets;

    StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;

    RWStructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint> reconnectionDataOffsets;
    StructuredBuffer<float4> doFReconnectionShiftBuffer;

    StructuredBuffer<NeighborValidMask> neighborValidMask;
    RWTexture2D<uint> sampleSurvivedFrames;

    RWStructuredBuffer<PrecomputedShiftData> precomputedShiftBuffer;

    int gSpatialRoundId;
    // for default pattern
    int gNeighborCount;
    float gGatherRadius;
    bool gUsePairedNeighbors;

    bool resetTemporalReservoirs;

    bool areaReservoirUseNeighborRejection;

    int getNeighborCount()
    {
        return gNeighborCount; 
    }

    void DoPass(const uint2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId));
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = prevReservoirs[centralOffset];

        float3 centralPrimaryWo;
        HitInfo centralPrimaryHit = getPixelHitInfo(params, vbuffer, pixel, centralPrimaryWo);
        VertexData centralPrimaryVertexData = getVertexData(centralPrimaryHit, gScene.camera.data.posW, -centralPrimaryWo);
        float centralPrimaryRayFootprint = restirpt.computePrimaryRayFootprint(centralPrimaryVertexData.faceNormalW, gScene.camera.data.posW,
                                                                                -centralPrimaryWo, centralPrimaryVertexData.posW);

        if (!centralPrimaryHit.isValid() && !restirpt.kUseAreaReservoirs) return;


        ReconnectionData dummyRcData;
        dummyRcData.init();

        int neighborCount = getNeighborCount();
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        float spatialRadius = gGatherRadius;

        float canonicalWeight = 1;

        float doFReconnectionShiftProbability = restirpt.doFReconnectionShiftProbability;
        if (restirpt.kUseAreaReservoirs && restirpt.useDoFReconnectionShift && restirpt.deriveDoFReconnectionShiftProbabilityFromCoC)
        {
            doFReconnectionShiftProbability =
                1.f - computeConfidenceWeightForTwoShifts(params.frameDim, centralPrimaryVertexData.posW, !centralPrimaryHit.isValid());
        }

        int doFReconnectionShiftCount = restirpt.useDoFReconnectionShift ? round(doFReconnectionShiftProbability * neighborCount) : 0;

        int selectedId = -1;

        var sharedSg = TinyUniformSampleGenerator(params.frameDim + 1, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId));
        PairedNeighborTransform transform = PairedNeighborTransform(sharedSg, params.frameDim.x, params.frameDim.y);

        for (int i = 0; i < neighborCount; ++i)
        {
            if (!neighborValidMask[centralOffset].isValid(i)) continue;

            int dx, dy;
            transform.readTransformedDelta(i, pixel.x, pixel.y, dx, dy);
            int2 neighborPixel = pixel + int2(dx, dy);

            uint shiftedPrimaryHitAddress = neighborCount * centralOffset + i;

            uint neighborOffset = params.getReservoirOffset(neighborPixel);
            PathReservoir neighborReservoir = prevReservoirs[neighborOffset];

            if (restirpt.kUseAreaReservoirs && restirpt.useDoFReconnectionShift && i < doFReconnectionShiftCount)
            {
                neighborReservoir.setLensSample(doFReconnectionShiftBuffer[shiftedPrimaryHitAddress].xy);
            }

            // overwrite as centralShiftedPrimaryHit
            if (restirpt.kUseAreaReservoirs)
            {
                centralPrimaryHit = LoadShiftedPrimaryHitAssumeAreaReSTIRAndLoadedLensSample( params, restirpt, shiftedPrimaryHitBuffer,
                    pixel, shiftedPrimaryHitAddress, neighborReservoir, centralPrimaryWo, centralPrimaryRayFootprint );
            }

            ReconnectionData rcData = loadReconnectionData(reconnectionDataBuffer, reconnectionDataOffsets, shiftedPrimaryHitAddress, neighborReservoir, centralPrimaryHit, centralPrimaryWo );

            float Jacobian = 1.f;
            float3 dstIntegrand = shiftPath( restirpt, params, Jacobian,
                                    neighborReservoir, rcData, centralPrimaryRayFootprint);

            PrecomputedShiftData spData = { dstIntegrand, Jacobian, neighborReservoir.reconnectionJacobian };
            precomputedShiftBuffer[shiftedPrimaryHitAddress] = spData;
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
    Pass gPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadId.xy);
}
