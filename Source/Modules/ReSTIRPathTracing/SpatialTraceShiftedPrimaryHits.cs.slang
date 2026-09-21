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
    Texture2D<PackedHitInfo> sampleVBuffer;
    Texture1D<float2> neighborOffsets;

    RWStructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;

    RWStructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;
    RWStructuredBuffer<float4> doFReconnectionShiftBuffer;

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

        HitInfo centralPrimaryCenterHit;
        ShadingData centralPrimaryCenterSd = getPixelShadingData(params, vbuffer, pixel, centralPrimaryCenterHit);

        int neighborCount = getNeighborCount();
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        float spatialRadius = gGatherRadius;

        // compute self Jacobian term
        PackedHitInfo centerPixelX1Hit = sampleVBuffer[pixel];
        Ray centerRay = gScene.camera.computeRayThinlensSubPixelRandom( pixel, params.frameDim, centralReservoir.filmSample, centralReservoir.lensSample );
        float centerJ = 0.f;
        if (restirpt.useDoFReconnectionShift)
            centerJ = computePrimaryReconnectionJacobian( pixel, params.frameDim, HitInfo(centerPixelX1Hit), centerRay.dir, centralReservoir.filmSample, false);

        float doFReconnectionShiftProbability = restirpt.deriveDoFReconnectionShiftProbabilityFromCoC ?
        1.f - computeConfidenceWeightForTwoShifts(params.frameDim, centralPrimaryCenterSd.posW, !centralPrimaryCenterHit.isValid()) :
        restirpt.doFReconnectionShiftProbability;

        int doFReconnectionShiftCount = round(doFReconnectionShiftProbability * neighborCount);
        var sharedSg = TinyUniformSampleGenerator(params.frameDim + 1, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId));
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

            if (!isValidScreenRegion(params, neighborPixel)) continue;

            PathReservoir neighborReservoir = prevReservoirs[params.getReservoirOffset(neighborPixel)];

            if (restirpt.useDoFReconnectionShift && i < doFReconnectionShiftCount)
            {
                PackedHitInfo neighborPixelX1Hit = sampleVBuffer[neighborPixel];
                Ray neighborRay = gScene.camera.computeRayThinlensSubPixelRandom( neighborPixel, params.frameDim, neighborReservoir.filmSample, neighborReservoir.lensSample );
                float3 X0_dstJ = solveLensSampleForPrimaryHit( pixel, params.frameDim, HitInfo(neighborPixelX1Hit), neighborRay.dir, neighborReservoir.filmSample, false );
                float srcJ = computePrimaryReconnectionJacobian( neighborPixel, params.frameDim, HitInfo(neighborPixelX1Hit), neighborRay.dir, neighborReservoir.filmSample, false );
                X0_dstJ.z /= srcJ;
                if (isinf(X0_dstJ.z) || isnan(X0_dstJ.z)) X0_dstJ.z = 0.f;
                doFReconnectionShiftBuffer[(gUsePairedNeighbors ? 1 : 2) * neighborCount * centralOffset + (gUsePairedNeighbors ? i : 2 * i + 1)] = float4(X0_dstJ, 0.f);
                shiftedPrimaryHitBuffer[(gUsePairedNeighbors ? 1 : 2) * neighborCount * centralOffset + (gUsePairedNeighbors ? i : 2 * i + 1)] = neighborPixelX1Hit;
            }
            else
            {
                // shift neighbor to center
                HitInfo hit = retracePrimaryRay(pixel, params.frameDim, neighborReservoir.filmSample, neighborReservoir.lensSample, false);
                shiftedPrimaryHitBuffer[(gUsePairedNeighbors ? 1 : 2) * neighborCount * centralOffset + (gUsePairedNeighbors ? i : 2 * i + 1)] = hit.data;
            }

            if (!gUsePairedNeighbors)
            {
                if (restirpt.useDoFReconnectionShift && i < doFReconnectionShiftCount)
                {
                    float3 X0_dstJ = solveLensSampleForPrimaryHit( neighborPixel, params.frameDim, HitInfo(centerPixelX1Hit), centerRay.dir, centralReservoir.filmSample, false );
                    X0_dstJ.z /= centerJ;
                    if (isinf(X0_dstJ.z) || isnan(X0_dstJ.z)) X0_dstJ.z = 0.f;
                    doFReconnectionShiftBuffer[2 * neighborCount * centralOffset + 2 * i] = float4(X0_dstJ, 0.f);
                    shiftedPrimaryHitBuffer[2 * neighborCount * centralOffset + 2 * i] = centerPixelX1Hit;
                }
                else
                {
                    // shift center to neighbor
                    HitInfo hit = retracePrimaryRay(neighborPixel, params.frameDim, centralReservoir.filmSample, centralReservoir.lensSample, false);
                    shiftedPrimaryHitBuffer[2 * neighborCount * centralOffset + 2 * i] = hit.data;
                }
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
    Pass gPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadId.xy);
}
