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

#if USE_DECOUPLED_SHADING
#define EXTRA_SHIFT_ARGS outColor
#else
#define EXTRA_SHIFT_ARGS 
#endif

struct SpatialResampling
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

    StructuredBuffer<PrecomputedShiftData> precomputedShiftBuffer;

#if USE_DECOUPLED_SHADING
    RWTexture2D<float4> outputColor;
#endif

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

    /** Per-pixel spatial resampling pass, one spatial round.
    */
    void DoPass(const uint2 pixel)
    {
        // ---- Phase 1: RNG + central reservoir + primary hit ----
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId));
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = prevReservoirs[centralOffset];

        float3 centralPrimaryWo;
        float centralPrimaryRayFootprint;
        float3 centralPrimaryHitPosition;
        HitInfo centralPrimaryHit = LoadPixelCenterPrimaryHit( params, restirpt, vbuffer, pixel, centralPrimaryWo, centralPrimaryRayFootprint, centralPrimaryHitPosition);

        if (!centralPrimaryHit.isValid() && !restirpt.kUseAreaReservoirs) return;

        PathReservoir dstReservoir;
        dstReservoir.init();

        // ---- Phase 2: Spatial radius + DoF reconnection setup ----
        int neighborCount = getNeighborCount();
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        float spatialRadius = gGatherRadius;

        float doFReconnectionShiftProbability = restirpt.doFReconnectionShiftProbability;
        if (restirpt.kUseAreaReservoirs && restirpt.useDoFReconnectionShift && restirpt.deriveDoFReconnectionShiftProbabilityFromCoC)
        {
            doFReconnectionShiftProbability =
                1.f - computeConfidenceWeightForTwoShifts(params.frameDim, centralPrimaryHitPosition, !centralPrimaryHit.isValid());
        }

        int doFReconnectionShiftCount = restirpt.useDoFReconnectionShift ? round(doFReconnectionShiftProbability * neighborCount) : 0;

        int selectedId = -1;

        // ---- Phase 3: Paired-neighbor transform (reciprocal-neighbor pattern) ----
        var sharedSg = TinyUniformSampleGenerator(params.frameDim + 1, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId));
        PairedNeighborTransform transform = PairedNeighborTransform(sharedSg, params.frameDim.x, params.frameDim.y);

        // ---- Phase 4: Pairwise canonical weight init ----
        PairwiseFactor C = { neighborValidMask[centralOffset].cSum, centralReservoir.M };
        float canonicalWeight = C.c_c / C.c_sum;

#if USE_DECOUPLED_SHADING
        float3 outColor = 0.f;
#endif

        // ---- Phase 5: Per-neighbor loop ----
        for (int i = 0; i < neighborCount; ++i)
        {
            if (!neighborValidMask[centralOffset].isValid(i)) continue;

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
            uint neighborOffset = params.getReservoirOffset(neighborPixel);

            uint shiftedPrimaryHitAddress = gUsePairedNeighbors ? neighborCount * centralOffset + i : 2 * neighborCount * centralOffset + 2 * i + 1;
            uint inverseShiftedPrimaryHitAddress = gUsePairedNeighbors ? neighborCount * neighborOffset + i : 2 * neighborCount * centralOffset + 2 * i;

            // Load this neighbor's reservoir and the primary hit (shifted from the canonical sample in area ReSTIR )
            float3 neighborShiftedPrimaryWo;
            float neighborShiftedPrimaryRayFootprint;
            HitInfo neighborShiftedPrimaryHit = LoadShiftedPrimaryHit( params, restirpt, shiftedPrimaryHitBuffer, vbuffer, doFReconnectionShiftBuffer, restirpt.useDoFReconnectionShift && i < doFReconnectionShiftCount, neighborPixel,
                inverseShiftedPrimaryHitAddress, centralReservoir, neighborShiftedPrimaryWo, neighborShiftedPrimaryRayFootprint);
            PathReservoir neighborReservoir = prevReservoirs[neighborOffset];

            // Inverse shift T_{c->i}: evaluate p_hat of the canonical sample at the neighbor pixel.
            float3 canonicalAtOtherIntegrand;
            float canonicalAtOtherJacobian = 1.f;

            if (!gUsePairedNeighbors)
            {
                ReconnectionData rcData = loadReconnectionData( reconnectionDataBuffer, reconnectionDataOffsets, inverseShiftedPrimaryHitAddress, centralReservoir, neighborShiftedPrimaryHit, neighborShiftedPrimaryWo );
                canonicalAtOtherIntegrand = computeShiftedIntegrand( restirpt, params, canonicalAtOtherJacobian, centralReservoir, rcData, neighborShiftedPrimaryRayFootprint);
            }
            else
            {
                PrecomputedShiftData spData = precomputedShiftBuffer[inverseShiftedPrimaryHitAddress];
                canonicalAtOtherIntegrand = computeShiftedIntegrandWithPrecomputedInfo(canonicalAtOtherJacobian, centralReservoir, spData);
            }

            // accumulate a new term for canonical's pairwise MIS weight
            float canonicalAtOtherPHat = PathRISState::toScalar(canonicalAtOtherIntegrand) * canonicalAtOtherJacobian;
            if (restirpt.kUseAreaReservoirs && restirpt.useDoFReconnectionShift && i < doFReconnectionShiftCount)
            {
                canonicalAtOtherPHat *= doFReconnectionShiftBuffer[inverseShiftedPrimaryHitAddress].z; // Jacobian
            }

            float currentWeightTerm = 1;
            currentWeightTerm -= (canonicalAtOtherPHat > 0.f)
                                   ? canonicalAtOtherPHat * (C.c_sum - C.c_c) /
                                         (canonicalAtOtherPHat * (C.c_sum - C.c_c) + PathRISState::toScalar(centralReservoir.integrand) * C.c_c)
                : 0;

            canonicalWeight += neighborReservoir.M / C.c_sum * currentWeightTerm;

            // Shift T_{i->c}: shift the neighbor sample into the canonical's domain

            // read shifted lens coordinates in the DoF case
            float splatOrLensShiftJacobian = 1.f;

            if (restirpt.kUseAreaReservoirs && restirpt.useDoFReconnectionShift && i < doFReconnectionShiftCount)
            {
                splatOrLensShiftJacobian = doFReconnectionShiftBuffer[shiftedPrimaryHitAddress].z;
                neighborReservoir.setLensSample(doFReconnectionShiftBuffer[shiftedPrimaryHitAddress].xy);
            }

            // load primary hit shifted from the neighbor sample in the area ReSTIR case 
            if (restirpt.kUseAreaReservoirs)
            {
                centralPrimaryHit = LoadShiftedPrimaryHitAssumeAreaReSTIRAndLoadedLensSample( params, restirpt, shiftedPrimaryHitBuffer, pixel, shiftedPrimaryHitAddress, neighborReservoir, centralPrimaryWo, centralPrimaryRayFootprint);
            }

            bool selected;
            if (!gUsePairedNeighbors)
            {
                ReconnectionData rcData = loadReconnectionData( reconnectionDataBuffer, reconnectionDataOffsets, shiftedPrimaryHitAddress, neighborReservoir, centralPrimaryHit, centralPrimaryWo );
                selected = shiftAndMergeReservoirSpatial( restirpt, params, dstReservoir, neighborReservoir, rcData, centralPrimaryRayFootprint, sg, C, splatOrLensShiftJacobian, EXTRA_SHIFT_ARGS );                
            }
            else
            {
                PrecomputedShiftData spData = precomputedShiftBuffer[shiftedPrimaryHitAddress];
                selected = shiftAndMergeReservoirSpatialWithPrecomputedInfo(dstReservoir, neighborReservoir, spData, sg, C, splatOrLensShiftJacobian, EXTRA_SHIFT_ARGS);
            }

            if (selected) selectedId = i;
        }

        // ---- Phase 6: Merge canonical sample with accumulated MIS weight ----
        bool selectOwnSample = dstReservoir.merge(centralReservoir.integrand, 1.f, centralReservoir, sg, canonicalWeight);
#if USE_DECOUPLED_SHADING
        outColor += centralReservoir.integrand * centralReservoir.weight * canonicalWeight;
#endif

        // ---- Phase 7: Finalize, write back, optional decoupled shading output ----
        dstReservoir.finalizeReservoirWeight();
#if USE_DECOUPLED_SHADING
        outColor = any(isnan(outColor) || isinf(outColor)) ? 0.f : outColor;
#endif
        dstReservoir.weight = (isnan(dstReservoir.weight) || isinf(dstReservoir.weight))
            ? 0.f
            : max(0.f, dstReservoir.weight);

        if (restirpt.kUseAreaReservoirs && (restirpt.useDoFReconnectionShift || restirpt.useReservoirSplatting) && !selectOwnSample)
            sampleVBuffer[pixel] = shiftedPrimaryHitBuffer[(gUsePairedNeighbors ? 1 : 2) * neighborCount * centralOffset + (gUsePairedNeighbors ? selectedId : 2 * selectedId + 1)];
        if (dstReservoir.initRandomSeed != centralReservoir.initRandomSeed)
            sampleSurvivedFrames[pixel] = 0;

        reservoirs[centralOffset] = dstReservoir;

#if USE_DECOUPLED_SHADING
        if (gSpatialRoundId == restirpt.numSpatialRounds - 1)
        {
            if (kSamplesPerPixel == 1)
            {
                outputColor[pixel].rgb += outColor;
            }
            else
            {
                outputColor[pixel].rgb = outColor;
            }
        }
#endif
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
    SpatialResampling gSpatialResampling;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gSpatialResampling.execute(dispatchThreadId.xy);
}
