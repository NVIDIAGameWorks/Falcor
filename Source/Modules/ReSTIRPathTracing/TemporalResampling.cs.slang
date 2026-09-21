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
import ResamplingCommon;

struct TemporalResampling
{
    ReSTIRPathTracingParams params;
    ReSTIRPT restirpt; 

    RWTexture2D<PackedHitInfo> vbuffer;                     
    Texture2D<PackedHitInfo> temporalVbuffer;
    StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;
    Texture2D<float2> motionVectors;
    Texture2D<uint> duplicationMap;

    Texture2D<uint> prevSampleSurvivedFrames;
    RWTexture2D<uint> sampleSurvivedFrames;

    RWStructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint> reconnectionDataOffsets;
    StructuredBuffer<NeighborValidMask> neighborValidMask;
    StructuredBuffer<float4> doFReconnectionShiftBuffer;

    Texture2D<uint> splatCellCounts;
    Texture2D<uint> splatCellOffsets;
    StructuredBuffer<uint2> sortedSplatSourcePixels;
    StructuredBuffer<float4> splatSampleInfo;
    StructuredBuffer<uint2> splatTargetPixel;

    bool gEnableTemporalReprojection;

    uint gTemporalHistoryLength;
    bool gStochasticReprojection;
    float duplicationMCapPower;
    bool enableDupMapMCapDownweighting;
    bool computeTemporalDuplication;

    float gGatherRadius;

    int temporalPassId;


    void writeTemporalOutputs(const uint2 pixel, float temporalSampleAge)
    {
        if (temporalPassId != 0) return;
        if (computeTemporalDuplication)
            sampleSurvivedFrames[pixel] = temporalSampleAge;
    }


    /** Per-pixel temporal resampling pass.
    */
    bool DoPass(const uint2 pixel)
    {
        // ---- Phase 1: RNG + central reservoir ----
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.temporalSgOffset(temporalPassId));

        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = reservoirs[centralOffset];
        uint vBufferWritingOffset = centralOffset;

        PathReservoir dstReservoir;
        dstReservoir.init();

        // ---- Phase 2: Motion-vector reprojection + validity early-out ----
        float2 motionVector = gEnableTemporalReprojection ? motionVectors[pixel] : 0.f;
        float2 prevPixel = reprojectPixelForReuse(pixel, motionVector, params.frameDim,
            restirpt.useAreaReSTIRBackprojection, gStochasticReprojection, sg);

        bool shouldReturnEarly = !restirpt.useReservoirSplatting && (
            !restirpt.useAreaReSTIRBackprojection && !isValidScreenRegion(params, int2(prevPixel)) ||
             restirpt.useAreaReSTIRBackprojection && !isValidScreenRegionFloat(params, prevPixel));

        if (shouldReturnEarly)
        {
            return false; 
        }

        const bool useTemporalDoFReconnectionShift = restirpt.kUseAreaReservoirs && restirpt.useDoFReconnectionShift && restirpt.useDoFReconnectionForTemporalReuse;
        const bool isAdditionalPassForDofShift = useTemporalDoFReconnectionShift && temporalPassId == 1;

        // ---- Phase 3: Temporal reservoir lookup ----
        uint prevOffset = 0;
        PathReservoir temporalReservoir;
        temporalReservoir.M = 0;

        int2 prevPixelForDupMap = reprojectPixelForDupMap(pixel, motionVector, params.frameDim,
            restirpt.useAreaReSTIRBackprojection, gStochasticReprojection, sg);

        if (!restirpt.useReservoirSplatting)
        {
            int2 fetchPixel = restirpt.useAreaReSTIRBackprojection ? pixel : int2(prevPixel);
            prevOffset = params.getReservoirOffset(fetchPixel);
            temporalReservoir = prevReservoirs[prevOffset];
        }
        else
        {
            if (isValidScreenRegion(params, prevPixelForDupMap)) temporalReservoir.M = prevReservoirs[params.getReservoirOffset(prevPixelForDupMap)].M;
            else temporalReservoir.M = 0;
            prevPixelForDupMap = min(params.frameDim - 1, max(int2(0, 0), prevPixelForDupMap));
        }

        float2 shiftToPrevfilmSample = centralReservoir.filmSample;
        if (restirpt.useReservoirSplatting)
        {
            prevPixel = splatTargetPixel[2 * centralOffset];
            shiftToPrevfilmSample = splatSampleInfo[2 * centralOffset].xy;
        }

        static const bool doTemporalUpdate = restirpt.kTemporalUpdateForDynamicScene;
        static const bool doInverseTemporalUpdate = restirpt.kUsePrevFrameSceneData ? restirpt.kTemporalUpdateForDynamicScene : false;

        // ---- Phase 4: Duplication-driven MCap ----
        float MCap = gTemporalHistoryLength;
        DetermineMCap( MCap, duplicationMap,
            prevPixelForDupMap, enableDupMapMCapDownweighting,
            duplicationMCapPower
        );
        temporalReservoir.M = min(MCap, temporalReservoir.M);

        if (!neighborValidMask[centralOffset].isValid(0) && !restirpt.kUseAreaReservoirs) return false;

        bool selectNewSample = false;

        // ---- Phase 5: DoF-reconnection MIS pre-scaling ----
        if (useTemporalDoFReconnectionShift)
        {
            float doFReconnectionShiftProbability = doFReconnectionShiftBuffer[2 * centralOffset].w;
            float2 misScales = float2(1.f - doFReconnectionShiftProbability, doFReconnectionShiftProbability);
            temporalReservoir.M *= misScales[temporalPassId];
            if (temporalReservoir.M == 0.f) return false;
        }

        // ---- Phase 6: Inverse-shift T_{c->t} ----
        // We need p_hat of the canonical sample evaluated at the temporal pixel to compute canonical's MIS weight 
        float3 temporalShiftedPrimaryWo;
        float temporalShiftedPrimaryRayFootprint;
        HitInfo temporalShiftedPrimaryHit = LoadShiftedPrimaryHitPrevFrame( params, restirpt, shiftedPrimaryHitBuffer, temporalVbuffer, doFReconnectionShiftBuffer, isAdditionalPassForDofShift,
            prevPixel, 2 * centralOffset, centralReservoir, shiftToPrevfilmSample, temporalShiftedPrimaryWo, temporalShiftedPrimaryRayFootprint);

        ReconnectionData rcData = loadReconnectionData( reconnectionDataBuffer, reconnectionDataOffsets, 2 * centralOffset, centralReservoir, temporalShiftedPrimaryHit, temporalShiftedPrimaryWo );

        float canonicalAtOtherJacobian = 1.f;
        if (restirpt.useReservoirSplatting)
            canonicalAtOtherJacobian = splatSampleInfo[2 * centralOffset].z; // splatting's Jacobian

        float3 canonicalAtOtherIntegrand = computeShiftedIntegrand( restirpt, params, canonicalAtOtherJacobian, centralReservoir, rcData, temporalShiftedPrimaryRayFootprint, true, doInverseTemporalUpdate);
        float canonicalAtOtherPHat = PathRISState::toScalar(canonicalAtOtherIntegrand) * canonicalAtOtherJacobian;
        if (isAdditionalPassForDofShift)
        {
            canonicalAtOtherPHat *= doFReconnectionShiftBuffer[2 * centralOffset].z;
            // if cameray is static and centralReservoir stored shifted temporalReservoir's path, we are getting back to the original path
            // (fix numerical precision which could lead to firefly explosion)
            if (shouldGetSamePath(temporalShiftedPrimaryHit, centralReservoir.initRandomSeed == temporalReservoir.initRandomSeed))
                canonicalAtOtherPHat = PathRISState::toScalar(centralReservoir.integrand);
        }

        // kill p_hat if the shifted film sample falls outside frame.
        if (restirpt.useAreaReSTIRBackprojection && !restirpt.useReservoirSplatting)
        {
            float2 canonicalFilmSampleAtFloating = prevPixel + centralReservoir.filmSample;
            if (!all(canonicalFilmSampleAtFloating >= 0.f && canonicalFilmSampleAtFloating < params.frameDim)) canonicalAtOtherPHat = 0.f;
        }

        // ---- Phase 7: Splatting/non-splatting merge ----
        if (restirpt.useReservoirSplatting)
        {
            // Splatting branch: aggregate the multiple temporal reservoirs whose splat targets fell on this pixel
            float historyM = temporalReservoir.M; // this must be independent of the shift mapping (splatting)

            uint32_t numReservoirs = splatCellCounts[pixel];
            uint32_t cellOffset = splatCellOffsets[pixel];

            for (uint32_t i = 0; i < numReservoirs; i++)
            {
                uint2 splatSourcePixel = sortedSplatSourcePixels[cellOffset + i];

                uint prevOffset = params.getReservoirOffset(splatSourcePixel);
                uint shiftedPrimaryHitAddress = 2 * prevOffset + 1;

                temporalReservoir = prevReservoirs[prevOffset];
                temporalReservoir.setFilmSample(splatSampleInfo[shiftedPrimaryHitAddress].xy);
                temporalReservoir.M = min(MCap, temporalReservoir.M);

                float3 centralShiftedPrimaryWo;
                float centralShiftedPrimaryRayFootprint;
                HitInfo centralShiftedPrimaryHit = LoadShiftedPrimaryHitAssumeAreaReSTIRAndLoadedLensSample( params, restirpt, shiftedPrimaryHitBuffer, pixel, shiftedPrimaryHitAddress,
                                    temporalReservoir, centralShiftedPrimaryWo, centralShiftedPrimaryRayFootprint );

                rcData = loadReconnectionData( reconnectionDataBuffer, reconnectionDataOffsets, shiftedPrimaryHitAddress, temporalReservoir, centralShiftedPrimaryHit, centralShiftedPrimaryWo );

                float splatOrLensShiftJacobian = splatSampleInfo[shiftedPrimaryHitAddress].z;
                bool select = shiftAndMergeReservoirTemporal( restirpt, params, dstReservoir,
                    temporalReservoir, rcData, centralShiftedPrimaryRayFootprint, sg, centralReservoir.M, splatOrLensShiftJacobian, doTemporalUpdate );
                vBufferWritingOffset = select ? prevOffset : vBufferWritingOffset;
            }

            float splattedPrevPixelM = min(MCap, splatSampleInfo[2 * centralOffset].w);
            float canonicalWeight = (canonicalAtOtherPHat > 0.f)
                                        ? 1.f - (canonicalAtOtherPHat * splattedPrevPixelM /
                                                    (canonicalAtOtherPHat * splattedPrevPixelM + float(centralReservoir.M) * PathRISState::toScalar(centralReservoir.integrand)))
                                        : 1.f;
            // merge canonical sample
            selectNewSample = dstReservoir.merge(centralReservoir.integrand, 1.f, centralReservoir, sg, canonicalWeight);
            dstReservoir.M = centralReservoir.M + historyM;
        }
        else
        {
            // Non-splatting branch: a single forward-shift T_{t->c} of the temporal sample to the central pixel.
            uint inverseShiftedPrimaryHitAddress = 2 * centralOffset + 1;
            float3 centralShiftedPrimaryWo;
            float centralShiftedPrimaryRayFootprint;
            HitInfo centralShiftedPrimaryHit = LoadShiftedPrimaryHit( params, restirpt, shiftedPrimaryHitBuffer, vbuffer,
                doFReconnectionShiftBuffer, isAdditionalPassForDofShift,
                pixel, inverseShiftedPrimaryHitAddress, temporalReservoir, centralShiftedPrimaryWo, centralShiftedPrimaryRayFootprint );

            if (!centralShiftedPrimaryHit.isValid() && !restirpt.kUseAreaReservoirs) return false; 

            float canonicalWeight = (canonicalAtOtherPHat > 0.f)
                                ? 1.f - (canonicalAtOtherPHat * temporalReservoir.M /
                                            (canonicalAtOtherPHat * temporalReservoir.M + float(centralReservoir.M) * PathRISState::toScalar(centralReservoir.integrand)))
                : 1.f;

            rcData = loadReconnectionData( reconnectionDataBuffer, reconnectionDataOffsets, inverseShiftedPrimaryHitAddress, temporalReservoir, centralShiftedPrimaryHit, centralShiftedPrimaryWo);

            float splatOrLensShiftJacobian = 1.f;
            if (isAdditionalPassForDofShift)
            {
                splatOrLensShiftJacobian = doFReconnectionShiftBuffer[inverseShiftedPrimaryHitAddress].z;
                temporalReservoir.setLensSample(doFReconnectionShiftBuffer[inverseShiftedPrimaryHitAddress].xy);
            }

            bool select = shiftAndMergeReservoirTemporal( restirpt, params, dstReservoir,
                temporalReservoir, rcData, centralShiftedPrimaryRayFootprint, sg, centralReservoir.M, splatOrLensShiftJacobian, doTemporalUpdate );

            // Merge canonical sample
            selectNewSample = dstReservoir.merge(centralReservoir.integrand, 1.f, centralReservoir, sg, canonicalWeight);
        }

        // ---- Phase 8: Finalize reservoir, write back ----
        dstReservoir.finalizeReservoirWeight();
        dstReservoir.weight = (isnan(dstReservoir.weight) || isinf(dstReservoir.weight))
            ? 0.f
            : max(0.f, dstReservoir.weight);

        reservoirs[centralOffset] = dstReservoir;

        // vbuffer stores the **selected** primary hit in this case
        if (!selectNewSample && restirpt.kUseAreaReservoirs && (restirpt.useDoFReconnectionShift || restirpt.useReservoirSplatting))
            vbuffer[pixel] = shiftedPrimaryHitBuffer[2 * vBufferWritingOffset + 1];

        float temporalSampleAge = 0;
        if (temporalPassId == 0 && !selectNewSample && computeTemporalDuplication)
        {
            temporalSampleAge = prevSampleSurvivedFrames[prevPixelForDupMap] + 1;
        }

        writeTemporalOutputs(pixel, temporalSampleAge);
        return true;
    }

    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        if (!DoPass(pixel))
        {
            writeTemporalOutputs(pixel, 0.f);
        }
    }
};

cbuffer CB
{
    TemporalResampling gTemporalResampling;
}


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gTemporalResampling.execute(dispatchThreadId.xy);
}
