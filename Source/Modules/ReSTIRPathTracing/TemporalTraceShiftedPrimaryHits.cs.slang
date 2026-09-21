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

struct Pass
{
    ReSTIRPathTracingParams params;
    ReSTIRPT restirpt; 

    Texture2D<PackedHitInfo> vbuffer;                     
    Texture2D<float2> motionVectors;
    RWStructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;

    RWStructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    Texture2D<PackedHitInfo> sampleVBuffer;
    RWStructuredBuffer<float4> doFReconnectionShiftBuffer;

    bool gEnableTemporalReprojection;
    bool gStochasticReprojection;
    int temporalPassId;

    void DoPass(const uint2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.temporalSgOffset(temporalPassId));

        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = reservoirs[centralOffset];

        PathReservoir dstReservoir;
        dstReservoir.init();

        int numNeighbors = 1;

        float2 motionVector = gEnableTemporalReprojection ? motionVectors[pixel] : 0.f;
        float2 prevPixel = reprojectPixelForReuse(pixel, motionVector, params.frameDim, restirpt.useAreaReSTIRBackprojection, gStochasticReprojection, sg);

        bool shouldReturnEarly = 
            !restirpt.useAreaReSTIRBackprojection && !isValidScreenRegion(params, int2(prevPixel)) ||
             restirpt.useAreaReSTIRBackprojection && !isValidScreenRegionFloat(params, prevPixel);
        if (shouldReturnEarly) return;

        int2 fetchPixel = restirpt.useAreaReSTIRBackprojection ? pixel : int2(prevPixel);
        PathReservoir temporalReservoir = prevReservoirs[params.getReservoirOffset(fetchPixel)];

        if (temporalPassId == 0)
        {
            // trace primary ray (shift neighbor to center)
            HitInfo hit = retracePrimaryRay(pixel, params.frameDim, temporalReservoir.filmSample, temporalReservoir.lensSample, false);
            shiftedPrimaryHitBuffer[2 * numNeighbors * centralOffset + 1] = hit.data;

            // trace primary ray (shift center to neighbor)
            hit = retracePrimaryRay(prevPixel, params.frameDim, centralReservoir.filmSample, centralReservoir.lensSample, true);
            shiftedPrimaryHitBuffer[2 * numNeighbors * centralOffset] = hit.data;

            if (restirpt.useDoFReconnectionShift)
            {
                float3 centralCenterPrimaryWo;
                HitInfo centralPrimaryCenterHit = getPixelHitInfo(params, vbuffer, pixel, centralCenterPrimaryWo);
                VertexData centralPrimaryCenterVertexData = getVertexData(centralPrimaryCenterHit, gScene.camera.data.posW, -centralCenterPrimaryWo, false);
                float doFReconnectionShiftProbability = restirpt.deriveDoFReconnectionShiftProbabilityFromCoC ?
                1.f - computeConfidenceWeightForTwoShifts( params.frameDim, centralPrimaryCenterVertexData.posW, !centralPrimaryCenterHit.isValid() ) :
                restirpt.doFReconnectionShiftProbability;
                doFReconnectionShiftProbability = restirpt.useDoFReconnectionForTemporalReuse ? doFReconnectionShiftProbability : 0.f;
                bool cameraChanged = hasCameraChanged();
                if (isBorderPixel(params, pixel) && !cameraChanged) doFReconnectionShiftProbability = 0.f;
                doFReconnectionShiftBuffer[2 * centralOffset].w = doFReconnectionShiftProbability;
            }
        }
        else
        {
            if (restirpt.useDoFReconnectionShift)
            {
                float doFReconnectionShiftProbability = doFReconnectionShiftBuffer[2 * centralOffset].w;

                if (doFReconnectionShiftProbability > 0.f)
                {
                    {
                        HitInfo neighborPixelX1Hit = retracePrimaryRay( prevPixel, params.frameDim, temporalReservoir.filmSample, temporalReservoir.lensSample, true );
                        Ray neighborRay = gScene.camera.computeRayThinlensSubPixelRandomPrevFrame(prevPixel, params.frameDim, temporalReservoir.filmSample, temporalReservoir.lensSample);
                        float3 X0_dstJ = solveLensSampleForPrimaryHit( pixel, params.frameDim, neighborPixelX1Hit, neighborRay.dir, temporalReservoir.filmSample, false );
                        float srcJ = computePrimaryReconnectionJacobian( prevPixel, params.frameDim, neighborPixelX1Hit, neighborRay.dir, temporalReservoir.filmSample, true );
                        X0_dstJ.z /= srcJ;
                        if (isinf(X0_dstJ.z) || isnan(X0_dstJ.z)) X0_dstJ.z = 0.f;
                        doFReconnectionShiftBuffer[2 * centralOffset + 1] = float4(X0_dstJ, 0.f);
                        shiftedPrimaryHitBuffer[2 * centralOffset + 1] = neighborPixelX1Hit.data;
                    }

                    {
                        HitInfo centerPixelX1Hit = retracePrimaryRay( pixel, params.frameDim, centralReservoir.filmSample, centralReservoir.lensSample, false );
                        Ray centerRay = gScene.camera.computeRayThinlensSubPixelRandom( pixel, params.frameDim, centralReservoir.filmSample, centralReservoir.lensSample );
                        float3 X0_dstJ = solveLensSampleForPrimaryHit( prevPixel, params.frameDim, centerPixelX1Hit, centerRay.dir, centralReservoir.filmSample, true );
                        float centerJ = computePrimaryReconnectionJacobian( pixel, params.frameDim, centerPixelX1Hit, centerRay.dir, centralReservoir.filmSample, false );
                        X0_dstJ.z /= centerJ;
                        if (isinf(X0_dstJ.z) || isnan(X0_dstJ.z)) X0_dstJ.z = 0.f;
                        doFReconnectionShiftBuffer[2 * centralOffset].xyz = X0_dstJ;
                        shiftedPrimaryHitBuffer[2 * centralOffset] = centerPixelX1Hit.data;
                    }
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
