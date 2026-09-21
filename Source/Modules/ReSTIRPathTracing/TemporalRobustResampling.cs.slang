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

    StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;

    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint> reconnectionDataOffsets;

    RWStructuredBuffer<float> temporalMISPDFs;

    static const int2 kPrevNeighborOffsets[8] = {
        int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0), int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1)
    };

    void DoPass(const uint2 pixel)
    {
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = prevReservoirs[centralOffset];

        for (int i = 0; i < 8; ++i)
        {
            int2 neighborPixel = pixel + kPrevNeighborOffsets[i];
            if (!isValidScreenRegion(params, neighborPixel))
            {
                continue;
            }

            float3 neighborPrimaryWo;
            HitInfo neighborPrimaryHit = getPixelTemporalHitInfo( params, shiftedPrimaryHitBuffer, neighborPixel, 8 * centralOffset + i, centralReservoir.filmSample, centralReservoir.lensSample, neighborPrimaryWo );
            VertexData neighborPrimaryVertexData = getVertexData(neighborPrimaryHit, gScene.camera.data.prevPosW, -neighborPrimaryWo, true);
            float neighborPrimaryRayFootprint = restirpt.computePrimaryRayFootprint( neighborPrimaryVertexData.faceNormalW, gScene.camera.data.prevPosW,
                                                                                                -neighborPrimaryWo, neighborPrimaryVertexData.posW);

            ReconnectionData rcData;
            if (centralReservoir.pathFlags.rcVertexLength() > 1)
            {
                rcData = reconnectionDataBuffer[reconnectionDataOffsets[8 * centralOffset + i]];
            }
            else
            {
                rcData.pathThroughput = 1.f;
                rcData.rcPrevHit = neighborPrimaryHit;
                rcData.rcPrevWo = neighborPrimaryWo;
                rcData.rcFootprintFactor = 0.f; // not used
            }

            float canonicalAtOtherJacobian = 1.f;
            float3 canonicalAtOtherIntegrand = computeShiftedIntegrand(restirpt, params, canonicalAtOtherJacobian, centralReservoir, rcData, neighborPrimaryRayFootprint, true);
            float canonicalAtOtherPHat = PathRISState::toScalar(canonicalAtOtherIntegrand) * canonicalAtOtherJacobian;
            // we just want to get canonical At other pHat

            temporalMISPDFs[8 * centralOffset + i] = canonicalAtOtherPHat;
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
