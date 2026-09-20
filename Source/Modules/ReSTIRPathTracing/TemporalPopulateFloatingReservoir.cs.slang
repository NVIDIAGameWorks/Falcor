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
    Texture2D<PackedHitInfo> sampleVBuffer;
    RWStructuredBuffer<float4> doFReconnectionShiftBuffer;

    Texture2D<float2> motionVectors;
    Texture2D<float2> deltaReflectionMotionVectors;
    Texture2D<float2> deltaTransmissionMotionVectors;

    RWStructuredBuffer<PathReservoir> temporaryReservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<float> temporalMISPDFs;
    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint> reconnectionDataOffsets;
    StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBufferRobust;
    RWStructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;

    static const uint2 kPrevNeighborOffsets[4] = {
        uint2(0, 0), uint2(1, 0), uint2(0, 1), uint2(1, 1)
    };


    void DoPass(const uint2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.temporalPopulateFloatingSgOffset());
        const uint centralOffset = params.getReservoirOffset(pixel);

        PathReservoir dstReservoir;
        dstReservoir.init();


        float totalM = 0.f;

        {
            float2 motionVector = motionVectors[pixel];
            float mvecWeight = 1.f;

            float2 floatingPixel = pixel + motionVector * params.frameDim;

            if (!isValidScreenRegionFloat(params, floatingPixel))
                return;

            int2 quadTopLeft = int2(floor(floatingPixel));
            float2 areaCoordInQuad = saturate(float2(int2(pixel) - quadTopLeft) + (motionVector * params.frameDim));
            // has to be in [(0,0),(1,1))

            int2 selectedPixel = -1;
            int selected_relative_i = -1;

            // what's my four neighbors?
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    int2 neighborPixel = quadTopLeft + int2(j, i);
                    if (!isValidScreenRegion(params, neighborPixel)) continue;
                    // is the neighbor's sample in the floating area?
                    PathReservoir neighborReservoir = prevReservoirs[params.getReservoirOffset(neighborPixel)];
                    float2 filmSampleNew = float2(j, i) + neighborReservoir.filmSample - areaCoordInQuad;
                    float bilinearWeight = lerp(1 - j, j, areaCoordInQuad.x) * lerp(1 - i, i, areaCoordInQuad.y);
                    totalM += bilinearWeight * neighborReservoir.M * mvecWeight;

                    // option 1

                    if (!restirpt.temporalRobustShift)
                    {
                        if (all(filmSampleNew >= 0.f && filmSampleNew < 1.f))
                        {
                            neighborReservoir.setFilmSample(filmSampleNew);
                            dstReservoir.merge(neighborReservoir.integrand, 1, neighborReservoir, sg, mvecWeight);
                        }
                    }
                    // option 2
                    // read from the reconnection buffer (we have already computed pHat times Jacobian)
                    else
                    {
                        uint pHatMemoryOffset = 8 * params.getReservoirOffset(neighborPixel);
                        float pHatSource = PathRISState::toScalar(neighborReservoir.integrand);
                        float pHatTarget = pHatSource;
                        // uv is in ([0,2),[0,2)) range
                        int2 neighborRelativeOffsets = int2(j, i);
                        int2 targetPixel = neighborRelativeOffsets;
                        float2 uv = targetPixel + neighborReservoir.filmSample;
                        if (uv.y < areaCoordInQuad.y) targetPixel.y += 1;
                        if (uv.y > 1 + areaCoordInQuad.y) targetPixel.y -= 1;
                        if (uv.x < areaCoordInQuad.x) targetPixel.x += 1;
                        if (uv.x > 1 + areaCoordInQuad.x) targetPixel.x -= 1;
                        // fetch the MIS weight

                        int2 temp = targetPixel - neighborRelativeOffsets + int2(1, 1); // 8 different possibilites
                        int memoryOffset = 3 * temp.y + temp.x;
                        int relative_i = 9;
                        if (memoryOffset != 4) // not "center"
                        {
                            memoryOffset = memoryOffset < 4 ? memoryOffset : memoryOffset - 1;
                            // fetch pHat from memory
                            pHatTarget = temporalMISPDFs[pHatMemoryOffset + memoryOffset];
                            relative_i = memoryOffset;
                        }

                        // 0,0  1,0  2,0
                        // 1,0       2,1
                        // 2,0  2,1  2,2

                        // sum up MIS weights
                        float pHatSum = bilinearWeight * pHatSource * neighborReservoir.M;
                        for (uint jdx = 0; jdx < 4; jdx++)
                        {
                            uint2 offsetPixel = quadTopLeft + kPrevNeighborOffsets[jdx];
                            if (!isValidScreenRegion(params, offsetPixel)) continue;
                            int2 diff = int2(kPrevNeighborOffsets[jdx]) - neighborRelativeOffsets;
                            if (diff.x == 0 && diff.y == 0) continue;
                            int2 temp = diff + int2(1, 1);
                            int memoryOffset_ = 3 * temp.y + temp.x;
                            memoryOffset_ = memoryOffset_ < 4 ? memoryOffset_ : memoryOffset_ - 1;
                            // fetch pHat from memory
                            int _i = jdx / 2;
                            int _j = jdx % 2;
                            PathReservoir offsetReservoir = prevReservoirs[params.getReservoirOffset(offsetPixel)];
                            float bilinearWeightJ = lerp(1 - _j, _j, areaCoordInQuad.x) * lerp(1 - _i, _i, areaCoordInQuad.y);
                            pHatSum += bilinearWeightJ * offsetReservoir.M * temporalMISPDFs[pHatMemoryOffset + memoryOffset_];
                        }

                        // fetch MIS weight
                        float misWeight = pHatSum > 0.0f ? bilinearWeight * neighborReservoir.M * pHatSource / pHatSum : 0.0f;
                        neighborReservoir.setFilmSample(saturate(targetPixel + neighborReservoir.filmSample - areaCoordInQuad));

                        bool selected = dstReservoir.mergeWithoutWriting(pHatTarget, neighborReservoir, sg, misWeight);

                        if (selected)
                        {
                            selectedPixel = neighborPixel;
                            selected_relative_i = relative_i;
                        }

                        // Start resampling
                        // only do reconnection with the selected neighbor
                    }
                }
            }

            // populate the reservoir with shifted quantities
            if (selected_relative_i != -1 && restirpt.temporalRobustShift) // doesn't support multi motion vectors yet
            {
                int neighborOffset = params.getReservoirOffset(selectedPixel);
                PathReservoir neighborReservoir = prevReservoirs[neighborOffset];
                float3 dstIntegrand = neighborReservoir.integrand;
                neighborReservoir.setFilmSample(dstReservoir.filmSample);
                neighborReservoir.setLensSample(dstReservoir.lensSample);

                if (selected_relative_i != 9)
                {
                    float3 centralShiftedPrimaryWo;

                    HitInfo centralShiftedPrimaryHit =
                        getPixelTemporalHitInfo( params, shiftedPrimaryHitBufferRobust, floatingPixel, 8 * neighborOffset + selected_relative_i, dstReservoir.filmSample, dstReservoir.lensSample, centralShiftedPrimaryWo );
                    VertexData centralShiftedPrimaryVertexData = getVertexData(centralShiftedPrimaryHit, gScene.camera.data.prevPosW, -centralShiftedPrimaryWo, true);
                    float centralShiftedPrimaryRayFootprint = restirpt.computePrimaryRayFootprint( centralShiftedPrimaryVertexData.faceNormalW, gScene.camera.data.prevPosW,
                        -centralShiftedPrimaryWo, centralShiftedPrimaryVertexData.posW );

                    ReconnectionData rcData;
                    if (neighborReservoir.pathFlags.rcVertexLength() > 1)
                    {
                        rcData = reconnectionDataBuffer[reconnectionDataOffsets[8 * neighborOffset + selected_relative_i]];
                    }
                    else
                    {
                        rcData.pathThroughput = 1.f;
                        rcData.rcPrevHit = centralShiftedPrimaryHit;
                        rcData.rcPrevWo = centralShiftedPrimaryWo;
                        rcData.rcFootprintFactor = 0.f; // not used
                    }
                    float Jacobian = 1.f;
                    dstIntegrand = shiftPath(restirpt, params, Jacobian, neighborReservoir, rcData, centralShiftedPrimaryRayFootprint, true);
                }
                dstReservoir.mergeDirectOverwrite(dstIntegrand, neighborReservoir);
            }
        }

        dstReservoir.finalizeReservoirWeight();
        dstReservoir.M = totalM;
        // save dstReservoir

        temporaryReservoirs[centralOffset] = dstReservoir;
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
