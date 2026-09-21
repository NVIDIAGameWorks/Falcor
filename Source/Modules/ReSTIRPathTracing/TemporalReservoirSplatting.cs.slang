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
import ReSTIRPT;
import PathReservoir;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import ResamplingCommon;
import Utils.Math.Ray;
import Scene.Scene;

groupshared uint sampleIDs[32][32];

struct Pass
{
    ReSTIRPathTracingParams params; 
    Texture2D<PackedHitInfo> sampleVBuffer;
    Texture2D<PackedHitInfo> temporalSampleVbuffer;
    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<PathReservoir> reservoirs;
    RWStructuredBuffer<uint> splatCellOffsetCounter;
    RWTexture2D<uint> splatCellCounts;
    RWTexture2D<uint> splatCellOffsets;
    RWStructuredBuffer<uint2> sortedSplatSourcePixels;
    RWTexture2D<uint2> splatCellIndices;
    RWStructuredBuffer<float4> splatSampleInfo;
    RWStructuredBuffer<uint2> splatTargetPixel;
    RWStructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer;

    void splatSamples(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        const uint offset = params.getReservoirOffset(pixel);

        HitInfo currentPixelX1Hit = HitInfo(sampleVBuffer[pixel]);
        PathReservoir currentReservoir = reservoirs[offset];
        Ray currentRay = gScene.camera.computeRayThinlensSubPixelRandom( pixel, params.frameDim, currentReservoir.filmSample, currentReservoir.lensSample );
        float3 prevImageCoordAndJacobian = splatHitToImagePlane( params.frameDim, currentPixelX1Hit, currentRay.dir, currentReservoir.lensSample, true);

        int2 splattedPrevPixel = int2(prevImageCoordAndJacobian.xy);
        float M = 0;
        if (prevImageCoordAndJacobian.z > 0) M = prevReservoirs[params.getReservoirOffset(splattedPrevPixel)].M;
        splatSampleInfo[2 * offset] = float4(frac(prevImageCoordAndJacobian.xy), prevImageCoordAndJacobian.z, M);
        splatTargetPixel[2 * offset] = splattedPrevPixel;
        shiftedPrimaryHitBuffer[2 * offset] = currentPixelX1Hit.data;

        PathReservoir temporalReservoir = prevReservoirs[offset];
        HitInfo temporalPixelX1Hit = HitInfo(temporalSampleVbuffer[pixel]);
        Ray temporalRay = gScene.camera.computeRayThinlensSubPixelRandomPrevFrame( pixel, params.frameDim, temporalReservoir.filmSample, temporalReservoir.lensSample );

        float3 imageCoordAndJacobian = splatHitToImagePlane( params.frameDim, temporalPixelX1Hit, temporalRay.dir, temporalReservoir.lensSample, false );
        int2 splattedPixel = int2(imageCoordAndJacobian.xy);

        splatSampleInfo[2 * offset + 1] = float4(frac(imageCoordAndJacobian.xy), imageCoordAndJacobian.z, 0);
        splatTargetPixel[2 * offset + 1] = splattedPixel;
        shiftedPrimaryHitBuffer[2 * offset + 1] = temporalPixelX1Hit.data;

        if (imageCoordAndJacobian.z > 0)
        {
            int splattedPixel1D = splattedPixel.y * params.frameDim.x + splattedPixel.x;
            uint indexInCell;
            InterlockedAdd(splatCellCounts[splattedPixel], 1, indexInCell);
            splatCellIndices[pixel] = uint2(splattedPixel1D, indexInCell);
        }
        else
        {
            splatCellIndices[pixel] = uint2(uint(-1), 0);
        }

    }

    void computeSplatCellOffsets(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        uint numElements = splatCellCounts[pixel];

        uint offset;
        InterlockedAdd(splatCellOffsetCounter[0], numElements, offset);

        splatCellOffsets[pixel] = offset;
    }

    void sortSplatSourcesByTargetPixel(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;
        uint2 prevPixel = pixel;
        const uint2 indices = splatCellIndices[prevPixel];
        if (indices.x == uint(-1)) return;

        const uint splattedPixel1D = indices.x;
        const uint2 splattedPixel = uint2(splattedPixel1D % params.frameDim.x, splattedPixel1D / params.frameDim.x);
        const uint indexInCell = indices.y;

        sortedSplatSourcePixels[splatCellOffsets[splattedPixel] + indexInCell] = prevPixel;
    }
};

cbuffer CB
{
    Pass gPass;
}

[numthreads(16, 16, 1)]
void splatSamples( uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gPass.params.frameDim)) return;
    printSetPixel(pixel);
    logSetPixel(pixel);
    gPass.splatSamples(dispatchThreadId.xy);
}

[numthreads(16, 16, 1)]
void computeSplatCellOffsets( uint3 dispatchThreadId: SV_DispatchThreadID )
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gPass.params.frameDim)) return;
    printSetPixel(pixel);
    logSetPixel(pixel);
    gPass.computeSplatCellOffsets(dispatchThreadId.xy);
}

[numthreads(16, 16, 1)]
void sortSplatSourcesByTargetPixel(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gPass.params.frameDim)) return;
    printSetPixel(pixel);
    logSetPixel(pixel);
    gPass.sortSplatSourcesByTargetPixel(dispatchThreadId.xy);
}
