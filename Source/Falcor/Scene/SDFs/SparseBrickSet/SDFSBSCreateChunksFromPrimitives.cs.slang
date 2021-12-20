/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Math/MathConstants.slangh"

import Scene.SDFs.SDF3DPrimitive;
import Utils.Math.IntervalArithmetic;
import Utils.Math.PackedFormats;

static const uint kGroupWidth = CHUNK_WIDTH;
static const uint kChunkSize = CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_WIDTH;
static const uint kBrickWidth = BRICK_WIDTH;

cbuffer CB
{
    uint gPrimitiveCount;
    uint gGridWidth;
    uint gGroupCount;
};

StructuredBuffer<SDF3DPrimitive> gPrimitives;
ByteAddressBuffer gChunkCoords;

RWByteAddressBuffer gSubChunkValidity;
RWByteAddressBuffer gSubChunkCoords;

groupshared uint gGroupSubChunkValidity[kChunkSize];
groupshared uint3 gGroupSubChunkCoords[kChunkSize];

void calculateSubChunkBounds(const uint3 coords, out float3 pCenter, out float3 pHalfExtent)
{
    pHalfExtent = 0.5f / float(gGridWidth);
    pCenter = -0.5f + (float3(coords) + 0.5f) / float(gGridWidth);
}

bool evalSubChunk(float3 pCenter, float3 pHalfExtent)
{
    float2 sd = FLT_MAX;

    for (uint primitiveID = 0; primitiveID < gPrimitiveCount; primitiveID++)
    {
        SDF3DPrimitive primitive = gPrimitives[primitiveID];
        sd = primitive.evalInterval(pCenter, pHalfExtent, sd);
    }

    return ivlContainsZero(sd);
}

// This function evaluates voxel validity inside of a 4x4x4 chunk of voxels with data at corners, valid voxels are made into sub chunks.
[numthreads(kGroupWidth, kGroupWidth, kGroupWidth)]
void rootEntryPoint(uint3 globalSubChunkCoords : SV_DispatchThreadID, uint subChunkID : SV_GroupIndex)
{
    float3 pCenter;
    float3 pHalfExtent;
    calculateSubChunkBounds(globalSubChunkCoords, pCenter, pHalfExtent);

    // Evaluate primitives for the current sub chunk.
    bool valid = evalSubChunk(pCenter, pHalfExtent);
    gGroupSubChunkValidity[subChunkID] = valid ? 1 : 0;
    gGroupSubChunkCoords[subChunkID] = valid ? globalSubChunkCoords : uint3(UINT32_MAX);

    GroupMemoryBarrierWithGroupSync();

    // 1 thread per chunk writes to output buffer.
    if (subChunkID == 0)
    {
        for (uint i = 0; i < kChunkSize; i += 4)
        {
            gSubChunkValidity.Store4(
                i << 2,
                uint4(gGroupSubChunkValidity[i],
                      gGroupSubChunkValidity[i + 1],
                      gGroupSubChunkValidity[i + 2],
                      gGroupSubChunkValidity[i + 3]));

            gSubChunkCoords.Store3((3 * i) << 2,      gGroupSubChunkCoords[i]);
            gSubChunkCoords.Store3((3 * i + 3) << 2,  gGroupSubChunkCoords[i + 1]);
            gSubChunkCoords.Store3((3 * i + 6) << 2,  gGroupSubChunkCoords[i + 2]);
            gSubChunkCoords.Store3((3 * i + 9) << 2,  gGroupSubChunkCoords[i + 3]);
        }
    }
}

[numthreads(kGroupWidth, kGroupWidth, kGroupWidth)]
void subdivideEntryPoint(uint3 groupID : SV_GroupID, uint3 localSubChunkCoords : SV_GroupThreadID, uint subChunkID : SV_GroupIndex)
{
    // Load global chunk coords.
    const uint chunkID = groupID.x;
    const uint3 chunkCoords = gChunkCoords.Load3((3 * chunkID) << 2);

    // Calculate virtual sub chunk coords
    const uint3 virtualSubChunkCoords = (chunkCoords << 2) + localSubChunkCoords;

    float3 pCenter;
    float3 pHalfExtent;
    calculateSubChunkBounds(virtualSubChunkCoords, pCenter, pHalfExtent);

    // Evaluate primitives for the current sub chunk.
    bool valid = evalSubChunk(pCenter, pHalfExtent);
    gGroupSubChunkValidity[subChunkID] = valid ? 1 : 0;
    gGroupSubChunkCoords[subChunkID] = valid ? virtualSubChunkCoords : uint3(UINT32_MAX);

    GroupMemoryBarrierWithGroupSync();

    // 1 thread per chunk writes to output buffer.
    if (subChunkID == 0)
    {
        const uint chunkOffset = chunkID * kChunkSize;

        for (uint i = 0; i < kChunkSize; i += 4)
        {
            const uint offset = chunkOffset + i;

            gSubChunkValidity.Store4(
                offset << 2,
                uint4(gGroupSubChunkValidity[i],
                      gGroupSubChunkValidity[i + 1],
                      gGroupSubChunkValidity[i + 2],
                      gGroupSubChunkValidity[i + 3]));

            gSubChunkCoords.Store3((3 * offset) << 2,      gGroupSubChunkCoords[i]);
            gSubChunkCoords.Store3((3 * offset + 3) << 2,  gGroupSubChunkCoords[i + 1]);
            gSubChunkCoords.Store3((3 * offset + 6) << 2,  gGroupSubChunkCoords[i + 2]);
            gSubChunkCoords.Store3((3 * offset + 9) << 2,  gGroupSubChunkCoords[i + 3]);
        }
    }
}
