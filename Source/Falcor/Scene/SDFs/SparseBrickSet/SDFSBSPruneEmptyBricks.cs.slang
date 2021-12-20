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
import Scene.SDFs.SDFVoxelCommon;

static const uint kGroupWidth = BRICK_WIDTH;
static const uint kGroupSize = BRICK_WIDTH * BRICK_WIDTH * BRICK_WIDTH;
static const uint kBrickWidthInVoxels = BRICK_WIDTH;
static const float kRootThree = sqrt(3.0f);

cbuffer CB
{
    uint gPrimitiveCount;
    uint gGridWidth;
    uint gGroupCount;
};

StructuredBuffer<SDF3DPrimitive> gPrimitives;
ByteAddressBuffer gChunkCoords;

RWByteAddressBuffer gChunkValidity;

groupshared uint gGroupChunkValidity[kGroupSize];

float evalCoords(const uint3 coords)
{
    const float3 p = -0.5f + float3(coords) / gGridWidth;

    float sd = FLT_MAX;

    for (uint primitiveID = 0; primitiveID < gPrimitiveCount; primitiveID++)
    {
        SDF3DPrimitive primitive = gPrimitives[primitiveID];
        sd = primitive.eval(p, sd);
    }

    return clamp(sd * 2.0f * gGridWidth / kRootThree, -1.0f, 1.0f);
}

[numthreads(kGroupWidth, kGroupWidth, kGroupWidth)]
void main(uint3 groupID : SV_GroupID, uint3 brickLocalVoxelCoords : SV_GroupThreadID, uint brickLocalID : SV_GroupIndex)
{
    // Load virtual brick coords.
    const uint brickID = groupID.x;
    const uint3 virtualBrickCoords = gChunkCoords.Load3((3 * brickID) << 2);

    // Calculate brick grid coords.
    const uint3 brickGridCoords = virtualBrickCoords * kBrickWidthInVoxels;
    const uint3 voxelGridCoords = brickGridCoords + brickLocalVoxelCoords;

    const float4 values0xx = float4(
        evalCoords(voxelGridCoords),
        evalCoords(voxelGridCoords + uint3(0, 0, 1)),
        evalCoords(voxelGridCoords + uint3(0, 1, 0)),
        evalCoords(voxelGridCoords + uint3(0, 1, 1)));

    const float4 values1xx = float4(
        evalCoords(voxelGridCoords + uint3(1, 0, 0)),
        evalCoords(voxelGridCoords + uint3(1, 0, 1)),
        evalCoords(voxelGridCoords + uint3(1, 1, 0)),
        evalCoords(voxelGridCoords + uint3(1, 1, 1)));

    gGroupChunkValidity[brickLocalID] = SDFVoxelCommon::containsSurface(values0xx, values1xx) ? 1 : 0;

    GroupMemoryBarrierWithGroupSync();

    if (brickLocalID == 0)
    {
        for (uint i = 0; i < kGroupSize; i++)
        {
            if (gGroupChunkValidity[i])
            {
                gChunkValidity.Store(brickID << 2, 1);
                return;
            }
        }
    }
}
