/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
import Utils.Math.PackedFormats;

static const uint kBrickWidthInVoxels = BRICK_WIDTH;
static const uint kBrickWidthInValues = BRICK_WIDTH + 1;
static const float kRootThree = sqrt(3.0f);

cbuffer CB
{
    uint gPrimitiveCount;
    uint gGridWidth;
    uint gBrickCount;
};

StructuredBuffer<SDF3DPrimitive> gPrimitives;
ByteAddressBuffer gChunkCoords;

RWByteAddressBuffer gChunkValidity;

groupshared float gGroupBrickValues[kBrickWidthInValues * kBrickWidthInValues * kBrickWidthInValues];
groupshared bool gGroupBrickValidity[kBrickWidthInVoxels * kBrickWidthInVoxels * kBrickWidthInVoxels];

float evalCoords(const uint3 coord)
{
    const float3 p = -0.5f + float3(coord) / gGridWidth;
    return evalPos(p);
}

float evalPos(const float3 p)
{
    float sd = FLT_MAX;

    for (uint primitiveID = 0; primitiveID < gPrimitiveCount; primitiveID++)
    {
        SDF3DPrimitive primitive = gPrimitives[primitiveID];
        sd = primitive.eval(p, sd);
    }

    return sd * 2.0f * gGridWidth / kRootThree;
}

[numthreads(256, 1, 1)]
void coarsePrune(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Load virtual brick coords.
    const uint brickID = dispatchThreadID.x;

    if (brickID >= gBrickCount) return;

    const uint3 virtualBrickCoords = gChunkCoords.Load3((3 * brickID) << 2);

    // Calculate brick grid coords.
    const uint3 brickGridCoords = virtualBrickCoords * kBrickWidthInVoxels;
    const float3 brickCenterPos = -0.5f + (float3(brickGridCoords) + 0.5f * float(kBrickWidthInVoxels)) / gGridWidth;

    float centerValue = evalPos(brickCenterPos);

    // We don't do abs here since some signed distance functions aren't bounded on the interior.
    bool valid = centerValue <= float(kBrickWidthInVoxels);
    gChunkValidity.Store(brickID << 2, valid ? 1 : 0);
}

[numthreads(kBrickWidthInValues, kBrickWidthInValues, kBrickWidthInValues)]
void finePrune(uint3 groupID : SV_GroupID, uint3 brickLocalValueCoords : SV_GroupThreadID, uint brickLocalID : SV_GroupIndex)
{
    // Load virtual brick coords.
    const uint brickID = groupID.x;
    const uint3 virtualBrickCoords = gChunkCoords.Load3((3 * brickID) << 2);

    // Calculate brick grid coords.
    const uint3 brickGridCoords = virtualBrickCoords * kBrickWidthInVoxels;
    const uint3 valueGridCoords = brickGridCoords + brickLocalValueCoords;

    gGroupBrickValues[brickLocalID] = evalCoords(valueGridCoords);

    GroupMemoryBarrierWithGroupSync();

    if (all(brickLocalValueCoords <= kBrickWidthInVoxels))
    {
        float4 values0xx = float4(
            gGroupBrickValues[flatten3D(brickLocalValueCoords, kBrickWidthInValues, kBrickWidthInValues)],
            gGroupBrickValues[flatten3D(brickLocalValueCoords + uint3(0, 0, 1), kBrickWidthInValues, kBrickWidthInValues)],
            gGroupBrickValues[flatten3D(brickLocalValueCoords + uint3(0, 1, 0), kBrickWidthInValues, kBrickWidthInValues)],
            gGroupBrickValues[flatten3D(brickLocalValueCoords + uint3(0, 1, 1), kBrickWidthInValues, kBrickWidthInValues)]);

        float4 values1xx = float4(
            gGroupBrickValues[flatten3D(brickLocalValueCoords + uint3(1, 0, 0), kBrickWidthInValues, kBrickWidthInValues)],
            gGroupBrickValues[flatten3D(brickLocalValueCoords + uint3(1, 0, 1), kBrickWidthInValues, kBrickWidthInValues)],
            gGroupBrickValues[flatten3D(brickLocalValueCoords + uint3(1, 1, 0), kBrickWidthInValues, kBrickWidthInValues)],
            gGroupBrickValues[flatten3D(brickLocalValueCoords + uint3(1, 1, 1), kBrickWidthInValues, kBrickWidthInValues)]);

        uint voxelID = flatten3D(brickLocalValueCoords, kBrickWidthInVoxels, kBrickWidthInVoxels);
        gGroupBrickValidity[voxelID] = SDFVoxelCommon::containsSurface(values0xx, values1xx);
    }

    GroupMemoryBarrierWithGroupSync();

    if (brickLocalID == 0)
    {
        for (uint voxelID = 0; voxelID < kBrickWidthInVoxels * kBrickWidthInVoxels * kBrickWidthInVoxels; voxelID++)
        {
            if (gGroupBrickValidity[voxelID])
            {
                gChunkValidity.Store(brickID << 2, 1);
                return;
            }
        }
    }
}
