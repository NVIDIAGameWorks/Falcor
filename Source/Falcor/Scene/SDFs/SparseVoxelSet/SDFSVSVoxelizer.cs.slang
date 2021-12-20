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
import Scene.SDFs.SDFVoxelTypes;
import Scene.SDFs.SDFVoxelCommon;
import Utils.Math.AABB;
import Utils.Math.FormatConversion;

#define GROUP_WIDTH 4
#define GROUP_SIZE GROUP_WIDTH * GROUP_WIDTH * GROUP_WIDTH

cbuffer CB
{
    uint gVirtualGridLevel;
    uint gVirtualGridWidth;
};

Texture3D<float> gSDFGrid;

RWStructuredBuffer<AABB> gVoxelAABBs;
RWStructuredBuffer<SDFSVSVoxel> gVoxels;

groupshared AABB gGroupVoxelAABBs[GROUP_SIZE];
groupshared SDFSVSVoxel gGroupVoxels[GROUP_SIZE];
groupshared uint gGroupNumAddedVoxels;

float safeLoadValue(int3 coord)
{
    if (any(coord < 0) || any(coord >= gVirtualGridWidth)) return 1.0f;
    return gSDFGrid[coord].x;
}

static uint4 packValues(float4 values0x, float4 values1x, float4 values2x, float4 values3x)
{
    uint4 packedValues;
    packedValues.x = packSnorm8(values0x[0]);
    packedValues.x |= (packSnorm8(values0x[1]) << 8);
    packedValues.x |= (packSnorm8(values0x[2]) << 16);
    packedValues.x |= (packSnorm8(values0x[3]) << 24);
    packedValues.y = packSnorm8(values1x[0]);
    packedValues.y |= (packSnorm8(values1x[1]) << 8);
    packedValues.y |= (packSnorm8(values1x[2]) << 16);
    packedValues.y |= (packSnorm8(values1x[3]) << 24);
    packedValues.z = packSnorm8(values2x[0]);
    packedValues.z |= (packSnorm8(values2x[1]) << 8);
    packedValues.z |= (packSnorm8(values2x[2]) << 16);
    packedValues.z |= (packSnorm8(values2x[3]) << 24);
    packedValues.w = packSnorm8(values3x[0]);
    packedValues.w |= (packSnorm8(values3x[1]) << 8);
    packedValues.w |= (packSnorm8(values3x[2]) << 16);
    packedValues.w |= (packSnorm8(values3x[3]) << 24);
    return packedValues;
}

uint4 loadAndPackSlice(int3 voxelCoords, int x)
{
    // voxelCoords corresponds to 111.
    float4 x0x; // Contains the values (x00, x01, x02, x03).
    x0x.x = safeLoadValue(voxelCoords + int3(x, -1, -1));
    x0x.y = safeLoadValue(voxelCoords + int3(x, -1,  0));
    x0x.z = safeLoadValue(voxelCoords + int3(x, -1,  1));
    x0x.w = safeLoadValue(voxelCoords + int3(x, -1,  2));

    float4 x1x; // Contains the values (x10, x11, x12, x13).
    x1x.x = safeLoadValue(voxelCoords + int3(x, 0, -1));
    x1x.y = safeLoadValue(voxelCoords + int3(x, 0,  0));
    x1x.z = safeLoadValue(voxelCoords + int3(x, 0,  1));
    x1x.w = safeLoadValue(voxelCoords + int3(x, 0,  2));

    float4 x2x; // Contains the values (x20, x21, x22, x23).
    x2x.x = safeLoadValue(voxelCoords + int3(x, 1, -1));
    x2x.y = safeLoadValue(voxelCoords + int3(x, 1,  0));
    x2x.z = safeLoadValue(voxelCoords + int3(x, 1,  1));
    x2x.w = safeLoadValue(voxelCoords + int3(x, 1,  2));

    float4 x3x; // Contains the values (x30, x31, x32, x33).
    x3x.x = safeLoadValue(voxelCoords + int3(x, 2, -1));
    x3x.y = safeLoadValue(voxelCoords + int3(x, 2,  0));
    x3x.z = safeLoadValue(voxelCoords + int3(x, 2,  1));
    x3x.w = safeLoadValue(voxelCoords + int3(x, 2,  2));

    return packValues(x0x, x1x, x2x, x3x);
}

bool voxelContainsSurface(int3 voxelCoords)
{
    if (any(voxelCoords < 0) || any(voxelCoords >= gVirtualGridWidth)) return false;

    // Load voxel corner values.
    float4 values0xx;
    float4 values1xx;
    values0xx[0] = gSDFGrid[voxelCoords].x;
    values0xx[1] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
    values0xx[2] = gSDFGrid[voxelCoords + uint3(0, 1, 0)].x;
    values0xx[3] = gSDFGrid[voxelCoords + uint3(0, 1, 1)].x;
    values1xx[0] = gSDFGrid[voxelCoords + uint3(1, 0, 0)].x;
    values1xx[1] = gSDFGrid[voxelCoords + uint3(1, 0, 1)].x;
    values1xx[2] = gSDFGrid[voxelCoords + uint3(1, 1, 0)].x;
    values1xx[3] = gSDFGrid[voxelCoords + uint3(1, 1, 1)].x;

    return SDFVoxelCommon::containsSurface(values0xx, values1xx);
}

/** Checks surrounding neighbors for surface containment (quite inefficient).
*/
uint createNeighborValidityMask(int3 voxelCoords)
{
    uint validityMask = 0;

    for (int z = 0; z <= 2; z++)
    {
        for (int y = 0; y <= 2; y++)
        {
            for (int x = 0; x <= 2; x++)
            {
                if (voxelContainsSurface(voxelCoords + int3(x - 1, y - 1, z - 1)))
                {
                    validityMask |= (1 << (z + 3 * (y + 3 * x)));
                }
            }
        }
    }

    return validityMask;
}

[numthreads(GROUP_WIDTH, GROUP_WIDTH, GROUP_WIDTH)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    if (any(dispatchThreadID >= gVirtualGridWidth))
        return;

    const int3 voxelCoords = dispatchThreadID;
    const bool isGroupManager = all(groupThreadID == uint3(0));

    // Initialize group values.
    if (isGroupManager)
    {
        gGroupNumAddedVoxels = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    // Process the voxel for this thread.
    {
        float4 values0xx;
        float4 values1xx;
        values0xx[0] = gSDFGrid[voxelCoords].x;
        values0xx[1] = gSDFGrid[voxelCoords + int3(0, 0, 1)].x;
        values0xx[2] = gSDFGrid[voxelCoords + int3(0, 1, 0)].x;
        values0xx[3] = gSDFGrid[voxelCoords + int3(0, 1, 1)].x;
        values1xx[0] = gSDFGrid[voxelCoords + int3(1, 0, 0)].x;
        values1xx[1] = gSDFGrid[voxelCoords + int3(1, 0, 1)].x;
        values1xx[2] = gSDFGrid[voxelCoords + int3(1, 1, 0)].x;
        values1xx[3] = gSDFGrid[voxelCoords + int3(1, 1, 1)].x;

        if (SDFVoxelCommon::containsSurface(values0xx, values1xx))
        {
            // Successfully inserted values.
            uint insertIndex;
            InterlockedAdd(gGroupNumAddedVoxels, 1, insertIndex);

            float3 p = float3(voxelCoords) - float(gVirtualGridWidth) * 0.5f;
            gGroupVoxelAABBs[insertIndex] = AABB(p / float(gVirtualGridWidth), (p + 1.0f) / float(gVirtualGridWidth));

            SDFSVSVoxel voxel;
            voxel.packedValuesSlices[0] = loadAndPackSlice(voxelCoords, -1);
            voxel.packedValuesSlices[1] = loadAndPackSlice(voxelCoords, 0);
            voxel.packedValuesSlices[2] = loadAndPackSlice(voxelCoords, 1);
            voxel.packedValuesSlices[3] = loadAndPackSlice(voxelCoords, 2);
            voxel.validNeighborsMask = createNeighborValidityMask(voxelCoords);
            gGroupVoxels[insertIndex] = voxel;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Let the group manager write group values to output buffers.
    if (isGroupManager)
    {
        if (gGroupNumAddedVoxels > 0)
        {
            for (uint i = 0; i < gGroupNumAddedVoxels; i++)
            {
                // TODO: Perhaps a better solution for this, increment counter once per group and perform compaction after?
                uint primitiveID = gVoxels.IncrementCounter();
                gVoxelAABBs[primitiveID] = gGroupVoxelAABBs[i];
                gVoxels[primitiveID] = gGroupVoxels[i];
            }
        }
    }
}
