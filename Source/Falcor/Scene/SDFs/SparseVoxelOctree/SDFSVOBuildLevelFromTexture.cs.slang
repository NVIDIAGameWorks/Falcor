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
import Scene.SDFs.SDFVoxelCommon;
import Scene.SDFs.SparseVoxelOctree.SDFSVOHashTable;

cbuffer CB
{
    uint gLevel;
    uint gNumLevels;
    uint gLevelWidth;
};

Texture3D<float> gSDFGrid;
ParameterBlock<SDFSVOHashTable> gVoxelHashTable;
RWByteAddressBuffer gLocationCodes;

#if !FINEST_LEVEL_PASS
RWByteAddressBuffer gVoxelCounts;
groupshared uint gGroupNumVoxels;
#endif

[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (any(dispatchThreadID >= gLevelWidth)) return;

#if !FINEST_LEVEL_PASS
    const bool isGroupManager = all(groupThreadID == uint3(0));

    if (isGroupManager)
    {
        gGroupNumVoxels = 0;
    }

    GroupMemoryBarrierWithGroupSync();
#endif

    const uint3 voxelCoords = dispatchThreadID;
    
#if FINEST_LEVEL_PASS
    // Load the 8 corner values of the voxel.
    float4 values0xx;
    float4 values1xx;
    values0xx[0] = gSDFGrid[voxelCoords + uint3(0, 0, 0)].x;
    values0xx[1] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
    values0xx[2] = gSDFGrid[voxelCoords + uint3(0, 1, 0)].x;
    values0xx[3] = gSDFGrid[voxelCoords + uint3(0, 1, 1)].x;
    values1xx[0] = gSDFGrid[voxelCoords + uint3(1, 0, 0)].x;
    values1xx[1] = gSDFGrid[voxelCoords + uint3(1, 0, 1)].x;
    values1xx[2] = gSDFGrid[voxelCoords + uint3(1, 1, 0)].x;
    values1xx[3] = gSDFGrid[voxelCoords + uint3(1, 1, 1)].x;

    // Check if the voxel contains the surface, i.e., at least one corner has a positive distance and another has a negative distance.
    if (SDFVoxelCommon::containsSurface(values0xx, values1xx))
    {
        uint2 locationCode = SDFVoxelCommon::encodeLocation(voxelCoords, gLevel);

        // Set a valid bit in the 2 unused bits of location code.
        locationCode.y |= (1 << 31);

        uint slot;
        gVoxelHashTable.insert(locationCode, 0, values0xx, values1xx, slot);

        // Write the location code so that it can later be sorted.
        gLocationCodes.Store2(slot * 8, locationCode);
    }
#else
    const uint2 locationCode = SDFVoxelCommon::encodeLocation(voxelCoords, gLevel);

    // Set a valid bit in the 2 unused bits of location code.
    locationCode.y |= (1 << 31);

    uint2 childLocationCodes[8];

    // Create location codes for each child.
    SDFVoxelCommon::createChildLocationCode(locationCode, 0, childLocationCodes[0]);
    SDFVoxelCommon::createChildLocationCode(locationCode, 1, childLocationCodes[1]);
    SDFVoxelCommon::createChildLocationCode(locationCode, 2, childLocationCodes[2]);
    SDFVoxelCommon::createChildLocationCode(locationCode, 3, childLocationCodes[3]);
    SDFVoxelCommon::createChildLocationCode(locationCode, 4, childLocationCodes[4]);
    SDFVoxelCommon::createChildLocationCode(locationCode, 5, childLocationCodes[5]);
    SDFVoxelCommon::createChildLocationCode(locationCode, 6, childLocationCodes[6]);
    SDFVoxelCommon::createChildLocationCode(locationCode, 7, childLocationCodes[7]);

    // Set valid bit for child location codes;
    childLocationCodes[0].y |= (1 << 31);
    childLocationCodes[1].y |= (1 << 31);
    childLocationCodes[2].y |= (1 << 31);
    childLocationCodes[3].y |= (1 << 31);
    childLocationCodes[4].y |= (1 << 31);
    childLocationCodes[5].y |= (1 << 31);
    childLocationCodes[6].y |= (1 << 31);
    childLocationCodes[7].y |= (1 << 31);

    // Create a validity mask where set bits represent valid children.
    uint validMask = 0;
    validMask |= (gVoxelHashTable.contains(childLocationCodes[0]) ? 0x1 : 0x0);
    validMask |= (gVoxelHashTable.contains(childLocationCodes[1]) ? 0x2 : 0x0);
    validMask |= (gVoxelHashTable.contains(childLocationCodes[2]) ? 0x4 : 0x0);
    validMask |= (gVoxelHashTable.contains(childLocationCodes[3]) ? 0x8 : 0x0);
    validMask |= (gVoxelHashTable.contains(childLocationCodes[4]) ? 0x10 : 0x0);
    validMask |= (gVoxelHashTable.contains(childLocationCodes[5]) ? 0x20 : 0x0);
    validMask |= (gVoxelHashTable.contains(childLocationCodes[6]) ? 0x40 : 0x0);
    validMask |= (gVoxelHashTable.contains(childLocationCodes[7]) ? 0x80 : 0x0);

    if (validMask > 0)
    {
        const uint hierarchy = (gNumLevels - gLevel - 1);
        const uint3 gridCoords = voxelCoords << hierarchy;
        const uint voxelWidth = 1 << hierarchy;

        // Load the 8 corner values of the voxel.
        float4 values0xx;
        float4 values1xx;
        values0xx[0] = gSDFGrid[gridCoords + uint3(0, 0, 0)].x;
        values0xx[1] = gSDFGrid[gridCoords + uint3(0, 0, voxelWidth)].x;
        values0xx[2] = gSDFGrid[gridCoords + uint3(0, voxelWidth, 0)].x;
        values0xx[3] = gSDFGrid[gridCoords + uint3(0, voxelWidth, voxelWidth)].x;
        values1xx[0] = gSDFGrid[gridCoords + uint3(voxelWidth, 0, 0)].x;
        values1xx[1] = gSDFGrid[gridCoords + uint3(voxelWidth, 0, voxelWidth)].x;
        values1xx[2] = gSDFGrid[gridCoords + uint3(voxelWidth, voxelWidth, 0)].x;
        values1xx[3] = gSDFGrid[gridCoords + uint3(voxelWidth, voxelWidth, voxelWidth)].x;

        uint slot;
        gVoxelHashTable.insert(locationCode, validMask, values0xx, values1xx, slot);

        // Write the location code so that it can later be sorted.
        gLocationCodes.Store2(slot * 8, locationCode);

        InterlockedAdd(gGroupNumVoxels, 1);
    }

    GroupMemoryBarrierWithGroupSync();

    if (isGroupManager)
    {
        gVoxelCounts.InterlockedAdd(4 * gLevel, gGroupNumVoxels);
    }
#endif
}
