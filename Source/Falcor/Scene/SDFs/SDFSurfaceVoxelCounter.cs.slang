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

#define GROUP_WIDTH 4

cbuffer CB
{
    uint gGridWidth;
};

Texture3D<float> gSDFGrid;
RWByteAddressBuffer gTotalVoxelCount;

groupshared uint gGroupVoxelCount;
groupshared float gGroupValues[(GROUP_WIDTH + 1) * (GROUP_WIDTH + 1) * (GROUP_WIDTH + 1)];

uint getGroupValueIndex(uint3 groupValueCoords)
{
    return groupValueCoords.x + (GROUP_WIDTH + 1) * (groupValueCoords.y + (GROUP_WIDTH + 1) * groupValueCoords.z);
}

[numthreads(GROUP_WIDTH, GROUP_WIDTH, GROUP_WIDTH)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (any(dispatchThreadID >= gGridWidth)) return;

    const uint3 voxelCoords = dispatchThreadID;
    const bool isGroupManager = all(groupThreadID == uint3(0));

    // Initialize group values.
    if (isGroupManager)
    {
        gGroupVoxelCount = 0;
    }

    gGroupValues[getGroupValueIndex(groupThreadID)] = gSDFGrid[voxelCoords].x;

    bool3 borderThread = groupThreadID == GROUP_WIDTH - 1 || dispatchThreadID == gGridWidth - 1;

    if (borderThread.x)
    {
        gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 0))] = gSDFGrid[voxelCoords + uint3(1, 0, 0)].x;

        if (borderThread.y)
        {
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 0))] = gSDFGrid[voxelCoords + uint3(0, 1, 0)].x;
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 1, 0))] = gSDFGrid[voxelCoords + uint3(1, 1, 0)].x;

            if (borderThread.z)
            {
                gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
                gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 1))] = gSDFGrid[voxelCoords + uint3(0, 1, 1)].x;
                gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 1))] = gSDFGrid[voxelCoords + uint3(1, 0, 1)].x;
                gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 1, 1))] = gSDFGrid[voxelCoords + uint3(1, 1, 1)].x;
            }
        }
        else if (borderThread.z)
        {
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 1))] = gSDFGrid[voxelCoords + uint3(1, 0, 1)].x;
        }
    }
    else if (borderThread.y)
    {
        gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 0))] = gSDFGrid[voxelCoords + uint3(0, 1, 0)].x;

        if (borderThread.z)
        {
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 1))] = gSDFGrid[voxelCoords + uint3(0, 1, 1)].x;
        }
    }
    else if (borderThread.z)
    {
        gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
    }

    GroupMemoryBarrierWithGroupSync();
    
    // Load voxel corner values.
    float4 values0xx;
    float4 values1xx;
    values0xx[0] = gGroupValues[getGroupValueIndex(groupThreadID)];
    values0xx[1] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))];
    values0xx[2] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 0))];
    values0xx[3] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 1))];
    values1xx[0] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 0))];
    values1xx[1] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 1))];
    values1xx[2] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 1, 0))];
    values1xx[3] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 1, 1))];

    if (SDFVoxelCommon::containsSurface(values0xx, values1xx))
    {
        InterlockedAdd(gGroupVoxelCount, 1);
    }

    GroupMemoryBarrierWithGroupSync();

    if (isGroupManager)
    {
        gTotalVoxelCount.InterlockedAdd(0, gGroupVoxelCount);
    }
}
