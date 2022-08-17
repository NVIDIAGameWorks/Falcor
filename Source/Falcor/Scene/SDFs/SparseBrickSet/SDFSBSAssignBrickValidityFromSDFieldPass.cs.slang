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
import Scene.SDFs.SDFVoxelCommon;

static const uint kGroupWidth = GROUP_WIDTH;

struct ParamBlock
{
    uint virtualGridWidth;
    uint virtualBricksPerAxis;
    uint brickWidthInVoxels;
    Texture3D<float> sdfGrid;
    RWByteAddressBuffer brickValidity;
};

ParameterBlock<ParamBlock> gParamBlock;

#if GROUP_BRICK_CREATION
groupshared uint gBrickValid;
#endif

[numthreads(kGroupWidth, kGroupWidth, kGroupWidth)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (any(dispatchThreadID >= gParamBlock.virtualGridWidth))
        return;

    const uint3 voxelCoords = dispatchThreadID;

    // Initialize group values.
#if GROUP_BRICK_CREATION
    const bool isGroupManager = all(groupThreadID == uint3(0));

    if (isGroupManager)
    {
        gBrickValid = 0;
    }

    GroupMemoryBarrierWithGroupSync();
#else
    uint voxelValid = 0;
#endif

    // Load voxel corner values.
    float4 values0xx;
    float4 values1xx;
    values0xx[0] = gParamBlock.sdfGrid[voxelCoords].x;
    values0xx[1] = gParamBlock.sdfGrid[voxelCoords + uint3(0, 0, 1)].x;
    values0xx[2] = gParamBlock.sdfGrid[voxelCoords + uint3(0, 1, 0)].x;
    values0xx[3] = gParamBlock.sdfGrid[voxelCoords + uint3(0, 1, 1)].x;
    values1xx[0] = gParamBlock.sdfGrid[voxelCoords + uint3(1, 0, 0)].x;
    values1xx[1] = gParamBlock.sdfGrid[voxelCoords + uint3(1, 0, 1)].x;
    values1xx[2] = gParamBlock.sdfGrid[voxelCoords + uint3(1, 1, 0)].x;
    values1xx[3] = gParamBlock.sdfGrid[voxelCoords + uint3(1, 1, 1)].x;

    if (SDFVoxelCommon.containsSurface(values0xx, values1xx))
    {
#if GROUP_BRICK_CREATION
        gBrickValid = 1;
#else
        voxelValid = 1;
#endif
    }

#if GROUP_BRICK_CREATION
    GroupMemoryBarrierWithGroupSync();
#endif

#if GROUP_BRICK_CREATION
    if (isGroupManager)
    {
        uint3 virtualBrickCoords = voxelCoords / gParamBlock.brickWidthInVoxels;
        uint virtualBrickID = virtualBrickCoords.x + gParamBlock.virtualBricksPerAxis * (virtualBrickCoords.y + gParamBlock.virtualBricksPerAxis * virtualBrickCoords.z);

        gParamBlock.brickValidity.Store(virtualBrickID * 4, gBrickValid);
    }
#else
    uint3 virtualBrickCoords = voxelCoords / gParamBlock.brickWidthInVoxels;
    uint virtualBrickID = virtualBrickCoords.x + gParamBlock.virtualBricksPerAxis * (virtualBrickCoords.y + gParamBlock.virtualBricksPerAxis * virtualBrickCoords.z);
    gParamBlock.brickValidity.InterlockedOr(virtualBrickID * 4, voxelValid);
#endif
}
