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

import Utils.Math.AABB;
import Utils.Math.FormatConversion;
import Scene.SDFs.SparseBrickSet.BC4Encode;

static const uint kCompressionWidth = 4;
static const float kRootThree = sqrt(3.0f);

struct ParamBlock
{
    uint virtualGridWidth;
    uint virtualBrickCount;
    uint virtualBricksPerAxis;
    uint brickCount;
    uint brickWidthInVoxels;
    uint2 bricksPerAxis;
    Texture3D<float> sdfGrid;
    Buffer<uint> indirectionBuffer;

    RWStructuredBuffer<AABB> brickAABBs;
#if COMPRESS_BRICKS
    RWTexture2D<uint2> bricks;
#else
    RWTexture2D<float> bricks;
#endif
};

ParameterBlock<ParamBlock> gParamBlock;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (dispatchThreadID.x >= gParamBlock.virtualBrickCount) return;

    const uint virtualBrickID = dispatchThreadID.x;

    uint brickID = gParamBlock.indirectionBuffer[virtualBrickID];
    if (brickID == UINT32_MAX) return;

    // Calculate virtual brick coords.
    uint3 virtualBrickCoords = uint3(virtualBrickID, gParamBlock.virtualBricksPerAxis * gParamBlock.virtualBricksPerAxis, 0);
    virtualBrickCoords.z = virtualBrickCoords.x / virtualBrickCoords.y;  // = virtualBrickID / (virtualBricksPerAxis * virtualBricksPerAxis)
    virtualBrickCoords.x -= virtualBrickCoords.z * virtualBrickCoords.y; // = brickCoords.z * (virtualBricksPerAxis * virtualBricksPerAxis)
    virtualBrickCoords.y = virtualBrickCoords.x / gParamBlock.virtualBricksPerAxis;
    virtualBrickCoords.x = virtualBrickCoords.x % gParamBlock.virtualBricksPerAxis;

    // Calculate the min corner of the brick in the grid.
    uint brickWidthInValues = gParamBlock.brickWidthInVoxels + 1;
    uint3 brickGridCoords = virtualBrickCoords * gParamBlock.brickWidthInVoxels;

    // Calculate the AABB min and max corners for the brick.
    float3 brickAABBMin = -0.5f + float3(virtualBrickCoords * gParamBlock.brickWidthInVoxels) / float(gParamBlock.virtualGridWidth);
    float3 brickAABBMax = min(brickAABBMin + float(gParamBlock.brickWidthInVoxels) / float(gParamBlock.virtualGridWidth), 0.5f);
    gParamBlock.brickAABBs[brickID] = AABB(brickAABBMin, brickAABBMax);

    // Calculate the min corner of the brick in the brick texture.
    uint2 brickTextureCoords = uint2(brickID % gParamBlock.bricksPerAxis.x, brickID / gParamBlock.bricksPerAxis.x) * uint2(brickWidthInValues * brickWidthInValues, brickWidthInValues);

    // Write brick values.
    for (uint z = 0; z < brickWidthInValues; ++z)
    {
        for (uint y = 0; y < brickWidthInValues; y += kCompressionWidth)
        {
            for (uint x = 0; x < brickWidthInValues; x += kCompressionWidth)
            {
                uint2 voxelTextureCoords = brickTextureCoords + uint2(x + z * brickWidthInValues, y);
                uint2 blockTextureCoords = voxelTextureCoords / kCompressionWidth;

                uint3 blockGridCoords = brickGridCoords + uint3(x, y, z);

                int4x4 block;
                for (uint bY = 0; bY < kCompressionWidth && bY + y < brickWidthInValues; ++bY)
                {
                    for (uint bX = 0; bX < kCompressionWidth && bX + x < brickWidthInValues; ++bX)
                    {
                        uint3 voxelGridCoords = blockGridCoords + uint3(bX, bY, 0);

#if COMPRESS_BRICKS
                        if (all(voxelGridCoords < gParamBlock.virtualGridWidth))
                        {
                            float cornerDistance = gParamBlock.sdfGrid[voxelGridCoords];

                            // Convert to snorm.
                            float intScale = cornerDistance * 127.0f;
                            block[bY][bX] = int(intScale >= 0.0f ? intScale + 0.5f : intScale - 0.5f);
                        }
                        else
                        {
                            block[bY][bX] = 127;
                        }
#else
                        if (all(voxelGridCoords < gParamBlock.virtualGridWidth))
                        {
                            float cornerDistance = gParamBlock.sdfGrid[voxelGridCoords];
                            float normalization = 2.0f * gParamBlock.virtualGridWidth / kRootThree;
                            gParamBlock.bricks[voxelTextureCoords + uint2(bX, bY)] = clamp(cornerDistance* normalization, -1.0f, 1.0f);
                        }
                        else
                        {
                            gParamBlock.bricks[voxelTextureCoords + uint2(bX, bY)] = 1.0f;
                        }
#endif
                    }
                }

#if COMPRESS_BRICKS
                gParamBlock.bricks[blockTextureCoords] = compressBlock(block);
#endif
            }
        }
    }
}
