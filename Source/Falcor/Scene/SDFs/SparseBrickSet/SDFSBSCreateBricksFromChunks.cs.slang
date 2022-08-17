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
import Scene.SDFs.SparseBrickSet.BC4Encode;
import Utils.Math.AABB;
import Utils.Math.PackedFormats;

static const uint kGroupWidth = BRICK_WIDTH + 1;
static const uint kChunkWidth = CHUNK_WIDTH;
static const uint kBrickWidthInVoxels = BRICK_WIDTH;
static const uint kBrickWidthInValues = BRICK_WIDTH + 1;
static const uint kCompressionWidth = 4;
static const float kRootThree = sqrt(3.0f);

struct ParamBlock
{
    uint primitiveCount;
    uint gridWidth;
    uint brickCount;
    uint2 bricksPerAxis;

    StructuredBuffer<SDF3DPrimitive> primitives;
    ByteAddressBuffer chunkCoords;

    RWStructuredBuffer<AABB> brickAABBs;
    RWTexture3D<uint> indirectionBuffer;
#if COMPRESS_BRICKS
    RWTexture2D<uint2> bricks;
#else
    RWTexture2D<float> bricks;
#endif

    Texture3D<float> sdfGrid;
};

ParameterBlock<ParamBlock> gParamBlock;

float evalCoords(const uint3 coords)
{
    const float3 p = -0.5f + float3(coords) / gParamBlock.gridWidth;

    float sd = FLT_MAX;

#ifdef _BUILD_WITH_SD_FIELD
    sd = gParamBlock.sdfGrid[coords];
#endif
    
    for (uint primitiveID = 0; primitiveID < gParamBlock.primitiveCount; primitiveID++)
    {
        SDF3DPrimitive primitive = gParamBlock.primitives[primitiveID];
        sd = primitive.eval(p, sd);
    }

    // Normalize the distance such that +-1 represent the half voxel distance to its diagonal. Any other distance greater than that is clamped to +-1.
    return clamp(sd * 2.0f * gParamBlock.gridWidth / kRootThree, -1.0f, 1.0f);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint brickID = dispatchThreadID.x;
    if (brickID >= gParamBlock.brickCount) return;

    // Load global virtualBrickCoords.
    const uint3 virtualBrickCoords = gParamBlock.chunkCoords.Load3((3 * brickID) << 2);

    // Calculate brick grid coords.
    uint3 brickGridCoords = virtualBrickCoords * kBrickWidthInVoxels;

    // Calculate the AABB min and max corners for the brick.
    float3 brickAABBMin = -0.5f + float3(brickGridCoords) / float(gParamBlock.gridWidth);
    float3 brickAABBMax = min(brickAABBMin + kBrickWidthInVoxels / float(gParamBlock.gridWidth), 0.5f);
    gParamBlock.brickAABBs[brickID] = AABB(brickAABBMin, brickAABBMax);

    // Write the brickID to the indirection buffer.
    gParamBlock.indirectionBuffer[virtualBrickCoords] = brickID;

    // Calculate the min corner of the brick in the brick texture.
    uint2 brickTextureCoords = uint2(brickID % gParamBlock.bricksPerAxis.x, brickID / gParamBlock.bricksPerAxis.x) * uint2(kBrickWidthInValues * kBrickWidthInValues, kBrickWidthInValues);

    // Write brick values.
    for (uint z = 0; z < kBrickWidthInValues; ++z)
    {
        for (uint y = 0; y < kBrickWidthInValues; y += kCompressionWidth)
        {
            for (uint x = 0; x < kBrickWidthInValues; x += kCompressionWidth)
            {
                uint2 voxelTextureCoords = brickTextureCoords + uint2(x + z * kBrickWidthInValues, y);
                uint2 blockTextureCoords = voxelTextureCoords / kCompressionWidth;

                uint3 blockGridCoords = brickGridCoords + uint3(x, y, z);

                int4x4 block;
                for (uint bY = 0; bY < kCompressionWidth && bY + y < kBrickWidthInValues; ++bY)
                {
                    for (uint bX = 0; bX < kCompressionWidth && bX + x < kBrickWidthInValues; ++bX)
                    {
                        uint3 voxelGridCoords = blockGridCoords + uint3(bX, bY, 0);

#if COMPRESS_BRICKS
                        if (all(voxelGridCoords < gParamBlock.gridWidth))
                        {
                            const float sd = evalCoords(voxelGridCoords);

                            // Convert to snorm.
                            float intScale = sd * 127.0f;
                            block[bY][bX] = int(intScale >= 0.0f ? intScale + 0.5f : intScale - 0.5f);
                        }
                        else
                        {
                            block[bY][bX] = 127;
                        }
#else
                        if (all(voxelGridCoords < gParamBlock.gridWidth))
                        {
                            const float sd = evalCoords(voxelGridCoords);
                            gParamBlock.bricks[voxelTextureCoords + uint2(bX, bY)] = sd;
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
