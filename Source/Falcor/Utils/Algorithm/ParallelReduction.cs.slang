/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/

/** Parallel reduction using shared memory and warp instructions.

    The host sets these defines:
    - FORMAT_CHANNELS <N>   Number of components in the data (N=1..4).
    - FORMAT_TYPE <type>    Texture format type. See ParallelReductionType.h.
*/
#include "ParallelReductionType.h"

cbuffer PerFrameCB
{
    uint2 gResolution;      // Pixel dimensions of input texture.
    uint2 gNumTiles;        // Number of tiles in input texture.
    uint gElems;            // Number of elements in input intermediate buffer.
};

// Typedef the data format as 'DataType'. The format is specified from the host.
#if FORMAT_TYPE == FORMAT_TYPE_FLOAT
typedef vector<float, 4> DataType;
#elif FORMAT_TYPE == FORMAT_TYPE_SINT
typedef vector<int, 4> DataType;
#elif FORMAT_TYPE == FORMAT_TYPE_UINT
typedef vector<uint, 4> DataType;
#endif

// Declare the input texture.
Texture2D<DataType> gInput;

Buffer<DataType> gInputBuffer;
RWBuffer<DataType> gResult;

groupshared DataType gIntermediateCache[32 /* = 1024 / 32 */];

DataType loadTexture(uint2 pixelCoords)
{
    DataType value = gInput[pixelCoords];
#if FORMAT_CHANNELS < 4
    // The default value for missing components is (0,0,0,1). Reset last component to zero for consistency.
    value.w = 0.f;
#endif
    return value;
}

/** Performs reduction within a thread group and writes single result to the results buffer at 'dstIdx'.
*/
void reduce(DataType value, uint dstIdx, uint groupThreadIdx)
{
    // Add all elements within warp. Store result to shared memory.
    {
        value = WaveActiveSum(value);
        if (WaveIsFirstLane() == 0) gIntermediateCache[groupThreadIdx / 32] = value;
    }
    GroupMemoryBarrierWithGroupSync();

    // Add all elements produced by the warps.
    if (groupThreadIdx < 32)
    {
        value = gIntermediateCache[groupThreadIdx];
        value = WaveActiveSum(value);
        if (groupThreadIdx == 0) gResult[dstIdx] = value;
    }
}

[numthreads(32, 32, 1)]
void initialPass(uint3 globalThreadId : SV_DispatchThreadID, uint groupThreadIdx : SV_GroupIndex, uint3 groupId : SV_GroupID)
{
    const uint2 pixelCoords = globalThreadId.xy;
    const uint tileIdx = groupId.y * gNumTiles.x + groupId.x;

    // Load input from texture in tiles of 32x32 pixels.
    DataType value = 0;
    if (all(pixelCoords < gResolution)) value = loadTexture(pixelCoords);

    reduce(value, tileIdx, groupThreadIdx);
}

[numthreads(1024, 1, 1)]
void finalPass(uint3 globalThreadId : SV_DispatchThreadID, uint groupThreadIdx : SV_GroupIndex, uint3 groupId : SV_GroupID)
{
    // Load input from buffer written in previous pass.
    DataType value = 0;
    if (globalThreadId.x < gElems) value = gInputBuffer[globalThreadId.x];

    reduce(value, groupId.x, groupThreadIdx);
}
