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

/** Parallel reduction using shared memory and warp instructions.

    The host sets these defines:
    - FORMAT_CHANNELS <N>   Number of components in the data (N=1..4).
    - FORMAT_TYPE <type>    Texture format type. See ParallelReductionType.slangh.
*/
#include "ParallelReductionType.slangh"
#include "Utils/Math/MathConstants.slangh"

cbuffer PerFrameCB
{
    uint2 gResolution;      // Pixel dimensions of input texture.
    uint2 gNumTiles;        // Number of tiles in input texture.
    uint gElems;            // Number of elements in input intermediate buffer.
};

// Typedef the data format as 'DataType'. The format is specified from the host.
#if FORMAT_TYPE == FORMAT_TYPE_FLOAT
typedef vector<float, 4> DataType;
static const float kMaxValue = FLT_MAX;
static const float kMinValue = -FLT_MAX;
#elif FORMAT_TYPE == FORMAT_TYPE_SINT
typedef vector<int, 4> DataType;
static const int kMinValue = INT32_MIN;
static const int kMaxValue = INT32_MAX;
#elif FORMAT_TYPE == FORMAT_TYPE_UINT
typedef vector<uint, 4> DataType;
static const uint kMinValue = 0;
static const uint kMaxValue = UINT32_MAX;
#endif

// Declare the input texture.
Texture2D<DataType> gInput;

Buffer<DataType> gInputBuffer;
RWBuffer<DataType> gResult;

#if REDUCTION_TYPE == REDUCTION_TYPE_SUM
groupshared DataType gIntermediateCache[32 /* = 1024 / 32 */];
#elif REDUCTION_TYPE == REDUCTION_TYPE_MINMAX
groupshared DataType gIntermediateCache[64 /* = 1024 / 32 * 2 */];
#endif


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
void reduceSum(DataType value, uint dstIdx, uint groupThreadIdx)
{
    // Add all elements within warp. Store result to shared memory.
    {
        value = WaveActiveSum(value);
        if (WaveIsFirstLane()) gIntermediateCache[groupThreadIdx / 32] = value;
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

/** Take the min/max of all elements within a thread group and writes single result to the results buffer at 'dstIdx'.
*/
void reduceMinMax(DataType minValue, DataType maxValue, uint dstIdx, uint groupThreadIdx)
{
    // Add all elements within warp. Store result to shared memory.
    {
        minValue = WaveActiveMin(minValue);
        maxValue = WaveActiveMax(maxValue);
        if (WaveIsFirstLane())
        {
            gIntermediateCache[(groupThreadIdx / 32) * 2] = minValue;
            gIntermediateCache[(groupThreadIdx / 32) * 2 + 1] = maxValue;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Take the min/max of all elements produced by the warps.
    if (groupThreadIdx < 32)
    {
        minValue = gIntermediateCache[groupThreadIdx * 2];
        maxValue = gIntermediateCache[groupThreadIdx * 2 + 1];
        minValue = WaveActiveMin(minValue);
        maxValue = WaveActiveMax(maxValue);
        if (groupThreadIdx == 0)
        {
            gResult[dstIdx * 2] = minValue;
            gResult[dstIdx * 2 + 1] = maxValue;
        }
    }
}

[numthreads(32, 32, 1)]
void initialPass(uint3 globalThreadId : SV_DispatchThreadID, uint groupThreadIdx : SV_GroupIndex, uint3 groupId : SV_GroupID)
{
    const uint2 pixelCoords = globalThreadId.xy;
    const uint tileIdx = groupId.y * gNumTiles.x + groupId.x;

#if REDUCTION_TYPE == REDUCTION_TYPE_SUM
    // Load input from texture in tiles of 32x32 pixels.
    DataType value = 0;
    if (all(pixelCoords < gResolution)) value = loadTexture(pixelCoords);

    reduceSum(value, tileIdx, groupThreadIdx);
#elif REDUCTION_TYPE == REDUCTION_TYPE_MINMAX
    // Load input from texture in tiles of 32x32 pixels.
    DataType minValue = kMaxValue;
    DataType maxValue = kMinValue;
    if (all(pixelCoords < gResolution)) minValue = maxValue = loadTexture(pixelCoords);

    reduceMinMax(minValue, maxValue, tileIdx, groupThreadIdx);
#endif
}

[numthreads(1024, 1, 1)]
void finalPass(uint3 globalThreadId : SV_DispatchThreadID, uint groupThreadIdx : SV_GroupIndex, uint3 groupId : SV_GroupID)
{
#if REDUCTION_TYPE == REDUCTION_TYPE_SUM
    // Load input from buffer written in previous pass.
    DataType value = 0;
    if (globalThreadId.x < gElems) value = gInputBuffer[globalThreadId.x];

    reduceSum(value, groupId.x, groupThreadIdx);
#elif REDUCTION_TYPE == REDUCTION_TYPE_MINMAX
    // Load input from buffer written in previous pass.
    DataType minValue = kMaxValue;
    DataType maxValue = kMinValue;
    if (globalThreadId.x < gElems)
    {
        minValue = gInputBuffer[globalThreadId.x * 2];
        maxValue = gInputBuffer[globalThreadId.x * 2 + 1];
    }

    reduceMinMax(minValue, maxValue, groupId.x, groupThreadIdx);
#endif
}
