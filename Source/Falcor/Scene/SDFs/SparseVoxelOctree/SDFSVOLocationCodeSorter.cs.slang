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

/** In place bitonic sort of 64-bit location codes.
    Adapted from https://poniesandlight.co.uk/reflect/bitonic_merge_sort/.
*/

cbuffer CB
{
    uint gAlgorithm;
    uint gComparisonHeight; 
};

RWByteAddressBuffer gBuffer;

groupshared uint2 gGroupBuffer[GROUP_SIZE * 2];

void localCompareSwap(const uint a, const uint b)
{
    uint2 aVal = gGroupBuffer[a];
    uint2 bVal = gGroupBuffer[b];

    if (bVal.y < aVal.y || (bVal.y == aVal.y && bVal.x < aVal.x))
    {
        gGroupBuffer[a] = bVal;
        gGroupBuffer[b] = aVal;
    }
}

void globalCompareSwap(const uint a, const uint b)
{
    uint2 aVal = gBuffer.Load2(a * 8);
    uint2 bVal = gBuffer.Load2(b * 8);

    if (bVal.y < aVal.y || (bVal.y == aVal.y && bVal.x < aVal.x))
    {
        gBuffer.Store2(a * 8, bVal);
        gBuffer.Store2(b * 8, aVal);
    }
}

void localFlip(const uint localThreadID, const uint comparisonHeight)
{
    GroupMemoryBarrierWithGroupSync();

    uint halfComparisonHeightMinus1 = (comparisonHeight >> 1) - 1;
    uint comparisonGroupOffset = comparisonHeight * ((localThreadID << 1) / comparisonHeight);

    uint a = comparisonGroupOffset + (localThreadID & halfComparisonHeightMinus1);
    uint b = (comparisonGroupOffset + comparisonHeight) - ((localThreadID & halfComparisonHeightMinus1) + 1);

    localCompareSwap(a, b);
}

void localDisperse(const uint localThreadID, const uint comparisonHeight)
{
    for (uint ch = comparisonHeight; ch > 1; ch >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();

        uint halfComparisonHeight = ch >> 1;
        uint comparisonGroupOffset = ch * ((localThreadID << 1) / ch);

        uint a = comparisonGroupOffset + (localThreadID & (halfComparisonHeight - 1));
        uint b = a + halfComparisonHeight;

        localCompareSwap(a, b);
    }
}

void localBMS(const uint localThreadID, const uint comparisonHeight)
{
    for (uint currentComparisonHeight = 2; currentComparisonHeight <= comparisonHeight; currentComparisonHeight <<= 1)
    {
        localFlip(localThreadID, currentComparisonHeight);
        localDisperse(localThreadID, currentComparisonHeight >> 1);
    }
}

void bigFlip(const uint globalThreadID, const uint comparisonHeight)
{
    if ((GROUP_SIZE << 1) > comparisonHeight) return;

    uint halfComparisonHeightMinus1 = (comparisonHeight >> 1) - 1;
    uint comparisonGroupOffset = comparisonHeight * ((globalThreadID << 1) / comparisonHeight);

    uint a = comparisonGroupOffset + (globalThreadID & halfComparisonHeightMinus1);
    uint b = comparisonGroupOffset + comparisonHeight - (globalThreadID & halfComparisonHeightMinus1) - 1;

    globalCompareSwap(a, b);
}

void bigDisperse(const uint globalThreadID, const uint comparisonHeight)
{
    if ((GROUP_SIZE << 1) > comparisonHeight) return;

    uint halfComparisonHeight = comparisonHeight >> 1;
    uint comparisonGroupOffset = comparisonHeight * ((globalThreadID << 1) / comparisonHeight);

    uint a = comparisonGroupOffset + (globalThreadID & (halfComparisonHeight - 1));
    uint b = a + halfComparisonHeight;

    globalCompareSwap(a, b);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    if (gAlgorithm == LOCAL_BMS || gAlgorithm == LOCAL_DISPERSE)
    {
        const uint localThreadID = groupThreadID.x;
        const uint localThreadOffset = localThreadID << 1;
        const uint groupOffset = 2 * groupID.x * GROUP_SIZE;

        uint4 globalData = gBuffer.Load4((groupOffset + localThreadOffset) * 8);
        gGroupBuffer[localThreadOffset] = globalData.xy;
        gGroupBuffer[localThreadOffset + 1] = globalData.zw;

        if (gAlgorithm == LOCAL_BMS) localBMS(localThreadID, gComparisonHeight);
        else localDisperse(localThreadID, gComparisonHeight);

        GroupMemoryBarrierWithGroupSync();

        gBuffer.Store4((groupOffset + localThreadOffset) * 8, uint4(gGroupBuffer[localThreadOffset], gGroupBuffer[localThreadOffset + 1]));
    }
    else
    {
        const uint globalThreadID = dispatchThreadID.x;

        if (gAlgorithm == BIG_FLIP) bigFlip(globalThreadID, gComparisonHeight);
        else bigDisperse(globalThreadID, gComparisonHeight);
    }
}
