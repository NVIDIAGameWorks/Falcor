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

/** In-place bitonic sort of 32-bit values in chunks of N elements.

    The host sets these two defines:
    CHUNK_SIZE (= N)    chunk size which must be a power-of-two, and
    GROUP_SIZE          thread group size, must be a power-of-two <= 1024.

    The code uses horizontal instructions to shuffle within the warp when possible,
    and shared memory to shuffle between warps.
    Shuffles are not yet available in shader model 6.0+, we therefore rely on NVAPI.
*/
#include "Utils/NVAPI.slangh" // We need this to get shuffle-xor operations.

#if (NV_WARP_SIZE != 32)
#error Kernel assumes warp size 32
#endif

// Check constraints.
// The kernel is currently written for a 1:1 mapping between elements to sort and threads.
#if (CHUNK_SIZE > GROUP_SIZE)
#error CHUNK_SIZE > GROUP_SIZE not supported
#endif

cbuffer CB
{
    uint gTotalSize;        ///< Total number of elements.
    uint gDispatchX;        ///< Number of thread groups in dispatch dimension X.
};

RWByteAddressBuffer gData;                      ///< The data buffer we're sorting in-place.

groupshared uint gSharedData[GROUP_SIZE * 2];   ///< Temporary working buffer in shared memory.


/** Within warp bitonic sort, for iterations {j, j/2, ..., 1}, where j <= 16.
    \param[in,out] value The current thread's value.
    \param[in] i Global element index.
    \param[in] j Start element offset j<=16.
    \param[in] dir Sorting ascending (true) or descending (false).
*/
void bitonicSortInWarp(inout uint value, uint i, uint j, bool dir)
{
    for (; j > 0; j >>= 1)
    {
        // Get index of sorting partner in chunk
        uint value_ixj = NvShflXor(value, j);       // Value from current lane ^ j

        // Decide whether to swap.
        bool pred = (((i & j) == 0) != dir) == value < value_ixj;
        if (pred) value = value_ixj;
    }
}

/** In-place bitonic sort.
*/
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIdx : SV_GroupIndex)
{
    const uint group = groupID.y * gDispatchX + groupID.x;  // Sequential group index.
    const uint thid = groupIdx;                             // Local thread index in group (range 0..GROUP_SIZE-1).

    const uint globalIdx = group * GROUP_SIZE + thid;       // Global element index in gData
    const uint globalAddr = globalIdx * 4;                  // Address of current element in gData

    const uint N = CHUNK_SIZE;                              // Number of elements per chunk to sort. Must be a power-of-two.
    const uint i = globalIdx & (N - 1);                     // i = local index of element in chunk, range [0,N).

    // Load value from memory.
    // Out-of-bounds elements are set to UINT_MAX (-1) to be placed last and allow data that is not a multiple of chunk size.
    uint value = uint(-1);
    if (globalIdx < gTotalSize)
    {
        value = gData.Load(globalAddr);
    }

    // Major steps for k = {2,4,...,32} are done within warp.
    for (uint k = 2; k <= min(N,32); k <<= 1)
    {
        // Minor steps for iterations j = {16, 8, ..., 1} in warp.
        const bool dir = ((i & k) == 0);   // Sort ascending (true) or descending (false)
        uint j = k >> 1;    // j <= 16
        bitonicSortInWarp(value, i, j, dir);
    }

#if (CHUNK_SIZE > 32)
    // Load data into shared memory.
    gSharedData[thid * 2] = value;
    GroupMemoryBarrierWithGroupSync();

    // Major steps for k = {64,128,...N} are done in shared memory.
    for (uint k = 64; k <= N; k <<= 1)
    {
        const bool dir = ((i & k) == 0);   // Sort ascending (true) or descending (false)

        // We ping-pong data in shared memory between adjacent addresses, using offset = {0, 1} to denote which one.
        uint offset = 0;

        // Minor steps for iterations j = {k/2, k/4, ..., 32} in shared memory.
        for (uint j = k >> 1; j >= 32; j >>= 1)
        {
            // Get sorting partner.
            uint value_ixj = gSharedData[(thid ^ j) * 2 + offset];

            // Decide whether to swap. See comments in bitonicSortInWarp().
            bool pred = (((i & j) == 0) != dir) == value < value_ixj;
            if (pred) value = value_ixj;

            // Store result for next minor step (except for last iteration). Write to offset address to avoid race condition.
            if (j > 32)
            {
                gSharedData[thid * 2 + (offset ^ 1)] = value;
                GroupMemoryBarrierWithGroupSync();
                offset ^= 1;
            }
        }

        // Minor steps for iterations j = {16, 8, ..., 1} in warp.
        uint jStart = min(k >> 1, 16);
        bitonicSortInWarp(value, i, jStart, dir);

        // Store result for major step (except for last iteration).
        if (k < N)
        {
            gSharedData[thid * 2] = value;
            GroupMemoryBarrierWithGroupSync();
        }
    }
#endif

    // Write result to memory.
    if (globalIdx < gTotalSize)
    {
        gData.Store(globalAddr, value);
    }
}
