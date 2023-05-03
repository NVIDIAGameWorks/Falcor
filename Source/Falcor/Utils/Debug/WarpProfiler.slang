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

/// Import this module to use the warp profiler.
/// The shader program must be compiled for shader model 6.5 or above.
/// The implementation currently assumes 32-wide warps for NVIDIA GPUs.

/// Global buffer for collecting warp histograms.
RWStructuredBuffer<uint> gWarpHistogram;

/**
 * @brief Call to measure the utilization in the current warp.
 * The utilization is measured as the number of active threads in the warp.
 * The histogram of active thread count can later be queried on the host.
 * @param[in] binIndex Which profiling bin to use.
 */
void warpUtilization(uint binIndex)
{
    uint activeCount = WaveActiveCountBits(true);
    if (WaveIsFirstLane())
    {
        binIndex = binIndex * 32 + activeCount - 1;
        InterlockedAdd(gWarpHistogram[binIndex], 1);
    }
}

/**
 * @brief Call to measure the amount of divergence in the current warp.
 * The divergence is measured as the number of different values seen among the active threads in the warp.
 * The histogram of divergence can later be queried on the host.
 * @param[in] binIndex Which profiling bin to use.
 * @param[in] value Value for which divergence is measured.
 */
void warpDivergence(uint binIndex, uint value)
{
    // We use wave match to identify each group of distinct values across the warp.
    // The first lane in each such group is active in the branch below and `warpUtilization()`
    // is called to count the number of active threads and record the result.
    uint groupMask = WaveMatch(value).x;
    uint firstLaneInGroup = firstbitlow(groupMask);
    if (WaveGetLaneIndex() == firstLaneInGroup)
    {
        warpUtilization(binIndex);
    }
}
