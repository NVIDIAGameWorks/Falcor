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

/** Functionality for collecting runtime stats in the path tracer.

    Note that the log*() functions must be called from a ray tracing
    program as they use DispatchRaysIndex() to identify the current pixel.

    The host sets the following defines:

    _LOGGING_ENABLE_STATS       Nonzero if stats should be collected.

*/

shared RWTexture2D<uint> gStatsRayCount;            // Per-pixel ray count stats.
shared RWTexture2D<uint> gStatsPathLength;          // Per-pixel path length.

void logTraceRay()
{
#if _LOGGING_ENABLE_STATS
    uint2 launchIndex = DispatchRaysIndex().xy;
    InterlockedAdd(gStatsRayCount[launchIndex], 1);
#endif
}

void logPathLength(uint pathLength)
{
#if _LOGGING_ENABLE_STATS
    uint2 launchIndex = DispatchRaysIndex().xy;
    gStatsPathLength[launchIndex] = pathLength;
#endif
}
