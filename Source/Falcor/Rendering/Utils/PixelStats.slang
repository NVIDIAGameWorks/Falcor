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

/** Functionality for collecting runtime stats in the path tracer.

    The host sets the following defines:

    _PIXEL_STATS_ENABLED       Nonzero if stats should be collected.

*/
__exported import PixelStatsShared;

RWTexture2D<uint> gStatsRayCount[(uint)PixelStatsRayType::Count]; // Per-pixel ray count stats.
RWTexture2D<uint> gStatsPathLength;          // Per-pixel path length.
RWTexture2D<uint> gStatsPathVertexCount;     // Per-pixel path vertex count.
RWTexture2D<uint> gStatsVolumeLookupCount;   // Per-pixel volume lookup count.

#ifdef _PIXEL_STATS_ENABLED
static uint2 gPixelStatsPixel;
#endif

void logSetPixel(uint2 pixel)
{
#ifdef _PIXEL_STATS_ENABLED
    gPixelStatsPixel = pixel;
#endif
}

void logTraceRay(PixelStatsRayType rayType)
{
#ifdef _PIXEL_STATS_ENABLED
    InterlockedAdd(gStatsRayCount[(uint)rayType][gPixelStatsPixel], 1);
#endif
}

void logPathLength(uint pathLength)
{
#ifdef _PIXEL_STATS_ENABLED
    gStatsPathLength[gPixelStatsPixel] = pathLength;
#endif
}

void logPathVertex()
{
#ifdef _PIXEL_STATS_ENABLED
    InterlockedAdd(gStatsPathVertexCount[gPixelStatsPixel], 1);
#endif
}

void logVolumeLookup()
{
#ifdef _PIXEL_STATS_ENABLED
    InterlockedAdd(gStatsVolumeLookupCount[gPixelStatsPixel], 1);
#endif
}
