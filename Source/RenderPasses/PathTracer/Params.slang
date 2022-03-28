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
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Per-sample color format.
*/
enum class ColorFormat : uint32_t
{
    RGBA32F         = 0,
    LogLuvHDR       = 1,
};

enum class MISHeuristic : uint32_t
{
    Balance     = 0,    ///< Balance heuristic.
    PowerTwo    = 1,    ///< Power heuristic (exponent = 2.0).
    PowerExp    = 2,    ///< Power heuristic (variable exponent).
};

// Define tile sizes in pixels.
// The frame is divided into tiles stored in scanline order, with pixels in tiles enumerated in Morton order.
static const uint2 kScreenTileDim = { 16, 16 };     ///< Screen-tile dimension in pixels.
static const uint2 kScreenTileBits = { 4, 4 };      ///< Bits needed to describe pixel position within a screen-tile.

// Define path configuration limits.
static const uint kMaxSamplesPerPixel = 16;         ///< Maximum supported sample count. We can use tiling to support large sample counts if needed.
static const uint kMaxFrameDimension = 4096;        ///< Maximum supported frame dimension in pixels along x or y. We can increase the bit allocation if needed.
static const uint kMaxBounces = 254;                ///< Maximum supported number of bounces per bounce category (value 255 is reserved for internal use). The resulting path length may be longer than this.
static const uint kMaxLightSamplesPerVertex = 8;    ///< Maximum number of shadow rays per path vertex for next-event estimation.

// Import static specialization constants.
#ifndef HOST_CODE
__exported import StaticParams;
__exported import Utils.Math.BitTricks;
#endif

/** Path tracer parameters shared between host/device code.
*/
struct PathTracerParams
{
    int     useFixedSeed = false;       ///< Use fixed random seed. This is useful for debugging.
    uint    fixedSeed = 1;              ///< The seed to use when 'useFixedSeed' is enabled.
    float   lodBias = 0.f;              ///< LOD bias applied to secondary hits.
    float   specularRoughnessThreshold = 0.25f; ///< Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.

    // Runtime values
    uint2   frameDim = { 0, 0 };        ///< Frame dimension in pixels.
    uint2   screenTiles = { 0, 0 };     ///< Number of screen-tiles. Screen tiles may extend outside the frame.

    uint    frameCount = 0;             ///< Frames rendered. This is used as random seed.
    uint    seed = 0;                   ///< Random seed. This will get updated from the host depending on settings.
    uint2   _pad0;

#ifndef HOST_CODE
    /** Computes the offset into the tiled sample buffer for a given tile.
        The samples for all pixels are stored consecutively after this offset.
        \param[in] tile Tile coordinates.
        \return Offset into tiled sample buffer.
    */
    uint getTileOffset(const uint2 tile)
    {
        uint maxSpp = kSamplesPerPixel > 0 ? kSamplesPerPixel : kMaxSamplesPerPixel;
        uint stride = kScreenTileDim.x * kScreenTileDim.y * maxSpp;
        uint tileIdx = tile.y * screenTiles.x + tile.x;
        return tileIdx * stride;
    }

    /** Computes the offset into the tiled sample buffer for a given pixel.
        \param[in] pixel Pixel on screen.
        \param[in] sampleOffset Per-pixel sample offset within tiles. Only used if kSamplesPerPixel == 0.
        \return Offset into tiled sample buffer.
    */
    uint getSampleOffset(const uint2 pixel, Texture2D<uint> sampleOffset)
    {
        uint2 tileID = pixel >> kScreenTileBits;
        uint tileOffset = getTileOffset(tileID);

        if (kSamplesPerPixel > 0)
        {
            uint tileBits = kScreenTileBits.x + kScreenTileBits.y;
            uint pixelIdx = interleave_16bit(pixel) & ((1 << tileBits) - 1); // TODO: Use interleave_8bit() if kScreenTileBits <= 4.
            return tileOffset + pixelIdx * kSamplesPerPixel;
        }
        else
        {
            return tileOffset + sampleOffset[pixel];
        }
    }
#endif
};

END_NAMESPACE_FALCOR
