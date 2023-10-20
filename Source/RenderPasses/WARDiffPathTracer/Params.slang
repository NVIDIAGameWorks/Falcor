/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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

// Import static specialization constants.
#ifndef HOST_CODE
__exported import RenderPasses.WARDiffPathTracer.StaticParams;
__exported import Utils.Math.BitTricks;
#endif

// Define path configuration limits.

/// Maximum supported sample count. We can use tiling to support large sample counts if needed.
static const uint kMaxSamplesPerPixel = 16;
/// Maximum supported frame dimension in pixels along x or y. We can increase the bit allocation if needed.
static const uint kMaxFrameDimension = 4096;
/// Maximum supported number of bounces per bounce category (value 255 is reserved for internal use).
/// The resulting path length may be longer than this.
static const uint kBounceLimit = 254;
/// Maximum number of shadow rays per path vertex for next-event estimation.
static const uint kMaxLightSamplesPerVertex = 8;

/**
 * Path tracer parameters shared between host/device code.
 */
struct WARDiffPathTracerParams
{
    /// Use fixed random seed. This is useful for debugging.
    int useFixedSeed = false;
    /// The seed to use when 'useFixedSeed' is enabled.
    uint fixedSeed = 1;
    /// Threshold for asserting on NaNs.
    float assertThreshold = 1e9f;
    /// A runtime varialbe to control running the backward pass or the primal pass.
    uint runBackward = 1;

    // Runtime values

    /// Frame dimension in pixels.
    uint2 frameDim = { 0, 0 };
    /// Number of screen-tiles. Screen tiles may extend outside the frame.
    uint2 screenTiles = { 0, 0 };

    /// Frames rendered. This is used as random seed.
    uint frameCount = 0;
    /// Random seed. This will get updated from the host depending on settings.
    uint seed = 0;
    uint2 _pad0;
};

END_NAMESPACE_FALCOR
