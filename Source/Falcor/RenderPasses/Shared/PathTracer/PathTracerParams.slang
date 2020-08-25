/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Experimental/Scene/Lights/EmissiveLightSamplerType.slangh"

BEGIN_NAMESPACE_FALCOR

// Define max supported path length and number of shadow rays per vertex.
// These limits depend on the bit layout for the packed path flags.
static const uint kMaxPathFlagsBits = 16;
static const uint kMaxPathLengthBits = 8;
static const uint kMaxPathLength = (1 << kMaxPathLengthBits) - 1;
static const uint kMaxLightSamplesPerVertex = 8;

// Define ray indices.
static const uint32_t kRayTypeScatter = 0;
static const uint32_t kRayTypeShadow = 1;

// TODO: Remove explicitly assigned values when we can use the enums as default initializer below
enum class MISHeuristic
{
    BalanceHeuristic = 0,       ///< Balance heuristic.
    PowerTwoHeuristic = 1,      ///< Power heuristic (exponent = 2.0).
    PowerExpHeuristic = 2,      ///< Power heuristic (variable exponent).
};

/** Path tracer parameters. Shared between host and device.

    Note that if you add configuration parameters, do not forget to register
    them with the scripting system in SCRIPT_BINDING() in PathTracer.cpp.
*/
struct PathTracerParams
{
    // Make sure struct layout follows the HLSL packing rules as it is uploaded as a memory blob.
    // Do not use bool's as they are 1 byte in Visual Studio, 4 bytes in HLSL.
    // https://msdn.microsoft.com/en-us/library/windows/desktop/bb509632(v=vs.85).aspx
    // Note that the default initializers are ignored by Slang but used on the host.

    // General
    uint    samplesPerPixel = 1;            ///< Number of samples (paths) per pixel. Use compile-time constant kSamplesPerPixel in shader.
    uint    lightSamplesPerVertex = 1;      ///< Number of light samples per path vertex. Use compile-time constant kLightSamplesPerVertex in shader.
    uint    maxBounces = 3;                 ///< Max number of indirect bounces (0 = none), up to kMaxPathLength. Use compile-time constant kMaxBounces in shader.
    uint    maxNonSpecularBounces = 3;      ///< Max number of non-specular indirect bounces (0 = none), up to kMaxPathLength. Use compile-time constant kMaxNonSpecularBounces in shader.

    int     useVBuffer = 1;                 ///< Use a V-buffer as input. Use compile-time constant kUseVBuffer (or preprocessor define USE_VBUFFER) in shader.
    int     useAlphaTest = 1;               ///< Use alpha testing on non-opaque triangles. Use compile-time constant kUseAlphaTest (or preprocessor define USE_ALPHA_TEST) in shader.
    int     _removed;
    int     forceAlphaOne = true;           ///< Force the alpha channel to 1.0. Otherwise background will have alpha 0.0 and covered samples 1.0 to allow compositing. Use compile-time constant kForceAlphaOne in shader.

    int     clampSamples = false;           ///< Clamp the per-path contribution to 'clampThreshold' to reduce fireflies.
    float   clampThreshold = 10.f;
    float   specularRoughnessThreshold = 0.25f; ///< Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold.
    float   _pad0;

    // Sampling
    int     useBRDFSampling = true;         ///< Use BRDF importance sampling (otherwise cosine-weighted hemisphere sampling). Use compile-time constant kUseBRDFSampling in shader.
    int     useNEE = true;                  ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
    int     useMIS = true;                  ///< Use multiple importance sampling (MIS) when NEE is enabled. This enables a scatter ray from the last path vertex. Use compile-time constant kUseMIS in shader.
    uint    misHeuristic = 1; /* (uint)MISHeuristic::PowerTwoHeuristic */   ///< MIS heuristic. Use compile-time constant kMISHeuristic in shader. TODO: Replace initializer value by enum when Slang supports it.

    float   misPowerExponent = 2.f;         ///< MIS exponent for the power heuristic. This is only used when 'PowerExpHeuristic' is chosen.
    int     useRussianRoulette = false;     ///< Use Russian roulette. Use compile-time constant kUseRussianRoulette in shader.
    float   probabilityAbsorption = 0.2f;   ///< Probability of absorption for Russian roulette.
    int     useFixedSeed = false;           ///< Use fixed random seed for the sample generator. This is useful for print() debugging.

    int     useLegacyBSDF = false;          ///< Use legacy BRDF sampling code (no support for specular transmission).
    int     useNestedDielectrics = true;    ///< Use algorithm to handle nested dielectric materials. Use compile-time constant kUseNestedDielectrics in shader.
    int     useLightsInVolumes = false;     ///< Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface. Use compile-time constant kUseLightsInVolumes in shader.
    int     disableCaustics = false;        ///< Disable sampling of caustics. Use compile-time constant kDisableCaustics in shader.

    // Ray footprint
    float   screenSpacePixelSpreadAngle = 0.0; ///< The angle an "average" pixel spans from camera (Used by ray footprint).
    uint    rayFootprintMode = 0;           ///< Ray footprint tracking mode used for Texture LOD. See RayFootprintModes.slangh.
    uint    rayConeMode = 2;                ///< Sub-mode for the Ray Cone mode of ray footprint. 0:RayTracingGems1 (unsupported), 1:Combo, 2:Unified. See RayFootprint.slang.
    int     rayFootprintUseRoughness = 0;   ///< Integrates material roughness into the footprint calculation (only used by Ray Cone right now).

    // Runtime data
    uint2   frameDim = uint2(0, 0);         ///< Current frame dimensions in pixels.
    uint    frameCount = 0;                 ///< Frame count since scene was loaded.
    uint    prngDimension = 0;              ///< First available PRNG dimension.
};

END_NAMESPACE_FALCOR
