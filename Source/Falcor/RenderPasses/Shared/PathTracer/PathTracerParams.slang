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
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"
#include "Experimental/Scene/Lights/EmissiveLightSamplerType.slangh"

BEGIN_NAMESPACE_FALCOR

// Define max supported path length and number of shadow rays per vertex.
// These limits depend on the bit layout for the packed path flags.
static const uint kMaxPathFlagsBits = 11;
static const uint kMaxPathLengthBits = 5;
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
#ifdef HOST_CODE
struct PathTracerParams : Falcor::ScriptBindings::enable_to_string
#else
struct PathTracerParams
#endif
{
    // Make sure struct layout follows the HLSL packing rules as it is uploaded as a memory blob.
    // Do not use bool's as they are 1 byte in Visual Studio, 4 bytes in HLSL.
    // https://msdn.microsoft.com/en-us/library/windows/desktop/bb509632(v=vs.85).aspx
    // Note that the default initializers are ignored by Slang but used on the host.

    // General
    uint    samplesPerPixel = 1;            ///< Number of samples (paths) per pixel. Use compile-time constant SAMPLES_PER_PIXEL in shader.
    uint    lightSamplesPerVertex = 1;      ///< Number of light samples per path vertex. Use compile-time constant LIGHT_SAMPLES_PER_VERTEX in shader. Note: This option is only supported in the WavefrontPathTracer.
    uint    maxBounces = 3;                 ///< Max number of indirect bounces (0 = none), up to kMaxPathLength. Use compile-time constant MAX_BOUNCES in shader.
    int     useVBuffer = 1;                 ///< Use a V-buffer as input. Use compile-time constant USE_VBUFFER in shader.

    int     forceAlphaOne = true;           ///< Force the alpha channel to 1.0. Otherwise background will have alpha 0.0 and covered samples 1.0 to allow compositing. Use compile-time constant FORCE_ALPHA_ONE in shader.
    int     clampSamples = false;           ///< Clamp the per-path contribution to 'clampThreshold' to reduce fireflys.
    float   clampThreshold = 10.f;
    int     _pad1;

    // Lighting
    int     useAnalyticLights = true;       ///< Use built-in analytic lights.
    int     useEmissiveLights = true;       ///< Use emissive geometry as light sources.
    int     useEnvLight = true;             ///< Use environment map as light source (if loaded). Use compile-time constant USE_ENV_LIGHT in shader.
    int     useEnvBackground = true;        ///< Use environment map as background (if loaded). Use compile-time constant USE_ENV_BACKGROUND in shader.

    // Sampling
    int     useBRDFSampling = true;         ///< Use BRDF importance sampling (otherwise cosine-weighted hemisphere sampling). Use compile-time constant USE_BRDF_SAMPLING in shader.
    int     useMIS = true;                  ///< Use multiple importance sampling (MIS). Use compile-time constant USE_MIS in shader.
    uint    misHeuristic = 1; /* (uint)MISHeuristic::PowerTwoHeuristic */   ///< MIS heuristic. Use compile-time constant MIS_HEURISTIC in shader. TODO: Replace initializer value by enum when Slang supports it.
    float   misPowerExponent = 2.f;         ///< MIS exponent for the power heuristic. This is only used when 'PowerExpHeuristic' is chosen.

    int     useEmissiveLightSampling = true;///< Use emissive light importance sampling.
    int     useRussianRoulette = false;     ///< Use Russian roulette. Use compile-time constant USE_RUSSIAN_ROULETTE in shader.
    float   probabilityAbsorption = 0.2f;   ///< Probability of absorption for Russian roulette.
    int     useFixedSeed = false;           ///< Use fixed random seed for the sample generator. This is useful for print() debugging.

    // Runtime data
    uint2   frameDim = uint2(0, 0);         ///< Current frame dimensions in pixels.
    uint    frameCount = 0;                 ///< Frame count since scene was loaded.
    uint    prngDimension = 0;              ///< First available PRNG dimension.

    uint    lightCountPoint = 0;            ///< Number of point lights in the scene.
    uint    lightCountDirectional = 0;      ///< Number of directional lights in the scene.
    uint    lightCountAnalyticArea = 0;     ///< Number of analytic area lights in the scene.
    uint    lightCountTriangle = 0;         ///< Number of triangle lights in the scene.

    int     useLegacyBSDF = false;          ///< Use legacy BRDF sampling code (no support for specular transmission).
    int     useNestedDielectrics = true;    ///< Use algorithm to handle nested dielectric materials.
    int     useLightSamplesInVolumes = false; ///< Use direct light sampling inside of volumes (transmissive materials), even though shadow rays are typically always occluded.
    int     _pad4;
};

END_NAMESPACE_FALCOR
