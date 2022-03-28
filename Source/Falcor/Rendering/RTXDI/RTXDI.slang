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
#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import PackedTypes;

// Set default value for RTXDI_INSTALLED.
// We want this shader module to compile even if no external defines are set.
#ifndef RTXDI_INSTALLED
#define RTXDI_INSTALLED 0
#endif

#if RTXDI_INSTALLED

// Thread groups size for launches. Not user controllable, due to internal RTXDI restrictions
static const uint kRTXDIGroupSize = 16;

/** Global data structure used to hold parameters and resources for RTXDI.
*/
struct RTXDI
{
    RTXDI_ResamplingRuntimeParameters params;           ///< RTXDI runtime parameters (potentially changing every frame).

    uint    frameIndex;                                 ///< Current frame index. Used to make sure we use different random numbers each frame.
    uint2   frameDim;                                   ///< Frame dimensions in pixels.
    uint    pixelCount;                                 ///< Equivalent to frameDim.x * frameDim.y.
    float   rayEpsilon;                                 ///< Our ray-epsilon.
    uint    currentSurfaceBufferIndex;                  ///< Index of the surface data buffer for the current frame (0 or 1)
    uint    prevSurfaceBufferIndex;                     ///< Index of the surface data buffer for the previous frame (0 or 1).

    // Camera data for the previous frame.
    float3  prevCameraU;
    float3  prevCameraV;
    float3  prevCameraW;
    float2  prevCameraJitter;

    bool    storeCompactLightInfo;                      ///< Enable storing compact light info for presampled light tiles.
    bool    useEmissiveTextures;                        ///< Lookup final triangle emission in emissive texture.

    // Parameters to control how RTXDI generates initial light candidate samples.
    uint    localLightCandidateCount;                   ///< Number of initial candidates (per pixel) on local lights.
    uint    infiniteLightCandidateCount;                ///< Number of initial candidates (per pixel) on infinite lights.
    uint    envLightCandidateCount;                     ///< Number of initial candidates (per pixel) on the environment light.
    uint    brdfCandidateCount;                         ///< Number of initial candidates (per pixel) from brdf.
    float   brdfCutoff;                                 ///< Value in range[0, 1] to determine how much to shorten BRDF rays. 0 to disable shortening.

    // Parameters needed to control RTXDI sample reuse (generally).
    uint    maxHistoryLength;                           ///< Maximum history length for temporal reuse, measured in frames.
    uint    biasCorrectionMode;                         ///< See definition of BiasCorrection in RTXDI.h.
    float   depthThreshold;                             ///< Relative depth difference at which pixels are classified too far apart to be reused (0.1 = 10%).
    float   normalThreshold;                            ///< Cosine of the angle between normals, below which pixels are classified too far apart to be reused.

    // Parameters needed to control RTXDI's spatial reuse (and spatiotemporal reuse).
    float   samplingRadius;                             ///< Screen-space radius for spatial resampling, measured in pixels.
    uint    spatialSampleCount;                         ///< Number of neighbor pixels considered for resampling.
    uint    disocclusionSampleCount;                    ///< Number of neighbor pixels considered for resampling in disoccluded regions.

    // Parameters needed to control RTXDI's temporal reuse (and spatiotemporal reuse).
    float   boilingFilterStrength;                      ///< 0 = off, 1 = full strength.
    bool    enableVisibilityShortcut;                   ///< Should we use a visibility shortcut?  (Degrades quality somewhat)
    bool    enablePermutationSampling;                  ///< Enables permuting the pixels sampled from the previous frame (noisier but more denoiser friendly).

    // Parameters only valid during shading (i.e., after calling RTXDI::update() in your C++ code).
    uint    finalShadingReservoir;                      ///< Reservoir where the final samples are stored (needed to extract them for the user).

    // Texture and buffer resources used in the RTXDI bridge.
    StructuredBuffer<PackedPolymorphicLight> lightInfo;             ///< Buffer storing information about all the lights in the scene.
    RWBuffer<uint2> risBuffer;                                      ///< Buffer storing presampled light tiles.
    RWStructuredBuffer<PackedPolymorphicLight> compactLightInfo;    ///< Optional buffer storing compact light info for samples in the light tile buffer for improved coherence.
    RWStructuredBuffer<PackedSurfaceData> surfaceData;              ///< Buffer storing the surface data for the current and previous frames.
    Buffer<float2> neighborOffsets;                                 ///< Buffer storing a poisson(-ish) distribution of offsets for sampling randomized neighbors.
    RWStructuredBuffer<RTXDI_PackedReservoir> reservoirs;           ///< Buffer storing reservoirs.
    Texture2D<float2> motionVectors;                                ///< Texture storing screen-space motion vectors.

    Texture2D localLightPdfTexture;                     ///< PDF texture for sampling local lights (emissive triangles, local analytical lights).
    Texture2D envLightLuminanceTexture;                 ///< Luminance texture for evaluating the environment light.
    Texture2D envLightPdfTexture;                       ///< PDF texture for sampling environment light.
};

// HACK: PolymorphicLight.slang contains light sampling functions accessing data in gRTXDI.
// We also use PolymorphicLight.slang in the LightUpdater.cs.slang module for setting up
// light information. The global gRTXDI resources are not available there, so we disable
// the light sampling functions by default.
// Here, we enable the light sampling functions and #include the module rather than
// importing it, so all the code is inlined into this module.
#define RTXDI_POLYMORPHIC_LIGHT_SAMPLING
#include "PolymorphicLight.slang"

#include "RTXDIApplicationBridge.slangh"

// The following defines are needed by ResamplingFunctions.hlsli.

// Define buffers for RTXDI to access data from.
#define RTXDI_RIS_BUFFER gRTXDI.risBuffer
#define RTXDI_LIGHT_RESERVOIR_BUFFER gRTXDI.reservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER gRTXDI.neighborOffsets

// Settings needed to use the boiling filter.
#define RTXDI_ENABLE_BOILING_FILTER
#define RTXDI_BOILING_FILTER_GROUP_SIZE kRTXDIGroupSize

#include "rtxdi/ResamplingFunctions.hlsli"

#else // RTXDI_INSTALLED

struct RTXDI {};

#endif // RTXDI_INSTALLED

/** Public interface to the RTXDI module.
    Note: All of these functions are no-ops if the RTXDI SDK is not installed.
*/
extension RTXDI
{
    /** Set the per-pixel surface data required for RTXDI at valid surface hits.
        Note: The provided position should be save to trace visibility rays from (i.e. is offseted from the true surface).
        \param[in] pixel Pixel coordinate.
        \param[in] position Position in world-space.
        \param[in] normal Shading normal in world-space.
        \param[in] diffuse Diffuse albedo.
        \param[in] specular Specular albedo.
        \param[in] specularRoughness Specular roughness (linear).
    */
    void setSurfaceData(const uint2 pixel, const float3 position, const float3 normal, const float3 diffuse, const float3 specular, const float specularRoughness)
    {
#if RTXDI_INSTALLED
        if (any(pixel >= gRTXDI.frameDim)) return;

        // Compute buffer index based on pixel and currently used surface buffer index.
        uint bufferIdx = pixel.x + (pixel.y * gRTXDI.frameDim.x) + gRTXDI.currentSurfaceBufferIndex * gRTXDI.pixelCount;

        SurfaceData sd = {};
        sd.position = position;
        sd.normal = normal;
        sd.viewDepth = distance(gScene.camera.getPosition(), position);

        float diffuseWeight = luminance(diffuse);
        float specularWeight = luminance(specular);
        float sumWeights = diffuseWeight + specularWeight;
        float diffuseProb = sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);

        sd.diffuse = diffuseWeight;
        sd.specular = specularWeight;
        sd.diffuseProb = diffuseProb;
        sd.roughness = specularRoughness;

        // Store surface data.
        gRTXDI.surfaceData[bufferIdx] = sd.pack();
#endif
    }

    /** Mark pixel as containing no valid surface data.
       \param[in] pixel Pixel coordinate.
    */
    void setInvalidSurfaceData(uint2 pixel)
    {
#if RTXDI_INSTALLED
        if (any(pixel >= gRTXDI.frameDim)) return;

        // Compute buffer index based on pixel and currently used surface buffer index.
        uint bufferIdx = pixel.x + (pixel.y * gRTXDI.frameDim.x) + gRTXDI.currentSurfaceBufferIndex * gRTXDI.pixelCount;

        // Store invalid surface data.
        gRTXDI.surfaceData[bufferIdx] = SurfaceData().pack();
#endif
    }

    /** Get the final sample for the given pixel computed using RTXDI.
        Note: The returned incident radiance is pre-divided by the PDF.
        \param[in] pixel Pixel coordinate.
        \param[out] dir Direction to light in world-space.
        \param[out] distance Distance to light.
        \param[out] Li Weighted unoccluded incident radiance (i.e., incident radiance / pdf)
        \return Returns true if a valid sample was returned.
    */
    bool getFinalSample(uint2 pixel, out float3 dir, out float distance, out float3 Li)
    {
#if RTXDI_INSTALLED
        // Get the final sample from RTXDI.
        RAB_Surface surface;
        RAB_LightSample lightSample;
        bool isValid = gRTXDI.getFinalSample(pixel, surface, lightSample, Li);

        // Return data.
        dir = lightSample.direction;
        distance = lightSample.distance - gRTXDI.rayEpsilon;
        return isValid;
#else
        dir = float3(0.f);
        distance = 0.f;
        Li = float3(0.f);
        return false;
#endif
    }
}

#if RTXDI_INSTALLED

/** "Private" interface to the RTXDI module.
    Note: The entire private API is not defined if RTXDI is not installed, as they all
    use RTXDI structures during computation and/or function input or output parameters.
*/
extension RTXDI
{
    /** Generate initial light sample candidates, select one to store in the reservoir.
    */
    void generateCandidates(uint2 pixelID, uint outputReservoirIndex)
    {
        uint2 pixelPosition = RTXDI_ReservoirToPixelPos(pixelID, gRTXDI.params);
        printSetPixel(pixelPosition);

        // Initialize random samplers.
        RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 2);
        RAB_RandomSamplerState tileRng = RAB_InitRandomSampler(pixelPosition / RTXDI_TILE_SIZE_IN_PIXELS, 3);

        // Load surface data.
        RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

        RTXDI_SampleParameters sampleParams = RTXDI_InitSampleParameters(
            0, // Number of samples to take from ReGIR.
            gRTXDI.localLightCandidateCount,
            gRTXDI.infiniteLightCandidateCount,
            gRTXDI.envLightCandidateCount,
            gRTXDI.brdfCandidateCount,
            gRTXDI.brdfCutoff,
            0.001f);

        // Generate initial candidates.
        RAB_LightSample lightSample;
        RTXDI_Reservoir reservoir = RTXDI_SampleLightsForSurface(
            rng, tileRng, surface, sampleParams, gRTXDI.params, lightSample);

        // Store selected light sample in a reservoir for later reuse and shading.
        RTXDI_StoreReservoir(reservoir, gRTXDI.params, pixelID, outputReservoirIndex);
    }

    /** Trace a visibility ray from the primary hit surface to the light sample in the reservoir.
        Update the reservoir if the light is not visible.
    */
    void testCandidateVisibility(uint2 pixelID, uint reservoirIndex)
    {
        // Load surface data and reservoir containing the light sample.
        RAB_Surface surface = RAB_GetGBufferSurface(RTXDI_ReservoirToPixelPos(pixelID, gRTXDI.params), false);
        RTXDI_Reservoir reservoir = RTXDI_LoadReservoir(gRTXDI.params, pixelID, reservoirIndex);

        if (RAB_IsSurfaceValid(surface) && RTXDI_IsValidReservoir(reservoir))
        {
            // Get the light sample, so we have data to construct our visibility query
            RAB_LightInfo lightInfo = RAB_LoadLightInfo(RTXDI_GetReservoirLightIndex(reservoir), false);
            RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, RTXDI_GetReservoirSampleUV(reservoir));

            // Trace a visibility ray and update the reservoir.
            if (!RAB_GetConservativeVisibility(surface, lightSample))
            {
                RTXDI_StoreVisibilityInReservoir(reservoir, 0, true);
                RTXDI_StoreReservoir(reservoir, gRTXDI.params, pixelID, reservoirIndex);
            }
        }
    }

    /** Presample local lights into presampled light tiles.
    */
    void presampleLocalLights(uint2 threadId)
    {
        uint2 texDim;
        gRTXDI.localLightPdfTexture.GetDimensions(texDim.x, texDim.y);

        RAB_RandomSamplerState rng = RAB_InitRandomSampler(threadId.xy, 0);

        RTXDI_PresampleLocalLights(
            rng,
            gRTXDI.localLightPdfTexture,
            texDim,
            threadId.y,         // tileIndex [0...tileCount)
            threadId.x,         // sampleInTile [0...tileSize)
            gRTXDI.params);
    }

    /** Presample environment light into presampled light tiles.
    */
    void presampleEnvLight(uint2 threadId)
    {
        uint2 texDim;
        gRTXDI.envLightPdfTexture.GetDimensions(texDim.x, texDim.y);

        RAB_RandomSamplerState rng = RAB_InitRandomSampler(threadId.xy, 1);

        RTXDI_PresampleEnvironmentMap(
            rng,
            gRTXDI.envLightPdfTexture,
            texDim,
            threadId.y,         // tileIndex [0...tileCount)
            threadId.x,         // sampleInTile [0...tileSize)
            gRTXDI.params);
    }

    /** Perform spatial resampling.
    */
    void doSpatialResampling(uint2 pixelID, uint inReservoirIndex, uint outReservoirIndex)
    {
        uint2 pixelPosition = RTXDI_ReservoirToPixelPos(pixelID, gRTXDI.params);

        // Initialize random sampler.
        RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 4);

        // Load surface data.
        RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

        RTXDI_Reservoir reservoir = {};

        if (RAB_IsSurfaceValid(surface))
        {
            // Load reservoir at the current pixel.
            RTXDI_Reservoir centerReservoir = RTXDI_LoadReservoir(gRTXDI.params, pixelID, inReservoirIndex);

            // Setup resampling parameters.
            RTXDI_SpatialResamplingParameters sparams;
            sparams.sourceBufferIndex = inReservoirIndex;
            sparams.numSamples = gRTXDI.spatialSampleCount;
            sparams.numDisocclusionBoostSamples = gRTXDI.disocclusionSampleCount;
            sparams.targetHistoryLength = gRTXDI.maxHistoryLength;
            sparams.biasCorrectionMode = gRTXDI.biasCorrectionMode;
            sparams.samplingRadius = gRTXDI.samplingRadius;
            sparams.depthThreshold = gRTXDI.depthThreshold;
            sparams.normalThreshold = gRTXDI.normalThreshold;

            // Execute resampling.
            RAB_LightSample selectedSample = {};
            reservoir = RTXDI_SpatialResampling(pixelPosition, surface, centerReservoir,
                rng, sparams, gRTXDI.params, selectedSample);
        }

        // Store selected light sample in the output reservoir.
        RTXDI_StoreReservoir(reservoir, gRTXDI.params, pixelID, outReservoirIndex);
    }

    /** Perform temporal resampling.
    */
    void doTemporalResampling(uint2 pixelID, uint2 groupId, uint candidateReservoirIndex, uint priorFrameReservoirIndex, uint outReservoirIndex)
    {
        uint2 pixelPosition = RTXDI_ReservoirToPixelPos(pixelID, gRTXDI.params);

        // Initialize random sampler.
        RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 4);

        // Load surface data.
        RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

        RTXDI_Reservoir reservoir = {};

        if (RAB_IsSurfaceValid(surface))
        {
            // Load reservoir containing the initial candidate at the current pixel.
            RTXDI_Reservoir candidateReservoir = RTXDI_LoadReservoir(gRTXDI.params, pixelID, candidateReservoirIndex);

            // Setup resampling parameters.
            RTXDI_TemporalResamplingParameters tparams;
            tparams.screenSpaceMotion = float3(gRTXDI.motionVectors[pixelPosition] * gRTXDI.frameDim, 0.f); // TODO handle linear z
            tparams.sourceBufferIndex = priorFrameReservoirIndex;
            tparams.maxHistoryLength = gRTXDI.maxHistoryLength;
            tparams.biasCorrectionMode = gRTXDI.biasCorrectionMode;
            tparams.depthThreshold = gRTXDI.depthThreshold;
            tparams.normalThreshold = gRTXDI.normalThreshold;
            tparams.enableVisibilityShortcut = gRTXDI.enableVisibilityShortcut;
            tparams.enablePermutationSampling = gRTXDI.enablePermutationSampling;

            // Execute resampling.
            RAB_LightSample selectedSample = {};
            int2 selectedSamplePos = 0;
            reservoir = RTXDI_TemporalResampling(pixelPosition, surface, candidateReservoir,
                rng, tparams, gRTXDI.params, selectedSamplePos, selectedSample);
        }

        // If requested (and if we're in a compute shader), spawn a filter to surpress boiling.
#ifdef RTXDI_ENABLE_BOILING_FILTER
        if (gRTXDI.boilingFilterStrength > 0)
        {
            RTXDI_BoilingFilter(groupId, gRTXDI.boilingFilterStrength, gRTXDI.params, reservoir);
        }
#endif

        // Store selected light sample in the output reservoir.
        RTXDI_StoreReservoir(reservoir, gRTXDI.params, pixelID, outReservoirIndex);
    }

    /** Perform spatio-temporal resampling.
    */
    void doSpatiotemporalResampling(uint2 pixelID, uint2 groupId, uint candidateReservoirIndex, uint priorFrameReservoirIndex, uint outReservoirIndex)
    {
        uint2 pixelPosition = RTXDI_ReservoirToPixelPos(pixelID, gRTXDI.params);

        // Initialize random sampler.
        RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 4);

        // Load surface data.
        RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

        RTXDI_Reservoir reservoir = {};

        if (RAB_IsSurfaceValid(surface))
        {
            // Load reservoir containing the initial candidate at the current pixel.
            RTXDI_Reservoir candidateReservoir = RTXDI_LoadReservoir(gRTXDI.params, pixelID, candidateReservoirIndex);

            // Setup resampling parameters.
            RTXDI_SpatioTemporalResamplingParameters stparams;
            stparams.screenSpaceMotion = float3(gRTXDI.motionVectors[pixelPosition] * gRTXDI.frameDim, 0.f); // TODO handle linear z
            stparams.sourceBufferIndex = priorFrameReservoirIndex;
            stparams.maxHistoryLength = gRTXDI.maxHistoryLength;
            stparams.biasCorrectionMode = gRTXDI.biasCorrectionMode;
            stparams.depthThreshold = gRTXDI.depthThreshold;
            stparams.normalThreshold = gRTXDI.normalThreshold;
            stparams.numSamples = gRTXDI.spatialSampleCount + 1;
            stparams.numDisocclusionBoostSamples = gRTXDI.disocclusionSampleCount;
            stparams.samplingRadius = gRTXDI.samplingRadius;
            stparams.enableVisibilityShortcut = gRTXDI.enableVisibilityShortcut;
            stparams.enablePermutationSampling = gRTXDI.enablePermutationSampling;

            // Execute resampling.
            RAB_LightSample selectedSample = {};
            int2 selectedSamplePos = 0;
            reservoir = RTXDI_SpatioTemporalResampling(pixelPosition, surface, candidateReservoir,
                rng, stparams, gRTXDI.params, selectedSamplePos, selectedSample);
        }

        // If requested (and if we're in a compute shader), spawn a filter to surpress boiling.
#ifdef RTXDI_ENABLE_BOILING_FILTER
        if (gRTXDI.boilingFilterStrength > 0)
        {
            RTXDI_BoilingFilter(groupId, gRTXDI.boilingFilterStrength, gRTXDI.params, reservoir);
        }
#endif

        // Store selected light sample in the output reservoir.
        RTXDI_StoreReservoir(reservoir, gRTXDI.params, pixelID, outReservoirIndex);
    }

    /** Returns the final sample; a more cryptic form of getFinalSample(), from above, exposing RTXDI bridge structures
        \param[in]  pixel Pixel coordinate.
        \param[out] surface An RTXDI RAB_Surface structure at this pixel.
        \param[out] lightSample An RTXDI RAB_LightSample structure for the final selected light.
        \param[out] Li Weighted unoccluded incident radiance (i.e., incident radiance / pdf)
        \return Returns true if a valid sample was returned.
    */
    bool getFinalSample(uint2 pixel, out RAB_Surface surface, out RAB_LightSample lightSample, out float3 Li)
    {
        // Zero initialize out parameters.
        lightSample = {};
        Li = {};

        // Load surface data and reservoir containing the final light sample.
        uint2 pixelPosition = RTXDI_ReservoirToPixelPos(pixel, gRTXDI.params);
        surface = RAB_GetGBufferSurface(pixelPosition, false);
        RTXDI_Reservoir reservoir = RTXDI_LoadReservoir(gRTXDI.params, pixel, gRTXDI.finalShadingReservoir);

        // Abort if we don't have a valid surface or reservoir.
        if (!RAB_IsSurfaceValid(surface) || !RTXDI_IsValidReservoir(reservoir)) return false;

        // Load the selected light and the specific light sample on it.
        uint lightIdx = RTXDI_GetReservoirLightIndex(reservoir);
        float2 lightUV = RTXDI_GetReservoirSampleUV(reservoir);
        RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIdx, false);
        lightSample = lightInfo.calcSample(surface.position, lightUV, false); // Make sure to sample full RGB incident radiance.

        // Compute incident radiance pre-divided by the PDF.
        // The incident radiance returned from the light sample is already pre-divided by the PDF.
        // We only need to apply the inverse PDF from the reservoir.
        Li = lightSample.Li * RTXDI_GetReservoirInvPdf(reservoir);

        return true;
    }
}

#endif // RTXDI_INSTALLED

// We use global resources such that RTXDI can access them.
ParameterBlock<RTXDI> gRTXDI;
