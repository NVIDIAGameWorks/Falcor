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
#include "Core/Macros.h"
#include "Utils/Debug/PixelDebug.h"
#include "Scene/Scene.h"
#include <memory>
#include <type_traits>
#include <vector>

#if FALCOR_HAS_RTXDI
#include "rtxdi/RTXDI.h"
#endif

namespace Falcor
{
    class RenderContext;

    /** Wrapper module for RTXDI. This allows easy integration of RTXDI into our path tracer.

        Usage requires a copy of the RTXDI SDK. See README for installation details.

        The host side (this class) is responsible for managing and updating the resources
        required by RTXDI. This includes:
        - Light information and sampling data structures
        - Surface data buffers (G-Buffer)
        - Additional internal buffers

        The device side (RTXDI.slang) acts as the interface between a renderer and RTXDI.
        It is responsible for:
        - Filling the surface data buffer
        - Get final light samples for shading

        Integrating this module into a renderer requires a few intialization steps:
        - Create the module with RTXDI::create().
        - When compiling shaders using this module, ensure you add the shader preprocessor
          defines provided by RTXDI::getDefines().
        - When executing shaders using this module, ensure you set the shader data
          using RTXDI::setShaderData().

        To render a frame, the following steps need to occur:

        1)  Start the frame by calling RTXDI::beginFrame().

        2)  In your shader, provide RTXDI with the surface data of primary hits to guide resampling
            by calling gRTXDI.setSurfaceData()/setInvalidSurfaceData() for each pixel.

        3)  Run the RTXDI resampling via RTXDI::update().

        4)  In your shader, shade using RTXDI's selected light sample by calling gRTXDI.getFinalSample().
            Two variants of getFinalSample() exist; one returns just the data
            needed to shade while the other retuns the RAB_Surface and RAB_LightSample
            RTXDI structures for the selected light to allow further processing.

        5) End the frame by calling RTXDI::endFrame().

        Also see the Source/RenderPasses/RTXDIPass/RTXDIPass.cpp for a minimal example on how to use the module.
    */
    class FALCOR_API RTXDI
    {
    public:
        using SharedPtr = std::shared_ptr<RTXDI>;

        /** RTXDI sampling modes.
        */
        enum class Mode
        {
            NoResampling                = 1, ///< No resampling (Talbot RIS from EGSR 2005 "Importance Resampling for Global Illumination").
            SpatialResampling           = 2, ///< Spatial resampling only.
            TemporalResampling          = 3, ///< Temporal resampling only.
            SpatiotemporalResampling    = 4, ///< Spatiotemporal resampling.
        };

        /** Bias correction modes.
        */
        enum class BiasCorrection
        {
            Off         = 0, ///< Use (1/M) normalization, which is very biased but also very fast.
            Basic       = 1, ///< Use MIS-like normalization but assume that every sample is visible.
            Pairwise    = 2, ///< Use pairwise MIS normalization.  Assumes every sample is visible.
            RayTraced   = 3, ///< Use MIS-like normalization with visibility rays. Unbiased.
        };

        /** Configuration options, with generally reasonable defaults.
        */
        struct Options
        {
            Mode mode = Mode::SpatiotemporalResampling; ///< RTXDI sampling mode.

            // Light presampling options.
            uint32_t presampledTileCount = 128;         ///< Number of precomputed light tiles.
            uint32_t presampledTileSize = 1024;         ///< Size of each precomputed light tile (number of samples).
            bool storeCompactLightInfo = true;          ///< Store compact light info for precomputed light tiles to improve coherence.

            // Initial candidate sampling options.
            uint32_t localLightCandidateCount = 24;     ///< Number of initial local light candidate samples.
            uint32_t infiniteLightCandidateCount = 8;   ///< Number of initial infinite light candidate samples.
            uint32_t envLightCandidateCount = 8;        ///< Number of initial environment light candidate samples.
            uint32_t brdfCandidateCount = 1;            ///< Number of initial brdf candidate samples.
            float brdfCutoff = 0.f;                     ///< Value in range[0, 1] to determine how much to shorten BRDF rays. 0 to disable shortening.
            bool testCandidateVisibility = true;        ///< Test visibility on selected candidate sample before doing resampling.

            // Resampling options.
            BiasCorrection biasCorrection = BiasCorrection::Basic; ///< Bias correction mode.
            float depthThreshold = 0.1f;                ///< Relative depth difference at which pixels are classified too far apart to be reused (0.1 = 10%).
            float normalThreshold = 0.5f;               ///< Cosine of the angle between normals, below which pixels are classified too far apart to be reused.

            // Spatial resampling options.
            float samplingRadius = 30.0f;               ///< Screen-space radius for spatial resampling, measured in pixels.
            uint32_t spatialSampleCount = 1;            ///< Number of neighbor pixels considered for resampling.
            uint32_t spatialIterations = 5;             ///< Number of spatial resampling passes (only used in SpatialResampling mode, Spatiotemporal mode always uses 1 iteration).

            // Temporal resampling options.
            uint32_t maxHistoryLength = 20;             ///< Maximum history length for temporal reuse, measured in frames.
            float  boilingFilterStrength = 0.0f;        ///< 0 = off, 1 = full strength.

            // Rendering options.
            float rayEpsilon = 1.0e-3f;                 ///< Ray epsilon for avoiding self-intersection of visibility rays.

            // Parameter controlling behavior of final shading. Lights can have an emissive texture containing arbitrarily high frequncies.
            // To improve convergence and significantly reduce texture lookup costs, this code always uses a preintegrated emissivity over each
            // triangle during resampling. This preintegrated value can *also* be used for final shading, which reduces noise (quite a lot if the
            // artists go crazy on the textures), at the tradeoff of losing high frequency details in the lighting.
            // We recommend using "false" -- if that overblurs, consider tessellating lights to better match lighting variations.
            // If dynamic textured lights make textured lookups vital for quality, we recommend enabling this setting on only lights where it is vital.
            bool useEmissiveTextures = false;           ///< Use emissive textures to return final sample incident radiance (true is slower and noisier).

            // RTXDI options not currently exposed in the sample, as it isn't configured for them to produce acceptable results.

            bool enableVisibilityShortcut = false;      ///< Reuse visibility across frames to reduce cost; requires careful setup to avoid bias / numerical blowups.
            bool enablePermutationSampling = false;     ///< Enables permuting the pixels sampled from the previous frame (noisier but more denoiser friendly).

            // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
            Options() {}
        };

        static_assert(std::is_trivially_copyable<Options>() , "Options needs to be trivially copyable");

        /** Check if the RTXDI SDK is installed.
            \return True if the RTXDI SDK is installed.
        */
        static bool isInstalled() { return (bool)FALCOR_HAS_RTXDI; }

        /** Create a new instance of the RTXDI sampler.
            \param[in] pScene Scene.
            \param[in] options Configuration options.
            \return A new instance.
        */
        static SharedPtr create(const Scene::SharedPtr& pScene, const Options& options = Options());

        /** Set the configuration options.
            \param[in] options Configuration options.
        */
        void setOptions(const Options& options);

        /** Returns the current configuration options.
            \return The configuration options.
        */
        const Options& getOptions() { return mOptions; }

        /** Render the GUI.
            \return True if options were changed, false otherwise.
        */
        bool renderUI(Gui::Widgets& widget);

        /** Get a list of shader defines for using the RTXDI sampler.
            \return List of shader defines.
        */
        Program::DefineList getDefines() const;

        /** Bind the RTXDI sampler to a given shader var.
            Note: RTXDI is always bound to the global "gRTXDI" variable, so we expect a root shader variable here.
            \param[in] rootVar The root shader variable to set the data into.
        */
        void setShaderData(const ShaderVar& rootVar);

        /** Begin a frame.
            Must be called once at the beginning of each frame.
            \param[in] pRenderContext Render context.
            \param[in] frameDim Current frame dimension.
        */
        void beginFrame(RenderContext* pRenderContext, const uint2& frameDim);

        /** End a frame.
            Must be called one at the end of each frame.
            \param[in] pRenderContext Render context.
        */
        void endFrame(RenderContext* pRenderContext);

        /** Update and run this frame's RTXDI resampling, allowing final samples to be queried afterwards.
            Must be called once between beginFrame() and endFrame().
            \param[in] pRenderContext Render context.
            \param[in] pMotionVectors Motion vectors for temporal reprojection.
        */
        void update(RenderContext* pRenderContext, const Texture::SharedPtr& pMotionVectors);

        /** Get the pixel debug component.
            \return Returns the pixel debug component.
        */
        const PixelDebug::SharedPtr& getPixelDebug() const { return mpPixelDebug; }

    private:
        RTXDI(const Scene::SharedPtr& pScene, const Options& options);

        Scene::SharedPtr                    mpScene;                ///< Scene (set on initialization).
        Options                             mOptions;               ///< Configuration options.

        PixelDebug::SharedPtr               mpPixelDebug;           ///< Pixel debug component.

        // If the SDK is not installed, we leave out most of the implementation.

#if FALCOR_HAS_RTXDI

        // RTXDI state.

        rtxdi::ContextParameters            mRTXGIContextParams;    ///< Parameters that largely stay constant during program execution.
        RTXDI_ResamplingRuntimeParameters   mRTXDIShaderParams;     ///< Structure passed to the GPU per frame.
        std::unique_ptr<rtxdi::Context>     mpRTXDIContext;         ///< The RTXDI context.

        // Runtime state.

        uint        mFrameIndex = 0;                                ///< Current frame index.
        uint2       mFrameDim = { 0, 0 };                           ///< Current frame dimension in pixels.
        uint32_t    mLastFrameReservoirID = 1;                      ///< Index of the reservoir containing last frame's output (for temporal reuse).
        uint32_t    mCurrentSurfaceBufferIndex = 0;                 ///< Index of the surface buffer used for the current frame (0 or 1).

        CameraData  mPrevCameraData;                                ///< Previous frame's camera data.

        // RTXDI categorizes lights into local, infinite and an environment light.
        // Falcor has emissive (triangle) lights, analytic lights (point, area, directional, distant) and an environment light.
        // This struct keeps track of the mapping from Falcor lights to RTXDI lights.
        // Falcors emissive lights and (local) analytic lights generate RTXDIs local lights.
        // Falcors directional and distant lights generate RTXDIs infinite lights.
        struct
        {
            uint32_t emissiveLightCount = 0;                        ///< Total number of local emissive lights (triangle lights).
            uint32_t localAnalyticLightCount = 0;                   ///< Total number of local analytic lights (point lights).
            uint32_t infiniteAnalyticLightCount = 0;                ///< Total number of infinite analytic lights (directional and distant lights).
            bool envLightPresent = false;                           ///< True if environment light is present.

            uint32_t prevEmissiveLightCount = 0;                    ///< Total number of local emissive lights (triangle lights) in previous frame.
            uint32_t prevLocalAnalyticLightCount = 0;               ///< Total number of local analytic lights (point lights) in previous frame.

            std::vector<uint32_t> analyticLightIDs;                 ///< List of analytic light IDs sorted for use with RTXDI.

            uint32_t getLocalLightCount() const { return emissiveLightCount + localAnalyticLightCount; }
            uint32_t getInfiniteLightCount() const { return infiniteAnalyticLightCount; }
            uint32_t getTotalLightCount() const { return getLocalLightCount() + getInfiniteLightCount() + (envLightPresent ? 1 : 0); }

            uint32_t getFirstLocalLightIndex() const { return 0; }
            uint32_t getFirstInfiniteLightIndex() const { return getLocalLightCount(); }
            uint32_t getEnvLightIndex() const { return getLocalLightCount() + getInfiniteLightCount(); }
        } mLights;

        // Flags for triggering various actions and updates.

        struct
        {
            bool    updateEmissiveLights = true;                    ///< Set if emissive triangles have changed (moved, enabled/disabled).
            bool    updateEmissiveLightsFlux = true;                ///< Set if emissive triangles have changed intensities.
            bool    updateAnalyticLights = true;                    ///< Set if analytic lights have changed (enabled/disabled).
            bool    updateAnalyticLightsFlux = true;                ///< Set if analytic lights have changed intensities.
            bool    updateEnvLight = true;                          ///< Set if environment light has changed (env map, intensity, enabled/disabled).
            bool    recompileShaders = true;                        ///< Set if shaders need recompilation on next beginFrame() call.
            bool    clearReservoirs = false;                        ///< Set if reservoirs need to be cleared on next beginFrame() call (useful when changing configuration).
        } mFlags;

        // Resources.

        Buffer::SharedPtr       mpAnalyticLightIDBuffer;            ///< Buffer storing a list of analytic light IDs used in the scene.
        Buffer::SharedPtr       mpLightInfoBuffer;                  ///< Buffer storing information about all the lights in the scene.
        Texture::SharedPtr      mpLocalLightPdfTexture;             ///< Texture storing the PDF for sampling local lights proportional to radiant flux.
        Texture::SharedPtr      mpEnvLightLuminanceTexture;         ///< Texture storing luminance of the environment light.
        Texture::SharedPtr      mpEnvLightPdfTexture;               ///< Texture storing the PDF for sampling the environment light proportional to luminance (times solid angle).

        Buffer::SharedPtr       mpLightTileBuffer;                  ///< Buffer storing precomputed light tiles (see presampleLights()). This is called "ris buffer" in RTXDI.
        Buffer::SharedPtr       mpCompactLightInfoBuffer;           ///< Optional buffer storing compact light info for samples in the light tile buffer for improved coherence.

        Buffer::SharedPtr       mpReservoirBuffer;                  ///< Buffer storing light reservoirs between kernels (and between frames)
        Buffer::SharedPtr       mpSurfaceDataBuffer;                ///< Buffer storing the surface data for the current and previous frames.
        Buffer::SharedPtr       mpNeighborOffsetsBuffer;            ///< Buffer storing a poisson(-ish) distribution of offsets for sampling randomized neighbors.

        // Compute passes.

        // Passes to pipe data from Falcor into RTXDI.

        ComputePass::SharedPtr  mpReflectTypes;                     ///< Helper pass for reflecting type information.
        ComputePass::SharedPtr  mpUpdateLightsPass;                 ///< Update the light infos and light PDF texture.
        ComputePass::SharedPtr  mpUpdateEnvLightPass;               ///< Update the environment light luminance and PDF texture.

        // Passes for all RTXDI modes.

        ComputePass::SharedPtr  mpPresampleLocalLightsPass;         ///< Presample local lights into light tiles.
        ComputePass::SharedPtr  mpPresampleEnvLightPass;            ///< Presample the environment light into light tiles.
        ComputePass::SharedPtr  mpGenerateCandidatesPass;           ///< Generate initial candidates.
        ComputePass::SharedPtr  mpTestCandidateVisibilityPass;      ///< Test visibility for selected candidate.

        // Passes for various types of reuse.

        ComputePass::SharedPtr  mpSpatialResamplingPass;            ///< Spatial only resampling.
        ComputePass::SharedPtr  mpTemporalResamplingPass;           ///< Temporal only resampling.
        ComputePass::SharedPtr  mpSpatiotemporalResamplingPass;     ///< Spatiotemporal resampling.

        // Compute pass launches.

        void setShaderDataInternal(const ShaderVar& rootVar, const Texture::SharedPtr& pMotionVectors);
        void updateLights(RenderContext* pRenderContext);
        void updateEnvLight(RenderContext* pRenderContext);
        void presampleLights(RenderContext* pRenderContext);
        void generateCandidates(RenderContext* pRenderContext, uint32_t outputReservoirID);
        void testCandidateVisibility(RenderContext* pRenderContext, uint32_t candidateReservoirID);
        uint32_t spatialResampling(RenderContext* pRenderContext, uint32_t inputReservoirID);
        uint32_t temporalResampling(RenderContext* pRenderContext, const Texture::SharedPtr& pMotionVectors, uint32_t candidateReservoirID, uint32_t lastFrameReservoirID);
        uint32_t spatiotemporalResampling(RenderContext* pRenderContext, const Texture::SharedPtr& pMotionVectors, uint32_t candidateReservoirID, uint32_t lastFrameReservoirID);

        // Internal routines.

        void loadShaders();
        void prepareResources(RenderContext* pRenderContext);
        void setRTXDIFrameParameters();

#endif // FALCOR_HAS_RTXDI
    };
}
