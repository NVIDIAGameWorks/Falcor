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
#include "RTXDI.h"
#include "Core/Error.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
#include "Utils/Timing/Profiler.h"
#include <fstd/bit.h> // TODO C++20: Replace with <bit>

namespace Falcor
{
    namespace
    {
        // Shader location
        const std::string kReflectTypesShaderFile = "Rendering/RTXDI/ReflectTypes.cs.slang";
        const std::string kRTXDIShadersFile = "Rendering/RTXDI/RTXDISetup.cs.slang";
        const std::string kLightUpdaterShaderFile = "Rendering/RTXDI/LightUpdater.cs.slang";
        const std::string kEnvLightUpdaterShaderFile = "Rendering/RTXDI/EnvLightUpdater.cs.slang";

        /** Config setting : Maximum number of unique screen-sized reservoir bufers needed by any
            RTXDI pipelines we create in this pass. Just controls memory allocation (not really perf).
        */
        const uint32_t kMaxReservoirs = 3;          ///< Number of reservoirs per pixel to allocate (and hence the max # used).
        const uint32_t kCandidateReservoirID = 2;   ///< We always store per-frame candidate lights in reservoir #2.

        const uint32_t kMinPresampledTileCount = 1;
        const uint32_t kMaxPresampledTileCount = 1024;

        const uint32_t kMinPresampledTileSize = 256;
        const uint32_t kMaxPresampledTileSize = 8192;

        const uint32_t kMinLightCandidateCount = 0;
        const uint32_t kMaxLightCandidateCount = 256;

        const float kMinSpatialRadius = 0.f;
        const float kMaxSpatialRadius = 50.f;

        const uint32_t kMinSpatialSampleCount = 0;
        const uint32_t kMaxSpatialSampleCount = 25;

        const uint32_t kMinSpatialIterations = 0;
        const uint32_t kMaxSpatialIterations = 10;

        const uint32_t kMinMaxHistoryLength = 0;
        const uint32_t kMaxMaxHistoryLength = 50;

        template<typename T>
        void validateRange(T& value, T minValue, T maxValue, const char* name)
        {
            if (value < minValue || value > maxValue)
            {
                logWarning("RTXDI: '{}' is {}. Clamping to [{},{}].", name, value, minValue, maxValue);
                value = std::clamp(value, minValue, maxValue);
            }
        };
    }

    RTXDI::RTXDI(const ref<Scene>& pScene, const Options& options)
        : mpScene(pScene)
        , mpDevice(mpScene->getDevice())
        , mOptions(options)
    {
        if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
            FALCOR_THROW("RTXDI requires Shader Model 6.5 support.");

        mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);

        FALCOR_ASSERT(pScene);
        setOptions(options);
        if (!isInstalled()) logWarning("RTXDI SDK is not installed.");
    }

    void RTXDI::setOptions(const Options& options)
    {
        Options newOptions = options;

        validateRange(newOptions.presampledTileCount, kMinPresampledTileCount, kMaxPresampledTileCount, "presampledTileCount");
        validateRange(newOptions.presampledTileSize, kMinPresampledTileSize, kMaxPresampledTileSize, "presampledTileSize");

        validateRange(newOptions.localLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount, "localLightCandidateCount");
        validateRange(newOptions.infiniteLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount, "infiniteLightCandidateCount");
        validateRange(newOptions.envLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount, "envLightCandidateCount");
        validateRange(newOptions.brdfCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount, "brdfCandidateCount");
        validateRange(newOptions.brdfCutoff, 0.f, 1.f, "brdfCutoff");

        validateRange(newOptions.depthThreshold, 0.f, 1.f, "depthThreshold");
        validateRange(newOptions.normalThreshold, 0.f, 1.f, "normalThreshold");

        validateRange(newOptions.samplingRadius, kMinSpatialRadius, kMaxSpatialRadius, "samplingRadius");
        validateRange(newOptions.spatialSampleCount, kMinSpatialSampleCount, kMaxSpatialSampleCount, "spatialSampleCount");
        validateRange(newOptions.spatialIterations, kMinSpatialIterations, kMaxSpatialIterations, "spatialIterations");

        validateRange(newOptions.maxHistoryLength, kMinMaxHistoryLength, kMaxMaxHistoryLength, "maxHistoryLength");
        validateRange(newOptions.boilingFilterStrength, 0.f, 1.f, "boilingFilterStrength");

#if FALCOR_HAS_RTXDI
        if (newOptions.mode != mOptions.mode)
        {
            mFlags.clearReservoirs = true;
            mLastFrameReservoirID = 0; // Switching out of Talbot mode can break without this.
        }

        if (newOptions.presampledTileCount != mOptions.presampledTileCount || newOptions.presampledTileSize != mOptions.presampledTileSize)
        {
            mpRTXDIContext = nullptr;
        }

        if (newOptions.envLightCandidateCount != mOptions.envLightCandidateCount && newOptions.envLightCandidateCount == 0)
        {
            // Avoid fadeout when disabling env sampling
            mFlags.clearReservoirs = true;
        }

        if (newOptions.testCandidateVisibility != mOptions.testCandidateVisibility)
        {
            mFlags.clearReservoirs = true;
        }
#endif

        mOptions = newOptions;
    }

    DefineList RTXDI::getDefines() const
    {
        DefineList defines;
#if FALCOR_HAS_RTXDI
        defines.add("RTXDI_INSTALLED", "1");
#else
        defines.add("RTXDI_INSTALLED", "0");
#endif
        return defines;
    }

    void RTXDI::bindShaderData(const ShaderVar& rootVar)
    {
#if FALCOR_HAS_RTXDI
        bindShaderDataInternal(rootVar, nullptr, false);
#endif
    }

    void RTXDI::beginFrame(RenderContext* pRenderContext, const uint2& frameDim)
    {
#if FALCOR_HAS_RTXDI
        // Check for scene changes that require shader recompilation.
        // TODO: We may want to reset other data that depends on the scene geometry or materials.
        if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
            is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
        {
            mFlags.recompileShaders = true;
        }

        // Make sure the light collection is created.
        mpScene->getLightCollection(pRenderContext);

        // Initialize previous frame camera data.
        if (mFrameIndex == 0) mPrevCameraData = mpScene->getCamera()->getData();

        // Update the screen resolution.
        if (any(frameDim != mFrameDim))
        {
            mFrameDim = frameDim;
            // Resizes require reallocating resources.
            mpRTXDIContext = nullptr;
        }

        // Load shaders if required.
        if (mFlags.recompileShaders) loadShaders();

        // Create RTXDI context and allocate resources if required.
        if (!mpRTXDIContext) prepareResources(pRenderContext);

        // Clear reservoir buffer if requested. This can be required when changing configuration options.
        if (mFlags.clearReservoirs)
        {
            pRenderContext->clearUAV(mpReservoirBuffer->getUAV().get(), uint4(0));
            mFlags.clearReservoirs = false;
        }

        // Determine what, if anything happened since last frame. TODO: Make more granular / flexible.
        const Scene::UpdateFlags updates = mpScene->getUpdates();

        // Emissive lights.
        if (is_set(updates, Scene::UpdateFlags::LightCollectionChanged)) mFlags.updateEmissiveLights = true;
        if (is_set(updates, Scene::UpdateFlags::EmissiveMaterialsChanged)) mFlags.updateEmissiveLightsFlux = true;
        // Analytic lights.
        if (is_set(updates, Scene::UpdateFlags::LightCountChanged)) mFlags.updateAnalyticLights = true;
        if (is_set(updates, Scene::UpdateFlags::LightPropertiesChanged)) mFlags.updateAnalyticLights = true;
        if (is_set(updates, Scene::UpdateFlags::LightIntensityChanged)) mFlags.updateAnalyticLightsFlux = true;
        // Env light. Update the env light PDF either if the env map changed or its tint/intensity changed.
        if (is_set(updates, Scene::UpdateFlags::EnvMapChanged)) mFlags.updateEnvLight = true;
        if (is_set(updates, Scene::UpdateFlags::EnvMapPropertiesChanged) && is_set(mpScene->getEnvMap()->getChanges(), EnvMap::Changes::Intensity)) mFlags.updateEnvLight = true;

        if (is_set(updates, Scene::UpdateFlags::RenderSettingsChanged))
        {
            mFlags.updateAnalyticLights = true;
            mFlags.updateAnalyticLightsFlux = true;
            mFlags.updateEmissiveLights = true;
            mFlags.updateEmissiveLightsFlux = true;
            mFlags.updateEnvLight = true;
        }

        mpPixelDebug->beginFrame(pRenderContext, mFrameDim);
#endif
    }

    void RTXDI::endFrame(RenderContext* pRenderContext)
    {
#if FALCOR_HAS_RTXDI
        // Increment our frame counter and swap surface buffers.
        mFrameIndex++;
        mCurrentSurfaceBufferIndex = 1 - mCurrentSurfaceBufferIndex;

        // Remember this frame's camera data for use next frame.
        mPrevCameraData = mpScene->getCamera()->getData();

        mpPixelDebug->endFrame(pRenderContext);
#endif
    }

    void RTXDI::update(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors)
    {
#if FALCOR_HAS_RTXDI
        FALCOR_PROFILE(pRenderContext, "RTXDI::update");

        // Create a PDF texture for our primitive lights (for now, just triangles)
        updateLights(pRenderContext);
        updateEnvLight(pRenderContext);

        // Update our parameters for the current frame and pass them into our GPU structure.
        setRTXDIFrameParameters();

        // Create tiles of presampled lights once per frame to improve per-pixel memory coherence.
        presampleLights(pRenderContext);

        // Reservoir buffer containing reservoirs after sampling/resampling.
        uint32_t outputReservoirID;

        switch (mOptions.mode)
        {
        case Mode::NoResampling:
            generateCandidates(pRenderContext, kCandidateReservoirID);
            outputReservoirID = kCandidateReservoirID;
            break;
        case Mode::SpatialResampling:
            generateCandidates(pRenderContext, kCandidateReservoirID);
            testCandidateVisibility(pRenderContext, kCandidateReservoirID);
            outputReservoirID = spatialResampling(pRenderContext, kCandidateReservoirID);
            break;
        case Mode::TemporalResampling:
            generateCandidates(pRenderContext, kCandidateReservoirID);
            testCandidateVisibility(pRenderContext, kCandidateReservoirID);
            outputReservoirID = temporalResampling(pRenderContext, pMotionVectors, kCandidateReservoirID, mLastFrameReservoirID);
            break;
        case Mode::SpatiotemporalResampling:
            generateCandidates(pRenderContext, kCandidateReservoirID);
            testCandidateVisibility(pRenderContext, kCandidateReservoirID);
            outputReservoirID = spatiotemporalResampling(pRenderContext, pMotionVectors, kCandidateReservoirID, mLastFrameReservoirID);
            break;
        }

        // Remember output reservoir buffer for the next frame (and shading this frame).
        mLastFrameReservoirID = outputReservoirID;
#endif
    }

#if FALCOR_HAS_RTXDI

    void RTXDI::bindShaderDataInternal(const ShaderVar& rootVar, const ref<Texture>& pMotionVectors, bool bindScene)
    {
        auto var = rootVar["gRTXDI"];

        // Send our parameter structure down
        var["params"].setBlob(&mRTXDIShaderParams, sizeof(mRTXDIShaderParams));

        // Parameters needed inside the core RTXDI application bridge
        var["frameIndex"] = mFrameIndex;
        var["rayEpsilon"] = mOptions.rayEpsilon;
        var["frameDim"] = mFrameDim;
        var["pixelCount"] = mFrameDim.x * mFrameDim.y;
        var["storeCompactLightInfo"] = mOptions.storeCompactLightInfo;
        var["useEmissiveTextures"] = mOptions.useEmissiveTextures;
        var["currentSurfaceBufferIndex"] = mCurrentSurfaceBufferIndex;
        var["prevSurfaceBufferIndex"] = 1 - mCurrentSurfaceBufferIndex;

        // Parameters for initial candidate samples
        var["localLightCandidateCount"] = mOptions.localLightCandidateCount;
        var["infiniteLightCandidateCount"] = mOptions.infiniteLightCandidateCount;
        var["envLightCandidateCount"] = mOptions.envLightCandidateCount;
        var["brdfCandidateCount"] = mOptions.brdfCandidateCount;

        // Parameters for general sample reuse
        var["maxHistoryLength"] = mOptions.maxHistoryLength;
        var["biasCorrectionMode"] = uint(mOptions.biasCorrection);

        // Parameter for final shading
        var["finalShadingReservoir"] = mLastFrameReservoirID;

        // Parameters for generally spatial sample reuse
        var["spatialSampleCount"] = mOptions.spatialSampleCount;
        var["disocclusionSampleCount"] = mOptions.spatialSampleCount;
        var["samplingRadius"] = mOptions.samplingRadius;
        var["depthThreshold"] = mOptions.depthThreshold;
        var["normalThreshold"] = mOptions.normalThreshold;
        var["boilingFilterStrength"] = mOptions.boilingFilterStrength;
        var["enableVisibilityShortcut"] = mOptions.enableVisibilityShortcut;
        var["enablePermutationSampling"] = mOptions.enablePermutationSampling;

        // Parameters for last frame's camera coordinate
        var["prevCameraU"] = mPrevCameraData.cameraU;
        var["prevCameraV"] = mPrevCameraData.cameraV;
        var["prevCameraW"] = mPrevCameraData.cameraW;
        var["prevCameraJitter"] = float2(mPrevCameraData.jitterX, mPrevCameraData.jitterY);

        // Setup textures and other buffers needed by the RTXDI bridge
        var["lightInfo"] = mpLightInfoBuffer;
        var["surfaceData"] = mpSurfaceDataBuffer;
        var["risBuffer"] = mpLightTileBuffer;
        var["compactLightInfo"] = mpCompactLightInfoBuffer;
        var["reservoirs"] = mpReservoirBuffer;
        var["neighborOffsets"] = mpNeighborOffsetsBuffer;
        var["motionVectors"] = pMotionVectors;

        // PDF textures for importance sampling. Some shaders need UAVs, some SRVs
        var["localLightPdfTexture"] = mpLocalLightPdfTexture;
        var["envLightLuminanceTexture"] = mpEnvLightLuminanceTexture;
        var["envLightPdfTexture"] = mpEnvLightPdfTexture;

        // Bind the scene.
        if (bindScene)
            mpScene->bindShaderData(rootVar["gScene"]);
    }

    void RTXDI::updateLights(RenderContext* pRenderContext)
    {
        FALCOR_PROFILE(pRenderContext, "updateLights");

        // First, update our list of analytic lights to use.
        if (mFlags.updateAnalyticLights)
        {
            if (mpScene->useAnalyticLights())
            {
                std::vector<uint32_t> localAnalyticLightIDs;
                std::vector<uint32_t> infiniteAnalyticLightIDs;

                for (uint32_t lightID = 0; lightID < mpScene->getActiveLightCount(); ++lightID)
                {
                    const auto& pLight = mpScene->getActiveLight(lightID);
                    switch (pLight->getType())
                    {
                    case LightType::Point:
                        localAnalyticLightIDs.push_back(lightID);
                        break;
                    case LightType::Directional:
                    case LightType::Distant:
                        infiniteAnalyticLightIDs.push_back(lightID);
                        break;
                    case LightType::Rect:
                    case LightType::Disc:
                    case LightType::Sphere:
                        // We currently ignore all analytic area lights.
                        break;
                    default:
                        break;
                    }
                }

                // Update light counts.
                mLights.localAnalyticLightCount = (uint32_t)localAnalyticLightIDs.size();
                mLights.infiniteAnalyticLightCount = (uint32_t)infiniteAnalyticLightIDs.size();

                // Update list of light IDs, local lights followed by infinite lights.
                mLights.analyticLightIDs.clear();
                mLights.analyticLightIDs.reserve(localAnalyticLightIDs.size() + infiniteAnalyticLightIDs.size());
                mLights.analyticLightIDs.insert(mLights.analyticLightIDs.end(), localAnalyticLightIDs.begin(), localAnalyticLightIDs.end());
                mLights.analyticLightIDs.insert(mLights.analyticLightIDs.end(), infiniteAnalyticLightIDs.begin(), infiniteAnalyticLightIDs.end());

                // Create GPU buffer for holding light IDs.
                if (!mLights.analyticLightIDs.empty() && (!mpAnalyticLightIDBuffer || mpAnalyticLightIDBuffer->getElementCount() < mLights.analyticLightIDs.size()))
                {
                    mpAnalyticLightIDBuffer = mpDevice->createStructuredBuffer(sizeof(uint32_t), (uint32_t)mLights.analyticLightIDs.size());
                }

                // Update GPU buffer.
                if (mpAnalyticLightIDBuffer) mpAnalyticLightIDBuffer->setBlob(mLights.analyticLightIDs.data(), 0, mLights.analyticLightIDs.size() * sizeof(uint32_t));
            }
            else
            {
                // Analytic lights are disabled.
                mLights.localAnalyticLightCount = 0;
                mLights.infiniteAnalyticLightCount = 0;
                mLights.analyticLightIDs.clear();
            }
        }

        // Update other light counts.
        mLights.emissiveLightCount = mpScene->useEmissiveLights() ? mpScene->getLightCollection(pRenderContext)->getActiveLightCount(pRenderContext) : 0;
        mLights.envLightPresent = mpScene->useEnvLight();

        uint32_t localLightCount = mLights.getLocalLightCount();
        uint32_t totalLightCount = mLights.getTotalLightCount();

        // Allocate buffer for light infos.
        if (!mpLightInfoBuffer || mpLightInfoBuffer->getElementCount() < totalLightCount)
        {
            mpLightInfoBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()["lightInfo"], totalLightCount);
        }

        // Allocate local light PDF texture, which RTXDI uses for importance sampling.
        {
            uint32_t width, height, mipLevels;
            rtxdi::ComputePdfTextureSize(localLightCount, width, height, mipLevels);
            if (!mpLocalLightPdfTexture || mpLocalLightPdfTexture->getWidth() != width || mpLocalLightPdfTexture->getHeight() != height || mpLocalLightPdfTexture->getMipCount() != mipLevels)
            {
                mpLocalLightPdfTexture = mpDevice->createTexture2D(width, height,
                    ResourceFormat::R16Float, 1, mipLevels, nullptr,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget);
            }
        }

        // If the layout of local lights has changed, we need to make sure to remove any extra non-zero entries in the local light PDF texture.
        // We simply clear the texture and populate it from scratch.
        if (mLights.prevEmissiveLightCount != mLights.emissiveLightCount || mLights.prevLocalAnalyticLightCount != mLights.localAnalyticLightCount)
        {
            mFlags.updateAnalyticLightsFlux = true;
            mFlags.updateEmissiveLightsFlux = true;
            pRenderContext->clearUAV(mpLocalLightPdfTexture->getUAV().get(), float4(0.f));
        }

        // If the number of emissive lights has changed, we need to update the analytic lights because they change position in the light info buffer.
        if (mLights.prevEmissiveLightCount != mLights.emissiveLightCount)
        {
            mFlags.updateAnalyticLights = true;
        }

        // Run the update pass if any lights have changed.
        if (mFlags.updateEmissiveLights || mFlags.updateEmissiveLightsFlux || mFlags.updateAnalyticLights || mFlags.updateAnalyticLightsFlux || mFlags.updateEnvLight)
        {
            // Compute launch dimensions.
            uint2 threadCount = { 8192u, div_round_up(totalLightCount, 8192u) };

            auto var = mpUpdateLightsPass->getRootVar()["gLightUpdater"];
            var["lightInfo"] = mpLightInfoBuffer;
            var["localLightPdf"] = mpLocalLightPdfTexture;
            var["analyticLightIDs"] = mpAnalyticLightIDBuffer;
            var["threadCount"] = threadCount;
            var["totalLightCount"] = mLights.getTotalLightCount();
            var["firstLocalAnalyticLight"] = mLights.emissiveLightCount;
            var["firstInfiniteAnalyticLight"] = mLights.emissiveLightCount + mLights.localAnalyticLightCount;
            var["envLightIndex"] = mLights.getEnvLightIndex();
            var["updateEmissiveLights"] = mFlags.updateEmissiveLights;
            var["updateEmissiveLightsFlux"] = mFlags.updateEmissiveLightsFlux;
            var["updateAnalyticLights"] = mFlags.updateAnalyticLights;
            var["updateAnalyticLightsFlux"] = mFlags.updateAnalyticLightsFlux;
            mpScene->bindShaderData(mpUpdateLightsPass->getRootVar()["gScene"]);
            mpUpdateLightsPass->execute(pRenderContext, threadCount.x, threadCount.y);
        }

        // Update the light PDF texture mipmap chain if necessary.
        if (mFlags.updateEmissiveLightsFlux | mFlags.updateAnalyticLightsFlux)
        {
            mpLocalLightPdfTexture->generateMips(pRenderContext);
        }

        // Keep track of the number of local lights for the next frame.
        mLights.prevEmissiveLightCount = mLights.emissiveLightCount;
        mLights.prevLocalAnalyticLightCount = mLights.localAnalyticLightCount;

        mFlags.updateEmissiveLights = false;
        mFlags.updateEmissiveLightsFlux = false;
        mFlags.updateAnalyticLights = false;
        mFlags.updateAnalyticLightsFlux = false;
    }

    void RTXDI::updateEnvLight(RenderContext* pRenderContext)
    {
        FALCOR_PROFILE(pRenderContext, "updateEnvLight");

        // If scene uses an environment light, create a luminance & pdf texture for sampling it.
        if (mpScene->useEnvLight() && mFlags.updateEnvLight)
        {
            const auto& pEnvMap = mpScene->getEnvMap()->getEnvMap();
            FALCOR_ASSERT(pEnvMap);
            auto& pLuminanceTexture = mpEnvLightLuminanceTexture;
            auto& pPdfTexture = mpEnvLightPdfTexture;

            // RTXDI expects power-of-two textures.
            uint32_t width = fstd::bit_ceil(pEnvMap->getWidth());
            uint32_t height = fstd::bit_ceil(pEnvMap->getHeight());

            // Create luminance texture if it doesn't exist yet or has the wrong dimensions.
            if (!pLuminanceTexture || pLuminanceTexture->getWidth() != width || pLuminanceTexture->getHeight() != height)
            {
                pLuminanceTexture = mpDevice->createTexture2D(
                    width, height, ResourceFormat::R32Float, 1, 1, nullptr,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget);
            }

            // Create pdf texture if it doesn't exist yet or has the wrong dimensions.
            if (!pPdfTexture || pPdfTexture->getWidth() != width || pPdfTexture->getHeight() != height)
            {
                pPdfTexture = mpDevice->createTexture2D(
                    width, height, ResourceFormat::R32Float, 1, Resource::kMaxPossible, nullptr,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget);
            }

            // Update env light textures.
            auto var = mpUpdateEnvLightPass->getRootVar()["gEnvLightUpdater"];
            var["envLightLuminance"] = pLuminanceTexture;
            var["envLightPdf"] = pPdfTexture;
            var["texDim"] = uint2(width, height);
            mpScene->bindShaderData(mpUpdateEnvLightPass->getRootVar()["gScene"]);
            mpUpdateEnvLightPass->execute(pRenderContext, width, height);

            // Create a mipmap chain for pdf texure.
            pPdfTexture->generateMips(pRenderContext);
        }

        mFlags.updateEnvLight = false;
    }

    void RTXDI::presampleLights(RenderContext* pRenderContext)
    {
        FALCOR_PROFILE(pRenderContext, "presampleLights");

        // Presample local lights.
        {
            auto var = mpPresampleLocalLightsPass->getRootVar();
            bindShaderDataInternal(var, nullptr);
            mpPresampleLocalLightsPass->execute(pRenderContext, mRTXDIContextParams.TileSize, mRTXDIContextParams.TileCount);
        }

        // Presample environment light.
        if (mLights.envLightPresent)
        {
            auto var = mpPresampleEnvLightPass->getRootVar();
            bindShaderDataInternal(var, nullptr);
            mpPresampleEnvLightPass->execute(pRenderContext, mRTXDIContextParams.EnvironmentTileSize, mRTXDIContextParams.EnvironmentTileCount);
        }
    }

    void RTXDI::generateCandidates(RenderContext* pRenderContext, uint32_t outputReservoirID)
    {
        FALCOR_PROFILE(pRenderContext, "generateCandidates");

        auto var = mpGenerateCandidatesPass->getRootVar();
        mpPixelDebug->prepareProgram(mpGenerateCandidatesPass->getProgram(), var);

        var["CB"]["gOutputReservoirID"] = outputReservoirID;
        bindShaderDataInternal(var, nullptr);
        mpGenerateCandidatesPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
    }

    void RTXDI::testCandidateVisibility(RenderContext* pRenderContext, uint32_t candidateReservoirID)
    {
        if (!mOptions.testCandidateVisibility) return;

        FALCOR_PROFILE(pRenderContext, "testCandidateVisibility");

        auto var = mpTestCandidateVisibilityPass->getRootVar();
        var["CB"]["gOutputReservoirID"] = candidateReservoirID;
        bindShaderDataInternal(var, nullptr);
        mpTestCandidateVisibilityPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
    }

    uint32_t RTXDI::spatialResampling(RenderContext* pRenderContext, uint32_t inputReservoirID)
    {
        FALCOR_PROFILE(pRenderContext, "spatialResampling");

        // We ping-pong between reservoir buffers, depending on # of spatial iterations.
        uint32_t inputID = inputReservoirID;
        uint32_t outputID = (inputID != 1) ? 1 : 0;

        auto var = mpSpatialResamplingPass->getRootVar();
        mpPixelDebug->prepareProgram(mpSpatialResamplingPass->getProgram(), var);

        for (uint32_t i = 0; i < mOptions.spatialIterations; ++i)
        {
            var["CB"]["gInputReservoirID"] = inputID;
            var["CB"]["gOutputReservoirID"] = outputID;
            bindShaderDataInternal(var, nullptr);
            mpSpatialResamplingPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

            // Ping pong our input and output buffers. (Generally between reservoirs 0 & 1).
            std::swap(inputID, outputID);
        }

        // Return the ID of the last buffer written into.
        return inputID;
    }

    uint32_t RTXDI::temporalResampling(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors,
        uint32_t candidateReservoirID, uint32_t lastFrameReservoirID)
    {
        FALCOR_PROFILE(pRenderContext, "temporalResampling");

        // This toggles between storing each frame's outputs between reservoirs 0 and 1.
        uint32_t outputReservoirID = 1 - lastFrameReservoirID;

        auto var = mpTemporalResamplingPass->getRootVar();
        mpPixelDebug->prepareProgram(mpTemporalResamplingPass->getProgram(), var);

        var["CB"]["gTemporalReservoirID"] = lastFrameReservoirID;
        var["CB"]["gInputReservoirID"] = candidateReservoirID;
        var["CB"]["gOutputReservoirID"] = outputReservoirID;
        bindShaderDataInternal(var, pMotionVectors);
        mpTemporalResamplingPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        return outputReservoirID;
    }

    uint32_t RTXDI::spatiotemporalResampling(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors,
        uint32_t candidateReservoirID, uint32_t lastFrameReservoirID)
    {
        FALCOR_PROFILE(pRenderContext, "spatiotemporalResampling");

        // This toggles between storing each frame's outputs between reservoirs 0 and 1.
        uint32_t outputReservoirID = 1 - lastFrameReservoirID;

        auto var = mpSpatiotemporalResamplingPass->getRootVar();
        mpPixelDebug->prepareProgram(mpSpatiotemporalResamplingPass->getProgram(), var);

        var["CB"]["gTemporalReservoirID"] = lastFrameReservoirID;
        var["CB"]["gInputReservoirID"] = candidateReservoirID;
        var["CB"]["gOutputReservoirID"] = outputReservoirID;
        bindShaderDataInternal(var, pMotionVectors);
        mpSpatiotemporalResamplingPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        return outputReservoirID;
    }

    void RTXDI::loadShaders()
    {
        FALCOR_ASSERT(mpScene);
        mpReflectTypes = ComputePass::create(mpDevice, kReflectTypesShaderFile);

        // Issue warnings if packed types are not aligned to 16B for best performance.
        auto pReflector = mpReflectTypes->getProgram()->getReflector();
        FALCOR_ASSERT(pReflector->findType("PackedPolymorphicLight"));
        FALCOR_ASSERT(pReflector->findType("PackedSurfaceData"));
        if (pReflector->findType("PackedPolymorphicLight")->getByteSize() % 16 != 0) logWarning("PackedPolymorphicLight struct size is not a multiple of 16B.");
        if (pReflector->findType("PackedSurfaceData")->getByteSize() % 16 != 0) logWarning("PackedSurfaceData struct size is not a multiple of 16B.");

        // Helper for creating compute passes.
        auto createComputePass = [&](const std::string& file, const std::string& entryPoint) {
            DefineList defines = mpScene->getSceneDefines();
            defines.add("RTXDI_INSTALLED", "1");

            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(file);
            desc.csEntry(entryPoint);
            desc.addTypeConformances(mpScene->getTypeConformances());
            ref<ComputePass> pPass = ComputePass::create(mpDevice, desc, defines);
            return pPass;
        };

        // Load compute passes for setting up RTXDI light information.
        mpUpdateLightsPass = createComputePass(kLightUpdaterShaderFile, "main");
        mpUpdateEnvLightPass = createComputePass(kEnvLightUpdaterShaderFile, "main");

        // Load compute passes for RTXDI sampling and resampling.
        mpPresampleLocalLightsPass = createComputePass(kRTXDIShadersFile, "presampleLocalLights");
        mpPresampleEnvLightPass = createComputePass(kRTXDIShadersFile, "presampleEnvLight");
        mpGenerateCandidatesPass = createComputePass(kRTXDIShadersFile, "generateCandidates");
        mpTestCandidateVisibilityPass = createComputePass(kRTXDIShadersFile, "testCandidateVisibility");
        mpSpatialResamplingPass = createComputePass(kRTXDIShadersFile, "spatialResampling");
        mpTemporalResamplingPass = createComputePass(kRTXDIShadersFile, "temporalResampling");
        mpSpatiotemporalResamplingPass = createComputePass(kRTXDIShadersFile, "spatiotemporalResampling");

        mFlags.recompileShaders = false;
    }

    void RTXDI::prepareResources(RenderContext* pRenderContext)
    {
        // Ask for some other refreshes elsewhere to make sure we're all consistent.
        mFlags.clearReservoirs = true;
        mFlags.updateEmissiveLights = true;
        mFlags.updateEmissiveLightsFlux = true;
        mFlags.updateAnalyticLights = true;
        mFlags.updateAnalyticLightsFlux = true;
        mFlags.updateEnvLight = true;

        // Make sure the RTXDI context has the current screen resolution.
        mRTXDIContextParams.RenderWidth = mFrameDim.x;
        mRTXDIContextParams.RenderHeight = mFrameDim.y;

        // Set the number and size of our presampled tiles.
        mRTXDIContextParams.TileSize = mOptions.presampledTileSize;
        mRTXDIContextParams.TileCount = mOptions.presampledTileCount;
        mRTXDIContextParams.EnvironmentTileSize = mOptions.presampledTileSize;
        mRTXDIContextParams.EnvironmentTileCount = mOptions.presampledTileCount;

        // Create a new RTXDI context.
        mpRTXDIContext = std::make_unique<rtxdi::Context>(mRTXDIContextParams);

        // Note: Additional resources are allocated lazily in updateLights() and updateEnvMap().

        // Allocate buffer for presampled light tiles (RTXDI calls this "RIS buffers").
        uint32_t lightTileSampleCount = std::max(mpRTXDIContext->GetRisBufferElementCount(), 1u);
        if (!mpLightTileBuffer || mpLightTileBuffer->getElementCount() < lightTileSampleCount)
        {
            mpLightTileBuffer = mpDevice->createTypedBuffer(ResourceFormat::RG32Uint, lightTileSampleCount);
        }

        // Allocate buffer for compact light info used to improve coherence for presampled light tiles.
        {
            uint32_t elementCount = lightTileSampleCount * 2;
            if (!mpCompactLightInfoBuffer || mpCompactLightInfoBuffer->getElementCount() < elementCount)
            {
                mpCompactLightInfoBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()["lightInfo"], elementCount);
            }
        }

        // Allocate buffer for light reservoirs. There are multiple reservoirs (specified by kMaxReservoirs) concatenated together.
        {
            uint32_t elementCount = mpRTXDIContext->GetReservoirBufferElementCount() * kMaxReservoirs;
            if (!mpReservoirBuffer || mpReservoirBuffer->getElementCount() < elementCount)
            {
                mpReservoirBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()["reservoirs"], elementCount);
            }
        }

        // Allocate buffer for surface data for current and previous frames.
        {
            uint32_t elementCount = 2 * mFrameDim.x * mFrameDim.y;
            if (!mpSurfaceDataBuffer || mpSurfaceDataBuffer->getElementCount() < elementCount)
            {
                mpSurfaceDataBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()["surfaceData"], elementCount);
            }
        }

        // Allocate buffer for neighbor offsets.
        if (!mpNeighborOffsetsBuffer)
        {
            std::vector<uint8_t> offsets(2 * (size_t)mRTXDIContextParams.NeighborOffsetCount);
            mpRTXDIContext->FillNeighborOffsetBuffer(offsets.data());
            mpNeighborOffsetsBuffer = mpDevice->createTypedBuffer(
                ResourceFormat::RG8Snorm,
                mRTXDIContextParams.NeighborOffsetCount,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                offsets.data());
        }
    }

    void RTXDI::setRTXDIFrameParameters()
    {
        rtxdi::FrameParameters frameParameters;

        // Set current frame index.
        frameParameters.frameIndex = mFrameIndex;

        // Always enable importance sampling for local lights.
        frameParameters.enableLocalLightImportanceSampling = true;

        // Set the range of local lights.
        frameParameters.firstLocalLight = mLights.getFirstLocalLightIndex();
        frameParameters.numLocalLights = mLights.getLocalLightCount();

        // Set the range of infinite lights.
        frameParameters.firstInfiniteLight = mLights.getFirstInfiniteLightIndex();
        frameParameters.numInfiniteLights = mLights.getInfiniteLightCount();

        // Set the environment light.
        frameParameters.environmentLightPresent = mLights.envLightPresent;
        frameParameters.environmentLightIndex = mLights.getEnvLightIndex();

        // In case we're using ReGIR, update the grid center to be at the camera.
        auto cameraPos = mpScene->getCamera()->getPosition();
        frameParameters.regirCenter = rtxdi::float3{ cameraPos.x, cameraPos.y, cameraPos.z };

        // Update the parameters RTXDI needs when we call its functions in our shaders.
        mpRTXDIContext->FillRuntimeParameters(mRTXDIShaderParams, frameParameters);
    }

#endif // FALCOR_HAS_RTXDI

    bool RTXDI::renderUI(Gui::Widgets& widget)
    {
        bool changed = false;

#if FALCOR_HAS_RTXDI
        // Edit a copy of the options and use setOptions() to validate the changes and trigger required
        // actions due to changing them. This unifies the logic independent of using the UI or setOptions() directly.
        Options options = mOptions;

        // Our user-controllable parameters vary depending on if we're doing reuse and what kind
        bool useResampling = (mOptions.mode != Mode::NoResampling);
        bool useTemporalResampling = (mOptions.mode == Mode::TemporalResampling || mOptions.mode == Mode::SpatiotemporalResampling);
        bool useSpatialResampling = (mOptions.mode == Mode::SpatialResampling || mOptions.mode == Mode::SpatiotemporalResampling);

        changed |= widget.dropdown("Mode", options.mode);
        widget.tooltip("Mode.\n\n"
            "NoResampling: No resampling (Talbot RIS from EGSR 2005 \"Importance Resampling for Global Illumination\").\n"
            "SpatialResampling: Spatial resampling only.\n"
            "TemporalResampling: Temporal resampling only.\n"
            "SpatiotemporalResampling: Spatiotemporal resampling."
        );

        if (auto group = widget.group("Light presampling", false))
        {
            changed |= group.var("Tile count", options.presampledTileCount, kMinPresampledTileCount, kMaxPresampledTileCount);
            group.tooltip("Number of precomputed light tiles.");

            changed |= group.var("Tile size", options.presampledTileSize, kMinPresampledTileSize, kMaxPresampledTileSize, 128u);
            group.tooltip("Size of each precomputed light tile (number of samples).");

            changed |= group.checkbox("Store compact light info", options.storeCompactLightInfo);
            group.tooltip("Store compact light info for precomputed light tiles to improve coherence.");
        }

        if (auto group = widget.group("Initial candidate sampling", false))
        {
            changed |= group.var("Local light samples", options.localLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
            group.tooltip("Number of initial local light candidate samples.");

            changed |= group.var("Infinite light samples", options.infiniteLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
            group.tooltip("Number of initial infinite light candidate samples.");

            changed |= group.var("Environment light samples", options.envLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
            group.tooltip("Number of initial environment light candidate samples.");

            changed |= group.var("BRDF samples", options.brdfCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
            group.tooltip("Number of initial BRDF candidate samples.");

            changed |= group.var("BRDF Cutoff", options.brdfCutoff, 0.f, 1.f);
            group.tooltip("Value in range [0,1] to determine how much to shorten BRDF rays");


            if (useResampling)
            {
                changed |= group.checkbox("Test selected candidate visibility", options.testCandidateVisibility);
                group.tooltip("Test visibility on selected candidate sample before doing resampling.\n\n"
                    "Occluded samples have their reseroirs zeroed out, so such a sample never has a chance to contribute "
                    "to neighbors. This is especially valuable in multi-room scenes, where occluded lights from a different "
                    "room are also unlikely to light neighbors.");
            }
        }

        if (useResampling)
        {
            if (auto group = widget.group("Resampling", false))
            {
                changed |= group.dropdown("Bias correction", options.biasCorrection);
                group.tooltip("Bias correction mode.\n\n"
                    "Off: Use (1/M) normalization, which is very biased but also very fast.\n"
                    "Basic: Use MIS-like normalization but assume that every sample is visible.\n"
                    "Pairwise: Use pairwise MIS normalization. Assumes every sample is visibile.\n"
                    "RayTraced: Use MIS-like normalization with visibility rays. Unbiased.\n");

                changed |= group.var("Depth threshold", options.depthThreshold, 0.0f, 1.0f, 0.001f);
                group.tooltip("Relative depth difference at which pixels are classified too far apart to be reused (0.1 = 10%).");

                changed |= group.var("Normal threshold", options.normalThreshold, 0.0f, 1.0f, 0.001f);
                group.tooltip("Cosine of the angle between normals, below which pixels are classified too far apart to be reused.");
            }
        }

        if (useSpatialResampling)
        {
            if (auto group = widget.group("Spatial resampling", false))
            {
                changed |= group.var("Sampling radius", options.samplingRadius, kMinSpatialRadius, kMaxSpatialRadius, 0.1f);
                group.tooltip("Screen-space radius for spatial resampling, measured in pixels.");

                changed |= group.var("Sample count", options.spatialSampleCount, kMinSpatialSampleCount, kMaxSpatialSampleCount);
                group.tooltip("Number of neighbor pixels considered for resampling.");

                if (options.mode == Mode::SpatialResampling)
                {
                    changed |= group.var("Iterations", options.spatialIterations, kMinSpatialIterations, kMaxSpatialIterations);
                    group.tooltip("Number of spatial resampling passes.");
                }
            }
        }

        if (useTemporalResampling)
        {
            if (auto group = widget.group("Temporal resampling", false))
            {
                changed |= group.var("Max history length", options.maxHistoryLength, kMinMaxHistoryLength, kMaxMaxHistoryLength);
                group.tooltip("Maximum history length for temporal reuse, measured in frames.");

                changed |= group.var("Boiling filter strength", options.boilingFilterStrength, 0.0f, 1.0f, 0.001f);
                group.tooltip("0 = off, 1 = full strength.");
            }
        }

        if (auto group = widget.group("Misc"))
        {
            changed |= group.checkbox("Use emissive textures", options.useEmissiveTextures);
            group.tooltip("Use emissive textures to return final sample incident radiance (true is slower and noisier).");

            changed |= group.checkbox("Enable permutation sampling", options.enablePermutationSampling);
            group.tooltip("Enables permuting the pixels sampled from the previous frame (noisier but more denoiser friendly).");
        }

        if (auto group = widget.group("Debugging"))
        {
            mpPixelDebug->renderUI(group);
        }

        if (changed) setOptions(options);
#else
        widget.textWrapped("The RTXDI SDK is not installed. See README for installation details.");
#endif

        return changed;
    }
}
