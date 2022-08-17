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
#include "RTXGIVolume.h"

#if FALCOR_HAS_D3D12

#include "UpdateProbesDebugData.slang"
#include "Core/API/API.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Core/API/Shared/D3D12DescriptorData.h"
#include "Utils/Logger.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Rendering/Lights/LightBVHSampler.h"

namespace Falcor
{
    namespace
    {
        // Ray tracing settings that affect the traversal stack size.
        // These should be set as small as possible.
        const uint32_t kMaxPayloadSizeBytes = 24;
        const uint32_t kMaxAttributeSizeBytes = 8;
        const uint32_t kMaxRecursionDepth = 1; // CHS/Miss only

        const std::string kProbeRadianceUpdateFilename = "Rendering/RTXGI/UpdateProbes.rt.slang";
        const std::string kParameterBlockName = "gRTXGIVolume";

        const uint32_t kMinNumIrradianceOrDistanceTexels = 8;   ///< Minimum supported size in current SDK version.
        const uint32_t kMaxNumRaysPerProbe = 256;               ///< Limitation in current SDK version.
    }

    RTXGIVolume::SharedPtr RTXGIVolume::create(RenderContext* pRenderContext, Scene::SharedPtr pScene, EnvMapSampler::SharedPtr pEnvMapSampler, EmissiveLightSampler::SharedPtr pEmissiveSampler, const Options& options)
    {
        return SharedPtr(new RTXGIVolume(pRenderContext, pScene, pEnvMapSampler, pEmissiveSampler, options));
    }

    void RTXGIVolume::update(RenderContext* pRenderContext)
    {
        FALCOR_ASSERT(mpScene);

        // Update the light sampler to the current frame.
        FALCOR_ASSERT(mpEmissiveSampler);
        mpEmissiveSampler->update(pRenderContext);

        const auto& pLights = mpScene->getLightCollection(pRenderContext);
        FALCOR_ASSERT(pLights);
        mUseEmissiveSampler = pLights->getActiveLightCount() > 0;

        if (mOptionsDirty)
        {
            // Re-create the RTXGI probe volume.
            destroyRTXGI();
            validateOptions();
            initRTXGI();

            // TODO: Remove flush and replace by necessary barriers.
            pRenderContext->flush(true);

            // Update our resources.
            updateParameterBlock();

            mProbeUpdateDispatchDims.x = mpDDGIVolume->GetNumRaysPerProbe();
            mProbeUpdateDispatchDims.y = mpDDGIVolume->GetNumProbes();
            mpProbeUpdateVars = nullptr; // Trigger re-creation of vars below

            mOptionsDirty = false;
        }
        FALCOR_ASSERT(mpDDGIVolume && mIsDDGIVolumeValid);

        /// If the classification or relocation has just been turned on, we need to reset the probe state.
        {
            if (mOptions.enableProbeClassification && !mpDDGIVolume->GetProbeClassificationEnabled())
            {
                mpDDGIVolume->SetProbeClassificationNeedsReset(true);
            }

            if (mOptions.enableProbeRelocation && !mpDDGIVolume->GetProbeRelocationEnabled())
            {
                mpDDGIVolume->SetProbeRelocationEnabled(true);
            }

            mpDDGIVolume->SetProbeClassificationEnabled(mOptions.enableProbeClassification);
            mpDDGIVolume->SetProbeRelocationEnabled(mOptions.enableProbeRelocation);
        }

        // Update the volume's random rotation and constant buffer.
        // Then copy the packed volume descriptor into both the ParamBlock and the SDK-internal
        // ddgiVolumeBlock
        {
            FALCOR_PROFILE("rtxgi::DDGIVolume::Update");
            mpDDGIVolume->Update();
            rtxgi::DDGIVolumeDescGPUPacked volumeDesc = mpDDGIVolume->GetDescGPUPacked();
            mpDDGIVolumeBlockSDK->setBlob(&volumeDesc, 0, sizeof(rtxgi::DDGIVolumeDescGPUPacked));
            mpParameterBlock->getRootVar()["volumePacked"].setBlob(volumeDesc);
        }

        // Falcor raytracing pass to fill RTXDI ray data.
        probeUpdatePass(pRenderContext);

        // RTXGI accesses the ray data texture, put a UAV barrier to synchronize with pass above.
        pRenderContext->uavBarrier(mpRayDataTex.get());

        // RTXGI expects these two textures to be in D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE state.
        pRenderContext->resourceBarrier(mpIrradianceTex.get(), Falcor::Resource::State::PixelShader);
        pRenderContext->resourceBarrier(mpProbeDistanceTex.get(), Falcor::Resource::State::PixelShader);


        // Update RTXGI data structure.
        {
            FALCOR_PROFILE("rtxgi::DDGIVolume::UpdateProbes");
            FALCOR_GET_COM_INTERFACE(pRenderContext->getLowLevelData()->getD3D12CommandList(), ID3D12GraphicsCommandList4, pList4);

            rtxgi::d3d12::DDGIVolume* pDDGIVolumes[1] = { mpDDGIVolume.get() };
#ifdef FALCOR_GFX
            pRenderContext->bindCustomGPUDescriptorPool();
#endif
            rtxgi::d3d12::UpdateDDGIVolumeProbes(pList4, 1, pDDGIVolumes);
            rtxgi::d3d12::RelocateDDGIVolumeProbes(pList4, 1, pDDGIVolumes);
            rtxgi::d3d12::ClassifyDDGIVolumeProbes(pList4, 1, pDDGIVolumes);
#ifdef FALCOR_GFX
            pRenderContext->unbindCustomGPUDescriptorPool();
#endif
        }

        // Flush command queue.
        pRenderContext->flush(false);

        mFrameCount++;
    }

    bool RTXGIVolume::renderUI(Gui::Widgets& widget)
    {
        // Debug button to force re-creation of probe volume.
        bool recomputeProbeGrid = false;
        if (widget.button("Re-create probe volume"))
        {
            mOptionsDirty = true;
            recomputeProbeGrid = true;
        }
        widget.tooltip("Forces re-creation of the RTXGI probe volume and all its resources. For debugging purposes.");

        // Probe grid settings
        if (widget.var("Grid size", mOptions.gridSize, 1, 1024))
        {
            mOptionsDirty = true;
            recomputeProbeGrid = true;
        }
        widget.tooltip("Number of probes on each axis.");

        // When the 'auto grid' box is checked, the grid spacing and origin is computed automatically
        // and the values shown in the UI. Changing any of the values manually disables 'auto grid'.
        if (widget.var("Grid origin", mOptions.gridOrigin))
        {
            mOptionsDirty = true;
            mOptions.useAutoGrid = false;
        }
        widget.tooltip("World-space origin of the grid, in the scene's coordinate system.");

        if (widget.var("Grid spacing", mOptions.gridSpacing))
        {
            mOptionsDirty = true;
            mOptions.useAutoGrid = false;
        }
        widget.tooltip("World-space distance between probes.");

        if (widget.checkbox("Compute probe placement automatically", mOptions.useAutoGrid)) recomputeProbeGrid = true;
        widget.tooltip("The probe grid origin and spacing are computed so that the probe volume exactly overlaps the scene bounding box.");

        if (mOptions.useAutoGrid && recomputeProbeGrid)
        {
            mOptionsDirty = true;
            computeProbeGrid();
        }

        mOptionsDirty |= widget.checkbox("Compute probe max ray distance", mOptions.useAutoMaxRayDistance);
        widget.tooltip("The probe ray distance is computed as 1.5 times the diagonal of a grid cell.");
        if (!mOptions.useAutoMaxRayDistance)
        {
            mOptionsDirty |= widget.var("Max ray distance", mOptions.probeMaxRayDistance, 0.f, std::numeric_limits<float>::max(), 1e-3f * mOptions.probeMaxRayDistance, false, "%.3e");
        }

        // Ray tracing budget
        mOptionsDirty |= widget.var("Num rays per probe", mOptions.numRaysPerProbe, 1u, kMaxNumRaysPerProbe);
        widget.tooltip("Number of rays cast per probe per frame. Independent of the number of probes or resolution of probe textures.");

        int64_t raysPerFrame = getProbeCount() * mOptions.numRaysPerProbe;
        FALCOR_ASSERT(raysPerFrame >= 0);
        widget.text("Rays per frame: " + std::to_string(raysPerFrame));

        // Probe texture resolutions
        mOptionsDirty |= widget.var("Irradiance texels", mOptions.numIrradianceTexels, kMinNumIrradianceOrDistanceTexels, 64u);
        widget.tooltip("Number of texels used in one dimension of the irradiance texture, not including the 1-pixel border on each side.");
        mOptionsDirty |= widget.var("Distance texels", mOptions.numDistanceTexels, kMinNumIrradianceOrDistanceTexels, 64u);
        widget.tooltip("Number of texels used in one dimension of the distance texture, not including the 1-pixel border on each side.");

        // RTXGI heuristics
        mOptionsDirty |= widget.var("Probe hysteresis", mOptions.probeHysteresis, 0.f, 1.f);
        widget.tooltip("Controls the influence of new rays when updating each probe. A value close to 1 will "
            "very slowly change the probe textures, improving stability but reducing accuracy when objects "
            "move in the scene. Values closer to 0.9 or lower will rapidly react to scene changes, "
            "but will exhibit flickering.");
        mOptionsDirty |= widget.var("Probe distance exponent", mOptions.probeDistanceExponent, 0.f);
        widget.tooltip("Exponent for depth testing. A high value will rapidly react to depth discontinuities, but risks causing banding.");
        mOptionsDirty |= widget.var("Probe irradiance encoding gamma", mOptions.probeIrradianceEncodingGamma, 0.f);
        widget.tooltip("Irradiance blending happens in post-tonemap space.");

        mOptionsDirty |= widget.var("Probe irradiance threshold", mOptions.probeIrradianceThreshold, 0.f);
        widget.tooltip("A threshold ratio used during probe radiance blending that determines if a large lighting change has happened. "
            "If the max color component difference is larger than this threshold, the hysteresis will be reduced.");

        mOptionsDirty |= widget.var("Probe brightness threshold", mOptions.probeBrightnessThreshold, 0.f);
        widget.tooltip("A threshold value used during probe radiance blending that determines the maximum allowed difference in brightness "
            "between the previous and current irradiance values. This prevents impulses from drastically changing a "
            "texel's irradiance in a single update cycle.");

        // Irradiance evaluation bias values.
        mOptionsDirty |= widget.var("View bias", mOptions.viewBias, 0.f);
        widget.tooltip("View direction bias used for computing the surface bias for irradiance evaluation.");
        mOptionsDirty |= widget.var("Normal bias", mOptions.normalBias, 0.f);
        widget.tooltip("Normal direction bias used for computing the surface bias for irradiance evaluation.");

        mOptionsDirty |= widget.var("Probe minimum frontface distance", mOptions.probeMinFrontfaceDistance, 0.f);
        widget.tooltip("Probe relocation moves probes that see front facing triangles closer than this value.");

        mOptionsDirty |= widget.var("Probe backface threshold", mOptions.probeBackfaceThreshold, 0.f);
        widget.tooltip("Probe relocation assumes probes with more than this ratio "
            "of backface hits are in walls, and will attempt to move them.");

        widget.checkbox("Enable probe relocation", mOptions.enableProbeRelocation);
        widget.tooltip("When enabled, probes are automatically relocated.");

        widget.checkbox("Enable probe classifcation", mOptions.enableProbeClassification);
        widget.tooltip("When enabled, probes are automatically enabled/disabled. Otherwise, probes are always active.");

        // Probe update settings.
        widget.checkbox("Recursive irradiance", mOptions.enableRecursiveIrradiance);
        widget.tooltip("When enabled, irradiance is computed recursively using data from the previous frame.");

        // Show debug data from probe update pass.
        bool debug = mDebug.enable;
        widget.checkbox("Debug probe update", mDebug.enable);
        if (debug)
        {
            uint2 dims = mProbeUpdateDispatchDims;
            widget.var("Probe update dims", dims);
            widget.var("Debug thread ID", mDebug.threadID);
            widget.text("Probe ray data:");

            UpdateProbesDebugData debugData = *static_cast<const UpdateProbesDebugData*>(mDebug.pData->map(Buffer::MapType::Read));
            widget.var("rayOrigin", debugData.rayOrigin);
            widget.var("rayDir", debugData.rayDir);
            widget.var("hitT", debugData.hitT);
            widget.var("hitKind", debugData.hitKind);
            widget.var("instanceID", debugData.instanceID);
            widget.var("primitiveIndex", debugData.primitiveIndex);
            widget.var("barycentrics", debugData.barycentrics);
            mDebug.pData->unmap();

            widget.dummy("#spacing0", { 1, 8 });
        }

        // Direct lighting options.
        widget.checkbox("Enable env map", mOptions.enableEnvMap);
        widget.tooltip("When enabled, direct lighting from the environment map is evaluated using 1 sample.");

        widget.checkbox("Enable analytic lights", mOptions.enableAnalyticLights);
        widget.tooltip("When enabled, direct lighting from analytic lights is evaluated using 1 sample per light.");

        widget.checkbox("Enable emissive lights", mOptions.enableEmissiveLights);
        widget.tooltip("When enabled, direct lighting from emissive geometry is evaluated using the specified number of samples.");

        if (mOptions.enableEmissiveLights)
        {
            widget.var("Emissive samples", mOptions.emissiveSampleCount, 0u, 256u);
            widget.tooltip("Emissive geometry is sampled using this many stochastic samples.");
        }

        return mOptionsDirty;
    }

    RTXGIVolume::RTXGIVolume(RenderContext* pRenderContext, Scene::SharedPtr pScene, EnvMapSampler::SharedPtr pEnvMapSampler, EmissiveLightSampler::SharedPtr pEmissiveSampler, const Options& options)
    {
        FALCOR_ASSERT(pRenderContext);
        FALCOR_ASSERT(pScene);

        mpScene = pScene;
        mOptions = options;

        mSceneBounds = mpScene->getSceneBounds();
        if (mOptions.useAutoGrid) computeProbeGrid();

        if (mpScene->hasProceduralGeometry()) logWarning("RTXGIVolume only supports triangles. Other types of geometry will be ignored.");

        initRTXGIShaders();

        // Create sample generator.
        mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_DEFAULT);
        FALCOR_ASSERT(mpSampleGenerator);

        // Create environment map sampler if required by the scene.
        mpEnvMapSampler = pEnvMapSampler;
        if (!mpEnvMapSampler && mpScene->getEnvMap()) mpEnvMapSampler = EnvMapSampler::create(pRenderContext, mpScene->getEnvMap());

        // Create emissive light sampler.
        mpEmissiveSampler = pEmissiveSampler;
        if (!mpEmissiveSampler) mpEmissiveSampler = LightBVHSampler::create(pRenderContext, mpScene);
        FALCOR_ASSERT(mpEmissiveSampler);

        // Prepare our programs for the scene.
        Shader::DefineList defines = mpScene->getSceneDefines();
        defines.add(mpSampleGenerator->getDefines());

        // Create program for probe radiance update.
        RtProgram::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProbeRadianceUpdateFilename);
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(kMaxAttributeSizeBytes);
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mpRtBindingTable = RtBindingTable::create(2, 2, pScene->getGeometryCount());
        auto& sbt = mpRtBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("shadowMiss"));
        sbt->setMiss(1, desc.addMiss("probeMiss"));
        sbt->setHitGroup(0, pScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowAnyHit"));
        sbt->setHitGroup(1, pScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("probeClosestHit", "probeAnyHit"));

        mpProbeUpdateProgram = RtProgram::create(desc, defines);

        // Prepare for light sampling.
        mpProbeUpdateProgram->addDefines(getDirectLightingDefines());
    }

    void RTXGIVolume::initRTXGIShaders()
    {
        // TODO: Remove after we have a wrapper for the SDK that can enfore these defines
#if !defined(RTXGI_DDGI_RESOURCE_MANAGEMENT) || (RTXGI_DDGI_RESOURCE_MANAGEMENT != 0)
#error "RTXGI_DDGI_RESOURCE_MANAGEMENT must be defined and set to 0, Falcor needs to manage resources on its own"
#endif
#if !defined(RTXGI_COORDINATE_SYSTEM) || (RTXGI_COORDINATE_SYSTEM != RTXGI_COORDINATE_SYSTEM_RIGHT)
#error "RTXGI_COORDINATE_SYSTEM must be defined"
#endif

        // The two blending shaders only differ in defines, but they are loaded as separate programs instead of program versions for consistency.
        Program::DefineList commonDefines;
        // common defines
        commonDefines.add("RTXGI_DDGI_RESOURCE_MANAGEMENT", std::to_string(RTXGI_DDGI_RESOURCE_MANAGEMENT));
        commonDefines.add("RTXGI_COORDINATE_SYSTEM", std::to_string(RTXGI_COORDINATE_SYSTEM));
        commonDefines.add("RTXGI_DDGI_SHADER_REFLECTION", "0");
        commonDefines.add("RTXGI_DDGI_BINDLESS_RESOURCES", "0");
        commonDefines.add("RTXGI_DDGI_USE_SHADER_CONFIG_FILE", "0");
        commonDefines.add("RTXGI_DDGI_DEBUG_BORDER_COPY_INDEXING", "0");
        commonDefines.add("RTXGI_DDGI_DEBUG_OCTAHEDRAL_INDEXING", "0");
        commonDefines.add("RTXGI_DDGI_DEBUG_PROBE_INDEXING", "0");
        commonDefines.add("SPIRV", "0");
        commonDefines.add("HLSL", "1");

        commonDefines.add("CONSTS_REGISTER", "b0");
        commonDefines.add("CONSTS_SPACE", "space1");
        commonDefines.add("VOLUME_CONSTS_REGISTER", "t0");
        commonDefines.add("VOLUME_CONSTS_SPACE", "space1");
        commonDefines.add("RAY_DATA_REGISTER", "u0");
        commonDefines.add("RAY_DATA_SPACE", "space1");
        //                 OUTPUT_REGISTER             Note: this register differs for irradiance vs. distance and should be defined per-shader
        commonDefines.add("OUTPUT_SPACE", "space1");
        commonDefines.add("PROBE_DATA_REGISTER", "u3");
        commonDefines.add("PROBE_DATA_SPACE", "space1");

        // Probe blending (irradiance)
        {
            Program::DefineList defines = commonDefines; // to avoid surprises of a left-over define
            defines.add("RTXGI_DDGI_BLEND_RADIANCE", "1");
            defines.add("RTXGI_DDGI_PROBE_NUM_TEXELS", std::to_string(mOptions.numIrradianceTexels));
            defines.add("RTXGI_DDGI_BLEND_SHARED_MEMORY", "1"); // without this the rays-per-probe would have no effect
            defines.add("RTXGI_DDGI_BLEND_RAYS_PER_PROBE", std::to_string(mOptions.numRaysPerProbe));

            defines.add("OUTPUT_REGISTER", "u1"); /// irradiance output
            mRadianceBlendingCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeBlendingCS.hlsl", "DDGIProbeBlendingCS", defines);
        }

        // Probe blending (distance)
        {
            Program::DefineList defines = commonDefines; // to avoid surprises of a left-over define
            defines.add("RTXGI_DDGI_BLEND_RADIANCE", "0");
            defines.add("RTXGI_DDGI_PROBE_NUM_TEXELS", std::to_string(mOptions.numDistanceTexels));
            defines.add("RTXGI_DDGI_BLEND_SHARED_MEMORY", "1"); // without this the rays-per-probe would have no effect
            defines.add("RTXGI_DDGI_BLEND_RAYS_PER_PROBE", std::to_string(mOptions.numRaysPerProbe));

            defines.add("OUTPUT_REGISTER", "u2"); /// distance output
            mDistanceBlendingCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeBlendingCS.hlsl", "DDGIProbeBlendingCS", defines);
        }

        // Border row update (irradiance)
        {
            Program::DefineList defines = commonDefines; // to avoid surprises of a left-over define
            defines.add("RTXGI_DDGI_BLEND_RADIANCE", "1");
            defines.add("RTXGI_DDGI_PROBE_NUM_TEXELS", std::to_string(mOptions.numIrradianceTexels));

            defines.add("OUTPUT_REGISTER", "u1"); /// distance output
            mRadianceBorderRowUpdateCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeBorderUpdateCS.hlsl", "DDGIProbeBorderRowUpdateCS", defines);
        }

        // Border column update (irradiance)
        {
            Program::DefineList defines = commonDefines; // to avoid surprises of a left-over define
            defines.add("RTXGI_DDGI_BLEND_RADIANCE", "1");
            defines.add("RTXGI_DDGI_PROBE_NUM_TEXELS", std::to_string(mOptions.numIrradianceTexels));

            defines.add("OUTPUT_REGISTER", "u1"); /// distance output
            mRadianceBorderColUpdateCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeBorderUpdateCS.hlsl", "DDGIProbeBorderColumnUpdateCS", defines);
        }

        // Border row update (distance)
        {
            Program::DefineList defines = commonDefines; // to avoid surprises of a left-over define
            defines.add("RTXGI_DDGI_BLEND_RADIANCE", "0");
            defines.add("RTXGI_DDGI_PROBE_NUM_TEXELS", std::to_string(mOptions.numDistanceTexels));

            defines.add("OUTPUT_REGISTER", "u2"); /// distance output
            mDistanceBorderRowUpdateCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeBorderUpdateCS.hlsl", "DDGIProbeBorderRowUpdateCS", defines);
        }

        // Border column update (distance)
        {
            Program::DefineList defines = commonDefines; // to avoid surprises of a left-over define
            defines.add("RTXGI_DDGI_BLEND_RADIANCE", "0");
            defines.add("RTXGI_DDGI_PROBE_NUM_TEXELS", std::to_string(mOptions.numDistanceTexels));

            defines.add("OUTPUT_REGISTER", "u2"); /// distance output
            mDistanceBorderColUpdateCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeBorderUpdateCS.hlsl", "DDGIProbeBorderColumnUpdateCS", defines);
        }

        // Probe relocation
        {
            Program::DefineList defines = commonDefines; // to avoid surprises of a left-over define
            mProbeRelocationUpdateCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeRelocationCS.hlsl", "DDGIProbeRelocationCS", defines);
            mProbeRelocationResetCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeRelocationCS.hlsl", "DDGIProbeRelocationResetCS", defines);
        }

        // Probe classification
        {
            Program::DefineList defines = commonDefines; // to avoid surprises of a left-over define
            mProbeClassificationUpdateCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeClassificationCS.hlsl", "DDGIProbeClassificationCS", defines);
            mProbeClassificationResetCS.pCS = ComputeProgram::createFromFile("rtxgi/shaders/ddgi/ProbeClassificationCS.hlsl", "DDGIProbeClassificationResetCS", defines);
        }
    }

    void RTXGIVolume::initRTXGI()
    {
        FALCOR_ASSERT(!mIsDDGIVolumeValid);

        // Create the probe volume object.
        mpDDGIVolume = std::make_unique<rtxgi::d3d12::DDGIVolume>();

        // Copy options to DDGI struct.
        mDDGIDesc.name = "DDGIVolume";
        mDDGIDesc.probeCounts = { mOptions.gridSize.x, mOptions.gridSize.y, mOptions.gridSize.z };
        mDDGIDesc.origin = { mOptions.gridOrigin.x, mOptions.gridOrigin.y, mOptions.gridOrigin.z };
        mDDGIDesc.probeSpacing = { mOptions.gridSpacing.x, mOptions.gridSpacing.y, mOptions.gridSpacing.z };
        mDDGIDesc.probeNumIrradianceTexels = mOptions.numIrradianceTexels;
        mDDGIDesc.probeNumDistanceTexels = mOptions.numDistanceTexels;
        mDDGIDesc.probeNumRays = mOptions.numRaysPerProbe;
        mDDGIDesc.probeHysteresis = mOptions.probeHysteresis;
        mDDGIDesc.probeDistanceExponent = mOptions.probeDistanceExponent;
        mDDGIDesc.probeIrradianceEncodingGamma = mOptions.probeIrradianceEncodingGamma;
        mDDGIDesc.probeIrradianceThreshold = mOptions.probeIrradianceThreshold;
        mDDGIDesc.probeBrightnessThreshold = mOptions.probeBrightnessThreshold;
        mDDGIDesc.probeViewBias = mOptions.viewBias;
        mDDGIDesc.probeNormalBias = mOptions.normalBias;
        mDDGIDesc.probeDistanceFormat = 1;
        mDDGIDesc.probeRayDataFormat = 1;
        mDDGIDesc.probeClassificationEnabled = mOptions.enableProbeClassification;
        mDDGIDesc.probeRelocationEnabled = mOptions.enableProbeRelocation;

        mDDGIDesc.probeMinFrontfaceDistance = mOptions.probeMinFrontfaceDistance;
        mDDGIDesc.probeBackfaceThreshold = mOptions.probeBackfaceThreshold;

        // We are manually managing the resources, so initialize them first.
        initRTXGIResources();
        initRTXGIPipelineStates();

        // Create the RTXGI volume using the resources we supply.
        rtxgi::ERTXGIStatus result = mpDDGIVolume->Create(mDDGIDesc, mDDGIResources);
        if (result != rtxgi::ERTXGIStatus::OK) throw RuntimeError("Failed to create RTXGI probe volume");

        // Set probe ray max distance if not computed automatically.
        if (!mOptions.useAutoMaxRayDistance) mpDDGIVolume->SetProbeMaxRayDistance(mOptions.probeMaxRayDistance);

        mIsDDGIVolumeValid = true;
    }

    void RTXGIVolume::initRTXGIResources()
    {
        // Manually create descriptor set to aid resource binding.
        // All RTXGI CS shaders use the same global set of resource bindings, but each shader only declares a subset of what's bound,
        // so we cannot rely on reflection to help us. They also use root constants, which Falcor doesn't properly support at this time.

        uint32_t numUAVDescriptors = rtxgi::GetDDGIVolumeNumUAVDescriptors();
        uint32_t numSRVDescriptors = rtxgi::GetDDGIVolumeNumSRVDescriptors();

        // Root index 0: Root Constants. Not required to be set by us.
        // Root index 1: Volume Constant Buffer.
        // Root index 2: Descriptor Set. Must be set by us.
        D3D12DescriptorSet::Layout layout;
        layout.addRange(ShaderResourceType::StructuredBufferSrv,  0,                     1,                 1); // StructuredBuffer : register(t0, space1);
        layout.addRange(ShaderResourceType::TextureUav,           1,                     numUAVDescriptors, 1); // RWTexture2D      : register(u0 ... uN, space1);
        layout.addRange(ShaderResourceType::TextureSrv,           1 + numUAVDescriptors, numSRVDescriptors, 1); // Texture2D        : register(t1 ... tN+1, space1);
        mpSet = D3D12DescriptorSet::create(layout, D3D12DescriptorSetBindingUsage::RootSignatureOffset);

        // Use the unmanaged resources (managed by Falcor rather than RTXGI)
        mDDGIResources.unmanaged.enabled = true;

        // Pass the GPU descriptor heap where the Set is allocated from to RTXGI
        // RTXGI expects the entire descriptor heap passed in. Actual descriptors are located through offsets baked into the root sig when we request the Root Sig desc
        mDDGIResources.descriptorHeapDesc.heap = gpDevice->getD3D12GpuDescriptorPool()->getApiData()->pHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV]->getApiHandle();
        mDDGIResources.descriptorHeapDesc.constsOffset = mpSet->getApiData()->pAllocation->getHeapEntryIndex(mpSet->getRange(0).baseRegIndex); // the Volume Constant Buffer
        mDDGIResources.descriptorHeapDesc.uavOffset    = mpSet->getApiData()->pAllocation->getHeapEntryIndex(mpSet->getRange(1).baseRegIndex); // the UAV descriptor set
        mDDGIResources.descriptorHeapDesc.srvOffset    = mpSet->getApiData()->pAllocation->getHeapEntryIndex(mpSet->getRange(2).baseRegIndex); // the SRV descriptor set

        // Create root signature using layout provided by RTXGI
        ID3DBlob* pSigBlob = nullptr;
        bool success = rtxgi::d3d12::GetDDGIVolumeRootSignatureDesc(mDDGIResources.descriptorHeapDesc.constsOffset, mDDGIResources.descriptorHeapDesc.uavOffset, pSigBlob);
        FALCOR_ASSERT(success);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateRootSignature(0, pSigBlob->GetBufferPointer(), pSigBlob->GetBufferSize(), IID_PPV_ARGS(&mpRootSig)));
        RTXGI_SAFE_RELEASE(pSigBlob);
        mDDGIResources.unmanaged.rootSignature = mpRootSig;
        mDDGIResources.unmanaged.rootParamSlotRootConstants = 0;
        mDDGIResources.unmanaged.rootParamSlotDescriptorTable = 1;

        // Create resources using RTXGI-specified formats and sizes, then set them into the appropriate location in the descriptor set

        uint32_t width = 0;
        uint32_t height = 0;
        ResourceFormat format = ResourceFormat::Unknown;

        // The constant buffer
        mpDDGIVolumeBlockSDK = Buffer::createStructured(sizeof(rtxgi::DDGIVolumeDescGPUPacked), 1, Resource::BindFlags::ShaderResource);
        mpSet->setSrv(0, 0, mpDDGIVolumeBlockSDK->getSRV().get());
        mpDDGIVolumeBlockSDK->setName("mpDDGIVolumeBlockSDK");

        // Probe Ray Data
        rtxgi::GetDDGIVolumeTextureDimensions(mDDGIDesc, rtxgi::EDDGIVolumeTextureType::RayData, width, height);
        format = getResourceFormat(rtxgi::d3d12::GetDDGIVolumeTextureFormat(rtxgi::EDDGIVolumeTextureType::RayData, mDDGIDesc.probeRayDataFormat));
        mpRayDataTex = Texture::create2D(width, height, format, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpRayDataTex->setName("mpRayDataTex");
        mpSet->setUav(1, 0, mpRayDataTex->getUAV().get()); // (u0, space1)
        mpSet->setSrv(2, 0, mpRayDataTex->getSRV().get()); // (t1, space1)
        mDDGIResources.unmanaged.probeRayData = mpRayDataTex->getD3D12Handle();

        // Probe Irradiance
        rtxgi::GetDDGIVolumeTextureDimensions(mDDGIDesc, rtxgi::EDDGIVolumeTextureType::Irradiance, width, height);
        format = getResourceFormat(rtxgi::d3d12::GetDDGIVolumeTextureFormat(rtxgi::EDDGIVolumeTextureType::Irradiance, mDDGIDesc.probeIrradianceFormat));
        mpIrradianceTex = Texture::create2D(width, height, format, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpIrradianceTex->setName("mpIrradianceTex");
        mpSet->setUav(1, 1, mpIrradianceTex->getUAV().get()); // (u1, space1)
        mpSet->setSrv(2, 1, mpIrradianceTex->getSRV().get()); // (t2, space1)
        mDDGIResources.unmanaged.probeIrradiance = mpIrradianceTex->getD3D12Handle();

        // Probe Distance
        rtxgi::GetDDGIVolumeTextureDimensions(mDDGIDesc, rtxgi::EDDGIVolumeTextureType::Distance, width, height);
        format = getResourceFormat(rtxgi::d3d12::GetDDGIVolumeTextureFormat(rtxgi::EDDGIVolumeTextureType::Distance, mDDGIDesc.probeDistanceFormat));
        mpProbeDistanceTex = Texture::create2D(width, height, format, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpProbeDistanceTex->setName("mpProbeDistanceTex");
        mpSet->setUav(1, 2, mpProbeDistanceTex->getUAV().get()); // (u2, space1)
        mpSet->setSrv(2, 2, mpProbeDistanceTex->getSRV().get()); // (t3, space1)
        mDDGIResources.unmanaged.probeDistance = mpProbeDistanceTex->getD3D12Handle();

        // Probe Data
        rtxgi::GetDDGIVolumeTextureDimensions(mDDGIDesc, rtxgi::EDDGIVolumeTextureType::Data, width, height);
        format = getResourceFormat(rtxgi::d3d12::GetDDGIVolumeTextureFormat(rtxgi::EDDGIVolumeTextureType::Data, mDDGIDesc.probeDataFormat));
        mpProbeDataTex = Texture::create2D(width, height, format, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpProbeDataTex->setName("mpProbeDataTex");
        mpSet->setUav(1, 3, mpProbeDataTex->getUAV().get()); // (u3, space1)
        mpSet->setSrv(2, 3, mpProbeDataTex->getSRV().get()); // (t4, space1)
        mDDGIResources.unmanaged.probeData = mpProbeDataTex->getD3D12Handle();

        // Now we create the RTV resources
        mpProbeIrradianceRTV = RenderTargetView::create(mpIrradianceTex, 0, 0, 1);
        mDDGIResources.unmanaged.probeIrradianceRTV = mpProbeIrradianceRTV->getD3D12CpuHeapHandle();

        mpProbeDistanceRTV   = RenderTargetView::create(mpProbeDistanceTex, 0, 0, 1);
        mDDGIResources.unmanaged.probeDistanceRTV = mpProbeDistanceRTV->getD3D12CpuHeapHandle();

    }

    void RTXGIVolume::initRTXGIPipelineStates()
    {
        // HACK: Dummy ProgramVars to allow retrieving ProgramKernels. These are raw HLSL so there's no specialization anyway.
        // TODO: Allow shader compilation without ProgramVars and emit error when specializations aren't satisfied. May need Slang update. (currently 0.12.25 at time of writing)
        if (mpDummyVars == nullptr) mpDummyVars = ComputeVars::create(mRadianceBlendingCS.pCS->getReflector());

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = mpRootSig;

        auto getCSBlob = [&](const RTXGIShader& shader) {
            return shader.pCS->getActiveVersion()->getKernels(mpDummyVars.get())->getShader(ShaderType::Compute)->getD3D12ShaderByteCode();
        };

        psoDesc.CS = getCSBlob(mRadianceBlendingCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mRadianceBlendingCS.pPSO)));
        mDDGIResources.unmanaged.probeBlendingIrradiancePSO = mRadianceBlendingCS.pPSO;

        psoDesc.CS = getCSBlob(mDistanceBlendingCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mDistanceBlendingCS.pPSO)));
        mDDGIResources.unmanaged.probeBlendingDistancePSO = mDistanceBlendingCS.pPSO;

        psoDesc.CS = getCSBlob(mDistanceBorderRowUpdateCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mDistanceBorderRowUpdateCS.pPSO)));
        mDDGIResources.unmanaged.probeBorderRowUpdateDistancePSO = mDistanceBorderRowUpdateCS.pPSO;

        psoDesc.CS = getCSBlob(mDistanceBorderColUpdateCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mDistanceBorderColUpdateCS.pPSO)));
        mDDGIResources.unmanaged.probeBorderColumnUpdateDistancePSO = mDistanceBorderColUpdateCS.pPSO;

        psoDesc.CS = getCSBlob(mRadianceBorderRowUpdateCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mRadianceBorderRowUpdateCS.pPSO)));
        mDDGIResources.unmanaged.probeBorderRowUpdateIrradiancePSO = mRadianceBorderRowUpdateCS.pPSO;

        psoDesc.CS = getCSBlob(mRadianceBorderColUpdateCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mRadianceBorderColUpdateCS.pPSO)));
        mDDGIResources.unmanaged.probeBorderColumnUpdateIrradiancePSO = mRadianceBorderColUpdateCS.pPSO;

        psoDesc.CS = getCSBlob(mProbeRelocationUpdateCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mProbeRelocationUpdateCS.pPSO)));
        mDDGIResources.unmanaged.probeRelocation.updatePSO = mProbeRelocationUpdateCS.pPSO;

        psoDesc.CS = getCSBlob(mProbeRelocationResetCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mProbeRelocationResetCS.pPSO)));
        mDDGIResources.unmanaged.probeRelocation.resetPSO = mProbeRelocationResetCS.pPSO;

        psoDesc.CS = getCSBlob(mProbeClassificationUpdateCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mProbeClassificationUpdateCS.pPSO)));
        mDDGIResources.unmanaged.probeClassification.updatePSO = mProbeClassificationUpdateCS.pPSO;

        psoDesc.CS = getCSBlob(mProbeClassificationResetCS);
        FALCOR_D3D_CALL(gpDevice->getD3D12Handle()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&mProbeClassificationResetCS.pPSO)));
        mDDGIResources.unmanaged.probeClassification.resetPSO = mProbeClassificationResetCS.pPSO;
    }

    void RTXGIVolume::destroyRTXGI()
    {
        // Graceful destruction of DDGIVolume.
        if (mpDDGIVolume && mIsDDGIVolumeValid)
        {
            mpDDGIVolume->Destroy();
        }
        mpDDGIVolume = nullptr;
        mIsDDGIVolumeValid = false;
    }

    void RTXGIVolume::updateParameterBlock()
    {
        // Create parameter block on the first call. The reflection info shouldn't change.
        if (mpParameterBlock == nullptr)
        {
            auto blockReflection = mpProbeUpdateProgram->getReflector()->getParameterBlock(kParameterBlockName);
            mpParameterBlock = ParameterBlock::create(blockReflection);
            if (!mpParameterBlock) throw RuntimeError("Failed to create parameter block");

            Sampler::Desc bilinearSamplerDesc;
            bilinearSamplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear).setMaxAnisotropy(1);
            mpParameterBlock["resources"]["bilinearSampler"] = Sampler::create(bilinearSamplerDesc);
        }

        // Bind the irradiance volume variables.
        auto var = mpParameterBlock->getRootVar();
        auto resources = var["resources"];
        resources["probeIrradiance"] = mpIrradianceTex;
        resources["probeDistance"] = mpProbeDistanceTex;
        resources["probeData"] = mpProbeDataTex;
    }

    void RTXGIVolume::validateOptions()
    {
        FALCOR_ASSERT(getProbeCount() > 0);
        mOptions.numIrradianceTexels = std::max(mOptions.numIrradianceTexels, kMinNumIrradianceOrDistanceTexels);
        mOptions.numDistanceTexels = std::max(mOptions.numDistanceTexels, kMinNumIrradianceOrDistanceTexels);
        mOptions.numRaysPerProbe = std::clamp(mOptions.numRaysPerProbe, 1u, kMaxNumRaysPerProbe);

        // Validate that our parameters are supported by the current SDK version.
        // TODO: Remove these check when the SDK has been fixed.
        if (mOptions.numIrradianceTexels != 8 || mOptions.numDistanceTexels != 16)
        {
            logWarning("Current RTXGI SDK requires numIrradianceTexels=8 and numDistanceTexels=16. Forcing these values.");
            mOptions.numIrradianceTexels = 8;
            mOptions.numDistanceTexels = 16;
            mOptionsDirty = true;
        }
        if (mOptions.numRaysPerProbe < 1 || mOptions.numRaysPerProbe > 256)
        {
            logWarning("Current RTXGI SDK requires numRaysPerProbe=1..256. Forcing it to 256.");
            mOptions.numRaysPerProbe = 256;
            mOptionsDirty = true;
        }
    }

    void RTXGIVolume::probeUpdatePass(RenderContext* pRenderContext)
    {
        FALCOR_PROFILE("probeUpdatePass");

        // Specialize the program for the emissive light sampler.
        // This function sets compile-time constants that may change from frame to frame.
        if (mpProbeUpdateProgram->addDefines(getDirectLightingDefines())) mpProbeUpdateVars = nullptr;

        // Re-create the program vars if needed.
        if (mpProbeUpdateVars == nullptr)
        {
            mpProbeUpdateVars = RtProgramVars::create(mpProbeUpdateProgram, mpRtBindingTable);
            auto var = mpProbeUpdateVars->getRootVar();

            var["DDGIProbeRayData"] = mpRayDataTex;
            var["DDGIProbeData"] = mpProbeDataTex;
            var[kParameterBlockName] = mpParameterBlock;

            mDebug.pData = Buffer::createStructured(var["gDebugData"], 1);
            var["gDebugData"] = mDebug.pData;

            mpSampleGenerator->setShaderData(var);
        }

        // Bind the resources.
        auto var = mpProbeUpdateVars->getRootVar();
        setDirectLightingShaderData(var["PerFrameCB"]["gDirectLighting"]);

        auto constant = var["PerFrameCB"];
        constant["gEnableRecursiveIrradiance"] = mOptions.enableRecursiveIrradiance;
        constant["gFrameCount"] = mFrameCount;
        constant["gDebugThreadID"] = mDebug.threadID;

        // Trace the rays
        mpScene->raytrace(pRenderContext, mpProbeUpdateProgram.get(), mpProbeUpdateVars, mProbeUpdateDispatchDims);
    }

    Program::DefineList RTXGIVolume::getDirectLightingDefines() const
    {
        Program::DefineList defines;
        if (mpEmissiveSampler) defines.add(mpEmissiveSampler->getDefines());
        return defines;
    }

    void RTXGIVolume::setDirectLightingShaderData(ShaderVar var) const
    {
        var["enableEnvMap"] = mOptions.enableEnvMap && mpEnvMapSampler != nullptr;
        var["enableAnalyticLights"] = mOptions.enableAnalyticLights;
        var["enableEmissiveLights"] = mOptions.enableEmissiveLights;
        var["emissiveSampleCount"] = mUseEmissiveSampler ? mOptions.emissiveSampleCount : 0;

        if (mpEnvMapSampler)
        {
            mpEnvMapSampler->setShaderData(var["envMapSampler"]);
        }

        if (mUseEmissiveSampler && mOptions.emissiveSampleCount > 0)
        {
            // TODO: Do we have to bind this every frame?
            FALCOR_ASSERT(mpEmissiveSampler);
            mpEmissiveSampler->setShaderData(var["emissiveSampler"]);
        }
    }

    void RTXGIVolume::computeProbeGrid()
    {
        // Set grid origin and spacing so that the probe volume exactly overlaps the scene bounding box.
        float3 spacing = { 0, 0, 0 };
        int3 size = mOptions.gridSize;
        float3 extent = mSceneBounds.extent();
        for (int i = 0; i < 3; i++)
        {
            if (size[i] > 1) spacing[i] = extent[i] / (size[i] - 1);
        }
        mOptions.gridOrigin = mSceneBounds.center();
        mOptions.gridSpacing = spacing;
    }

    FALCOR_SCRIPT_BINDING(RTXGIVolume)
    {
        ScriptBindings::SerializableStruct<RTXGIVolume::Options> options(m, "RTXGIVolumeOptions");
#define field(f_) field(#f_, &RTXGIVolume::Options::f_)
        options.field(useAutoGrid);
        options.field(gridSize);
        options.field(gridOrigin);
        options.field(gridSpacing);
        options.field(numIrradianceTexels);
        options.field(numDistanceTexels);
        options.field(numRaysPerProbe);
        options.field(useAutoMaxRayDistance);
        options.field(probeMaxRayDistance);
        options.field(probeHysteresis);
        options.field(probeDistanceExponent);
        options.field(probeIrradianceEncodingGamma);
        options.field(probeIrradianceThreshold);
        options.field(probeBrightnessThreshold);
        options.field(viewBias);
        options.field(normalBias);
        options.field(probeMinFrontfaceDistance);
        options.field(probeBackfaceThreshold);
        options.field(enableProbeRelocation);
        options.field(enableProbeClassification);
        options.field(enableRecursiveIrradiance);
        options.field(enableAnalyticLights);
        options.field(enableEmissiveLights);
        options.field(emissiveSampleCount);
#undef field
    }
}

#endif // FALCOR_HAS_D3D12
