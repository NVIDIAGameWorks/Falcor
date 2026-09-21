/***************************************************************************
 # Copyright (c) 2015-26, NVIDIA CORPORATION. All rights reserved.
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
#include "ReSTIRPathTracing.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Color/ColorHelpers.slang"
#include <array>
namespace Falcor
{
    namespace
    {
        const char kReflectTypesFile[] = "ReSTIRPathTracing/ReflectTypes.cs.slang";
        const char kTemporalResamplingFile[] = "ReSTIRPathTracing/TemporalResampling.cs.slang";
        const char kSpatialPrecomputeShiftBufferFile[] = "ReSTIRPathTracing/SpatialPrecomputeShiftBuffer.cs.slang";
        const char kSpatialResamplingFile[] = "ReSTIRPathTracing/SpatialResampling.cs.slang";
        const char kTemporalRetraceFile[] = "ReSTIRPathTracing/TemporalPathRetrace.cs.slang";
        const char kSpatialRetraceFile[] = "ReSTIRPathTracing/SpatialPathRetrace.cs.slang";
        const char kSpatialGenerateRetraceWorkload[] = "ReSTIRPathTracing/GenerateRetraceWorkload.cs.slang";
        const char kFetchDenoiserData[] = "ReSTIRPathTracing/FetchDenoiserData.cs.slang";
        const char kFillSampleID[] = "ReSTIRPathTracing/FillSampleID.cs.slang";
        const char kComputeDuplicationMap[] = "ReSTIRPathTracing/ComputeDuplicationMap.cs.slang";
        const char kTemporalTraceShiftedPrimaryHits[] = "ReSTIRPathTracing/TemporalTraceShiftedPrimaryHits.cs.slang";
        const char kSpatialTraceShiftedPrimaryHits[] = "ReSTIRPathTracing/SpatialTraceShiftedPrimaryHits.cs.slang";
        const char kTemporalPopulateFloatingReservoir[] = "ReSTIRPathTracing/TemporalPopulateFloatingReservoir.cs.slang";
        const char kTemporalRobustResampling[] = "ReSTIRPathTracing/TemporalRobustResampling.cs.slang";
        const char kTemporalRobustPathRetrace[] = "ReSTIRPathTracing/TemporalRobustPathRetrace.cs.slang";
        const char kTemporalRobustCreateShiftVBuffer[] = "ReSTIRPathTracing/TemporalRobustCreateShiftVBuffer.cs.slang";
        const char kTemporalRobustProduceWorkload[] = "ReSTIRPathTracing/TemporalRobustProduceWorkload.cs.slang";
        const char kComputeReuseMotionVector[] = "ReSTIRPathTracing/ComputeReuseMotionVector.cs.slang";
        const char kTemporalReservoirSplatting[] = "ReSTIRPathTracing/TemporalReservoirSplatting.cs.slang"; // 3 in 1

        const ShaderModel kShaderModel = ShaderModel::SM6_5;

        const Gui::DropdownList kShiftMappingList =
        {
            { (uint32_t)ReSTIRPT::ShiftMapping::Reconnection, "Reconnection" },
            { (uint32_t)ReSTIRPT::ShiftMapping::Hybrid, "Hybrid" },
        };

        const Gui::DropdownList kRetraceScheduleType =
        {
            { (uint32_t)ReSTIRPT::RetraceScheduleType::Naive, "Naive" },
            { (uint32_t)ReSTIRPT::RetraceScheduleType::Compact, "Compact" },
        };

        const Gui::DropdownList kTemporalAreaReuseType =
        {
            {(uint32_t)ReSTIRPT::TemporalAreaReuseType::Point, "Point"},
            {(uint32_t)ReSTIRPT::TemporalAreaReuseType::Fast, "Fast"},
            {(uint32_t)ReSTIRPT::TemporalAreaReuseType::Robust, "Robust"},
            {(uint32_t)ReSTIRPT::TemporalAreaReuseType::Splat, "Splat"},
        };

        const Gui::DropdownList kPointReprojectionMode =
        {
            {(uint32_t)ReSTIRPT::PointReprojectionMode::Nearest, "Nearest"},
            {(uint32_t)ReSTIRPT::PointReprojectionMode::Stochastic, "Stochastic"},
        };

        const uint32_t kNeighborOffsetCount = 8192;

        constexpr uint32_t kTemporalNeighborMultiplier  = 2;
        constexpr uint32_t kTemporalRobustNeighborMultiplier = 8;
        constexpr uint32_t kSpatialNeighborMultiplier   = 2;

        constexpr uint32_t kPairedNeighborTextureExtent = 256;
        constexpr uint32_t kPairedNeighborStdevLevels   = 10;
        constexpr int      kPairedNeighborMinRadius     = 5;
        constexpr int      kPairedNeighborMaxRadius     = 50;
        constexpr int      kPairedNeighborRadiusStep    = 5;
        constexpr uint32_t kPairedNeighborMaxCount      = 5;

        constexpr uint32_t kDefaultMaxSamplesPerPixel = 16;
    }

    ReSTIRPathTracing::ReSTIRPathTracing(const ref<IScene>& pScene, const DefineList& ownerDefines, const Options& options)
        : mpScene(pScene),
        mpDevice(pScene->getDevice()),
        mOptions(options)
    {
        FALCOR_ASSERT(mpScene);

        mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);

        // Create compute pass for reflecting data types.
        ProgramDesc desc;
        DefineList defines;
        defines.add(ownerDefines);
        defines.add(getDefines());
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main").setShaderModel(kShaderModel);
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines); 

        // Create neighbor offset texture.
        mpNeighborOffsets = createNeighborOffsetTexture(kNeighborOffsetCount);
        mpPairedNeighborDeltas = createPairedNeighborDeltasTexture(mpDevice, mOptions.spatialNeighborCount, mOptions.spatialPairingStdevLevel);
    }

    DefineList ReSTIRPathTracing::getDefines() const
    {
        DefineList defines;
        defines.add("RESTIRPT_SHIFT_MAPPING", std::to_string((uint32_t)mOptions.shiftMapping));
        defines.add("TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE", mOptions.temporalUpdateForDynamicScene ? "1": "0");
        defines.add("USE_PREV_FRAME_SCENE_DATA", mOptions.temporalUpdateForDynamicScene ? "1" : "0");
        defines.add("USE_RESERVOIR_COMPRESSION", mOptions.useReservoirCompression ? "1" : "0");
        defines.add("RETRACE_SCHEDULE_TYPE", std::to_string((uint32_t)mOptions.retraceScheduleType));
        defines.add("USE_AREA_RESERVOIRS", mOptions.useAreaReservoirs ? "1" : "0");
        defines.add("USE_DECOUPLED_SHADING", mOptions.useDecoupledShading ? "1" : "0");
        defines.add("PAD_RANDOM_NUMBERS", "1");
        defines.add("FETCH_NRD_DATA_PASS", "0");

        return defines;
    }

    void ReSTIRPathTracing::setShaderData(const ShaderVar& var) const
    {
        var["settings"]["localStrategyType"] = mOptions.shiftMappingSettings.localStrategyType;
        var["settings"]["specularRoughnessThreshold"] = mOptions.shiftMappingSettings.specularRoughnessThreshold;
        var["settings"]["nearFieldDistanceThreshold"] = mOptions.shiftMappingSettings.nearFieldDistanceThreshold / 100.f;
        var["settings"]["specularRoughnessSigma"] = mOptions.shiftMappingSettings.specularRoughnessSigma;
        var["settings"]["nearFieldDistanceSigma"] = mOptions.shiftMappingSettings.nearFieldDistanceSigma;
        var["settings"]["additionalShadowRayOffset"] = mOptions.shiftMappingSettings.additionalShadowRayOffset;
        var["numSpatialRounds"] = mOptions.spatialIterations;
        var["initialLightSampleCount"] = exInitialLightSampleCount;
        var["sampleDI"] = mOptions.sampleDI;
        var["useAreaReSTIRBackprojection"] = mOptions.useAreaReSTIRBackprojection();
        var["temporalRobustShift"]         = mOptions.useAreaReservoirs && mOptions.temporalRobustShift();
        var["useReservoirSplatting"]       = mOptions.useAreaReservoirs && mOptions.useReservoirSplatting();

        var["samplesPerPixel"] = mStaticParams.samplesPerPixel != 0 ? mStaticParams.samplesPerPixel : kDefaultMaxSamplesPerPixel;
        var["specularMotionVectorRoughnessThreshold"] = mOptions.specularMotionVectorRoughnessThreshold;
        var["pathReservoirs"] = mpReservoirs;
        var["enableSpecularMotionVectors"] = mOptions.enableSpecularMotionVectors;
        var["enableDisocclusionMotionVectors"] = mOptions.enableDisocclusionMotionVectors;
        var["useDoFReconnectionShift"] = mOptions.useDoFReconnectionShift;
        var["useDoFReconnectionForTemporalReuse"] = useDoFReconnectionForTemporalReuseEffective();
        var["doFReconnectionShiftProbability"] = mOptions.doFReconnectionShiftProbability;
        var["deriveDoFReconnectionShiftProbabilityFromCoC"] = mOptions.deriveDoFReconnectionShiftProbabilityFromCoC;
    }

    void ReSTIRPathTracing::setPathTracerParams(ReSTIRPathTracingParams params)
    {
        mPathTracerParams = params;
    }

    void ReSTIRPathTracing::setOwnerDefines(DefineList defines)
    {
        mOwnerDefines = defines;
    }

    void ReSTIRPathTracing::setSharedStaticParams(uint32_t samplesPerPixel, uint32_t maxSurfaceBounces, bool useNEE)
    {
        mStaticParams.maxSurfaceBounces = maxSurfaceBounces;
        mStaticParams.useNEE = useNEE;
        mStaticParams.samplesPerPixel = samplesPerPixel;
    }

    void ReSTIRPathTracing::createPathTracerBlock()
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("pathTracer");
        mpPathTracerBlock = ParameterBlock::create(mpDevice, reflector);
    }

    ref<ParameterBlock> ReSTIRPathTracing::getPathTracerBlock()
    {
        return mpPathTracerBlock;
    }

    void ReSTIRPathTracing::setReservoirData(const ShaderVar& var) const
    {
        var["pathReservoirs"] = mpReservoirs;
        var["useAreaReservoirs"] = mOptions.useAreaReservoirs;
        var["useDecoupledShading"] = mOptions.useDecoupledShading && mOptions.useSpatialResampling;
    }

    bool ReSTIRPathTracing::renderUI(Gui::Widgets& widget)
    {
        bool dirty = false;

        dirty |= widget.checkbox("Include direct lighting in reservoirs", mOptions.sampleDI);

        mRecompile |= widget.checkbox("Use Decoupled Shading", mOptions.useDecoupledShading);

        mReallocate |= widget.checkbox("Use area reservoirs", mOptions.useAreaReservoirs);

        if (auto group = widget.group("Performance settings", true))
        {
            mReallocate |= group.checkbox("Use reservoir compression", mOptions.useReservoirCompression);
            mReallocate |= group.dropdown("Retrace Schedule Type", kRetraceScheduleType, reinterpret_cast<uint32_t&>(mOptions.retraceScheduleType));
        }

        if (auto group = widget.group("Shift mapping options", true))
        {
            mRecompile |= group.dropdown("Shift Mapping", kShiftMappingList, reinterpret_cast<uint32_t&>(mOptions.shiftMapping));

            if (mOptions.shiftMapping == ReSTIRPT::ShiftMapping::Hybrid)
            {
                dirty |= group.var("Distance Threshold", mOptions.shiftMappingSettings.nearFieldDistanceThreshold, 0.f, 10000.f);
                dirty |= group.var("Distance Sigma", mOptions.shiftMappingSettings.nearFieldDistanceSigma, 0.f, 1.f);
                dirty |= group.var("Roughness Threshold", mOptions.shiftMappingSettings.specularRoughnessThreshold, 0.f, 1.f);
                dirty |= group.var("Roughness Sigma", mOptions.shiftMappingSettings.specularRoughnessSigma, 0.f, 1.f);
            }

            dirty |= group.checkbox("Additional Shadow Ray Offset", mOptions.shiftMappingSettings.additionalShadowRayOffset);

            mReallocate |= group.checkbox("Use primary hit reconnection shift for DoF", mOptions.useDoFReconnectionShift);
            if (mOptions.useDoFReconnectionShift)
            {
                dirty |= group.checkbox("Use DoF reconnection for temporal reuse", mOptions.useDoFReconnectionForTemporalReuse);
                dirty |= group.checkbox("Derive DoF reconnection shift probability from CoC", mOptions.deriveDoFReconnectionShiftProbabilityFromCoC);
                dirty |= group.var("DoF reconnection shift probability", mOptions.doFReconnectionShiftProbability, 0.f, 1.f);
            }
        }

        if (auto group = widget.group("Temporal resampling", true))
        {
            dirty |= group.checkbox("Use temporal resampling", mOptions.useTemporalResampling);

            dirty |= group.var("Max history length", mOptions.maxHistoryLength, 0u, 100u);
            group.tooltip("Maximum temporal history length.");


            dirty |= group.dropdown("Point Reprojection Mode", kPointReprojectionMode, reinterpret_cast<uint32_t&>(mOptions.pointReprojectionMode));

            mReallocate |= group.checkbox("Update RcVertex Radiance for Dynamic Scene", mOptions.temporalUpdateForDynamicScene);
            dirty |= group.checkbox("Enable Specular Motion Vectors", mOptions.enableSpecularMotionVectors);
            if (mOptions.enableSpecularMotionVectors)
            {
                dirty |= group.checkbox("Handle Refraction in Specular Motion Vectors", mOptions.handleTransmissionInSpecMVec);
                dirty |= group.var("Specular Motion Vectors Roughness Threshold", mOptions.specularMotionVectorRoughnessThreshold, 0.f, 1.f);
            }
            dirty |= group.checkbox("Enable Disocclusion MVec", mOptions.enableDisocclusionMotionVectors);


            ReSTIRPT::TemporalAreaReuseType displayed = mOptions.useAreaReservoirs
                ? mOptions.temporalAreaReuseType
                : ReSTIRPT::TemporalAreaReuseType::Point;
            if (group.dropdown("Temporal Area Reuse Type", kTemporalAreaReuseType, reinterpret_cast<uint32_t&>(displayed))
                && mOptions.useAreaReservoirs)
            {
                mRecompile = true;
                mOptions.temporalAreaReuseType = displayed;
            }
            dirty |= group.checkbox("Duplication-based MCap downweighting", mOptions.enableDupMapMCapDownweighting);

            dirty |= group.var("Duplication-based M-cap downweighting power", mOptions.duplicationMCapPower, 0.f, 10.f);
        }

        if (auto group = widget.group("Spatial resampling", true))
        {
            dirty |= group.checkbox("Use spatial resampling", mOptions.useSpatialResampling);

            dirty |= group.var("Iterations", mOptions.spatialIterations, 0u, 8u);
            group.tooltip("Number of spatial resampling iterations.");

            dirty |= group.checkbox("Use reciprocal neighbor selection", mOptions.usePairedNeighbors);

            bool neighborCountChanged = group.var("Neighbor count", mOptions.spatialNeighborCount, 1u, (mOptions.usePairedNeighbors ? 5u : 16u));
            group.tooltip("Number of neighbor samples to resample per pixel and iteration.");

            mReallocate |= neighborCountChanged;

            if (mOptions.usePairedNeighbors)
            {
                bool stdevLevelChanged = false;
                int avgRadius = kPairedNeighborMinRadius + mOptions.spatialPairingStdevLevel * kPairedNeighborRadiusStep;
                stdevLevelChanged = group.var("Gather radius", avgRadius, kPairedNeighborMinRadius, kPairedNeighborMaxRadius, kPairedNeighborRadiusStep);
                dirty |= stdevLevelChanged;
                if (neighborCountChanged || stdevLevelChanged)
                {
                    mOptions.spatialPairingStdevLevel = avgRadius / kPairedNeighborRadiusStep - 1;
                    mpPairedNeighborDeltas =
                        createPairedNeighborDeltasTexture(mpDevice, mOptions.spatialNeighborCount, mOptions.spatialPairingStdevLevel);
                }
                group.tooltip("Average spatial neighbor distance of the pairing texture");
            }
            else
            {
                dirty |= group.var("Gather radius", mOptions.spatialGatherRadius, 0u, 200u);
                group.tooltip("Radius to gather samples from.");
            }
            dirty |= group.checkbox("Area Reservoir use Neighbor Rejection", mOptions.areaReservoirUseNeighborRejection);
        }

        if (auto group = widget.group("Debugging"))
        {
            mpPixelDebug->renderUI(group);
        }

        mRecompile |= mReallocate;
        dirty |= mRecompile;

        if (dirty)
        {
            mResetTemporalReservoirs = true;
        }

        return dirty;
    }

    void ReSTIRPathTracing::setOptions(const Options& options)
    {
        if (std::memcmp(&options, &mOptions, sizeof(Options)) != 0)
        {
            mOptions = options;
            mRecompile = true;
        }
    }

    void ReSTIRPathTracing::beginFrame(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles, bool needRecompile)
    {
        mRecompile |= needRecompile;

        mFrameDim = frameDim;

        prepareResources(pRenderContext, frameDim, screenTiles);

        if (mResetTemporalReservoirs) mFrameIndex = 0;


        mpPixelDebug->beginFrame(pRenderContext, mFrameDim);
    }

    void ReSTIRPathTracing::endFrame(RenderContext* pRenderContext)
    {
        mFrameIndex++;

        // Swap reservoirs.
        std::swap(mpReservoirs, mpPrevReservoirs);
        if (mOptions.computeTemporalDuplication)
            std::swap(mpTemporalCorrelationMap, mpPrevTemporalCorrelationMap);
        if (mOptions.useAreaReservoirs)
            std::swap(mpRandomVBuffer, mpTemporalRandomVBuffer);

        mpPixelDebug->endFrame(pRenderContext);
    }


    void ReSTIRPathTracing::updateReSTIRPT(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const ref<Texture>& pVBuffer, const ref<Texture>& pColorBuffer)
    {
        FALCOR_PROFILE(pRenderContext, "ReSTIRPathTracing::updateReSTIRPT");

        temporalResampling(pRenderContext, pMotionVectors, pVBuffer);
        spatialResampling(pRenderContext, pVBuffer, pColorBuffer);

        // prepare temporal data
        pRenderContext->copyResource(mpTemporalVBuffer.get(), pVBuffer.get());

        if (mOptions.enableDupMapMCapDownweighting || mOptions.computeTemporalDuplication)
            fillSampleID(pRenderContext);

        return;
    }

    void ReSTIRPathTracing::createOrDestroyBuffer(ref<Buffer>& pBuffer, std::string_view reflectVarName, int requiredElementCount, bool keepCondition)
    {
        if (!keepCondition)
        {
            pBuffer = nullptr;
            return;
        }
        if (mReallocate || !pBuffer || pBuffer->getElementCount() != requiredElementCount)
        {
            pBuffer = mpDevice->createStructuredBuffer(
                mpReflectTypes->getRootVar()[std::string(reflectVarName)],
                requiredElementCount,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr,
                false);
        }
    }

    void ReSTIRPathTracing::createOrDestroyRawBuffer(ref<Buffer>& pBuffer, size_t requiredSize, bool keepCondition)
    {
        if (keepCondition && (mReallocate || !pBuffer || pBuffer->getSize() != requiredSize))
            pBuffer = mpDevice->createBuffer(requiredSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ReSTIRPathTracing::createTexture2D(ref<Texture>& pTexture, uint32_t width, uint32_t height, ResourceFormat format, bool keepCondition)
    {
        if (keepCondition && (mReallocate || !pTexture || pTexture->getWidth() != width || pTexture->getHeight() != height))
            pTexture = mpDevice->createTexture2D(width, height, format, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        if (!keepCondition) pTexture = nullptr;
    }


    void ReSTIRPathTracing::createComputePass(ref<ComputePass>& pPass, std::string shaderFile, DefineList defines, ProgramDesc baseDesc, std::string entryFunction)
    {
        if (!pPass)
        {
            ProgramDesc desc = baseDesc;
            desc.addShaderLibrary(shaderFile).csEntry(entryFunction);
            pPass = ComputePass::create(mpDevice, desc, defines, false);
        }
        pPass->getProgram()->addDefines(defines);
        pPass->setVars(nullptr);
    }

    void ReSTIRPathTracing::prepareResources(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles)
    {
        // Create screen sized buffers.
        uint32_t tileCount = screenTiles.x * screenTiles.y;
        const uint32_t elementCount = tileCount * mPathTracerParams.kScreenTileDim.x * mPathTracerParams.kScreenTileDim.y;

        if (mReallocate && mpReservoirs) updatePrograms();

        // The retrace queue must be sized for the worst case across temporal/spatial passes.
        const uint32_t temporalPerPixel = mOptions.temporalRobustShift() ? kTemporalRobustNeighborMultiplier : kTemporalNeighborMultiplier;
        const uint32_t spatialPerPixel  = kSpatialNeighborMultiplier * mOptions.spatialNeighborCount;
        const uint32_t shiftCount = elementCount * std::max(temporalPerPixel, spatialPerPixel);
        const uint32_t splatShiftCount = elementCount * kTemporalNeighborMultiplier;

        // Floating reservoirs need to exist when any temporal mode that consumes them is enabled,
        // even if the current frame routes through the splatting path (which doesn't read them).
        const bool needFloatingReservoirs =
            mOptions.useAreaReSTIRBackprojection()
            ;

        prepareReservoirAndPathBuffers(elementCount, shiftCount, needFloatingReservoirs);
        prepareDuplicationMapResources(frameDim);
        prepareVBufferTextures(frameDim);
        prepareReservoirSplattingResources(frameDim, splatShiftCount);
        prepareMotionVectorTextures(frameDim);

        if (mReallocate)
            mRecompile = true;
        mReallocate = false;
    }

    void ReSTIRPathTracing::prepareReservoirAndPathBuffers(uint32_t elementCount, uint32_t pathCount, bool needFloatingReservoirs)
    {
        // Per-reservoir buffers.
        createOrDestroyBuffer(mpReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpPrevReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpFloatingReservoirs, "pathReservoirs", elementCount, needFloatingReservoirs);
        createOrDestroyBuffer(mpNeighborValidMaskBuffer, "neighborValidMask", elementCount);

        // retrace work queues.
        createOrDestroyRawBuffer(mpRetraceWorkItems, pathCount * sizeof(uint32_t), useCompactRetraceSchedule());
        createOrDestroyRawBuffer(mpCounters, sizeof(uint32_t), useCompactRetraceSchedule());

        // Reconnection / shifted-hit / paired-shift buffers.
        createOrDestroyBuffer(mpReconnectionDataBuffer, "reconnectionDataBuffer", pathCount, useReconnectionDataBuffers());
        createOrDestroyBuffer(mpReconnectionDataOffsets, "reconnectionDataOffsets", pathCount, useReconnectionDataBuffers());
        createOrDestroyBuffer(mpShiftedPrimaryHitBuffer, "shiftedPrimaryHitBuffer", pathCount, mOptions.useAreaReservoirs);
        createOrDestroyBuffer(mpShiftedPrimaryHitBufferRobust, "shiftedPrimaryHitBuffer", pathCount, useRobustTemporalAreaReuse());
        createOrDestroyBuffer(mpPrecomputedShiftDataBuffer, "precomputedShiftBuffer", elementCount * mOptions.spatialNeighborCount, mOptions.usePairedNeighbors);

        // DoF / robust temporal MIS scratch buffers
        createOrDestroyBuffer(mpDoFReconnectionShiftBuffer, "doFReconnectionShiftBuffer", pathCount, mOptions.useAreaReservoirs && mOptions.useDoFReconnectionShift);
        createOrDestroyBuffer(mpTemporalMISPDFs, "temporalMISPDFs", pathCount, useRobustTemporalAreaReuse());
    }

    void ReSTIRPathTracing::prepareDuplicationMapResources(const uint2& frameDim)
    {
        createTexture2D(mpPrevTemporalCorrelationMap, frameDim.x, frameDim.y, ResourceFormat::R32Uint);
        createTexture2D(mpTemporalCorrelationMap, frameDim.x, frameDim.y, ResourceFormat::R32Uint);

        const bool dupMapEnabled = useDuplicationMap();
        createTexture2D(mpSampleIDTexture, frameDim.x, frameDim.y, ResourceFormat::R32Uint, dupMapEnabled);
        createTexture2D(mpDuplicationMap, frameDim.x, frameDim.y, ResourceFormat::R32Uint, dupMapEnabled);

    }

    void ReSTIRPathTracing::prepareVBufferTextures(const uint2& frameDim)
    {
        const auto& pScene = dynamic_ref_cast<Scene>(mpScene);
        FALCOR_ASSERT(pScene);
        const HitInfo hitInfo = pScene->getHitInfo();
        createTexture2D(mpTemporalVBuffer, frameDim.x, frameDim.y, hitInfo.getFormat());
        createTexture2D(mpRandomVBuffer, frameDim.x, frameDim.y, hitInfo.getFormat(), mOptions.useAreaReservoirs);
        createTexture2D(mpTemporalRandomVBuffer, frameDim.x, frameDim.y, hitInfo.getFormat(), mOptions.useAreaReservoirs);
    }

    void ReSTIRPathTracing::prepareReservoirSplattingResources(const uint2& frameDim, uint32_t temporalPathCount)
    {
        createOrDestroyBuffer(mpReservoirSplatInfo, "splatSampleInfo", temporalPathCount, mOptions.useReservoirSplatting());
        createOrDestroyBuffer(mpReservoirSplatPixel, "splatTargetPixel", temporalPathCount, mOptions.useReservoirSplatting());
        createOrDestroyBuffer(mpSortedScatteringPixels, "sortedSplatSourcePixels", frameDim.x * frameDim.y, mOptions.useReservoirSplatting());
        createTexture2D(mpMappingIndices, frameDim.x, frameDim.y, ResourceFormat::RG32Uint, mOptions.useReservoirSplatting());
        createTexture2D(mpCellCounters, frameDim.x, frameDim.y, ResourceFormat::R32Uint, mOptions.useReservoirSplatting());
        createTexture2D(mpCellOffsets, frameDim.x, frameDim.y, ResourceFormat::R32Uint, mOptions.useReservoirSplatting());
        createOrDestroyBuffer(mpCellOffsetCounter, "splatCellOffsetCounter", 1, mOptions.useReservoirSplatting());
    }

    void ReSTIRPathTracing::prepareMotionVectorTextures(const uint2& frameDim)
    {
        // internal motion vector buffer used for specular/disocclusion motion vectors
        createTexture2D(mpMVecBuffer, frameDim.x, frameDim.y, ResourceFormat::RG32Float);
        // Delta reflection / transmission motion vectors written by the path tracer's delta guide passes.
        createTexture2D(mpDeltaReflectionMotionVectors, frameDim.x, frameDim.y, ResourceFormat::RG16Float, mOptions.enableSpecularMotionVectors);
        createTexture2D(mpDeltaTransmissionMotionVectors, frameDim.x, frameDim.y, ResourceFormat::RG16Float, mOptions.enableSpecularMotionVectors);
    }

    void ReSTIRPathTracing::updatePrograms()
    {
        if (!mRecompile || !mpReservoirs)
            return;

        DefineList commonDefines;
        commonDefines.add(getDefines());
        commonDefines.add(mOwnerDefines);

        TypeConformanceList typeConformances;
        mpScene->getTypeConformances(typeConformances);

        ProgramDesc baseDesc;
        mpScene->getShaderModules(baseDesc.shaderModules);
        baseDesc.addTypeConformances(typeConformances);

        DefineList defines = commonDefines;
        defines.add("NEIGHBOR_OFFSET_COUNT", std::to_string(mpNeighborOffsets->getWidth()));

        createComputePass(mpReflectTypes, kReflectTypesFile, defines, baseDesc);

        compileTemporalPasses(defines, baseDesc);
        compileSpatialPasses(defines, baseDesc);

        if (mOptions.useTemporalResampling || mOptions.useSpatialResampling)
            createComputePass(mpGenerateRetraceWorkload, kSpatialGenerateRetraceWorkload, defines, baseDesc);

        createComputePass(mpFetchDenoiserData, kFetchDenoiserData, defines, baseDesc);
        createComputePass(mpFillSampleId, kFillSampleID, defines, baseDesc);
        createComputePass(mpComputeDuplicationMap, kComputeDuplicationMap, defines, baseDesc);

        if (mOptions.useAreaReservoirs)
            compileAreaReservoirPasses(defines, baseDesc);

        createComputePass(mpComputeReuseMotionVector, kComputeReuseMotionVector, defines, baseDesc);

        mRecompile = false;
        mResetTemporalReservoirs = true;
    }

    void ReSTIRPathTracing::compileTemporalPasses(const DefineList& defines, const ProgramDesc& baseDesc)
    {
        if (!mOptions.useTemporalResampling) return;
        createComputePass(mpTemporalRetrace, kTemporalRetraceFile, defines, baseDesc);
        createComputePass(mpTemporalResampling, kTemporalResamplingFile, defines, baseDesc);
    }

    void ReSTIRPathTracing::compileSpatialPasses(const DefineList& defines, const ProgramDesc& baseDesc)
    {
        if (!mOptions.useSpatialResampling) return;
        createComputePass(mpSpatialRetrace, kSpatialRetraceFile, defines, baseDesc);
        createComputePass(mpSpatialResampling, kSpatialResamplingFile, defines, baseDesc);

        if (mOptions.usePairedNeighbors)
            createComputePass(mpSpatialPrecomputeShiftBufferPass, kSpatialPrecomputeShiftBufferFile, defines, baseDesc);
    }

    void ReSTIRPathTracing::compileAreaReservoirPasses(const DefineList& defines, const ProgramDesc& baseDesc)
    {
        createComputePass(mpSpatialTraceShiftedPrimaryHits, kSpatialTraceShiftedPrimaryHits, defines, baseDesc);
        createComputePass(mpTemporalTraceShiftedPrimaryHits, kTemporalTraceShiftedPrimaryHits, defines, baseDesc);
        createComputePass(mpTemporalPopulateFloatingReservoir, kTemporalPopulateFloatingReservoir, defines, baseDesc);

        if (mOptions.temporalRobustShift())
            compileRobustShiftPasses(defines, baseDesc);

        if (mOptions.useReservoirSplatting())
            compileReservoirSplattingPasses(defines, baseDesc);
    }

    void ReSTIRPathTracing::compileRobustShiftPasses(const DefineList& defines, const ProgramDesc& baseDesc)
    {
        createComputePass(mpTemporalRobustCreateShiftVBuffer, kTemporalRobustCreateShiftVBuffer, defines, baseDesc);
        createComputePass(mpTemporalRobustProduceWorkload, kTemporalRobustProduceWorkload, defines, baseDesc);
        createComputePass(mpTemporalRobustPathRetrace, kTemporalRobustPathRetrace, defines, baseDesc);
        createComputePass(mpTemporalRobustResampling, kTemporalRobustResampling, defines, baseDesc);
    }

    void ReSTIRPathTracing::compileReservoirSplattingPasses(const DefineList& defines, const ProgramDesc& baseDesc)
    {
        createComputePass(mpTemporalSplatSamples, kTemporalReservoirSplatting, defines, baseDesc, "splatSamples");
        createComputePass(mpTemporalComputeCellOffsets, kTemporalReservoirSplatting, defines, baseDesc, "computeSplatCellOffsets");
        createComputePass(mpTemporalSortCellData, kTemporalReservoirSplatting, defines, baseDesc, "sortSplatSourcesByTargetPixel");
    }

    ShaderVar ReSTIRPathTracing::bindCommonPassVars(
        RenderContext* pRenderContext,
        ref<ComputePass> pPass,
        const std::string& cbName,
        bool bindPathTracer)
    {
        auto rootVar = pPass->getRootVar();
        mpScene->bindShaderData(rootVar["gScene"]);
        mpScene->bindShaderDataForRaytracing(pRenderContext, rootVar["gScene"]);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];
        var["params"].setBlob(mPathTracerParams);
        setShaderData(var["restirpt"]);

        if (bindPathTracer)
            rootVar["gPathTracer"] = mpPathTracerBlock;

        return var;
    }

    ShaderVar ReSTIRPathTracing::bindTemporalRobustVars(
        RenderContext* pRenderContext,
        ref<ComputePass> pPass,
        std::string cbName,
        bool bindPathTracer
    )
    {
        ShaderVar var = bindCommonPassVars(pRenderContext, pPass, cbName, bindPathTracer);
        var["shiftedPrimaryHitBuffer"] = mpShiftedPrimaryHitBufferRobust;
        var["prevReservoirs"] = mpPrevReservoirs;
        return var;
    }

    ShaderVar ReSTIRPathTracing::bindSpatialResamplingVars(
        RenderContext* pRenderContext,
        ref<ComputePass> pPass,
        std::string cbName,
        const ref<Texture>& pVBuffer,
        bool bindPathTracer,
        bool isShiftVBufferPass
    )
    {
        // Paired-neighbor delta tables live at the root scope, not inside the constant buffer.
        auto rootVar = pPass->getRootVar();
        rootVar["pairingDeltaTextures"] = mpPairedNeighborDeltas;
        rootVar["pairingDeltaTextureSizes"] = mpPairedNeighborTextureSizes;

        ShaderVar var = bindCommonPassVars(pRenderContext, pPass, cbName, bindPathTracer);
        var["neighborOffsets"] = mpNeighborOffsets;
        var["gNeighborCount"] = mOptions.spatialNeighborCount;
        var["gGatherRadius"] = (float)mOptions.spatialGatherRadius;
        var["vbuffer"] = pVBuffer;
        var["resetTemporalReservoirs"] = mResetTemporalReservoirs;
        var["shiftedPrimaryHitBuffer"] = mpShiftedPrimaryHitBuffer;
        var["areaReservoirUseNeighborRejection"] = mOptions.areaReservoirUseNeighborRejection;
        var["gUsePairedNeighbors"] = mOptions.usePairedNeighbors;
        return var;
    }

    void ReSTIRPathTracing::spatialResampling(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const ref<Texture>& pColorBuffer)
    {
        FALCOR_PROFILE(pRenderContext, "spatialResampling");

        if (!mOptions.useSpatialResampling) return;

        // Bind the per-pass shader-var sets once outside the loop. Each iteration only writes the
        // few fields that change between rounds (reservoirs, gSpatialRoundId, ...).
        ShaderVar var = bindSpatialResamplingVars(pRenderContext, mpSpatialResampling, "gSpatialResampling", pVBuffer, true);
        var["neighborValidMask"] = mpNeighborValidMaskBuffer;
        var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        var["reconnectionDataOffsets"] = mpReconnectionDataOffsets;

        ShaderVar precomputeVar;
        if (mOptions.usePairedNeighbors)
        {
            precomputeVar = bindSpatialResamplingVars(pRenderContext, mpSpatialPrecomputeShiftBufferPass, "gPass", pVBuffer, true);
            precomputeVar["neighborValidMask"] = mpNeighborValidMaskBuffer;
            precomputeVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
            precomputeVar["reconnectionDataOffsets"] = mpReconnectionDataOffsets;
        }

        ShaderVar retraceVar = bindSpatialResamplingVars(pRenderContext, mpSpatialRetrace, "gSpatialPathRetrace", pVBuffer, true);
        retraceVar["neighborValidMask"] = mpNeighborValidMaskBuffer;
        retraceVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        retraceVar["reconnectionDataOffsets"] = mpReconnectionDataOffsets;
        retraceVar["numKeysBuffer"] = mpNumKeysBuffer;
        retraceVar["queues"]["counters"] = mpCounters;
        retraceVar["queues"]["retraceWorkItems"] = mpRetraceWorkItems;

        ShaderVar workLoadVar;
        if (useCompactRetraceSchedule())
        {
            workLoadVar = bindSpatialResamplingVars(pRenderContext, mpGenerateRetraceWorkload, "gRetraceWorkloadGenerator", pVBuffer, false);
            workLoadVar["queues"]["counters"] = mpCounters;
            workLoadVar["queues"]["retraceWorkItems"] = mpRetraceWorkItems;
            workLoadVar["neighborValidMask"] = mpNeighborValidMaskBuffer;
            workLoadVar["isSpatialPass"] = true;
        }

        ShaderVar shiftVar;
        if (mOptions.useAreaReservoirs)
            shiftVar = bindSpatialResamplingVars(pRenderContext, mpSpatialTraceShiftedPrimaryHits, "gPass", pVBuffer, true, true);

        mResetTemporalReservoirs = false;

        for (uint32_t iteration = 0; iteration < mOptions.spatialIterations; ++iteration)
        {
            std::swap(mpReservoirs, mpPrevReservoirs);

            if (mOptions.useAreaReservoirs)
                spatialDispatchTraceShiftedPrimaryHits(pRenderContext, shiftVar, iteration);

            if (useCompactRetraceSchedule())
                spatialDispatchGenerateRetraceWorkload(pRenderContext, workLoadVar, iteration);

            spatialDispatchRetrace(pRenderContext, retraceVar, iteration);

            if (mOptions.usePairedNeighbors)
                spatialDispatchPrecomputeShiftBuffer(pRenderContext, precomputeVar, iteration);

            spatialDispatchResamplingPass(pRenderContext, var, iteration, pColorBuffer);
        }
    }

    void ReSTIRPathTracing::spatialDispatchTraceShiftedPrimaryHits(RenderContext* pRenderContext, ShaderVar shiftVar, uint32_t iteration)
    {
        FALCOR_PROFILE(pRenderContext, "spatialCreateShiftVBuffer");
        shiftVar["prevReservoirs"] = mpPrevReservoirs;
        shiftVar["gSpatialRoundId"] = iteration;
        shiftVar["sampleVBuffer"] = mpRandomVBuffer;
        shiftVar["doFReconnectionShiftBuffer"] = mpDoFReconnectionShiftBuffer;
        mpSpatialTraceShiftedPrimaryHits->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    void ReSTIRPathTracing::spatialDispatchGenerateRetraceWorkload(RenderContext* pRenderContext, ShaderVar workLoadVar, uint32_t iteration)
    {
        FALCOR_PROFILE(pRenderContext, "produceRetraceWorkload");

        if (mpCounters)
            pRenderContext->clearUAV(mpCounters->getUAV().get(), uint4(0));

        workLoadVar["prevReservoirs"] = mpPrevReservoirs;
        workLoadVar["gSpatialRoundId"] = iteration;
        const uint32_t tileSize = mPathTracerParams.kScreenTileDim.x * mPathTracerParams.kScreenTileDim.y;
        mpGenerateRetraceWorkload->execute(pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1);
    }

    void ReSTIRPathTracing::spatialDispatchRetrace(RenderContext* pRenderContext, ShaderVar retraceVar, uint32_t iteration)
    {
        FALCOR_PROFILE(pRenderContext, "spatialRetrace");
        retraceVar["prevReservoirs"] = mpPrevReservoirs;
        retraceVar["gSpatialRoundId"] = iteration;
        retraceVar["doFReconnectionShiftBuffer"] = mpDoFReconnectionShiftBuffer;
        if (useCompactRetraceSchedule())
            mpSpatialRetrace->execute(pRenderContext, kSpatialNeighborMultiplier * mOptions.spatialNeighborCount * mFrameDim.x * mFrameDim.y, 1, 1);
        else
            mpSpatialRetrace->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    void ReSTIRPathTracing::spatialDispatchPrecomputeShiftBuffer(RenderContext* pRenderContext, ShaderVar precomputeVar, uint32_t iteration)
    {
        FALCOR_PROFILE(pRenderContext, "spatialPrecomputeShiftBuffer");
        precomputeVar["reservoirs"] = mpReservoirs;
        precomputeVar["prevReservoirs"] = mpPrevReservoirs;
        precomputeVar["sampleSurvivedFrames"] = mpTemporalCorrelationMap;
        precomputeVar["gSpatialRoundId"] = iteration;
        precomputeVar["doFReconnectionShiftBuffer"] = mpDoFReconnectionShiftBuffer;
        precomputeVar["sampleVBuffer"] = mpRandomVBuffer;
        precomputeVar["precomputedShiftBuffer"] = mpPrecomputedShiftDataBuffer;
        mpSpatialPrecomputeShiftBufferPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    void ReSTIRPathTracing::spatialDispatchResamplingPass(RenderContext* pRenderContext, ShaderVar var, uint32_t iteration, const ref<Texture>& pColorBuffer)
    {
        FALCOR_PROFILE(pRenderContext, "spatialResampling");
        var["reservoirs"] = mpReservoirs;
        var["prevReservoirs"] = mpPrevReservoirs;
        var["sampleSurvivedFrames"] = mpTemporalCorrelationMap;
        var["gSpatialRoundId"] = iteration;
        var["doFReconnectionShiftBuffer"] = mpDoFReconnectionShiftBuffer;
        var["sampleVBuffer"] = mpRandomVBuffer;
        var["precomputedShiftBuffer"] = mpPrecomputedShiftDataBuffer;

        if (mOptions.useDecoupledShading && pColorBuffer)
            var["outputColor"] = pColorBuffer;

        mpSpatialResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }



    ShaderVar ReSTIRPathTracing::bindReservoirSplattingVars(
        RenderContext* pRenderContext,
        ref<ComputePass> pPass,
        std::string cbName
    )
    {
        // Reservoir splatting passes don't bind the path tracer block or the `restirpt` settings struct,
        // so the common helper isn't a clean fit; just inline the small preamble.
        auto rootVar = pPass->getRootVar();
        mpScene->bindShaderData(rootVar["gScene"]);
        mpScene->bindShaderDataForRaytracing(pRenderContext, rootVar["gScene"]);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];
        var["params"].setBlob(mPathTracerParams);
        var["sampleVBuffer"] = mpRandomVBuffer;
        var["temporalSampleVbuffer"] = mpTemporalRandomVBuffer;
        var["prevReservoirs"] = mpPrevReservoirs;
        var["reservoirs"] = mpReservoirs;
        var["splatCellOffsetCounter"] = mpCellOffsetCounter;
        var["splatCellCounts"] = mpCellCounters;
        var["splatCellOffsets"] = mpCellOffsets;
        var["sortedSplatSourcePixels"] = mpSortedScatteringPixels;
        var["splatCellIndices"] = mpMappingIndices;
        var["splatSampleInfo"] = mpReservoirSplatInfo;
        var["splatTargetPixel"] = mpReservoirSplatPixel;
        var["shiftedPrimaryHitBuffer"] = mpShiftedPrimaryHitBuffer;
        return var;
    }

    ShaderVar ReSTIRPathTracing::bindTemporalResamplingVars(
        RenderContext* pRenderContext,
        ref<ComputePass> pPass,
        std::string cbName,
        const ref<Texture>& pVBuffer,
        const ref<Texture>& pMotionVectors,
        bool bindPathTracer,
        bool isShiftVBufferPass
    )
    {
        ShaderVar var = bindCommonPassVars(pRenderContext, pPass, cbName, bindPathTracer);

        if (!isShiftVBufferPass)
        {
            var["vbuffer"] = (mOptions.useAreaReservoirs && bindPathTracer) ? mpRandomVBuffer : pVBuffer;
            var["temporalVbuffer"] = mpTemporalVBuffer;
            var["gTemporalHistoryLength"] = mOptions.maxHistoryLength;
        }
        else
        {
            var["vbuffer"] = pVBuffer;
        }

        var["motionVectors"] = useCustomMotionVectorPass() ? mpMVecBuffer : pMotionVectors;
        var["gEnableTemporalReprojection"] = true;
        var["gStochasticReprojection"] = mOptions.pointReprojectionMode == ReSTIRPT::PointReprojectionMode::Stochastic;

        var["shiftedPrimaryHitBuffer"] = mpShiftedPrimaryHitBuffer;
        var["reservoirs"] = mpReservoirs;
        var["prevReservoirs"] = useFloatingReservoirs() ? mpFloatingReservoirs : mpPrevReservoirs;
        return var;
    }

    void ReSTIRPathTracing::temporalResampling(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const ref<Texture>& pVBuffer)
    {
        FALCOR_PROFILE(pRenderContext, "temporalResampling");

        if (mResetTemporalReservoirs)
        {
            pRenderContext->clearUAV(mpTemporalCorrelationMap->getUAV().get(), uint4(0));
            if (!mOptions.useSpatialResampling) mResetTemporalReservoirs = false;
            return;
        }

        if (!mOptions.useTemporalResampling) return;

        // A second temporal pass is required when DoF reconnection produces a separate proposal stream.
        const int numPasses = mOptions.useAreaReservoirs
            && useDoFReconnectionForTemporalReuseEffective()
            && mOptions.useDoFReconnectionShift ? 2 : 1;

        for (int i = 0; i < numPasses; i++)
        {
            if (useCustomMotionVectorPass())
                temporalDispatchComputeReuseMotionVector(pRenderContext, pVBuffer, pMotionVectors);

            if (useRobustTemporalAreaReuse() && i == 0)
                temporalDispatchRobustChain(pRenderContext);

            // The populate-floating-reservoir pass writes into mpFloatingReservoirs whenever the temporal
            // path will read from it. The standard backprojection path only runs on the first pass (i==0),
            // but the multi-motion-vectors path runs on every temporal pass.
            bool executePopulatePass = !mOptions.useReservoirSplatting() && mOptions.useAreaReSTIRBackprojection() && i == 0;
            if (executePopulatePass)
                temporalDispatchPopulateFloatingReservoir(pRenderContext, pVBuffer, pMotionVectors);

            if (mOptions.useReservoirSplatting())
                temporalDispatchSplattingChain(pRenderContext);

            if (mOptions.useAreaReservoirs && !mOptions.useReservoirSplatting())
                temporalDispatchTraceShiftedPrimaryHits(pRenderContext, pVBuffer, pMotionVectors, i);

            if (useCompactRetraceSchedule())
                temporalDispatchGenerateRetraceWorkload(pRenderContext, pVBuffer, pMotionVectors, i);

            temporalDispatchRetrace(pRenderContext, pVBuffer, pMotionVectors, i);
            temporalDispatchResamplingPass(pRenderContext, pVBuffer, pMotionVectors, i);
        }
    }

    void ReSTIRPathTracing::temporalDispatchComputeReuseMotionVector(
        RenderContext* pRenderContext,
        const ref<Texture>& pVBuffer,
        const ref<Texture>& pMotionVectors)
    {
        FALCOR_PROFILE(pRenderContext, "ComputeReuseMotionVector");
        ShaderVar var = bindCommonPassVars(pRenderContext, mpComputeReuseMotionVector, "gPass", false);
        var["originalMotionVectors"] = pMotionVectors;
        var["deltaReflectionMotionVectors"] = mpDeltaReflectionMotionVectors;
        var["deltaTransmissionMotionVectors"] = mOptions.handleTransmissionInSpecMVec ? mpDeltaTransmissionMotionVectors : pMotionVectors;
        var["motionVectors"] = mpMVecBuffer;
        var["vbuffer"] = pVBuffer;
        var["temporalVBuffer"] = mpTemporalVBuffer;

        if (mOptions.enableSpecularMotionVectors && (!mpDeltaReflectionMotionVectors || !mpDeltaTransmissionMotionVectors))
        {
            logWarning("Specular motion vector textures are not allocated; ComputeReuseMotionVector will read from unbound buffers.");
        }

        mpComputeReuseMotionVector->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    void ReSTIRPathTracing::temporalDispatchRobustChain(RenderContext* pRenderContext)
    {
        const uint32_t tileSize = mPathTracerParams.kScreenTileDim.x * mPathTracerParams.kScreenTileDim.y;

        {
            FALCOR_PROFILE(pRenderContext, "TemporalRobustCreateShiftVBuffer");
            ShaderVar var = bindTemporalRobustVars(pRenderContext, mpTemporalRobustCreateShiftVBuffer, "gPass", true);
            mpTemporalRobustCreateShiftVBuffer->execute(pRenderContext, mFrameDim.x, mFrameDim.y, kTemporalRobustNeighborMultiplier);
        }

        {
            FALCOR_PROFILE(pRenderContext, "TemporalRobustProduceWorkload");
            ShaderVar var = bindTemporalRobustVars(pRenderContext, mpTemporalRobustProduceWorkload, "gPass", false);
            if (mpCounters)
                pRenderContext->clearUAV(mpCounters->getUAV().get(), uint4(0));
            var["queues"]["counters"] = mpCounters;
            var["queues"]["retraceWorkItems"] = mpRetraceWorkItems;
            mpTemporalRobustProduceWorkload->execute(pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1);
        }

        {
            FALCOR_PROFILE(pRenderContext, "TemporalRobustPathRetrace");
            ShaderVar var = bindTemporalRobustVars(pRenderContext, mpTemporalRobustPathRetrace, "gPass", true);
            var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
            var["reconnectionDataOffsets"] = mpReconnectionDataOffsets;
            var["queues"]["counters"] = mpCounters;
            var["queues"]["retraceWorkItems"] = mpRetraceWorkItems;
            mpTemporalRobustPathRetrace->execute(pRenderContext, kTemporalRobustNeighborMultiplier * mFrameDim.x * mFrameDim.y, 1, 1);
        }

        {
            FALCOR_PROFILE(pRenderContext, "TemporalRobustResampling");
            ShaderVar var = bindTemporalRobustVars(pRenderContext, mpTemporalRobustResampling, "gPass", true);
            var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
            var["reconnectionDataOffsets"] = mpReconnectionDataOffsets;
            var["temporalMISPDFs"] = mpTemporalMISPDFs;
            mpTemporalRobustResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }
    }

    void ReSTIRPathTracing::temporalDispatchPopulateFloatingReservoir(
        RenderContext* pRenderContext,
        const ref<Texture>& pVBuffer,
        const ref<Texture>& pMotionVectors)
    {
        FALCOR_PROFILE(pRenderContext, "TemporalPopulateFloatingReservoir");
        ShaderVar var = bindCommonPassVars(pRenderContext, mpTemporalPopulateFloatingReservoir, "gPass", false);

        bool useCustomMVecBuffer = useCustomMotionVectorPass();
        var["motionVectors"] = useCustomMVecBuffer ? mpMVecBuffer : pMotionVectors;
        var["deltaTransmissionMotionVectors"] = mOptions.handleTransmissionInSpecMVec ? mpDeltaTransmissionMotionVectors : pMotionVectors;
        var["deltaReflectionMotionVectors"] = mpDeltaReflectionMotionVectors;
        var["temporaryReservoirs"] = mpFloatingReservoirs;
        var["prevReservoirs"] = mpPrevReservoirs;
        var["temporalMISPDFs"] = mpTemporalMISPDFs;
        var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        var["reconnectionDataOffsets"] = mpReconnectionDataOffsets;
        var["shiftedPrimaryHitBufferRobust"] = mpShiftedPrimaryHitBufferRobust;
        var["shiftedPrimaryHitBuffer"] = mpShiftedPrimaryHitBuffer;
        var["reservoirs"] = mpReservoirs;

        var["sampleVBuffer"] = mpRandomVBuffer;
        var["doFReconnectionShiftBuffer"] = mpDoFReconnectionShiftBuffer;
        var["vbuffer"] = pVBuffer;
        mpTemporalPopulateFloatingReservoir->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    void ReSTIRPathTracing::temporalDispatchSplattingChain(RenderContext* pRenderContext)
    {
        pRenderContext->clearUAV(mpCellOffsetCounter->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(mpCellCounters->getUAV().get(), uint4(0));

        {
            FALCOR_PROFILE(pRenderContext, "TemporalSplatSamples");
            ShaderVar var = bindReservoirSplattingVars(pRenderContext, mpTemporalSplatSamples, "gPass");
            mpTemporalSplatSamples->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }

        {
            FALCOR_PROFILE(pRenderContext, "TemporalComputeCellOffsets");
            ShaderVar var = bindReservoirSplattingVars(pRenderContext, mpTemporalComputeCellOffsets, "gPass");
            mpTemporalComputeCellOffsets->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }

        {
            FALCOR_PROFILE(pRenderContext, "TemporalSortCellData");
            ShaderVar var = bindReservoirSplattingVars(pRenderContext, mpTemporalSortCellData, "gPass");
            mpTemporalSortCellData->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }
    }

    void ReSTIRPathTracing::temporalDispatchTraceShiftedPrimaryHits(
        RenderContext* pRenderContext,
        const ref<Texture>& pVBuffer,
        const ref<Texture>& pMotionVectors,
        int temporalPassId)
    {
        FALCOR_PROFILE(pRenderContext, "TemporalTraceShiftedPrimaryHits");
        ShaderVar var = bindTemporalResamplingVars(
            pRenderContext, mpTemporalTraceShiftedPrimaryHits, "gPass", pVBuffer, pMotionVectors, true, true
        );
        var["sampleVBuffer"] = mpRandomVBuffer;
        var["doFReconnectionShiftBuffer"] = mpDoFReconnectionShiftBuffer;
        var["temporalPassId"] = temporalPassId;
        mpTemporalTraceShiftedPrimaryHits->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    void ReSTIRPathTracing::temporalDispatchGenerateRetraceWorkload(
        RenderContext* pRenderContext,
        const ref<Texture>& pVBuffer,
        const ref<Texture>& pMotionVectors,
        int temporalPassId)
    {
        FALCOR_PROFILE(pRenderContext, "GenerateRetraceWorkload");

        if (mpCounters)
            pRenderContext->clearUAV(mpCounters->getUAV().get(), uint4(0));

        ShaderVar var = bindTemporalResamplingVars(pRenderContext, mpGenerateRetraceWorkload, "gRetraceWorkloadGenerator", pVBuffer, pMotionVectors, false);
        var["queues"]["counters"] = mpCounters;
        var["queues"]["retraceWorkItems"] = mpRetraceWorkItems;
        var["neighborValidMask"] = mpNeighborValidMaskBuffer;
        var["isSpatialPass"] = false;
        var["temporalPassId"] = temporalPassId;
        var["splatSampleInfo"] = mpReservoirSplatInfo;

        const uint32_t tileSize = mPathTracerParams.kScreenTileDim.x * mPathTracerParams.kScreenTileDim.y;
        mpGenerateRetraceWorkload->execute(pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1);
    }

    void ReSTIRPathTracing::temporalDispatchRetrace(
        RenderContext* pRenderContext,
        const ref<Texture>& pVBuffer,
        const ref<Texture>& pMotionVectors,
        int temporalPassId)
    {
        FALCOR_PROFILE(pRenderContext, "temporalRetrace");

        ShaderVar var = bindTemporalResamplingVars(pRenderContext, mpTemporalRetrace, "gTemporalPathRetrace", pVBuffer, pMotionVectors, true);
        var["numKeysBuffer"] = mpNumKeysBuffer;
        var["neighborValidMask"] = mpNeighborValidMaskBuffer;
        var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        var["reconnectionDataOffsets"] = mpReconnectionDataOffsets;
        var["queues"]["counters"] = mpCounters;
        var["queues"]["retraceWorkItems"] = mpRetraceWorkItems;
        var["doFReconnectionShiftBuffer"] = mpDoFReconnectionShiftBuffer;
        var["splatSampleInfo"] = mpReservoirSplatInfo;
        var["splatTargetPixel"] = mpReservoirSplatPixel;
        var["temporalPassId"] = temporalPassId;

        if (useCompactRetraceSchedule())
            mpTemporalRetrace->execute(pRenderContext, kTemporalNeighborMultiplier * mFrameDim.x * mFrameDim.y, 1, 1);
        else
            mpTemporalRetrace->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    void ReSTIRPathTracing::temporalDispatchResamplingPass(
        RenderContext* pRenderContext,
        const ref<Texture>& pVBuffer,
        const ref<Texture>& pMotionVectors,
        int temporalPassId)
    {
        FALCOR_PROFILE(pRenderContext, "temporalResampling");

        ShaderVar var = bindTemporalResamplingVars(pRenderContext, mpTemporalResampling, "gTemporalResampling", pVBuffer, pMotionVectors, true);
        var["neighborValidMask"] = mpNeighborValidMaskBuffer;
        var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        var["reconnectionDataOffsets"] = mpReconnectionDataOffsets;
        var["enableDupMapMCapDownweighting"] = mOptions.enableDupMapMCapDownweighting;
        var["computeTemporalDuplication"] = mOptions.computeTemporalDuplication;
        var["duplicationMCapPower"] = mOptions.duplicationMCapPower;
        var["duplicationMap"] = mpDuplicationMap;
        var["sampleSurvivedFrames"] = mpTemporalCorrelationMap;
        var["prevSampleSurvivedFrames"] = mpPrevTemporalCorrelationMap;
        var["gGatherRadius"] = (float)mOptions.spatialGatherRadius;
        var["doFReconnectionShiftBuffer"] = mpDoFReconnectionShiftBuffer;
        var["splatSampleInfo"] = mpReservoirSplatInfo;
        var["splatTargetPixel"] = mpReservoirSplatPixel;
        var["splatCellCounts"] = mpCellCounters;
        var["splatCellOffsets"] = mpCellOffsets;
        var["sortedSplatSourcePixels"] = mpSortedScatteringPixels;
        var["temporalPassId"] = temporalPassId;

        mpTemporalResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    void ReSTIRPathTracing::fetchDenoiserData(
        RenderContext* pRenderContext,
        const ref<Texture>& pVBuffer,
        const ref<Buffer>&  pSampleGuideData,
        const ref<Texture>& pEmission,
        const ref<Texture>& pDiffuseRadianceHitDist,
        const ref<Texture>& pSpecularRadianceHitDist,
        const ref<Texture>& pDeltaReflectionRadianceHitDist,
        const ref<Texture>& pDeltaTransmissionRadianceHitDist,
        const ref<Texture>& pResidualRadianceHitDist,
        const ref<Texture>& pOutputColor
    )
    {
        FALCOR_PROFILE(pRenderContext, "fetchDenoiserData");

        mpFetchDenoiserData->addDefine("OUTPUT_GUIDE_DATA", (pSampleGuideData != nullptr ? "1" : "0"));
        mpFetchDenoiserData->addDefine("FETCH_NRD_DATA_PASS", (pDiffuseRadianceHitDist != nullptr ? "1" : "0"));

        ShaderVar var = bindCommonPassVars(pRenderContext, mpFetchDenoiserData, "gFetchDenoiserData", true);
        var["vbuffer"] = pVBuffer;
        var["reservoirs"] = mpReservoirs;
        var["sampleGuideData"] = pSampleGuideData;

        var["outputEmission"] = pEmission;
        var["outputDiffuseRadianceHitDist"] = pDiffuseRadianceHitDist;
        var["outputSpecularRadianceHitDist"] = pSpecularRadianceHitDist;

        var["outputDeltaReflectionRadianceHitDist"] = pDeltaReflectionRadianceHitDist;
        var["outputDeltaTransmissionRadianceHitDist"] = pDeltaTransmissionRadianceHitDist;
        var["outputResidualRadianceHitDist"] = pResidualRadianceHitDist;
        var["outputColor"] = pOutputColor;

        mpFetchDenoiserData->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
    }

    void ReSTIRPathTracing::fillSampleID(RenderContext* pRenderContext)
    {
        if (!mpReservoirs) return;

        {
            FALCOR_PROFILE(pRenderContext, "fillSampleID");

            auto rootVar = mpFillSampleId->getRootVar();
            mpPixelDebug->prepareProgram(mpFillSampleId->getProgram(), rootVar);
            auto var = rootVar["CB"]["gPass"];
            var["params"].setBlob(mPathTracerParams);
            var["reservoirs"] = mpReservoirs;
            var["sampleIDTexture"] = mpSampleIDTexture;
            mpFillSampleId->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }

        {
            FALCOR_PROFILE(pRenderContext, "computeDuplicationMap");

            auto rootVar = mpComputeDuplicationMap->getRootVar();
            mpPixelDebug->prepareProgram(mpComputeDuplicationMap->getProgram(), rootVar);
            auto var = rootVar["CB"]["gPass"];
            var["params"].setBlob(mPathTracerParams);
            var["sampleIDTexture"] = mpSampleIDTexture;
            var["duplicationMap"] = mpDuplicationMap;
            mpComputeDuplicationMap->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }
    }

    ref<Texture> ReSTIRPathTracing::createNeighborOffsetTexture(uint32_t sampleCount)
    {
        std::unique_ptr<int8_t[]> offsets(new int8_t[sampleCount * 2]);
        const int R = 254;
        const float phi2 = 1.f / 1.3247179572447f;
        float u = 0.5f;
        float v = 0.5f;
        for (uint32_t index = 0; index < sampleCount * 2;)
        {
            u += phi2;
            v += phi2 * phi2;
            if (u >= 1.f) u -= 1.f;
            if (v >= 1.f) v -= 1.f;

            float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
            if (rSq > 0.25f) continue;

            offsets[index++] = int8_t((u - 0.5f) * R);
            offsets[index++] = int8_t((v - 0.5f) * R);
        }

        return mpDevice->createTexture1D(sampleCount, ResourceFormat::RG8Snorm, 1, 1, offsets.get());
    }

    ref<Buffer> ReSTIRPathTracing::createPairedNeighborDeltasTexture(ref<Device> pDevice, uint32_t neighborCount, int stdevLevel)
    {
        // Avg radii: kPairedNeighborMinRadius, +step, +step, ... (kPairedNeighborStdevLevels entries).
        static const auto avg_radii = []() {
            std::array<int, kPairedNeighborStdevLevels> r{};
            for (size_t i = 0; i < r.size(); ++i)
                r[i] = kPairedNeighborMinRadius + int(i) * kPairedNeighborRadiusStep;
            return r;
        }();
        //allocate space
        std::vector<uint32_t> arr(kPairedNeighborTextureExtent * kPairedNeighborTextureExtent * neighborCount);
        std::vector<uint32_t> sizes(kPairedNeighborMaxCount);

        for (uint32_t i = 0; i < neighborCount; i++)
        {
            float stdev = sqrt(8 / (9 * M_PI)) * float(avg_radii[stdevLevel]);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << stdev;
            std::string stdevname = oss.str();

            std::filesystem::path filename;
            findFileInShaderDirectories(
                "ReSTIRPathTracing/PairedReusePattern/neighborCount" + std::to_string(neighborCount) + "-" + "stdev" + stdevname + "/neighbor" + std::to_string(i) + ".png",
                filename
            );

            // read texture
            Bitmap::UniqueConstPtr bitmap = Bitmap::createFromFile(filename, true, Bitmap::ImportFlags::None);

            uint32_t height = bitmap->getHeight();
            uint32_t width = bitmap->getWidth();
            uint16_t* data = (uint16_t*)bitmap->getData();

            // assume height == width

            sizes[i] = height;

            uint32_t counter = 0;

            for (uint32_t y = 0; y < height; y++)
            {
                for (uint32_t x = 0; x < width; x++)
                {
                    arr[kPairedNeighborTextureExtent * kPairedNeighborTextureExtent * i + width * y + x] = data[width * y + x];
                    counter++;
                }
            }
        }

        mpPairedNeighborTextureSizes = mpDevice->createStructuredBuffer(4, 5, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, sizes.data(), false);

        return mpDevice->createStructuredBuffer(
            4, 256 * 256 * neighborCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, arr.data(), false
        );
    }

    void ReSTIRPathTracing::scriptBindings(pybind11::module& m)
    {

    }
}
