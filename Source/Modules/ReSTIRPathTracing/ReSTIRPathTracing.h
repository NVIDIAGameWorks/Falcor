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
#pragma once
#include "Utils/Sampling/AliasTable.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Scene/Scene.h"
#include "Scene/Lights/LightCollection.h"
#include "Scene/Lights/Light.h"
#include "ReSTIRPT.slang"
#include "ReSTIRPathTracingParams.slang"
#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <vector>
#include "Utils/Algorithm/ParallelReduction.h"

namespace Falcor
{
    /** Implementation of ReSTIR PT Enhanced, with area reservoir support.
     * reference: "Generalized Resampled Importance Sampling: Foundations of ReSTIR"
     * "Area ReSTIR: Resampling for Real-Time Defocus and Antialiasing"
     * "Reservoir Splatting for Temporal Path Resampling and Motion Blur"
     * "ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing"
    */
    class ReSTIRPathTracing
    {
    public:
        using SharedPtr = std::shared_ptr<ReSTIRPathTracing>;

        /** Enumeration of available debug outputs.
            Note: Keep in sync with definition in Params.slang
        */
        enum class DebugOutput
        {
            Disabled,
            Position,
            Depth,
            Normal,
            FaceNormal,
            DiffuseWeight,
            SpecularWeight,
            SpecularRoughness,
            PackedNormal,
            PackedDepth,
            InitialWeight,
            TemporalReuse,
            SpatialReuse,
            FinalSampleDir,
            FinalSampleDistance,
            FinalSampleLi,
        };

        /** Configuration options.
        */
        struct Options
        {
            bool sampleDI = true; // RTXDI results will be overwritten if this is set to true

            // Temporal resampling options.

            bool useTemporalResampling = true;
            bool temporalUpdateForDynamicScene = false;
            ReSTIRPT::PointReprojectionMode pointReprojectionMode = ReSTIRPT::PointReprojectionMode::Nearest;
            ReSTIRPT::TemporalAreaReuseType temporalAreaReuseType = ReSTIRPT::TemporalAreaReuseType::Fast;

            bool enableSpecularMotionVectors = false;
            bool handleTransmissionInSpecMVec = false;
            float specularMotionVectorRoughnessThreshold = 0.1f;
            bool enableDisocclusionMotionVectors = false;
            uint32_t maxHistoryLength = 20;
            bool enableDupMapMCapDownweighting = false;
            bool computeTemporalDuplication = false;
            float duplicationMCapPower = 0.1f;
            bool useDoFReconnectionShift = true;
            bool useDoFReconnectionForTemporalReuse = true;
            float doFReconnectionShiftProbability = 0.3f;
            bool deriveDoFReconnectionShiftProbabilityFromCoC = true;

            bool useAreaReSTIRBackprojection() const
            {
                return useAreaReservoirs && temporalAreaReuseType != ReSTIRPT::TemporalAreaReuseType::Point;
            }
            bool temporalRobustShift()        const { return temporalAreaReuseType == ReSTIRPT::TemporalAreaReuseType::Robust; }
            bool useReservoirSplatting()      const { return temporalAreaReuseType == ReSTIRPT::TemporalAreaReuseType::Splat; }

            // Spatial resampling options.
            bool useSpatialResampling = true;           ///< Enable spatial resampling.
            uint32_t spatialIterations = 1;             ///< Number of spatial resampling iterations.
            uint32_t spatialNeighborCount = 3;          ///< Number of neighbor samples to resample per pixel and iteration.
            uint32_t spatialGatherRadius = 30;          ///< Radius to gather samples from.
            bool usePairedNeighbors = true;
            int spatialPairingStdevLevel = 5;
            bool areaReservoirUseNeighborRejection = false;


            // Options for ReSTIR PT.
            ReSTIRPT::ShiftMappingSettings shiftMappingSettings;

            uint32_t reservoirCountPerPixel = 1;                ///< Number of reservoirs per pixel.

            // static params
            ReSTIRPT::ShiftMapping shiftMapping = ReSTIRPT::ShiftMapping::Hybrid;

            bool useReservoirCompression = false;

            bool useAreaReservoirs = false;

            ReSTIRPT::RetraceScheduleType retraceScheduleType = ReSTIRPT::RetraceScheduleType::Compact;

            bool useDecoupledShading = false;

            // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
            Options() {}


            template<typename Archive>
            void serialize(Archive& ar)
            {
                ar("sampleDI", sampleDI);
                ar("useAreaReservoirs", useAreaReservoirs);
                ar("useTemporalResampling", useTemporalResampling);
                ar("maxHistoryLength", maxHistoryLength);
                ar("temporalAreaReuseType", temporalAreaReuseType);
                ar("pointReprojectionMode", pointReprojectionMode);
                ar("enableSpecularMotionVectors", enableSpecularMotionVectors);
                ar("handleTransmissionInSpecMVec", handleTransmissionInSpecMVec);
                ar("enableDisocclusionMotionVectors", enableDisocclusionMotionVectors);
                ar("useSpatialResampling", useSpatialResampling);
                ar("spatialIterations", spatialIterations);
                ar("spatialNeighborCount", spatialNeighborCount);
                ar("spatialGatherRadius", spatialGatherRadius);
                ar("shiftMappingSettings", shiftMappingSettings);
                ar("shiftMapping", shiftMapping);
                ar("useDoFReconnectionShift", useDoFReconnectionShift);
                ar("useDoFReconnectionForTemporalReuse", useDoFReconnectionForTemporalReuse);
                ar("deriveDoFReconnectionShiftProbabilityFromCoC", deriveDoFReconnectionShiftProbabilityFromCoC);
                ar("useDecoupledShading", useDecoupledShading);
                ar("usePairedNeighbors", usePairedNeighbors);
            }
        };

        // static params shared with path tracer
        struct SharedStaticParams
        {
            uint32_t    samplesPerPixel;                        ///< Number of samples (paths) per pixel, unless a sample density map is used.
            uint32_t    maxSurfaceBounces;                      ///< Max number of surface bounces (diffuse + specular + transmission), up to kMaxPathLenth. This will be initialized at startup.
            bool        useNEE;                              ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
        };

        /** Create a new instance of the ReSTIR sampler.
            \param[in] pScene Scene.
            \param[in] options Configuration options.
        */
        ReSTIRPathTracing(const ref<IScene>& pScene, const DefineList& ownerDefines, const Options& options);

        /** Get a list of shader defines for using the ReSTIR sampler.
            \return Returns a list of defines.
        */
        DefineList getDefines() const;

        /** Bind the ReSTIR sampler to a given shader var.
            \param[in] var The shader variable to set the data into.
        */
        void setShaderData(const ShaderVar& var) const;

        void setReservoirData(const ShaderVar& var) const;

        /** Render the GUI.
            \return True if options were changed, false otherwise.
        */
        bool renderUI(Gui::Widgets& widget);

        /** Returns the current configuration.
        */
        const Options& getOptions() const { return mOptions; }

        /** Set the configuration.
        */
        void setOptions(const Options& options);

        /** Begin a frame.
            Must be called once at the beginning of each frame.
            \param[in] pRenderContext Render context.
            \param[in] frameDim Current frame dimension.
        */
        void beginFrame(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles, bool needRecompile);

        /** End a frame.
            Must be called one at the end of each frame.
            \param[in] pRenderContext Render context.
        */
        void endFrame(RenderContext* pRenderContext);


        /** Run the ReSTIR PT algorithm.
            Must be called once between beginFrame() and endFrame().
            \param[in] pRenderContext Render context.
            \param[in] pMotionVectors Surface motion vectors for temporal reprojection.
            \param[in] vBuffer V-buffer texture.
            \param[in] pColorBuffer Output color texture.
        */
        void updateReSTIRPT(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const ref<Texture>& vBuffer, const ref<Texture>& pColorBuffer);

        void fetchDenoiserData(
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
        );

        void fillSampleID(RenderContext* pRenderContext);

        /** Get the debug output texture.
            \return Returns the debug output texture.
        */
        const ref<Texture>& getDebugOutputTexture() const { return mpDebugOutputTexture; }

        /** Get the pixel debug component.
            \return Returns the pixel debug component.
        */
        const std::unique_ptr<PixelDebug>& getPixelDebug() const { return mpPixelDebug; }

        /** Delta-reflection screen-space motion vector texture consumed by `ComputeReuseMotionVector`.
            Allocated when `enableSpecularMotionVectors` is enabled; the path tracer writes to it from
            the delta reflection guide trace pass. Returns nullptr when not allocated.
        */
        const ref<Texture>& getDeltaReflectionMotionVectors() const { return mpDeltaReflectionMotionVectors; }

        /** Delta-transmission screen-space motion vector texture consumed by `ComputeReuseMotionVector`.
            Allocated when `enableSpecularMotionVectors` is enabled; the path tracer writes to it from
            the delta transmission guide trace pass. Returns nullptr when not allocated.
        */
        const ref<Texture>& getDeltaTransmissionMotionVectors() const { return mpDeltaTransmissionMotionVectors; }

        /** Register script bindings.
        */
        static void scriptBindings(pybind11::module& m);

        void setPathTracerParams(ReSTIRPathTracingParams params);

        void setInitialLightSampleCount(uint32_t initialLightSampleCount)
        {
            exInitialLightSampleCount = initialLightSampleCount;
        }

        void setOwnerDefines(DefineList defines);

        void setSharedStaticParams(uint32_t samplesPerPixel, uint32_t maxSurfaceBounces, bool useNEE);

        void createPathTracerBlock();

        ref<ParameterBlock> getPathTracerBlock();

        void updatePrograms();

        void setShaderVarForDuplicationMap(const ShaderVar& var)
        {
            var["duplicationMap"] = mpDuplicationMap;
            var["sampleSurvivedFrames"] = mpTemporalCorrelationMap;
        }

        const ref<Texture>& getVBuffer(const ref<Texture>& pInputVBuffer, bool usePredefinedJitterPattern = false) const
        {
            return (mOptions.useAreaReservoirs && !usePredefinedJitterPattern) ? mpRandomVBuffer : pInputVBuffer;
        }

    private:
        void createComputePass(ref<ComputePass>& pPass, std::string shaderFile, DefineList defines, ProgramDesc desc, std::string entryFunction = "main");

        void prepareResources(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles);
        void prepareReservoirAndPathBuffers(uint32_t elementCount, uint32_t pathCount, bool needFloatingReservoirs);
        void prepareDuplicationMapResources(const uint2& frameDim);
        void prepareVBufferTextures(const uint2& frameDim);
        void prepareReservoirSplattingResources(const uint2& frameDim, uint32_t temporalPathCount);
        void prepareMotionVectorTextures(const uint2& frameDim);

        void compileTemporalPasses(const DefineList& defines, const ProgramDesc& baseDesc);
        void compileSpatialPasses(const DefineList& defines, const ProgramDesc& baseDesc);
        void compileAreaReservoirPasses(const DefineList& defines, const ProgramDesc& baseDesc);
        void compileRobustShiftPasses(const DefineList& defines, const ProgramDesc& baseDesc);
        void compileReservoirSplattingPasses(const DefineList& defines, const ProgramDesc& baseDesc);

        void temporalResampling(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const ref<Texture>& pVBuffer);
        void temporalDispatchComputeReuseMotionVector(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors);
        void temporalDispatchRobustChain(RenderContext* pRenderContext);
        void temporalDispatchPopulateFloatingReservoir(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors);
        void temporalDispatchSplattingChain(RenderContext* pRenderContext);
        void temporalDispatchTraceShiftedPrimaryHits(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, int temporalPassId);
        void temporalDispatchGenerateRetraceWorkload(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, int temporalPassId);
        void temporalDispatchRetrace(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, int temporalPassId);
        void temporalDispatchResamplingPass(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, int temporalPassId);

        void spatialResampling(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const ref<Texture>& pColorBuffer);
        void spatialDispatchTraceShiftedPrimaryHits(RenderContext* pRenderContext, Falcor::ShaderVar shiftVar, uint32_t iteration);
        void spatialDispatchGenerateRetraceWorkload(RenderContext* pRenderContext, Falcor::ShaderVar workLoadVar, uint32_t iteration);
        void spatialDispatchRetrace(RenderContext* pRenderContext, Falcor::ShaderVar retraceVar, uint32_t iteration);
        void spatialDispatchPrecomputeShiftBuffer(RenderContext* pRenderContext, Falcor::ShaderVar precomputeVar, uint32_t iteration);
        void spatialDispatchResamplingPass(RenderContext* pRenderContext, Falcor::ShaderVar var, uint32_t iteration, const ref<Texture>& pColorBuffer);

        Falcor::ShaderVar bindCommonPassVars(RenderContext* pRenderContext, ref<ComputePass> pPass, const std::string& cbName, bool bindPathTracer);
        Falcor::ShaderVar bindTemporalRobustVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, bool bindPathTracer);
        Falcor::ShaderVar bindSpatialResamplingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, bool bindPathTracer, bool isShiftVBufferPass=false);
        Falcor::ShaderVar bindTemporalResamplingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, bool bindPathTracer, bool isShiftVBufferPass=false);
        Falcor::ShaderVar bindReservoirSplattingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName);

        /** Create or release a structured buffer that mirrors `reflectVarName` from the reflect shader.
            Honors `mReallocate` (rebuilds the buffer when sizes change). When `keepCondition` is
            false, the buffer is released and set to nullptr.
        */
        void createOrDestroyBuffer(ref<Buffer>& pBuffer, std::string_view reflectVarName, int requiredElementCount, bool keepCondition = true);

        void createOrDestroyRawBuffer(ref<Buffer>& pBuffer, size_t requiredSize, bool keepCondition = true);
        void createTexture2D(ref<Texture>& pTexture, uint32_t width, uint32_t height, ResourceFormat format, bool keepCondition = true);

        /** Create a 1D texture with random offsets within a unit circle around (0,0).
            The texture is RG8Snorm for compactness and has no mip maps.
            \param[in] sampleCount Number of samples in the offset texture.
        */
        ref<Texture> createNeighborOffsetTexture(uint32_t sampleCount);
        ref<Buffer> createPairedNeighborDeltasTexture(ref<Device> pDevice, uint32_t neighborCount, int stdevLevel);

        // -------- Predicate helpers --------

        bool useCompactRetraceSchedule() const { return mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact; }

        /// True when a custom `mpMVecBuffer` motion vector pass needs to run before temporal reuse.
        bool useCustomMotionVectorPass() const
        {
            return mOptions.enableSpecularMotionVectors
                || mOptions.enableDisocclusionMotionVectors;
        }

        bool useRobustTemporalAreaReuse() const
        {
            return mOptions.useAreaReSTIRBackprojection() && mOptions.temporalRobustShift();
        }

        bool useFloatingReservoirs() const
        {
            if (mOptions.useReservoirSplatting() || !mOptions.useAreaReservoirs) return false;
            if (mOptions.useAreaReSTIRBackprojection()) return true;
            return false;
        }

        bool useReconnectionDataBuffers() const
        {
            return mOptions.temporalRobustShift() || mOptions.shiftMapping == ReSTIRPT::ShiftMapping::Hybrid;
        }

        bool useDuplicationMap() const
        {
            return mOptions.enableDupMapMCapDownweighting || mOptions.computeTemporalDuplication;
        }

        bool useDoFReconnectionForTemporalReuseEffective() const
        {
            return mOptions.useDoFReconnectionForTemporalReuse
                && mpScene->getCamera()->getApertureRadius() > 0.f
                && !mOptions.useReservoirSplatting();
        }



        ref<IScene> mpScene;                                ///< Scene.
        ref<Device>                         mpDevice; ///< GPU device.
        Options mOptions;                                   ///< Configuration options.
        SharedStaticParams mStaticParams;

        ReSTIRPathTracingParams          mPathTracerParams;

        DefineList mOwnerDefines;                   ///< Share defines with inline path tracer

        std::unique_ptr<PixelDebug> mpPixelDebug;                 ///< Pixel debug component.

        uint2 mFrameDim = uint2(0);                         ///< Current frame dimensions.
        uint32_t mFrameIndex = 0;                           ///< Current frame index.

        ref<ComputePass> mpReflectTypes;              ///< Pass for reflecting types.

        // ReSTIR PT passes.

        ref<ComputePass> mpTemporalResampling;        
        ref<ComputePass> mpTemporalRetrace;           

        ref<ComputePass> mpSpatialResampling;       
        ref<ComputePass> mpSpatialPrecomputeShiftBufferPass;
        ref<ComputePass> mpSpatialRetrace;            
        ref<ComputePass> mpGenerateRetraceWorkload;

        ref<ComputePass> mpFetchDenoiserData;
        ref<ComputePass> mpFillSampleId;
        ref<ComputePass> mpComputeDuplicationMap;

        ref<ComputePass> mpTemporalTraceShiftedPrimaryHits;
        ref<ComputePass> mpSpatialTraceShiftedPrimaryHits;
        ref<ComputePass> mpTemporalPopulateFloatingReservoir;

        ref<ComputePass> mpTemporalRobustCreateShiftVBuffer;
        ref<ComputePass> mpTemporalRobustPathRetrace;
        ref<ComputePass> mpTemporalRobustProduceWorkload;
        ref<ComputePass> mpTemporalRobustResampling;

        ref<ComputePass> mpComputeReuseMotionVector;

        // reservoir splatting
        ref<ComputePass> mpTemporalSplatSamples;
        ref<ComputePass> mpTemporalComputeCellOffsets;
        ref<ComputePass> mpTemporalSortCellData;

        ref<ParameterBlock>       mpPathTracerBlock;          ///< Parameter block for the path tracer.

        ref<Buffer> mpReservoirs;                     ///< Buffer containing the current reservoirs.
        ref<Buffer> mpPrevReservoirs;                 ///< Buffer containing the previous reservoirs.
        ref<Buffer> mpFloatingReservoirs;
        ref<Buffer> mpReconnectionDataBuffer;          ///< Buffer containing the reconnection data for retrace result.
        ref<Buffer> mpReconnectionDataOffsets;          ///< Buffer containing the reconnection data for retrace result.
        ref<Buffer> mpPrecomputedShiftDataBuffer;
        ref<Buffer> mpNumKeysBuffer;
        ref<Buffer> mpNeighborValidMaskBuffer;
        ref<Buffer>               mpRetraceWorkItems;             ///< Paths starting from primary hits on general materials (all types).
        ref<Buffer>               mpCounters;                 ///< Atomic counters (32-bit).

        ref<Texture> mpRandomVBuffer;          ///< Randomized V-buffer for area reservoirs.
        ref<Texture> mpTemporalRandomVBuffer;
        // for reservoir splatting
        ref<Buffer> mpReservoirSplatInfo;
        ref<Buffer> mpReservoirSplatPixel;
        ref<Buffer> mpSortedScatteringPixels;
        ref<Texture> mpMappingIndices;
        ref<Texture> mpCellCounters;
        ref<Texture> mpCellOffsets;
        ref<Buffer>  mpCellOffsetCounter;

        ref<Texture> mpDebugOutputTexture;            ///< Debug output texture.
        ref<Texture> mpNeighborOffsets;               ///< 1D texture containing neighbor offsets within a unit circle.
        ref<Buffer> mpPairedNeighborDeltas;
        ref<Buffer> mpPairedNeighborTextureSizes;


        // related to sample duplication map
        ref<Texture> mpSampleIDTexture;
        ref<Texture> mpDuplicationMap;
        ref<Texture> mpTemporalCorrelationMap;
        ref<Texture> mpPrevTemporalCorrelationMap;

        ref<Texture> mpTemporalVBuffer;

        ref<Buffer> mpShiftedPrimaryHitBuffer;
        ref<Buffer> mpShiftedPrimaryHitBufferRobust;
        ref<Buffer> mpDoFReconnectionShiftBuffer;
        ref<Buffer> mpTemporalMISPDFs;
        ref<Texture> mpMVecBuffer;
        /// Screen-space delta reflection/transmission motion vectors consumed by `ComputeReuseMotionVector`.
        /// Allocated when `enableSpecularMotionVectors` is enabled; the consumer's delta-path guide trace passes write to them.
        ref<Texture> mpDeltaReflectionMotionVectors;
        ref<Texture> mpDeltaTransmissionMotionVectors;

        bool mRecompile = true;                             ///< Recompile programs on next frame if set to true.
        bool mReallocate = true;                            ///< Reallocate the reservoirs since sizes change
        bool mResetTemporalReservoirs = true;               ///< Reset temporal reservoir buffer on next frame if set to true.
        uint32_t exInitialLightSampleCount = 1;
    };
}
