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
#include "Falcor.h"

#include "ReSTIR_FPDG.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"

#include "Rendering/RTXDI/RTXDI.h"
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Rendering/AccelerationStructure/CustomAccelerationStructure.h"

using namespace Falcor;

class ReSTIR_FPDG : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIR_FPDG, "ReSTIR_FPDG", "Insert pass description here.");

    static ref<ReSTIR_FPDG> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<ReSTIR_FPDG>(pDevice, props);
    }

    ReSTIR_FPDG(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

public:
    void beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame();

private:
    void updatePrograms();

    void setupLights(RenderContext* pRenderContext);
    void setupResources(RenderContext* pRenderContext, const RenderData& renderData);

    // Handles the photon acceleration structure
    void setupPhotonDifferentialsAS();

    // Traces the photons and builds the photon Acceleration Structure
    void tracePhotonDifferentialsPass(RenderContext* pRenderContext, const RenderData& renderData, bool analyticOnly = false, bool buildAS = true);

    // Handles readback of the photon counter and adjusts dispatched photons dynamically for the next trace photons pass
    void handlePhotonCounter(RenderContext* pRenderContext);

    // Generates the initial Samples for the reservoirs (FG) and initializes RTXDI surfaces
    void generateInitialSamplesPass(RenderContext* pRenderContext, const RenderData& renderData);

    // Reservoir Resampling for Final Gather Samples
    void resampleReservoirFGPass(RenderContext* pRenderContext, const RenderData& renderData);

    // Reservoir Resampling for Caustic Samples
    void resampleReservoirCausticPass(RenderContext* pRenderContext, const RenderData& renderData);

    // Evaluate Reservoirs
    void evaluateReservoirsPass(RenderContext* pRenderContext, const RenderData& renderData);

    DefineList getMaterialDefines();

private:
    struct ResamplingSettings
    {
        bool enable = true;
        uint confidenceCap = 20;                // Maximum confidence allowed
        uint spatialSamples = 1;                // Number of spatial samples
        uint disocclusionBoostExtraSamples = 1; // Number of spatial samples if no temporal surface was found
        float samplingRadius = 20.f;            // Sampling radius in pixel
    };

    enum class DirectLightingMode
    {
        None = 0u,
        RTXDI = 1u,
        AnalyticDirect = 2u,
    };

    enum class ResamplingMode
    {
        Temporal = 0u,
        Spatial = 1u,
        SpatioTemporal = 2u,
    };

    enum class RenderMode
    {
        FinalGather = 0,
        ReSTIRFG = 1u,
        ReSTIRGI = 2u
    };

    enum PhotonType
    {
        GLOBAL = 0u,
        CAUSTIC,
        MAX
    };

    //
    // Parameters
    //

    RenderMode mRenderMode = RenderMode::ReSTIRFG;
    DirectLightingMode mDirectLightMode = DirectLightingMode::RTXDI;
    ResamplingMode mResamplingMode = ResamplingMode::SpatioTemporal;

    uint mFrameCount = 0u;
    uint2 mScreenRes = uint2(0u, 0u);

    float mSpecularRoughnessThreshold = 0.25f; // Any material below this is considered specular

    // ReSTIR-FG Reservoirs. See ReSTIR-FG paper
    ResamplingSettings mResampleSettingsFG = {};
    ResamplingSettings mResampleSettingsCaustic = {};

    uint mFGRayMaxPathLength = 10;                      // Max path length for the final gather ray
    bool mRebuildReservoirBuffer = false;               // Rebuild the reservoir buffer
    bool mClearReservoir = true;                        // Clears both reservoirs
    bool mCanResample = false;                          // Resampling is only allowed if last iterations reservoir was created
    float mRelativeDepthThreshold = 0.15f;              // Relative Depth threshold (is neighbor 0.1 = 10% as near as the current depth)
    float mNormalThreshold = 0.906f;// 0.6f was before  // Cosine of maximum angle between both normals allowed (not more than 25 degrees)
    float mJacobianDistanceThreshold = 0.001f;          // Threshold for Jacobian distances
    bool mUsePathThreshold = false;                     // Enable resampling only if path length are the same
    bool mUsePhotonsForDirectLightInReflections = true; // Uses photons for direct light in reflections, else the final gather sample is used

    // Photon Distribution
    uint mPhotonMaxBounces = 4u;                       // Number of photon bounces
    float mGlobalPhotonRejection = 0.3f;                // Probability a global photon is stored
    uint mNumDispatchedPhotons = 2000000u;              // Number of photons dispatched
    uint2 mNumMaxPhotons = uint2(400000u, 300000u);     // Size of the photon buffer
    uint2 mNumMaxPhotonsUI = mNumMaxPhotons;            // For UI, as changing happens with a button
    bool mChangePhotonLightBufferSize = true;           // True if buffer size has changed
    uint2 mCurrentPhotonCount = mNumMaxPhotons;         // For dynamic photon propagation
    float mASBuildBufferPhotonOverestimate = 1.15f;     // Guard percentage for AS building
    float2 mPhotonRadius = float2(0.020f, 0.005f);      // Global/Caustic Radius.
    float mPhotonAnalyticRatio = 0.5f;                  // Analytic photon distribution ratio in a mixed light case. E.g. 0.3 -> 30% analytic, 70% emissive

    bool mUseDynamicPhotonDispatchCount = true;         // Dynamically change the number of photons to fit the max photon number
    uint mPhotonDynamicDispatchMax = 4000000u;          // Max value for dynamically dispatched photons
    float mPhotonDynamicGuardPercentage = 0.08f;        // Determines how much space of the buffer is used to guard against buffer overflows
    float mPhotonDynamicChangePercentage = 0.04f;       // The percentage the buffer is increased/decreased per frame

    ref<Scene> mpScene;                                                                 // Scene ptr
    ref<SampleGenerator> mpSampleGenerator;                                             // GPU Sample Gen
    std::unique_ptr<EmissiveLightSampler> mpEmissiveLightSampler;                       // Light sampler
    EmissiveLightSamplerType mEmissiveSamplerType = EmissiveLightSamplerType::Power;
    std::unique_ptr<CustomAccelerationStructure> mpPhotonDifferentialAS;                // Accel Pointer. Custom type, better use intrinsic Falcor tools
    std::unique_ptr<RTXDI> mpRTXDI;                                                     // Ptr to RTXDI for direct use
    RTXDI::Options mRTXDIOptions;                                                       // Options for RTXDI

    bool mResetScreenRes = false;
    bool mRecompile = false;
    bool mOptionsChanged = false;

    // Material Settings
    bool mUseLambertianDiffuse = true; // Diffuse BRDF used by ReSTIR PT and SuffixReSTIR
    bool mDisableDiffuse = false;
    bool mDisableSpecular = false;
    bool mDisableTranslucency = false;
    bool mStoreSampleGenState = false; // Stores samples GenStates

    //
    // Lights
    //
    bool mHasLights = false;
    bool mHasEmissiveLights = false;
    bool mHasAnalyticLights = false;
    bool mHasMixedLights = false;
    bool mHasEnvBackground = false;

    //
    // Resources
    //
    // IMPORTANT: Keep track of shader structs size when creating resources on the CPU side
    ref<Buffer> mpPhotonAABB[2];           // Photon AABBs for Acceleration Structure building
    ref<Buffer> mpPhotonData[2];           // Additional Photon data (flux, dir)
    ref<Buffer> mpPhotonCounter;           // Photon Counter
    ref<Buffer> mpPhotonCounterCPU;        // CPU copy of counter for readback
    ref<Buffer> mpFinalGatherReservoir[2]; // Reservoir for the Final Gather sample
    ref<Buffer> mpCausticReservoir[2];     // Reservoir for the Caustic sample
    ref<Texture> mpEmission;               // Emission for paths that travel through highly specular materials

    struct RaytracingProgramHelper
    {
        ref<Program> pProgram = nullptr;
        ref<RtBindingTable> pBindingTable = nullptr;
        ref<RtProgramVars> pVars = nullptr;

        static const RaytracingProgramHelper create()
        {
            RaytracingProgramHelper r;
            r.pProgram = nullptr;
            r.pBindingTable = nullptr;
            r.pVars = nullptr;
            return r;
        }

        void initProgramVars(ref<Device>& pDevice, ref<Scene>& pScene, ref<SampleGenerator>& pSampleGenerator);
    };

    RaytracingProgramHelper mTracePhotonDifferentialsPass;
    RaytracingProgramHelper mGenerateInitialSamplesPass;

    ref<ComputePass> mpResamplePhotonDifferentialsReservoirFGPass;
    ref<ComputePass> mpResamplePhotonDifferentialsReservoirCausticPass;
    ref<ComputePass> mpEvaluatePhotonDifferentialsReservoirsPass;
};
