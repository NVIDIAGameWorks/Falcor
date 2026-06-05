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
#include "ReSTIRFGPass.h"

#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
const std::string kShaderFolder = "RenderPasses/ReSTIRFGPass/";
const std::string kShaderTracePhotons = kShaderFolder + "TracePhotons.rt.slang";
const std::string kShaderGenInitialSamples = kShaderFolder + "GenerateInitialSamples.rt.slang";
const std::string kShaderResamplingReservoirFG = kShaderFolder + "ResampleReservoirFG.cs.slang";
const std::string kShaderResamplingReservoirCaustic = kShaderFolder + "ResampleReservoirCaustic.cs.slang";
const std::string kShaderEvaluateReservoirs = kShaderFolder + "EvaluateReservoirs.cs.slang";

const ShaderModel kShaderModel = ShaderModel::SM6_5;

// Render Pass inputs and outputs
const std::string kInputVBuffer = "vbuffer";
const std::string kInputMotionVectors = "mvec";

const Falcor::ChannelList kInputChannels{
    {kInputVBuffer, "gVBuffer", "Visibility buffer in packed format"},
    {kInputMotionVectors, "gMotionVectors", "Motion vector buffer (float format)", true /* optional */},
};

// Outputs
const std::string kOutputColor = "color";

const Falcor::ChannelList kOutputChannels{{kOutputColor, "gOutColor", "HDR output color", false /*optional*/, ResourceFormat::RGBA32Float}};

}; // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRFGPass>();
}

ReSTIRFGPass::ReSTIRFGPass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
    {
        throw RuntimeError("ReSTIR_FG: Shader Model 6.5 is not supported by the current device");
    }
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw RuntimeError("ReSTIR_FG: Raytracing Tier 1.1 is not supported by the current device");
    }

    // Caustic Reservoir default values
    mResampleSettingsCaustic.spatialSamples = 0;
    mResampleSettingsCaustic.disocclusionBoostExtraSamples = 1;
    mResampleSettingsCaustic.samplingRadius = 4.f;

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
}

Properties ReSTIRFGPass::getProperties() const
{
    return {};
}

RenderPassReflection ReSTIRFGPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void ReSTIRFGPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    // auto& pTexture = renderData.getTexture("src");

    if (!mpScene) return;

    // Add refresh flag if options changed
    auto& dict = renderData.getDictionary();
    auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
    if (mOptionsChanged)
    {
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // Init ReSTIR DI
    const auto& pMotionVectors = renderData[kInputMotionVectors]->asTexture();
    if (!mpRTXDI) mpRTXDI = std::make_unique<RTXDI>(mpScene, mRTXDIOptions);

    setupLights(pRenderContext);
    if (!mHasLights)
    {
        logWarningOnce("Scene has no lights, pass will not execute!");
        return;
    }

    setupResources(pRenderContext, renderData);

    setupPhotonAS(pRenderContext);

    if (!mpRTXDI) mpRTXDI = std::make_unique<RTXDI>(mpScene, mRTXDIOptions);

    mpRTXDI->beginFrame(pRenderContext, mScreenRes);

    // Clear Photon Counter before tracing the Photons for this frame
    pRenderContext->clearUAV(mpPhotonCounter->getUAV().get(), uint4(0));

    // Trace Photons. Up to two passes may be executed, depending on the light types in the scene
    //(one for emissive triangles and one for analytic point/spot lights)
    tracePhotonsPass(pRenderContext, renderData, !mMixedLights && mHasAnalyticLights, !mMixedLights && mPhotonAnalyticRatio > 0);
    if (mMixedLights && mPhotonAnalyticRatio > 0)
        tracePhotonsPass(pRenderContext, renderData, true); // Second pass. Always Analytic

    // Initial Samples for ReSTIR FG (1SPP Photon Final Gathering) and inti RTXDI structs
    generateInitialSamplesPass(pRenderContext, renderData);

    // ReSTIR DI pass
    mpRTXDI->update(pRenderContext, pMotionVectors);

    // Spatiotemporal resampling for final gather samples and caustics
    resampleReservoirFGPass(pRenderContext, renderData);

    resampleReservoirCausticPass(pRenderContext, renderData);

    // Finalize Reservoirs
    evaluateReservoirsPass(pRenderContext, renderData);

    mpRTXDI->endFrame(pRenderContext);

    mFrameCount++;
    mCanResample = true;
}

void ReSTIRFGPass::setupLights(RenderContext* pRenderContext)
{
    // Make sure that the emissive light is up to date
    auto& pLights = mpScene->getLightCollection(pRenderContext);
    pLights->prepareSyncCPUData(pRenderContext);

    bool emissiveUsed = mpScene->useEmissiveLights();
    bool analyticUsed = mpScene->useAnalyticLights();

    mHasLights = analyticUsed || emissiveUsed;
    mHasAnalyticLights = analyticUsed;
    mMixedLights = emissiveUsed && analyticUsed;

    if (emissiveUsed)
    {
        if (!mpEmissiveLightSampler)
        {
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            mpEmissiveLightSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, pLights);
        }
    }
    else
    {
        if (mpEmissiveLightSampler)
        {
            mpEmissiveLightSampler = nullptr;
            mTracePhotonPass.pVars.reset();
        }
    }
    if (mpEmissiveLightSampler)
    {
        mpEmissiveLightSampler->update(pRenderContext, pLights);
    }
}

void ReSTIRFGPass::setupResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Reset buffers when screen resolution or certain options changed
    auto& screenDims = renderData.getDefaultTextureDims();
    if (screenDims.x != mScreenRes.x || screenDims.y != mScreenRes.y)
    {
        mScreenRes = screenDims;
        mResetScreenTex = true;
    }

    if (mChangePhotonLightBufferSize)
    {
        mNumMaxPhotons = mNumMaxPhotonsUI;
        mpPhotonAABB[0].reset();
        mpPhotonAABB[1].reset();
        mpPhotonData[0].reset();
        mpPhotonData[1].reset();
        // Flag will be reset in preparePhotonAccelerationStructure()
    }

    // Buffers that exist two times
    for (uint i = 0; i < 2; i++)
    {
        if (!mpPhotonAABB[i])
        {
            mpPhotonAABB[i] = mpDevice->createStructuredBuffer(sizeof(AABB), mNumMaxPhotons[i], 
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mpPhotonAABB[i]->setName("PhotonAABB" + std::to_string(i));
        }
        if (!mpPhotonData[i])
        {
            mpPhotonData[i] = mpDevice->createStructuredBuffer(sizeof(float) * 12, mNumMaxPhotons[i],
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mpPhotonData[i]->setName("PhotonData" + std::to_string(i));
        }
        if (!mpFinalGatherReservoir[i] || mResetScreenTex)
        {
            mCanResample = false;
            mpFinalGatherReservoir[i] = mpDevice->createStructuredBuffer(112u, mScreenRes.x * mScreenRes.y,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mpFinalGatherReservoir[i]->setName("FinalGatherReservoir" + std::to_string(i));
        }
        if (!mpCausticReservoir[i] || mResetScreenTex)
        {
            mCanResample = false;
            mpCausticReservoir[i] = mpDevice->createStructuredBuffer(112u, mScreenRes.x * mScreenRes.y,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mpCausticReservoir[i]->setName("CausticReservoir" + std::to_string(i));
        }
    }

    // Photon Counters
    if (!mpPhotonCounter)
    {
        mpPhotonCounter = mpDevice->createStructuredBuffer(sizeof(uint), 2, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal, nullptr, false);
        mpPhotonCounter->setName("PhotonCounter");

        mpPhotonCounterCPU = mpDevice->createStructuredBuffer(sizeof(uint), 2, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);
        mpPhotonCounterCPU->setName("PhotonCounterCPU");
    }

    // Emission Texture
    if (!mpEmission || mResetScreenTex)
    {
        mpEmission = mpDevice->createTexture2D(mScreenRes.x, mScreenRes.y, ResourceFormat::RGBA32Float,
            1u, 1u, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );
        mpEmission->setName("EmissionTexture");
    }

    mResetScreenTex = false;
}

void ReSTIRFGPass::setupPhotonAS(RenderContext* pRenderContext)
{
    // Delete the Photon AS if max Buffer size changes
    if (mChangePhotonLightBufferSize)
    {
        mpPhotonAS.reset();
        mChangePhotonLightBufferSize = false;
    }

    // Create the Photon Acceleration Structure
    if (!mpPhotonAS)
    {
        std::vector<uint64_t> aabbCount = {mNumMaxPhotons[0], mNumMaxPhotons[1]};
        std::vector<uint64_t> aabbGPUAddress = {mpPhotonAABB[0]->getGpuAddress(), mpPhotonAABB[1]->getGpuAddress()};
        mpPhotonAS = std::make_unique<CustomAccelerationStructure>(
            mpDevice, aabbCount, aabbGPUAddress, CustomAccelerationStructure::BuildMode::FastBuild, CustomAccelerationStructure::UpdateMode::TLASOnly
        );
    }
}

void ReSTIRFGPass::tracePhotonsPass(RenderContext* pRenderContext, const RenderData& renderData, bool analyticOnly, bool buildAS)
{
    FALCOR_PROFILE(pRenderContext, "TracePhotons");

    // Init Shader
    if (!mTracePhotonPass.pProgram)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderTracePhotons);
        desc.setMaxPayloadSize(sizeof(float) * 4);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(1);
        if (!mpScene->hasProceduralGeometry())
            desc.setRtPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

        mTracePhotonPass.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& sbt = mTracePhotonPass.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen", mpScene->getTypeConformances()));
        sbt->setMiss(0, desc.addMiss("miss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        }
        DefineList defines;
        defines.add("USE_EMISSIVE_LIGHT", mpScene->useEmissiveLights() ? "1" : "0");
        defines.add(mpScene->getSceneDefines());

        mTracePhotonPass.pProgram = Program::create(mpDevice, desc, defines);
    }
    // Defines
    mTracePhotonPass.pProgram->addDefine("PHOTON_BUFFER_SIZE_GLOBAL", std::to_string(mNumMaxPhotons[0]));
    mTracePhotonPass.pProgram->addDefine("PHOTON_BUFFER_SIZE_CAUSTIC", std::to_string(mNumMaxPhotons[1]));
    mTracePhotonPass.pProgram->addDefine("ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
    mTracePhotonPass.pProgram->addDefines(getMaterialDefines());
    if (mpEmissiveLightSampler)
        mTracePhotonPass.pProgram->addDefines(mpEmissiveLightSampler->getDefines());

    // Program Vars
    if (!mTracePhotonPass.pVars)
        mTracePhotonPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);

    FALCOR_ASSERT(mTracePhotonPass.pVars);
    auto var = mTracePhotonPass.pVars->getRootVar();
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    // Handle shader dimension
    uint dispatchedPhotons = mNumDispatchedPhotons;
    if (mMixedLights)
    {
        float dispatchedF = float(dispatchedPhotons);
        dispatchedF *= analyticOnly ? mPhotonAnalyticRatio : 1.f - mPhotonAnalyticRatio;
        dispatchedPhotons = uint(dispatchedF);
    }
    uint shaderDispatchDim = static_cast<uint>(std::floor(sqrt(dispatchedPhotons)));
    shaderDispatchDim = std::max(32u, shaderDispatchDim);

    // Constant Buffer
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPhotonRadius"] = mPhotonRadius;
    var["CB"]["gMaxBounces"] = mPhotonMaxBounces;
    var["CB"]["gGlobalRejectionProb"] = mGlobalPhotonRejection;
    var["CB"]["gUseAnalyticLights"] = analyticOnly;
    var["CB"]["gDispatchDimension"] = shaderDispatchDim;

    // Structures
    if (mpEmissiveLightSampler)
        mpEmissiveLightSampler->bindShaderData(var["Light"]["gEmissiveSampler"]);

    // Output Buffers
    for (uint32_t i = 0; i < 2; i++)
    {
        var["gPhotonAABB"][i] = mpPhotonAABB[i];
        var["gPhotonData"][i] = mpPhotonData[i];
    }
    var["gPhotonCounter"] = mpPhotonCounter;

    // Dispatch raytracing shader
    mpScene->raytrace(
        pRenderContext, mTracePhotonPass.pProgram.get(), mTracePhotonPass.pVars, uint3(shaderDispatchDim, shaderDispatchDim, 1)
    );

    // If two passes are dispatched, the acceleration structure is build on the second dispatch
    if (buildAS)
    {
        // Clear values after the counter
        std::vector<ref<Buffer>> aabbs = {mpPhotonAABB[0], mpPhotonAABB[1]};
        mpPhotonAS->clearAABBBuffers(pRenderContext, aabbs, true, mpPhotonCounter);

        // Copy counter to CPU
        handlePhotonCounter(pRenderContext);

        // Build acceleration structure
        uint2 currentPhotons = mFrameCount > 0 ? uint2(float2(mCurrentPhotonCount) * mASBuildBufferPhotonOverestimate) : mNumMaxPhotons;
        std::vector<uint64_t> photonBuildSize = {
            std::min(mNumMaxPhotons[0], currentPhotons[0]), std::min(mNumMaxPhotons[1], currentPhotons[1])
        };
        mpPhotonAS->update(pRenderContext, photonBuildSize);
    }
}

void ReSTIRFGPass::handlePhotonCounter(RenderContext* pRenderContext)
{
    // Copy the photonCounter to a CPU Buffer (asynchronous, read GPU value can be a couple of frames old)
    pRenderContext->copyBufferRegion(mpPhotonCounterCPU.get(), 0, mpPhotonCounter.get(), 0, sizeof(uint2));

    void* data = mpPhotonCounterCPU->map(/*Buffer::MapType::Read*/);
    std::memcpy(&mCurrentPhotonCount, data, sizeof(uint2));
    mpPhotonCounterCPU->unmap();

    // Change Photon dispatch count dynamically.
    if (mUseDynamicPhotonDispatchCount)
    {
        // Only use global photons for the dynamic dispatch count
        uint globalPhotonCount = mCurrentPhotonCount[0];
        uint globalMaxPhotons = mNumMaxPhotons[0];
        uint causticPhotonCount = mCurrentPhotonCount[1];
        uint causticMaxPhotons = mNumMaxPhotons[1];
        // If counter is invalid, reset
        if (globalPhotonCount == 0)
        {
            mNumDispatchedPhotons = mPhotonDynamicDispatchMax / 2;
        }
        uint globBufferSizeCompValue = (uint)(globalMaxPhotons * (1.f - mPhotonDynamicGuardPercentage));
        uint globChangeSize = (uint)(globalMaxPhotons * mPhotonDynamicChangePercentage);
        uint causticBufferSizeCompValue = (uint)(causticMaxPhotons * (1.f - mPhotonDynamicGuardPercentage));
        uint causticChangeSize = (uint)(causticMaxPhotons * mPhotonDynamicChangePercentage);
        uint changeSize = std::max(globChangeSize, causticChangeSize);

        // If smaller, increase dispatch size
        if ((globalPhotonCount < globBufferSizeCompValue) && (causticPhotonCount < causticBufferSizeCompValue))
        {
            uint newDispatched = (uint)(mNumDispatchedPhotons + changeSize);
            mNumDispatchedPhotons = std::min(newDispatched, mPhotonDynamicDispatchMax);
        }
        // Reduce dispatch size
        else
        {
            uint newDispatched = (uint)(mNumDispatchedPhotons - changeSize);
            mNumDispatchedPhotons = std::max(newDispatched, 1024u);
        }
    }
}

void ReSTIRFGPass::generateInitialSamplesPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "InitialSamples");

    // Init Shader
    if (!mGenerateInitialSamplesPass.pProgram)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderGenInitialSamples);
        desc.setMaxPayloadSize(sizeof(float) * 4);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(1);
        if (!mpScene->hasProceduralGeometry())
            desc.setRtPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

        mGenerateInitialSamplesPass.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& sbt = mGenerateInitialSamplesPass.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen", mpScene->getTypeConformances()));
        sbt->setMiss(0, desc.addMiss("miss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        }

        mGenerateInitialSamplesPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
    }

    // Defines that can change on runtime
    mGenerateInitialSamplesPass.pProgram->addDefines(mpRTXDI->getDefines());
    mGenerateInitialSamplesPass.pProgram->addDefine("ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
    mGenerateInitialSamplesPass.pProgram->addDefines(getMaterialDefines());

    // Program Vars
    if (!mGenerateInitialSamplesPass.pVars)
        mGenerateInitialSamplesPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);

    FALCOR_ASSERT(mGenerateInitialSamplesPass.pVars);
    auto var = mGenerateInitialSamplesPass.pVars->getRootVar();

    // Constant Buffer
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFGRayMaxPathLength"] = mFGRayMaxPathLength;

    // RTXDI Resources
    mpRTXDI->bindShaderData(var);

    // Input Resources
    var["gVBuffer"] = renderData[kInputVBuffer]->asTexture();
    mpPhotonAS->bindTlas(var, "gPhotonAS");
    for (uint32_t i = 0; i < 2; i++)
    {
        var["gPhotonAABB"][i] = mpPhotonAABB[i];
        var["gPhotonData"][i] = mpPhotonData[i];
    }

    // Output Resources
    var["gFinalGatherReservoir"] = mpFinalGatherReservoir[mFrameCount % 2];
    var["gCausticReservoir"] = mpCausticReservoir[mFrameCount % 2];
    var["gEmission"] = mpEmission;

    // Dispatch Shader
    mpScene->raytrace(pRenderContext, mGenerateInitialSamplesPass.pProgram.get(), mGenerateInitialSamplesPass.pVars, uint3(mScreenRes, 1));

    // Reservoir barrier
    pRenderContext->uavBarrier(mpFinalGatherReservoir[mFrameCount % 2].get());
    pRenderContext->uavBarrier(mpCausticReservoir[mFrameCount % 2].get());
}

void ReSTIRFGPass::resampleReservoirFGPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "Resampling Final Gather");
    // Initialize compute pass
    if (!mpResampleReservoirFGPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderResamplingReservoirFG).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add("USE_ENV_BACKROUND", mpScene->useEnvBackground() ? "1" : "0");
        defines.add(mpRTXDI->getDefines());
        defines.add(getMaterialDefines());

        mpResampleReservoirFGPass = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpResampleReservoirFGPass);
    mpResampleReservoirFGPass->getProgram()->addDefines(getMaterialDefines()); // Runtime define

    // Return early if there is no previous reservoir or resampling is disabled
    if ((!mCanResample) || !mResampleSettingsFG.enable)
    {
        return;
    }

    // Set variables
    auto var = mpResampleReservoirFGPass->getRootVar();
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]); // Set scene data
    mpSampleGenerator->bindShaderData(var);                    // Sample generator

    // Constant Buffer
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFrameDim"] = mScreenRes;
    var["CB"]["gConfidenceLimit"] = mResampleSettingsFG.confidenceCap;
    var["CB"]["gSpatialRadius"] = mResampleSettingsFG.samplingRadius;
    var["CB"]["gSpatialSamples"] = mResampleSettingsFG.spatialSamples;
    var["CB"]["gDisocclusionBoostSpatialSamples"] = mResampleSettingsFG.disocclusionBoostExtraSamples;
    var["CB"]["gNormalThreshold"] = mNormalThreshold;
    var["CB"]["gJacobianDistanceThreshold"] = mJacobianDistanceThreshold;
    var["CB"]["gUsePathThreshold"] = mUsePathThreshold;

    // Input Resources
    var["gFinalGatherReservoirPrev"] = mpFinalGatherReservoir[(mFrameCount + 1) % 2];
    var["gMVec"] = renderData[kInputMotionVectors]->asTexture();

    // In-/Output Resources
    var["gFinalGatherReservoir"] = mpFinalGatherReservoir[mFrameCount % 2];

    // Execute Compute Pass
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpResampleReservoirFGPass->execute(pRenderContext, uint3(targetDim, 1));
}

void ReSTIRFGPass::resampleReservoirCausticPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "Resampling Caustics");
    // Initialize compute pass
    if (!mpResampleReservoirCausticPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderResamplingReservoirCaustic).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(mpRTXDI->getDefines());
        defines.add(getMaterialDefines());

        mpResampleReservoirCausticPass = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpResampleReservoirCausticPass);
    mpResampleReservoirCausticPass->getProgram()->addDefines(getMaterialDefines()); // Runtime define

    // Return early if there is no previous reservoir or resampling is disabled
    if ((!mCanResample) || !mResampleSettingsCaustic.enable)
    {
        return;
    }

    // Set variables
    auto var = mpResampleReservoirCausticPass->getRootVar();
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]); // Set scene data
    mpSampleGenerator->bindShaderData(var);                    // Sample generator

    // Constant Buffer
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFrameDim"] = mScreenRes;
    var["CB"]["gConfidenceLimit"] = mResampleSettingsCaustic.confidenceCap;
    var["CB"]["gSpatialRadius"] = mResampleSettingsCaustic.samplingRadius;
    var["CB"]["gSpatialSamples"] = mResampleSettingsCaustic.spatialSamples;
    var["CB"]["gDisocclusionBoostSpatialSamples"] = mResampleSettingsCaustic.disocclusionBoostExtraSamples;
    var["CB"]["gNormalThreshold"] = mNormalThreshold;
    var["CB"]["gPhotonRadius"] = mPhotonRadius;

    // Input Resources
    var["gCausticReservoirPrev"] = mpCausticReservoir[(mFrameCount + 1) % 2];
    var["gMVec"] = renderData[kInputMotionVectors]->asTexture();

    // In-/Output Resources
    var["gCausticReservoir"] = mpCausticReservoir[mFrameCount % 2];

    // Execute
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpResampleReservoirCausticPass->execute(pRenderContext, uint3(targetDim, 1));
}

void ReSTIRFGPass::evaluateReservoirsPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "EvaluateReservoirs");

    // Create compute pass
    if (!mpEvaluateReservoirsPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderEvaluateReservoirs).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add("USE_ENV_BACKROUND", mpScene->useEnvBackground() ? "1" : "0");
        defines.add(mpRTXDI->getDefines());
        defines.add(getMaterialDefines());

        mpEvaluateReservoirsPass = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpEvaluateReservoirsPass);

    // Runtime Defines
    mpEvaluateReservoirsPass->getProgram()->addDefines(mpRTXDI->getDefines());
    mpEvaluateReservoirsPass->getProgram()->addDefines(getMaterialDefines());
    mpEvaluateReservoirsPass->getProgram()->addDefine("USE_ENV_BACKROUND", mpScene->useEnvBackground() ? "1" : "0");

    // Set variables
    auto var = mpEvaluateReservoirsPass->getRootVar();
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]); // Set scene data
    mpSampleGenerator->bindShaderData(var);                    // Sample generator

    // Constant Buffer
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFrameDim"] = mScreenRes;

    // RTXDI resources
    mpRTXDI->bindShaderData(var);

    // Input
    var["gVBuffer"] = renderData[kInputVBuffer]->asTexture();
    var["gFinalGatherReservoir"] = mpFinalGatherReservoir[mFrameCount % 2];
    var["gCausticReservoir"] = mpCausticReservoir[mFrameCount % 2];
    var["gEmission"] = mpEmission;

    // Output
    var["gOutColor"] = renderData[kOutputColor]->asTexture();

    // Execute
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpEvaluateReservoirsPass->execute(pRenderContext, uint3(targetDim, 1));
}

DefineList ReSTIRFGPass::getMaterialDefines()
{
    DefineList defines;
    defines.add("DiffuseBrdf", mUseLambertianDiffuse ? "DiffuseBrdfLambert" : "DiffuseBrdfFrostbite");
    defines.add("enableDiffuse", "1");
    defines.add("enableSpecular", "1");
    defines.add("enableTranslucency", "1");
    return defines;
}

void ReSTIRFGPass::renderUI(Gui::Widgets& widget)
{
    bool changed = false;

    if (auto group = widget.group("Photon Options"))
    {
        if (mUseDynamicPhotonDispatchCount)
        {
            group.text("Dispatched Photons: " + std::to_string(mNumDispatchedPhotons));
        }
        else
        {
            group.var("Dispatched Photons", mNumDispatchedPhotons, 1024u, 67108864u, 1u); // Max is 8192^2
        }

        group.text("Global Photons: " + std::to_string(mCurrentPhotonCount[0]) + " / " + std::to_string(mNumMaxPhotons[0]));
        group.text("Caustic photons: " + std::to_string(mCurrentPhotonCount[1]) + " / " + std::to_string(mNumMaxPhotons[1]));
        group.text("Photon Buffer Size:");
        group.indent(10.f);
        group.var(" ##MaxPhotonUI", mNumMaxPhotonsUI, 100u, 100000000u, 100);
        group.tooltip("First -> Global, Second -> Caustic");
        mChangePhotonLightBufferSize = group.button("Apply", true);
        group.indent(-10.f);
        if (auto groupGen = group.group("Generation Settings", true))
        {
            if (mMixedLights)
            {
                changed |= groupGen.var("Mixed Analytic Ratio", mPhotonAnalyticRatio, 0.f, 1.f, 0.01f);
                groupGen.tooltip("Analytic photon distribution ratio in a mixed light case. E.g. 0.3 -> 30% analytic, 70% emissive");
            }

            changed |= groupGen.checkbox("Enable dynamic photon dispatch", mUseDynamicPhotonDispatchCount);
            groupGen.tooltip("Changed the number of dispatched photons dynamically. Tries to fill the photon buffer");
            if (mUseDynamicPhotonDispatchCount)
            {
                if (auto groupDynChange = groupGen.group("DynamicDispatchOptions"))
                {
                    changed |= groupDynChange.var("Max dispatched", mPhotonDynamicDispatchMax, 1024u, 67108864u);
                    changed |= groupDynChange.var("Guard Percentage", mPhotonDynamicGuardPercentage, 0.0f, 1.f, 0.001f);
                    groupDynChange.tooltip(
                        "If current fill rate is under PhotonBufferSize * (1-pGuard), the values are accepted. Reduces the changes "
                        "every frame"
                    );
                    changed |= groupDynChange.var("Percentage Change", mPhotonDynamicChangePercentage, 0.01f, 10.f, 0.01f);
                    groupDynChange.tooltip(
                        "Increase/Decrease percentage from the Buffer Size. With current value a increase/decrease of :" +
                        std::to_string(mPhotonDynamicChangePercentage * mNumMaxPhotons[0]) + "is expected"
                    );
                }
            }

            changed |= groupGen.var("Light Store Probability", mGlobalPhotonRejection, 0.f, 1.f, 0.0001f);
            group.tooltip("Probability a photon light is stored on diffuse hit. Flux is scaled up appropriately");

            changed |= groupGen.var("Max Bounces", mPhotonMaxBounces, 0u, 32u);

            groupGen.separator();
        }
        group.text("Photon Radius(Global / Caustic):");
        group.indent(10.f);
        group.var(" ##PhotonRadius", mPhotonRadius, 0, FLT_MAX, 0.0001f, false, "%.6f");
        group.indent(-10.f);
    }

    if (auto group = widget.group("RTXDI"))
    {
        if (mpRTXDI)
        {
            mpRTXDI->renderUI(group);
        }
        else
        {
            group.text("Load a scene for RTXDI options");
        }
    }

    if (auto group = widget.group("ReSTIR FG"))
    {
        group.var("Final Gather Path Length", mFGRayMaxPathLength, 1u, 64u, 1u);
        group.tooltip(
            "Path length for a final gather sample. A final gather sample stops when it encounters a rough enough surface (see Material "
            "Options)"
        );

        auto resampleUI = [](ResamplingSettings& settings, Gui::Widgets& widget)
        {
            widget.checkbox("Enable Resampling", settings.enable);
            widget.var("Confidence Cap", settings.confidenceCap, 1u, UINT_MAX, 1u);
            widget.tooltip("Maximum confidence a reservoir can have");
            widget.var("Spatial Samples", settings.spatialSamples, 0u, 64u, 1u);
            widget.var("Disocclusion additional spatial samples", settings.disocclusionBoostExtraSamples, 0u, 16u, 1u);
            widget.tooltip("Extra spatial samples if temporal resampling fails");
            widget.var("Spatial Sample Radius", settings.samplingRadius, 0.f, FLT_MAX, 1.f);
        };

        if (auto group2 = group.group("Resampling FG options"))
        {
            resampleUI(mResampleSettingsFG, group2);
        }
        if (auto group2 = group.group("Resampling Caustic options"))
        {
            resampleUI(mResampleSettingsCaustic, group2);
        }

        group.separator();
        group.text("Surface Rejection Options:");
        group.var("Normal Rejection Threshold", mNormalThreshold, 0.f, 1.0f, 0.001f);
        group.tooltip("Threshold of dot product between both reservoir face normals");
        group.var("Sample Distance Threshold", mJacobianDistanceThreshold, 0.f, FLT_MAX, 0.001f);
        group.checkbox("Use Path Threshold", mUsePathThreshold);
        group.tooltip(
            "Only resamples if the surfaces used for generating the Final Gather samples have the same path length. Always enabled for "
            "caustic collection"
        );
    }

    if (auto group = widget.group("Material Options"))
    {
        group.checkbox("Use Lambertian Diffuse BSDF", mUseLambertianDiffuse);
        group.tooltip("BSDF used by ReSTIR PT and Suffix ReSTIR prototype");

        group.text("Diffuse Classification Roughness Threshold:");
        group.tooltip("Surfaces with roughness above this threshold are considered diffuse");
        group.indent(10.f);
        group.var("##RoughnessThreshold", mSpecularRoughnessThreshold, 0.f, 1.f, 0.001f);
        group.indent(-10.f);
    }
}

void ReSTIRFGPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

     // Reset all passes and sampling helpers
    mpPhotonAS.reset();
    mpEmissiveLightSampler.reset();
    mpRTXDI.reset();
    mResetScreenTex = true;
    mChangePhotonLightBufferSize = true;

    mTracePhotonPass = RayTraceProgramHelper::create();
    mGenerateInitialSamplesPass = RayTraceProgramHelper::create();
    mpResampleReservoirFGPass.reset();
    mpResampleReservoirCausticPass.reset();
    mpEvaluateReservoirsPass.reset();

    if (mpScene)
    {
        if (mpScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }
    }
}

void ReSTIRFGPass::RayTraceProgramHelper::initProgramVars(ref<Device> pDevice, ref<Scene> pScene, ref<SampleGenerator> pSampleGenerator)
{
    FALCOR_ASSERT(pProgram);

    // Configure program.
    pProgram->addDefines(pSampleGenerator->getDefines());
    pProgram->setTypeConformances(pScene->getTypeConformances());
    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);

    // Bind utility classes into shared data.
    auto var = pVars->getRootVar();
    pSampleGenerator->bindShaderData(var);
}
