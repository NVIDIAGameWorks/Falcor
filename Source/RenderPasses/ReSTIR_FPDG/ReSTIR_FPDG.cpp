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

#include "ReSTIR_FPDG.h"

#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"

namespace
{
const std::string kShaderFolder = "RenderPasses/ReSTIR_FPDG/";
const std::string kShaderTracePhotons = kShaderFolder + "TracePass.rt.slang";
const std::string kShaderGenInitialSamples = kShaderFolder + "GenerateInitialSamples.rt.slang";
const std::string kShaderResamplingReservoirFG = kShaderFolder + "ResampleReservoirFG.cs.slang";
const std::string kShaderResamplingReservoirCaustic = kShaderFolder + "ResamplePhotonDifferentialsReservoirCaustic.cs.slang";
const std::string kShaderEvaluateReservoirs = kShaderFolder + "EvaluateReservoirs.cs.slang";

const ShaderModel kShaderModel = ShaderModel::SM6_5;

// Render Pass inputs and outputs
// To acquire initial hit information
const std::string kInputVBuffer = "vbuffer";
// For temporal resampling
const std::string kInputMotionVectors = "mvec";

const Falcor::ChannelList kInputChannels{
    {kInputVBuffer, "gVBuffer", "Visibility buffer in packed format"},
    {kInputMotionVectors, "gMotionVectors", "Motion vector buffer (float format)", true /* optional */},
};

// Outputs
const std::string kOutputColor = "color";
const std::string kOutputEmission = "emission";
const std::string kOutputDiffuseRadiance = "diffuseRadiance";
const std::string kOutputSpecularRadiance = "specularRadiance";
const std::string kOutputDiffuseReflectance = "diffuseReflectance";
const std::string kOutputSpecularReflectance = "specularReflectance";
const std::string kOutputResidualRadiance = "residualRadiance";     //The rest (transmission, delta)

const Falcor::ChannelList kOutputChannels{
    {kOutputColor,                  "gOutColor",                "HDR output color", false /*optional*/, ResourceFormat::RGBA32Float},
    {kOutputEmission,               "gOutEmission",             "Output Emission", true /*optional*/, ResourceFormat::RGBA32Float},
    {kOutputDiffuseRadiance,        "gOutDiffuseRadiance",      "Output demodulated diffuse color (linear)", true /*optional*/, ResourceFormat::RGBA32Float},
    {kOutputSpecularRadiance,       "gOutSpecularRadiance",     "Output demodulated specular color (linear)", true /*optional*/, ResourceFormat::RGBA32Float},
    {kOutputDiffuseReflectance,     "gOutDiffuseReflectance",   "Output primary surface diffuse reflectance", true /*optional*/, ResourceFormat::RGBA16Float},
    {kOutputSpecularReflectance,    "gOutSpecularReflectance",  "Output primary surface specular reflectance", true /*optional*/, ResourceFormat::RGBA16Float},
    {kOutputResidualRadiance,       "gOutResidualRadiance",     "Output residual color (transmission/delta)", true /*optional*/, ResourceFormat::RGBA32Float},
};
}; // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIR_FPDG>();
}

ReSTIR_FPDG::ReSTIR_FPDG(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
}

Properties ReSTIR_FPDG::getProperties() const
{
    return {};
}

RenderPassReflection ReSTIR_FPDG::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void ReSTIR_FPDG::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mScreenRes = compileData.defaultTexDims;
}

void ReSTIR_FPDG::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    // Add refresh flag if options changed
    auto& dict = renderData.getDictionary();
    auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
    if (mOptionsChanged)
    {
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    beginFrame(pRenderContext, renderData);

    // Init RTXDI if it is enabled
    if (mDirectLightMode == DirectLightingMode::RTXDI && !mpRTXDI)
    {
        mpRTXDI = std::make_unique<RTXDI>(mpScene, mRTXDIOptions);
    }
    // Delete RTXDI if it is set and the mode changed
    if (mDirectLightMode != DirectLightingMode::RTXDI && mpRTXDI) mpRTXDI = nullptr;

    if (mpRTXDI) mpRTXDI->beginFrame(pRenderContext, mScreenRes);

    pRenderContext->clearUAV(mpPhotonCounter->getUAV().get(), uint4(0u));

    tracePhotonDifferentialsPass(pRenderContext, renderData, !mHasMixedLights && mHasAnalyticLights, !mHasMixedLights && mPhotonAnalyticRatio > 0);
    if (mHasMixedLights && mPhotonAnalyticRatio > 0)
        tracePhotonDifferentialsPass(pRenderContext, renderData, true, true); // Second pass. Always Analytic
    generateInitialSamplesPass(pRenderContext, renderData);
    
    const auto& motionVectors = renderData.getResource(kInputMotionVectors)->asTexture();
    if (mpRTXDI) mpRTXDI->update(pRenderContext, motionVectors);

    // Spatiotemporal resampling for final gather samples and caustics
    resampleReservoirFGPass(pRenderContext, renderData);
    resampleReservoirCausticPass(pRenderContext, renderData);

    // Final shading
    evaluateReservoirsPass(pRenderContext, renderData);

    if (mpRTXDI) mpRTXDI->endFrame(pRenderContext);

    mFrameCount++;
    mCanResample = true;
}

void ReSTIR_FPDG::renderUI(Gui::Widgets& widget)
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
            if (mHasMixedLights)
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

            changed |= groupGen.var("Max Bounces", mPhotonMaxBounces, 0u, 10u); // set a cap of 10 bounces, since performance will drop drastically otherways

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

void ReSTIR_FPDG::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mFrameCount = 0u;

    mpRTXDI.reset();
    mpEmissiveLightSampler.reset();
    mResetScreenRes = true;

    mTracePhotonDifferentialsPass = RaytracingProgramHelper ::create();
    mGenerateInitialSamplesPass = RaytracingProgramHelper::create();

    mpResamplePhotonDifferentialsReservoirFGPass.reset();
    mpResamplePhotonDifferentialsReservoirCausticPass.reset();
    mpEvaluatePhotonDifferentialsReservoirsPass.reset();

    mChangePhotonLightBufferSize = true;

    if (mpScene && mpScene->hasGeometryType(Scene::GeometryType::Custom))
    {
        logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        mRecompile = true;
    }

    setupLights(pRenderContext);
}

void ReSTIR_FPDG::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    //setupLights(pRenderContext);
    //updatePrograms();
    setupResources(pRenderContext, renderData);

    setupPhotonDifferentialsAS();
}

void ReSTIR_FPDG::endFrame()
{
    // for PixelDebug and PixelStats
}

void ReSTIR_FPDG::updatePrograms()
{
    if (!mRecompile) return;

    mRecompile = false;
}

void ReSTIR_FPDG::setupLights(RenderContext* pRenderContext)
{
    auto& lightCollection = mpScene->getLightCollection(pRenderContext);
    lightCollection->prepareSyncCPUData(pRenderContext);

    bool emissiveLightsUsed = mpScene->useEmissiveLights();
    bool analyticLightsUsed = mpScene->useAnalyticLights();

    mHasEnvBackground = mpScene->useEnvBackground();
    mHasAnalyticLights = analyticLightsUsed;
    mHasEmissiveLights = emissiveLightsUsed;
    mHasMixedLights = analyticLightsUsed && emissiveLightsUsed;
    mHasLights = emissiveLightsUsed || analyticLightsUsed;

    if (!mHasLights)
    {
        logError("Scene has no lights. Pass will not be executed.");
    }

    if (emissiveLightsUsed)
    {
        if (!mpEmissiveLightSampler)
        {
            switch (mEmissiveSamplerType)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveLightSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, lightCollection);
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveLightSampler = std::make_unique<LightBVHSampler>(pRenderContext, lightCollection);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveLightSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, lightCollection);
                break;
            default:
                logError("Unknown emissive light sampler type!");
            }
        }
    }
    else
    {
        if (mpEmissiveLightSampler)
        {
            mpEmissiveLightSampler = nullptr;

            // TO DO: reset programVars of light sample rt shader
        }
    }

    if (mpEmissiveLightSampler)
    {
        mpEmissiveLightSampler->update(pRenderContext, lightCollection);
    }

    mRecompile = true;
}

void ReSTIR_FPDG::setupResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& screenDims = renderData.getDefaultTextureDims();
    if (mScreenRes.x != screenDims.x || mScreenRes.y != screenDims.y)
    {
        mScreenRes = screenDims;
        mResetScreenRes = true;
    }

    if (mChangePhotonLightBufferSize)
    {
        mNumMaxPhotons = mNumMaxPhotonsUI; 
        for (uint i = 0; i < 2; i++)
        {
            mpPhotonAABB[i].reset();
            mpPhotonData[i].reset();
        }
    }

    for (uint i = 0u; i < 2; i++)
    {
        if (!mpPhotonAABB[i])
        {
            mpPhotonAABB[i] = mpDevice->createStructuredBuffer(sizeof(AABB), mNumMaxPhotons[i], ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal, nullptr, false);
            mpPhotonAABB[i]->setName("PhotonAABB" + std::to_string(i));
        }

        if (!mpPhotonData[i])
        {
            mpPhotonData[i] = mpDevice->createStructuredBuffer(64u /*See the photonData struct in shader*/,
                mNumMaxPhotons[i], ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mpPhotonData[i]->setName("PhotonData" + std::to_string(i));
        }

        if (!mpCausticReservoir[i] || mResetScreenRes)
        {
            mCanResample = false;
            mpCausticReservoir[i] = mpDevice->createStructuredBuffer(
                128u, mScreenRes.x * mScreenRes.y, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mpCausticReservoir[i]->setName("CausticReservoir" + std::to_string(i));
        }

        if (!mpFinalGatherReservoir[i] || mResetScreenRes)
        {
            mCanResample = false;
            mpFinalGatherReservoir[i] = mpDevice->createStructuredBuffer(
                112u, mScreenRes.x * mScreenRes.y, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mpFinalGatherReservoir[i]->setName("FinalGatherReservoir" + std::to_string(i));
        }
    }

    // Photon counters
    if (!mpPhotonCounter)
    {
        mpPhotonCounter = mpDevice->createStructuredBuffer(sizeof(uint), 2u, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal, nullptr, false);
        mpPhotonCounter->setName("PhotonCounter");
    }

    if (!mpPhotonCounterCPU)
    {
        mpPhotonCounterCPU = mpDevice->createStructuredBuffer(sizeof(uint), 2u, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);
        mpPhotonCounterCPU->setName("PhotonCounterCPU");
    }

    if (!mpEmission || mResetScreenRes)
    {
        mpEmission = mpDevice->createTexture2D(mScreenRes.x, mScreenRes.y, ResourceFormat::RGBA32Float, 1u, 1u, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpEmission->setName("EmissionTexture");
    }

    mResetScreenRes = false;
}

void ReSTIR_FPDG::setupPhotonDifferentialsAS()
{
    // Delete the Photon differentials AS if max Buffer size changes
    if (mChangePhotonLightBufferSize)
    {
        mpPhotonDifferentialAS.reset();
        mChangePhotonLightBufferSize = false;
    }

    // Create AS
    if (!mpPhotonDifferentialAS)
    {
        std::vector<uint64_t> aabbCount = {mNumMaxPhotons[PhotonType::GLOBAL], mNumMaxPhotons[PhotonType::CAUSTIC]};
        std::vector<uint64_t> aabbGPUAddress = {mpPhotonAABB[PhotonType::GLOBAL]->getGpuAddress(), mpPhotonAABB[PhotonType::CAUSTIC]->getGpuAddress()};

        mpPhotonDifferentialAS = std::make_unique<CustomAccelerationStructure>(mpDevice, aabbCount, aabbGPUAddress,
            CustomAccelerationStructure::BuildMode::FastBuild, CustomAccelerationStructure::UpdateMode::TLASOnly);
    }
}

void ReSTIR_FPDG::tracePhotonDifferentialsPass(RenderContext* pRenderContext, const RenderData& renderData, bool analyticOnly, bool buildAS)
{
    FALCOR_PROFILE(pRenderContext, "TracePhotonDifferentials");

    TypeConformanceList sceneTypeConformances = mpScene->getTypeConformances();

    if (!mTracePhotonDifferentialsPass.pProgram)
    {
        DefineList globalDefines;
        globalDefines.add("USE_EMISSIVE_LIGHT", mHasEmissiveLights ? "1" : "0");
        globalDefines.add(mpScene->getSceneDefines());
        globalDefines.add("PHOTON_BUFFER_SIZE_GLOBAL", std::to_string(mNumMaxPhotons[0]));
        globalDefines.add("PHOTON_BUFFER_SIZE_CAUSTIC", std::to_string(mNumMaxPhotons[1]));
        globalDefines.add("ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
        globalDefines.add(getMaterialDefines());
        if (mpEmissiveLightSampler)
            globalDefines.add(mpEmissiveLightSampler->getDefines());

        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules())
            .addShaderLibrary(kShaderTracePhotons)
            .setMaxPayloadSize(sizeof(float) * 4) // PackedHitInfo is uint4 (16 byte)
            .setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize())
            .setMaxTraceRecursionDepth(1u);
        if (mpScene->hasProceduralGeometry())
            desc.setRtPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

        mTracePhotonDifferentialsPass.pBindingTable = RtBindingTable::create(1u, 1u, mpScene->getGeometryCount());

        // Specify entry point for raygen and miss shaders.
        // The raygen shader needs type conformances for *all* materials in the scene.
        mTracePhotonDifferentialsPass.pBindingTable->setRayGen(desc.addRayGen("rayGen", sceneTypeConformances));
        // The miss shader doesn't need type conformances as it doesn't access any materials.
        mTracePhotonDifferentialsPass.pBindingTable->setMiss(0u, desc.addMiss("miss"));
        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            mTracePhotonDifferentialsPass.pBindingTable->setHitGroup(
                0u, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit")
            );
        }

        mTracePhotonDifferentialsPass.pProgram = Program::create(mpDevice, desc, globalDefines);
    }

    if (!mTracePhotonDifferentialsPass.pVars)
        mTracePhotonDifferentialsPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);

    FALCOR_ASSERT(mTracePhotonDifferentialsPass.pVars);
    auto var = mTracePhotonDifferentialsPass.pVars->getRootVar();

    if (mpEmissiveLightSampler)
        mpEmissiveLightSampler->bindShaderData(var["Light"]["gEmissiveSampler"]);
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    // Handle shader dimension
    uint dispatchedPhotons = mNumDispatchedPhotons;
    if (mHasMixedLights)
    {
        float dispatchedF = float(dispatchedPhotons);
        dispatchedF *= analyticOnly ? mPhotonAnalyticRatio : 1.f - mPhotonAnalyticRatio;
        dispatchedPhotons = uint(dispatchedF);
    }
    uint shaderDispatchDim = static_cast<uint>(std::floor(sqrt(dispatchedPhotons)));
    shaderDispatchDim = std::max(32u, shaderDispatchDim);

    // CB
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gMaxBounces"] = mPhotonMaxBounces;
    var["CB"]["gPhotonRadius"] = mPhotonRadius;
    var["CB"]["gDispatchDimension"] = shaderDispatchDim;
    var["CB"]["gGlobalRejectionProb"] = mGlobalPhotonRejection;
    var["CB"]["gUseAnalyticLights"] = analyticOnly;   

    for (uint i = 0; i < 2; i++)
    {
        var["gPhotonAABB"][i] = mpPhotonAABB[i];
        var["gPhotonData"][i] = mpPhotonData[i];
    }
    var["gPhotonCounter"] = mpPhotonCounter;

    mpScene->raytrace(pRenderContext, mTracePhotonDifferentialsPass.pProgram.get(), mTracePhotonDifferentialsPass.pVars, uint3(shaderDispatchDim, shaderDispatchDim, 1u));
    
    // If two passes are dispatched, the acceleration structure is build on the second dispatch
    if (buildAS)
    {
        // Clear values after the counter
        std::vector<ref<Buffer>> aabbs = {mpPhotonAABB[PhotonType::GLOBAL], mpPhotonAABB[PhotonType::CAUSTIC]};
        mpPhotonDifferentialAS->clearAABBBuffers(pRenderContext, aabbs, true, mpPhotonCounter);

        // Copy counter to CPU
        handlePhotonCounter(pRenderContext);

        // Build acceleration structure
        uint2 currentPhotons = mFrameCount > 0 ? uint2(float2(mCurrentPhotonCount) * mASBuildBufferPhotonOverestimate) : mNumMaxPhotons;
        std::vector<uint64_t> photonBuildSize = {
            std::min(mNumMaxPhotons[PhotonType::GLOBAL], currentPhotons[PhotonType::GLOBAL]),
            std::min(mNumMaxPhotons[PhotonType::CAUSTIC], currentPhotons[PhotonType::CAUSTIC])
        };
        mpPhotonDifferentialAS->update(pRenderContext, photonBuildSize);
    }
}

// Works with 2-3 frames delay
// TODO: Possible performance bottleneck: gpu-cpu stall -> need for ring buffer[3]
void ReSTIR_FPDG::handlePhotonCounter(RenderContext* pRenderContext)
{
    // Try this with raw d3d12 in your framework
    pRenderContext->copyBufferRegion(mpPhotonCounterCPU.get(), 0u, mpPhotonCounter.get(), (uint64_t)0u, (uint64_t)sizeof(uint2));

    void* data = mpPhotonCounterCPU->map();
    std::memcpy(&mCurrentPhotonCount, data, sizeof(uint2));
    mpPhotonCounterCPU->unmap();

    if (mUseDynamicPhotonDispatchCount)
    {
        // Only use global photons for the dynamic dispatch count
        uint globalPhotonCount = mCurrentPhotonCount[0];
        uint globalMaxPhotons = mNumMaxPhotons[0];
        uint causticPhotonCount = mCurrentPhotonCount[1];
        uint causticMaxPhotons = mNumMaxPhotons[1];
        // If counter is invalid, reset
        if (globalPhotonCount == 0u)
        {
            mNumDispatchedPhotons = mPhotonDynamicDispatchMax * 0.5f;
        }
        uint globBufferSizeCompValue = (uint)(globalMaxPhotons * (1.f - mPhotonDynamicGuardPercentage));        // 92% of max global photons
        uint causticBufferSizeCompValue = (uint)(causticMaxPhotons * (1.f - mPhotonDynamicGuardPercentage));    // 92% of max caustic photons
        uint globChangeSize = (uint)(globalMaxPhotons * mPhotonDynamicChangePercentage);                        // 4% of current photons count
        uint causticChangeSize = (uint)(causticMaxPhotons * mPhotonDynamicChangePercentage);                    // 4% of current caustic photons count
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

// Generates the initial Samples for the reservoirs (FG) and initializes RTXDI surfaces
void ReSTIR_FPDG::generateInitialSamplesPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "GenerateInitialSamples");

    if (!mGenerateInitialSamplesPass.pProgram)
    {
        TypeConformanceList sceneTypeConformances = mpScene->getTypeConformances();

        DefineList globalDefines;
        globalDefines.add("USE_EMISSIVE_LIGHT", mHasEmissiveLights ? "1" : "0");
        globalDefines.add(mpScene->getSceneDefines());
        globalDefines.add("ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
        globalDefines.add(mpRTXDI->getDefines());
        globalDefines.add(getMaterialDefines());

        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules())
            .addShaderLibrary(kShaderGenInitialSamples)
            .addTypeConformances(mpScene->getTypeConformances())
            .setMaxPayloadSize(sizeof(float) * 4) // PackedHitInfo is uint4 (16 byte)
            .setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize())
            .setMaxTraceRecursionDepth(1u);
        if (mpScene->hasProceduralGeometry())
            desc.setRtPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

        mGenerateInitialSamplesPass.pBindingTable = RtBindingTable::create(1u, 1u, mpScene->getGeometryCount());
        mGenerateInitialSamplesPass.pBindingTable->setRayGen(desc.addRayGen("rayGen", sceneTypeConformances));
        mGenerateInitialSamplesPass.pBindingTable->setMiss(0u, desc.addMiss("miss"));
        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            mGenerateInitialSamplesPass.pBindingTable->setHitGroup(
                0u, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit")
            );
        }

        mGenerateInitialSamplesPass.pProgram = Program::create(mpDevice, desc, globalDefines);
    }

    if (!mGenerateInitialSamplesPass.pVars)
        mGenerateInitialSamplesPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);

    FALCOR_ASSERT(mGenerateInitialSamplesPass.pVars);

    // Set variables
    auto var = mGenerateInitialSamplesPass.pVars->getRootVar();

    // Set RTXDI resources
    mpRTXDI->bindShaderData(var);

    // CB
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFGRayMaxPathLength"] = mFGRayMaxPathLength;

    // Input resources
    // Input V-Buffer
    var["gVBuffer"] = renderData.getResource(kInputVBuffer)->asTexture();
    // Input TLAS of photon differentials
    mpPhotonDifferentialAS->bindTlas(var, "gPhotonAS");
    // Input AABB and photon differentials data (output from Trace pass)
    for (uint i = 0; i < 2; i++)
    {
        var["gPhotonAABB"][i] = mpPhotonAABB[i];
        var["gPhotonData"][i] = mpPhotonData[i];
    }

    // Output
    var["gFinalGatherReservoir"] = mpFinalGatherReservoir[mFrameCount % 2];
    var["gCausticReservoir"] = mpCausticReservoir[mFrameCount % 2];
    var["gEmission"] = mpEmission;

    mpScene->raytrace(pRenderContext, mGenerateInitialSamplesPass.pProgram.get(), mGenerateInitialSamplesPass.pVars, uint3(mScreenRes, 1u));

    // D3D12 Resource barriers
    pRenderContext->uavBarrier(mpFinalGatherReservoir[mFrameCount % 2].get());
    pRenderContext->uavBarrier(mpCausticReservoir[mFrameCount % 2].get());
}

// Reservoir Resampling for Final Gather Samples
void ReSTIR_FPDG::resampleReservoirFGPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ResampleReservoirFG");

    if (!mpResamplePhotonDifferentialsReservoirFGPass)
    {
        DefineList globalDefines;
        globalDefines.add(mpScene->getSceneDefines());
        globalDefines.add(mpSampleGenerator->getDefines());
        globalDefines.add(getMaterialDefines());
        globalDefines.add(mpRTXDI->getDefines());

        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules())
            .addTypeConformances(mpScene->getTypeConformances())
            .addShaderLibrary(kShaderResamplingReservoirFG).csEntry("main").setShaderModel(ShaderModel::SM6_5);

        mpResamplePhotonDifferentialsReservoirFGPass = ComputePass::create(mpDevice, desc, globalDefines, true);
    }
    FALCOR_ASSERT(mpResamplePhotonDifferentialsReservoirFGPass);

    if (!mCanResample || !mResampleSettingsFG.enable) return;

    // Set variables
    auto var = mpResamplePhotonDifferentialsReservoirFGPass->getRootVar();

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
    mpSampleGenerator->bindShaderData(var);

    // CB
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFrameDim"] = mScreenRes;
    var["CB"]["gConfidenceLimit"] = mResampleSettingsFG.confidenceCap;
    var["CB"]["gSpatialRadius"] = mResampleSettingsFG.samplingRadius;
    var["CB"]["gSpatialSamples"] = mResampleSettingsFG.spatialSamples;
    var["CB"]["gDisocclusionBoostSpatialSamples"] = mResampleSettingsFG.disocclusionBoostExtraSamples;
    var["CB"]["gNormalThreshold"] = mNormalThreshold;
    var["CB"]["gJacobianDistanceThreshold"] = mJacobianDistanceThreshold;
    var["CB"]["gUsePathThreshold"] = mUsePathThreshold;

    // Input resources
    var["gMVec"] = renderData.getResource(kInputMotionVectors)->asTexture();
    var["gFinalGatherReservoirPrev"] = mpFinalGatherReservoir[(mFrameCount + 1) % 2];

    // In/Out resources
    var["gFinalGatherReservoir"] = mpFinalGatherReservoir[mFrameCount % 2];

    // Execute Compute Pass
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpResamplePhotonDifferentialsReservoirFGPass->execute(pRenderContext, uint3(targetDim, 1u));
}

// Reservoir Resampling for Caustic Samples
void ReSTIR_FPDG::resampleReservoirCausticPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ResampleReservoirCaustic");

    if (!mpResamplePhotonDifferentialsReservoirCausticPass)    
    {
        DefineList globalDefines;
        globalDefines.add(mpScene->getSceneDefines());
        globalDefines.add(mpSampleGenerator->getDefines());
        globalDefines.add(getMaterialDefines());
        globalDefines.add(mpRTXDI->getDefines());

        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules())
            .addTypeConformances(mpScene->getTypeConformances())
            .addShaderLibrary(kShaderResamplingReservoirCaustic).csEntry("main").setShaderModel(ShaderModel::SM6_5);

        mpResamplePhotonDifferentialsReservoirCausticPass = ComputePass::create(mpDevice, desc, globalDefines, true);
    }

    FALCOR_ASSERT(mpResamplePhotonDifferentialsReservoirCausticPass);

    if (!mCanResample || !mResampleSettingsCaustic.enable) return;

    // Set variables
    auto var = mpResamplePhotonDifferentialsReservoirCausticPass->getRootVar();

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
    mpSampleGenerator->bindShaderData(var);

    // CB
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFrameDim"] = mScreenRes;
    var["CB"]["gConfidenceLimit"] = mResampleSettingsCaustic.confidenceCap;
    var["CB"]["gSpatialRadius"] = mResampleSettingsCaustic.samplingRadius;
    var["CB"]["gSpatialSamples"] = mResampleSettingsCaustic.spatialSamples;
    var["CB"]["gDisocclusionBoostSpatialSamples"] = mResampleSettingsCaustic.disocclusionBoostExtraSamples;
    var["CB"]["gNormalThreshold"] = mNormalThreshold;
    var["CB"]["gPhotonRadius"] = mPhotonRadius;

    // Input resources
    var["gMVec"] = renderData.getResource(kInputMotionVectors)->asTexture();
    var["gCausticReservoirPrev"] = mpCausticReservoir[(mFrameCount + 1) % 2];

    // In/Out resources
    var["gCausticReservoir"] = mpCausticReservoir[mFrameCount % 2];

    // Execute Compute Pass
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpResamplePhotonDifferentialsReservoirCausticPass->execute(pRenderContext, uint3(targetDim, 1u));
}

// Evaluate Reservoirs
void ReSTIR_FPDG::evaluateReservoirsPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "EvaluateReservoirs");

    if (!mpEvaluatePhotonDifferentialsReservoirsPass)
    {
        DefineList globalDefines;
        globalDefines.add(mpScene->getSceneDefines());
        globalDefines.add(mpSampleGenerator->getDefines());
        globalDefines.add(getMaterialDefines());
        globalDefines.add(mpRTXDI->getDefines());
        globalDefines.add("USE_ENV_BACKROUND", mHasEnvBackground ? "1" : "0");

        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules())
            .addTypeConformances(mpScene->getTypeConformances())
            .addShaderLibrary(kShaderEvaluateReservoirs).csEntry("main").setShaderModel(ShaderModel::SM6_5);

        mpEvaluatePhotonDifferentialsReservoirsPass = ComputePass::create(mpDevice, desc, globalDefines, true);
    }
    FALCOR_ASSERT(mpEvaluatePhotonDifferentialsReservoirsPass);

    // Set variables
    auto var = mpEvaluatePhotonDifferentialsReservoirsPass->getRootVar();
    // Set RTXDI resources
    mpRTXDI->bindShaderData(var);

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
    mpSampleGenerator->bindShaderData(var);

    // CB
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFrameDim"] = mScreenRes;

    // Input resources
    var["gVBuffer"] = renderData.getResource(kInputVBuffer)->asTexture();
    var["gFinalGatherReservoir"] = mpFinalGatherReservoir[mFrameCount % 2];
    var["gCausticReservoir"] = mpCausticReservoir[mFrameCount % 2];
    var["gEmission"] = mpEmission;

    // Output resources
    var["gOutColor"] = renderData.getResource(kOutputColor)->asTexture();

    // Execute Compute Pass
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpEvaluatePhotonDifferentialsReservoirsPass->execute(pRenderContext, uint3(targetDim, 1u));
}

DefineList ReSTIR_FPDG::getMaterialDefines()
{
    DefineList defines;
    defines.add("DiffuseBrdf", mUseLambertianDiffuse ? "DiffuseBrdfLambert" : "DiffuseBrdfFrostbite");
    defines.add("enableDiffuse", "1");
    defines.add("enableSpecular", "1");
    defines.add("enableTranslucency", "1");
    return defines;
}

void ReSTIR_FPDG::RaytracingProgramHelper::initProgramVars(ref<Device>& pDevice, ref<Scene>& pScene, ref<SampleGenerator>& pSampleGenerator)
{
    FALCOR_ASSERT(pProgram);

    pProgram->addDefines(pSampleGenerator->getDefines());
    pProgram->setTypeConformances(pScene->getTypeConformances()); //?

    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);

    auto var = pVars->getRootVar();
    pSampleGenerator->bindShaderData(var);
}
