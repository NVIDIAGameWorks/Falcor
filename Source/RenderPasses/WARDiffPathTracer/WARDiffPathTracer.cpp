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
#include "WARDiffPathTracer.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, WARDiffPathTracer>();
    ScriptBindings::registerBinding(WARDiffPathTracer::registerBindings);
}

namespace
{
const char kShaderFile[] = "RenderPasses/WARDiffPathTracer/WARDiffPathTracer.rt.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

const ChannelList kInputChannels = {};

const ChannelList kOutputChannels = {
    {"color", "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float},
    {"dColor", "gOutputDColor", "Output derivatives computed via auto-diff", false, ResourceFormat::RGBA32Float},
};

const std::string kSamplesPerPixel = "samplesPerPixel";
const std::string kMaxBounces = "maxBounces";

const std::string kDiffMode = "diffMode";
const std::string kDiffVarName = "diffVarName";

const std::string kSampleGenerator = "sampleGenerator";
const std::string kFixedSeed = "fixedSeed";
const std::string kUseBSDFSampling = "useBSDFSampling";
const std::string kUseNEE = "useNEE";
const std::string kUseMIS = "useMIS";

const std::string kUseWAR = "useWAR";
const std::string kAuxSampleCount = "auxSampleCount";
const std::string kLog10vMFConcentration = "Log10vMFConcentration";
const std::string kLog10vMFConcentrationScreen = "Log10vMFConcentrationScreen";
const std::string kBoundaryTermBeta = "boundaryTermBeta";
const std::string kUseAntitheticSampling = "useAntitheticSampling";
} // namespace

WARDiffPathTracer::WARDiffPathTracer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);

    FALCOR_ASSERT(mpSampleGenerator);

    // Set differentiable rendering debug parameters if needed.
    if (mStaticParams.diffVarName == "CBOX_BUNNY_MATERIAL")
    {
        // Albedo value with materialID = 0
        setDiffDebugParams(DiffVariableType::Material, uint2(0, 0), 0, float4(1.f, 1.f, 1.f, 0.f));
    }
    else if (mStaticParams.diffVarName == "CBOX_BUNNY_TRANSLATION")
    {
        // Vertical translation with meshID = 0
        setDiffDebugParams(DiffVariableType::GeometryTranslation, uint2(0, 0), 0, float4(0.f, 1.f, 0.f, 0.f));
    }
}

void WARDiffPathTracer::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        // Rendering parameters
        if (key == kSamplesPerPixel)
            mStaticParams.samplesPerPixel = value;
        else if (key == kMaxBounces)
            mStaticParams.maxBounces = value;

        // Differentiable rendering parameters
        else if (key == kDiffMode)
            mStaticParams.diffMode = value;
        else if (key == kDiffVarName)
            mStaticParams.diffVarName = value.operator std::string();

        // Sampling parameters
        else if (key == kSampleGenerator)
            mStaticParams.sampleGenerator = value;
        else if (key == kFixedSeed)
        {
            mParams.fixedSeed = value;
            mParams.useFixedSeed = true;
        }
        else if (key == kUseBSDFSampling)
            mStaticParams.useBSDFSampling = value;
        else if (key == kUseNEE)
            mStaticParams.useNEE = value;
        else if (key == kUseMIS)
            mStaticParams.useMIS = value;

        // WAR parameters
        else if (key == kUseWAR)
            mStaticParams.useWAR = value;
        else if (key == kAuxSampleCount)
            mStaticParams.auxSampleCount = value;
        else if (key == kLog10vMFConcentration)
            mStaticParams.log10vMFConcentration = value;
        else if (key == kLog10vMFConcentrationScreen)
            mStaticParams.log10vMFConcentrationScreen = value;
        else if (key == kBoundaryTermBeta)
            mStaticParams.boundaryTermBeta = value;
        else if (key == kUseAntitheticSampling)
            mStaticParams.useAntitheticSampling = value;

        else
            logWarning("Unknown property '{}' in WARDiffPathTracer properties.", key);
    }
}

Properties WARDiffPathTracer::getProperties() const
{
    Properties props;

    // Rendering parameters
    props[kSamplesPerPixel] = mStaticParams.samplesPerPixel;
    props[kMaxBounces] = mStaticParams.maxBounces;

    // Differentiable rendering parameters
    props[kDiffMode] = mStaticParams.diffMode;
    props[kDiffVarName] = mStaticParams.diffVarName;

    // Sampling parameters
    props[kSampleGenerator] = mStaticParams.sampleGenerator;
    if (mParams.useFixedSeed)
        props[kFixedSeed] = mParams.fixedSeed;
    props[kUseBSDFSampling] = mStaticParams.useBSDFSampling;
    props[kUseNEE] = mStaticParams.useNEE;
    props[kUseMIS] = mStaticParams.useMIS;

    // WAR parameters
    props[kUseWAR] = mStaticParams.useWAR;
    props[kAuxSampleCount] = mStaticParams.auxSampleCount;
    props[kLog10vMFConcentration] = mStaticParams.log10vMFConcentration;
    props[kLog10vMFConcentrationScreen] = mStaticParams.log10vMFConcentrationScreen;
    props[kBoundaryTermBeta] = mStaticParams.boundaryTermBeta;
    props[kUseAntitheticSampling] = mStaticParams.useAntitheticSampling;

    return props;
}

RenderPassReflection WARDiffPathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void WARDiffPathTracer::setFrameDim(const uint2 frameDim)
{
    auto prevFrameDim = mParams.frameDim;
    mParams.frameDim = frameDim;

    if (any(mParams.frameDim != prevFrameDim))
    {
        mVarsChanged = true;
    }
}

void WARDiffPathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mParams.frameCount = 0;
    mParams.frameDim = {};

    // Need to recreate the trace passes because the shader binding table changes.
    mpTracePass = nullptr;

    resetLighting();

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logError("WARDiffPathTracer: This render pass does not support custom primitives.");
        }

        mRecompile = true;
    }
}

void WARDiffPathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!beginFrame(pRenderContext, renderData))
        return;

    // Update shader program specialization.
    updatePrograms();

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    // Prepare the differentiable path tracer parameter block.
    // This should be called after all resources have been created.
    prepareDiffPathTracer(renderData);

    // Trace pass.
    FALCOR_ASSERT(mpTracePass);
    tracePass(pRenderContext, renderData, *mpTracePass);

    endFrame(pRenderContext, renderData);
}

void WARDiffPathTracer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    // Rendering options.
    dirty |= renderRenderingUI(widget);

    // Debug options.
    dirty |= renderDebugUI(widget);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

bool WARDiffPathTracer::renderRenderingUI(Gui::Widgets& widget)
{
    bool dirty = false;
    bool runtimeDirty = false;

    dirty |= widget.var("Samples/pixel", mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);
    widget.tooltip("Number of samples per pixel. One path is traced for each sample.");

    dirty |= widget.var("Max bounces", mStaticParams.maxBounces, 0u, kBounceLimit);
    widget.tooltip("Maximum number of surface bounces\n0 = direct only\n1 = one indirect bounce etc.");

    // Differentiable rendering options.

    dirty |= widget.dropdown("Diff mode", mStaticParams.diffMode);
    widget.text("Diff variable name: " + mStaticParams.diffVarName);

    dirty |= widget.checkbox("Antithetic sampling", mStaticParams.useAntitheticSampling);
    widget.tooltip(
        "Use antithetic sampling.\n"
        "When enabled, two correlated paths are traced per pixel, one with the original sample and one with the negated sample.\n"
        "This can be used to reduce variance in gradient estimation."
    );

    // Sampling options.

    if (widget.dropdown("Sample generator", SampleGenerator::getGuiDropdownList(), mStaticParams.sampleGenerator))
    {
        mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);
        dirty = true;
    }

    dirty |= widget.checkbox("BSDF importance sampling", mStaticParams.useBSDFSampling);
    widget.tooltip(
        "BSDF importance sampling should normally be enabled.\n\n"
        "If disabled, cosine-weighted hemisphere sampling is used for debugging purposes"
    );

    dirty |= widget.checkbox("Next-event estimation (NEE)", mStaticParams.useNEE);
    widget.tooltip("Use next-event estimation.\nThis option enables direct illumination sampling at each path vertex.");

    if (mStaticParams.useNEE)
    {
        dirty |= widget.checkbox("Multiple importance sampling (MIS)", mStaticParams.useMIS);
        widget.tooltip(
            "When enabled, BSDF sampling is combined with light sampling for the environment map and emissive lights.\n"
            "Note that MIS has currently no effect on analytic lights."
        );
    }

    if (dirty)
        mRecompile = true;
    return dirty || runtimeDirty;
}

bool WARDiffPathTracer::renderDebugUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Debugging"))
    {
        dirty |= group.checkbox("Use fixed seed", mParams.useFixedSeed);
        group.tooltip(
            "Forces a fixed random seed for each frame.\n\n"
            "This should produce exactly the same image each frame, which can be useful for debugging."
        );
        if (mParams.useFixedSeed)
        {
            dirty |= group.var("Seed", mParams.fixedSeed);
        }

        mpPixelDebug->renderUI(group);
    }

    return dirty;
}

bool WARDiffPathTracer::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpPixelDebug->onMouseEvent(mouseEvent);
}

WARDiffPathTracer::TracePass::TracePass(
    ref<Device> pDevice,
    const std::string& name,
    const std::string& passDefine,
    const ref<Scene>& pScene,
    const DefineList& defines,
    const TypeConformanceList& globalTypeConformances
)
    : name(name), passDefine(passDefine)
{
    const uint32_t kRayTypeScatter = 0;
    const uint32_t kMissScatter = 0;

    ProgramDesc desc;
    desc.addShaderModules(pScene->getShaderModules());
    desc.addShaderLibrary(kShaderFile);
    desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
    desc.setMaxAttributeSize(pScene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    if (!pScene->hasProceduralGeometry())
        desc.setRtPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

    // Create ray tracing binding table.
    pBindingTable = RtBindingTable::create(0, 1, pScene->getGeometryCount());

    // Specify entry point for raygen shader.
    // The raygen shader needs type conformances for *all* materials in the scene.
    pBindingTable->setRayGen(desc.addRayGen("rayGen", globalTypeConformances));

    pProgram = Program::create(pDevice, desc, defines);
}

void WARDiffPathTracer::TracePass::prepareProgram(ref<Device> pDevice, const DefineList& defines)
{
    FALCOR_ASSERT(pProgram != nullptr && pBindingTable != nullptr);
    pProgram->setDefines(defines);
    if (!passDefine.empty())
        pProgram->addDefine(passDefine);
    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);
}

void WARDiffPathTracer::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mRecompile == false)
        return;

    auto defines = mStaticParams.getDefines(*this);
    auto globalTypeConformances = mpScene->getTypeConformances();

    // Create trace pass.
    mpTracePass = std::make_unique<TracePass>(mpDevice, "tracePass", "", mpScene, defines, globalTypeConformances);
    mpTracePass->prepareProgram(mpDevice, defines);

    // Create compute passes.
    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(globalTypeConformances);

    // TODO: Create preprocessing compute passes for WAR.

    mVarsChanged = true;
    mRecompile = false;
}

void WARDiffPathTracer::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    // TODO: Prepare buffers for WAR.
}

void WARDiffPathTracer::prepareDiffPathTracer(const RenderData& renderData)
{
    if (!mpDiffPTBlock || mVarsChanged)
    {
        auto pReflection = mpTracePass->pProgram->getReflector();
        auto pBlockReflection = pReflection->getParameterBlock("gDiffPTData");
        FALCOR_ASSERT(pBlockReflection);
        mpDiffPTBlock = ParameterBlock::create(mpDevice, pBlockReflection);
        FALCOR_ASSERT(mpDiffPTBlock);
        mVarsChanged = true;
    }

    // Bind resources.
    auto var = mpDiffPTBlock->getRootVar();
    bindShaderData(var, renderData);
}

void WARDiffPathTracer::resetLighting()
{
    // We only use a uniform emissive sampler for now.
    mpEmissiveSampler = nullptr;
    mRecompile = true;
}

void WARDiffPathTracer::prepareMaterials(RenderContext* pRenderContext)
{
    // This functions checks for material changes and performs any necessary update.
    // For now all we need to do is to trigger a recompile so that the right defines get set.
    // In the future, we might want to do additional material-specific setup here.

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded))
    {
        mRecompile = true;
    }
}

bool WARDiffPathTracer::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }

    if (mpScene->useEnvLight())
    {
        logError("WARDiffPathTracer: This render pass does not support environment lights.");
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getLightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            FALCOR_ASSERT(!mpEmissiveSampler);

            // We only use a uniform emissive sampler for now.
            // LightBVH seems buggy with the Cornell box bunny example.
            mpEmissiveSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene);

            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            // We only use a uniform emissive sampler for now.
            mpEmissiveSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext);
        auto defines = mpEmissiveSampler->getDefines();
        if (mpTracePass && mpTracePass->pProgram->addDefines(defines))
            mRecompile = true;
    }

    return lightingChanged;
}

void WARDiffPathTracer::bindShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling) const
{
    var["params"].setBlob(mParams);

    if (useLightSampling && mpEmissiveSampler)
    {
        mpEmissiveSampler->bindShaderData(var["emissiveSampler"]);
    }
}

bool WARDiffPathTracer::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Reset output textures.
    bool dontClearOutputs = mStaticParams.diffMode == DiffMode::BackwardDiff && mParams.runBackward == 1;
    if (!dontClearOutputs)
    {
        for (const auto& channel : kOutputChannels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            pRenderContext->clearUAV(pTexture->getUAV().get(), float4(0.f));
        }
    }

    const auto& pOutputColor = renderData.getTexture("color");
    FALCOR_ASSERT(pOutputColor);

    // Set output frame dimension.
    setFrameDim(uint2(pOutputColor->getWidth(), pOutputColor->getHeight()));

    // Validate all I/O sizes match the expected size.
    // If not, we'll disable the path tracer to give the user a chance to fix the configuration before re-enabling it.
    bool resolutionMismatch = false;
    auto validateChannels = [&](const auto& channels)
    {
        for (const auto& channel : channels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            if (pTexture && (pTexture->getWidth() != mParams.frameDim.x || pTexture->getHeight() != mParams.frameDim.y))
                resolutionMismatch = true;
        }
    };
    validateChannels(kInputChannels);
    validateChannels(kOutputChannels);

    if (mEnabled && resolutionMismatch)
    {
        logError("WARDiffPathTracer I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    if (mpScene == nullptr || !mEnabled)
    {
        // Set refresh flag if changes that affect the output have occured.
        // This is needed to ensure other passes get notified when the path tracer is enabled/disabled.
        if (mOptionsChanged)
        {
            auto& dict = renderData.getDictionary();
            auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
            if (mOptionsChanged)
                flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
            dict[Falcor::kRenderPassRefreshFlags] = flags;
        }

        return false;
    }

    // Update materials.
    prepareMaterials(pRenderContext);

    // Update the emissive sampler to the current frame.
    bool lightingChanged = prepareLighting(pRenderContext);

    // Update refresh flag if changes that affect the output have occured.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged || lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        if (mOptionsChanged)
            flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        if (lightingChanged)
            flags |= Falcor::RenderPassRefreshFlags::LightingChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    mpPixelDebug->beginFrame(pRenderContext, mParams.frameDim);

    // Update the random seed.
    mParams.seed = mParams.useFixedSeed ? mParams.fixedSeed : mParams.frameCount;
    return true;
}

void WARDiffPathTracer::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpPixelDebug->endFrame(pRenderContext);

    mVarsChanged = false;
    mParams.frameCount++;
}

void WARDiffPathTracer::tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass)
{
    FALCOR_PROFILE(pRenderContext, tracePass.name);

    FALCOR_ASSERT(tracePass.pProgram != nullptr && tracePass.pBindingTable != nullptr && tracePass.pVars != nullptr);

    // Bind global resources.
    auto var = tracePass.pVars->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);

    if (mVarsChanged)
        mpSampleGenerator->bindShaderData(var);

    mpPixelDebug->prepareProgram(tracePass.pProgram, var);

    // Bind the differentiable path tracer data block;
    var["gDiffPTData"] = mpDiffPTBlock;

    var["gDiffDebug"].setBlob(mDiffDebugParams);
    var["dLdI"] = mpdLdI;
    if (mpSceneGradients)
        mpSceneGradients->bindShaderData(var["gSceneGradients"]);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : kInputChannels)
        bind(channel);
    for (auto channel : kOutputChannels)
        bind(channel);

    // Full screen dispatch.
    mpScene->raytrace(pRenderContext, tracePass.pProgram.get(), tracePass.pVars, uint3(mParams.frameDim, 1));
}

DefineList WARDiffPathTracer::StaticParams::getDefines(const WARDiffPathTracer& owner) const
{
    DefineList defines;

    defines.add("SAMPLES_PER_PIXEL", std::to_string(samplesPerPixel));
    defines.add("MAX_BOUNCES", std::to_string(maxBounces));

    defines.add("DIFF_MODE", std::to_string((uint32_t)diffMode));
    defines.add(diffVarName);

    defines.add("USE_BSDF_SAMPLING", useBSDFSampling ? "1" : "0");
    defines.add("USE_NEE", useNEE ? "1" : "0");
    defines.add("USE_MIS", useMIS ? "1" : "0");

    // WAR parameters configuration.
    defines.add("USE_WAR", useWAR ? "1" : "0");
    defines.add("AUX_SAMPLE_COUNT", std::to_string(auxSampleCount));
    defines.add("LOG10_VMF_CONCENTRATION", std::to_string(log10vMFConcentration));
    defines.add("LOG10_VMF_CONCENTRATION_SCREEN", std::to_string(log10vMFConcentrationScreen));
    defines.add("BOUNDARY_TERM_BETA", std::to_string(boundaryTermBeta));
    defines.add("USE_ANTITHETIC_SAMPLING", useAntitheticSampling ? "1" : "0");
    defines.add("HARMONIC_GAMMA", std::to_string(harmonicGamma));

    // Sampling utilities configuration.
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    if (owner.mpEmissiveSampler)
        defines.add(owner.mpEmissiveSampler->getDefines());

    // Scene-specific configuration.
    const auto& scene = owner.mpScene;
    if (scene)
        defines.add(scene->getSceneDefines());
    defines.add("USE_ENV_LIGHT", scene && scene->useEnvLight() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", scene && scene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", scene && scene->useEmissiveLights() ? "1" : "0");

    return defines;
}

void WARDiffPathTracer::registerBindings(pybind11::module& m)
{
    if (!pybind11::hasattr(m, "DiffMode"))
    {
        pybind11::enum_<DiffMode> diffMode(m, "DiffMode");
        diffMode.value("Primal", DiffMode::Primal);
        diffMode.value("BackwardDiff", DiffMode::BackwardDiff);
        diffMode.value("ForwardDiffDebug", DiffMode::ForwardDiffDebug);
        diffMode.value("BackwardDiffDebug", DiffMode::BackwardDiffDebug);
    }

    pybind11::class_<WARDiffPathTracer, RenderPass, ref<WARDiffPathTracer>> pass(m, "WARDiffPathTracer");
    pass.def_property("scene_gradients", &WARDiffPathTracer::getSceneGradients, &WARDiffPathTracer::setSceneGradients);
    pass.def_property("run_backward", &WARDiffPathTracer::getRunBackward, &WARDiffPathTracer::setRunBackward);
    pass.def_property("dL_dI", &WARDiffPathTracer::getdLdI, &WARDiffPathTracer::setdLdI);
}

void WARDiffPathTracer::setDiffDebugParams(DiffVariableType varType, uint2 id, uint32_t offset, float4 grad)
{
    mDiffDebugParams.varType = varType;
    mDiffDebugParams.id = id;
    mDiffDebugParams.offset = offset;
    mDiffDebugParams.grad = grad;
}
