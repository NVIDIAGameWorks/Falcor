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
#include "RTXGIPass.h"
#include "RenderGraph/RenderPassLibrary.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

const RenderPass::Info RTXGIPass::kInfo { "RTXGIPass", "Indirect diffuse lighing using RTXGI." };

namespace
{
    const std::string kComputeIndirectPassFilename = "RenderPasses/RTXGIPass/ComputeIndirect.cs.slang";

    // Probe visualizer
    const uint32_t kMaxPayloadSizeBytes = 4;
    const uint32_t kMaxAttributeSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1; // CHS/Miss only

    const std::string kSphereMeshFilename = "sphere.fbx";
    const std::string kVisualizeDirectFilename = "RenderPasses/RTXGIPass/VisualizeDirect.rt.slang";
    const std::string kVisualizeProbesFilename = "RenderPasses/RTXGIPass/VisualizeProbes.3d.slang";
    const std::string kVisualizeBackgroundFilename = "RenderPasses/RTXGIPass/VisualizeBackground.cs.slang";

    // List of inputs we need.
    const std::string kDepthChannel = "depth";

    const ChannelList kGBufferInputChannels =
    {
        { "posW",           "gWorldPosition",             "World-space position (xyz) and foreground flag (w)"  },
        { "normalW",        "gWorldShadingNormal",        "World-space shading normal (xyz)"                    },
        { "tangentW",       "gWorldShadingTangent",       "World-space shading tangent (xyz) and sign (w)",     },
        { "faceNormalW",    "gWorldFaceNormal",           "Face normal in world space (xyz)",                   },
        { "texC",           "gTextureCoord",              "Texture coordinate",                                 },
        { "texGrads",       "gTextureGrads",              "Texture gradients", true /* optional */              },
        { "mtlData",        "gMaterialData",              "Material data"                                       },
        { kDepthChannel,    "",                           "Depth buffer", true /* optional */                   },
    };

    const Falcor::ChannelList kVBufferInputChannels =
    {
        { "vbuffer",        "gVBuffer",                   "Visibility buffer in packed format"                  },
        { kDepthChannel,    "",                           "Depth buffer", true /* optional */                   },
    };

    const ChannelDesc kOutput = { "output", "gOutput", "Indirect diffuse illumination", true /* optional */, ResourceFormat::RGBA32Float };

    // Scripting options.
    const char* kEnablePass = "enablePass";
    const char* kUseVBuffer = "useVBuffer";
    const char* kVolumeOptions = "volumeOptions";

    // UI elements.
    const Gui::DropdownList kVisualizerSceneModeList =
    {
        { (uint32_t)VisualizerSceneMode::IndirectDiffuse, "Indirect diffuse" },
        { (uint32_t)VisualizerSceneMode::DirectDiffuse, "Direct diffuse" },
        { (uint32_t)VisualizerSceneMode::RadianceTexture, "Radiance texture" },
        { (uint32_t)VisualizerSceneMode::Albedo, "Albedo" },
        { (uint32_t)VisualizerSceneMode::Normals, "Normals" },
    };

    const Gui::DropdownList kVisualizerProbeModeList =
    {
        { (uint32_t)VisualizerProbeMode::Direction, "Direction" },
        { (uint32_t)VisualizerProbeMode::Irradiance, "Irradiance" },
    };
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerPass(RTXGIPass::kInfo, RTXGIPass::create);
}

RTXGIPass::SharedPtr RTXGIPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new RTXGIPass(dict));
}

RTXGIPass::RTXGIPass(const Dictionary& dict)
    : RenderPass(kInfo)
{
    parseDictionary(dict);

    mInputChannels = mUseVBuffer ? kVBufferInputChannels : kGBufferInputChannels;

    init();
}

void RTXGIPass::init()
{
    // Load scene with a single unit sphere to use as probe proxy.
    mpSphereScene = SceneBuilder::create(kSphereMeshFilename, SceneBuilder::Flags::Force32BitIndices)->getScene();
    if (mpSphereScene->getGeometryInstanceCount() != 1)
    {
        throw RuntimeError("RTXGIPass: Expected sphere scene to have one mesh instance");
    }

    // Create probe visualization pass.
    Program::Desc progDesc;
    progDesc.addShaderLibrary(kVisualizeProbesFilename).vsEntry("vsMain").psEntry("psMain");
    progDesc.setShaderModel("6_2");
    mVisualizeProbes.pProgram = GraphicsProgram::create(progDesc, mpSphereScene->getSceneDefines());
    mVisualizeProbes.pVars = GraphicsVars::create(mVisualizeProbes.pProgram.get());

    if (!mVisualizeProbes.pVars->setParameterBlock("gScene", mpSphereScene->getParameterBlock()))
    {
        throw RuntimeError("RTXGIPass: Failed to bind parameter block");
    }

    // Setup grapics state.
    mVisualizeProbes.pState = GraphicsState::create();
    mVisualizeProbes.pState->setProgram(mVisualizeProbes.pProgram);

    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(RasterizerState::CullMode::Back); // Default, but set it for clarity.
    mVisualizeProbes.pState->setRasterizerState(RasterizerState::create(rsDesc));

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(DepthStencilState::Func::Less); // Default, but set it for clarity.
    mVisualizeProbes.pState->setDepthStencilState(DepthStencilState::create(dsDesc));

    mVisualizeProbes.pState->setVao(mpSphereScene->getMeshVao());
    mVisualizeProbes.pState->setFbo(Fbo::create()); // Set empty FBO, we'll populate it later when we have the buffers.
}

void RTXGIPass::parseDictionary(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kEnablePass) mEnablePass = value;
        else if (key == kUseVBuffer) mUseVBuffer = value;
        else if (key == kVolumeOptions) mVolumeOptions = value;
        else logWarning("Unknown field '{}' in RTXGIPass dictionary.", key);
    }
}

Dictionary RTXGIPass::getScriptingDictionary()
{
    Dictionary d;
    d[kEnablePass] = mEnablePass;
    d[kUseVBuffer] = mUseVBuffer;
    d[kVolumeOptions] = mpVolume->getOptions();
    return d;
}

RenderPassReflection RTXGIPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    auto& output = reflector.addOutput(kOutput.name, kOutput.desc).format(kOutput.format);
    output.bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget);
    if (kOutput.optional) output.flags(RenderPassReflection::Field::Flags::Optional);

    addRenderPassInputs(reflector, mInputChannels);

    return reflector;
}

void RTXGIPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;

    mpDepthStencil = nullptr;
    if (!compileData.connectedResources.getField(kDepthChannel))
    {
        mpDepthStencil = Texture::create2D(mFrameDim.x, mFrameDim.y, ResourceFormat::D32Float, 1, 1, nullptr, Resource::BindFlags::DepthStencil);
    }
}

void RTXGIPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mpVolume = nullptr;

    // Clear old data and vars to trigger re-creation.
    mpVisualizeBackground = nullptr;
    mpComputeIndirectPass = nullptr;
    mVisualizeDirect.pProgram = nullptr;
    mVisualizeDirect.pVars = nullptr;

    if (mpScene)
    {
        if (mpScene->hasProceduralGeometry())
        {
            logWarning("RTXGIPass: This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        auto shaderModules = mpScene->getShaderModules();
        auto typeConformances = mpScene->getTypeConformances();
        Shader::DefineList defines = mpScene->getSceneDefines();

        // Create background visualization pass.
        {
            Program::Desc desc;
            desc.addShaderModules(shaderModules);
            desc.addShaderLibrary(kVisualizeBackgroundFilename).csEntry("main");
            desc.addTypeConformances(typeConformances);
            mpVisualizeBackground = ComputePass::create(desc, defines, false);
        }

        // Both programs below optionally use the V-buffer. Configure its usage via a define.
        defines.add("USE_VBUFFER", mUseVBuffer ? "1" : "0");

        // Create program for the indirect illumination pass.
        {
            Program::Desc desc;
            desc.addShaderModules(shaderModules);
            desc.addShaderLibrary(kComputeIndirectPassFilename).csEntry("main");
            desc.addTypeConformances(typeConformances);
            mpComputeIndirectPass = ComputePass::create(desc, defines, false);
        }

        // Create direct lighting visualization pass.
        RtProgram::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kVisualizeDirectFilename);
        desc.addTypeConformances(typeConformances);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(kMaxAttributeSizeBytes);
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mVisualizeDirect.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& sbt = mVisualizeDirect.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("shadowMiss"));
        sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowAnyHit"));

        mVisualizeDirect.pProgram = RtProgram::create(desc, defines);

        // Heuristic to set default probe radius for visualization.
        // Determine size based on shortest axis to avoid long, skinny scenes to get too large probe radiuses.
        auto extent = mpScene->getSceneBounds().extent();
        float minExtent = std::min(std::min(extent.x, extent.y), extent.z);
        mVisualizerOptions.probeRadius = std::max(0.001f, minExtent / 50.f);
    }
}

void RTXGIPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mEnablePass == false || mpScene == nullptr)
    {
        Texture::SharedPtr pOutput = renderData.getTexture(kOutput.name);
        pRenderContext->clearRtv(pOutput->getRTV().get(), float4(0));
        return;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        throw RuntimeError("RTXGIPass: This render pass does not support scene geometry changes.");
    }

    // Check if GBuffer has adjusted shading normals enabled.
    auto& dict = renderData.getDictionary();
    mGBufferAdjustShadingNormals = dict.getValue(Falcor::kRenderPassGBufferAdjustShadingNormals, false);

    // Keep a copy of the volume options.
    if (mpVolume) mVolumeOptions = mpVolume->getOptions();

    // Recreate volume if render settings or env map changed.
    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::EnvMapChanged))
    {
        mpVolume = nullptr;
    }

    // Update light sampling options.
    mVolumeOptions.enableAnalyticLights = mpScene->useAnalyticLights();
    mVolumeOptions.enableEmissiveLights = mpScene->useEmissiveLights();
    mVolumeOptions.enableEnvMap = mpScene->useEnvLight();

    // (Re-)create volume if requested or light sampler has changed.
    // Note that RTXGIVolume creates its own envmap/emissive sampler if none is supplied.
    if (mpVolume == nullptr ||
        mpVolume->getOptions().enableEnvMap != mVolumeOptions.enableEnvMap ||
        mpVolume->getOptions().enableAnalyticLights != mVolumeOptions.enableAnalyticLights ||
        mpVolume->getOptions().enableEmissiveLights != mVolumeOptions.enableEmissiveLights)
    {
        mpVolume = RTXGIVolume::create(pRenderContext, mpScene, nullptr, nullptr, mVolumeOptions);

        mVisualizeDirect.pProgram->addDefines(mpVolume->getSampleGenerator()->getDefines());
        mVisualizeDirect.pProgram->addDefines(mpVolume->getDirectLightingDefines());
        mVisualizeDirect.pVars = nullptr;
    }

    // Update RTXGI.
    mpVolume->update(pRenderContext);

    // Render indirect lighting.
    if (!mVisualizerOptions.enableVisualizer || mVisualizerOptions.sceneMode == VisualizerSceneMode::IndirectDiffuse)
    {
        computeIndirectPass(pRenderContext, renderData);
    }

    // Render visualizer.
    if (mVisualizerOptions.enableVisualizer)
    {
        probeVisualizerPass(pRenderContext, renderData);
    }

    mFrameCount++;
}

void RTXGIPass::bindPassIO(ShaderVar var, const RenderData& renderData) const
{
    for (const auto& it : mInputChannels)
    {
        // Only bind if variable exists, we don't need all G-buffer channels
        if (!it.texname.empty() && var.findMember(it.texname).isValid())
        {
            var[it.texname] = renderData.getTexture(it.name);
        }
    }

    Texture::SharedPtr pOutput = renderData.getTexture(kOutput.name);
    var[kOutput.texname] = pOutput;
}

void RTXGIPass::probeVisualizerPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE("probeVisualizerPass");

    Texture::SharedPtr pOutput = renderData.getTexture(kOutput.name);
    Texture::SharedPtr pDepthStencil = renderData.getTexture(kDepthChannel);

    if (pDepthStencil == nullptr)
    {
        // Depth input wasn't bound to pass, use internal depth buffer instead.
        FALCOR_ASSERT(mpDepthStencil && mpDepthStencil->getWidth() == renderData.getDefaultTextureDims().x && mpDepthStencil->getHeight() == renderData.getDefaultTextureDims().y);
        pRenderContext->clearDsv(mpDepthStencil->getDSV().get(), 1.f, 0);
        pDepthStencil = mpDepthStencil;

        // TODO: Run a depth pre-pass here to populate the depth buffer. Issue a warning for now.
        logWarning("No depth buffer bound. Probe visualization will ignore depth.");
    }

    // Draw the scene as background to the probe visualization.
    // We have two separate shaders for this: one for direct illumination evaluation
    // and one for everything else. This might be unified in the future.
    // If the selected mode is to show indirect illumination, we use the existing output.
    if (mVisualizerOptions.sceneMode == VisualizerSceneMode::DirectDiffuse)
    {
        mVisualizeDirect.pProgram->addDefine("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");

        // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
        mVisualizeDirect.pProgram->addDefines(getValidResourceDefines(mInputChannels, renderData));

        // Re-create the program vars if needed.
        if (mVisualizeDirect.pVars == nullptr)
        {
            mVisualizeDirect.pVars = RtProgramVars::create(mVisualizeDirect.pProgram, mVisualizeDirect.pBindingTable);

            auto var = mVisualizeDirect.pVars->getRootVar();
            mpVolume->setShaderData(var);
            mpVolume->getSampleGenerator()->setShaderData(var);
        }

        // Bind the resources.
        auto var = mVisualizeDirect.pVars->getRootVar();
        bindPassIO(var, renderData);
        mpVolume->setDirectLightingShaderData(var["CB"]["gDirectLighting"]);

        var["CB"]["gFrameDim"] = mFrameDim;
        var["CB"]["gFrameCount"] = mFrameCount;

        // Trace the rays
        mpScene->raytrace(pRenderContext, mVisualizeDirect.pProgram.get(), mVisualizeDirect.pVars, uint3(mFrameDim, 1));
    }
    else if (mVisualizerOptions.sceneMode != VisualizerSceneMode::IndirectDiffuse)
    {
        mpVisualizeBackground->addDefine("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");

        // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
        mpVisualizeBackground->getProgram()->addDefines(getValidResourceDefines(mInputChannels, renderData));

        if (!mpVisualizeBackground->hasVars()) mpVisualizeBackground->setVars(nullptr); // Re-create vars

        // Bind the resources.
        auto var = mpVisualizeBackground->getRootVar();
        bindPassIO(var, renderData);

        const auto& rayDataTexture = mpVolume->getRayDataTexture();

        var["gScene"] = mpScene->getParameterBlock(); // Bind the scene manually since it's a fullscreen compute pass.
        var["gRadianceTex"] = rayDataTexture;
        var["CB"]["gFrameDim"] = mFrameDim;
        var["CB"]["gRadianceTexDim"] = uint2(rayDataTexture->getWidth(), rayDataTexture->getHeight());
        var["CB"]["gVisualizerSceneMode"] = (uint32_t)mVisualizerOptions.sceneMode;

        // Run the program.
        mpVisualizeBackground->execute(pRenderContext, uint3(mFrameDim, 1));
    }

    // Rasterize the probes on top of the scene.
    // There is one special case here: when directly viewing the radiance
    // texture in 2D, we don't really want an overlay with the probes.
    if (mVisualizerOptions.sceneMode != VisualizerSceneMode::RadianceTexture)
    {
        // Bind the resources.
        auto pFbo = mVisualizeProbes.pState->getFbo();
        pFbo->attachColorTarget(pOutput, 0);
        pFbo->attachDepthStencilTarget(pDepthStencil);
        mVisualizeProbes.pState->setFbo(pFbo); // Sets the viewport

        mpVolume->setShaderData(mVisualizeProbes.pVars->getRootVar());

        auto var = mVisualizeProbes.pVars->getRootVar()["PerFrameCB"];
        mpScene->getCamera()->setShaderData(var["gCamera"]);
        var["gVisualizerProbeMode"] = (uint32_t)mVisualizerOptions.probeMode;
        var["gShowProbeStates"] = mVisualizerOptions.showProbeStates;
        var["gHighlightProbe"] = mVisualizerOptions.highlightProbe;
        var["gProbeIndex"] = mVisualizerOptions.probeIndex;
        var["gProbeRadius"] = mVisualizerOptions.probeRadius;
        var["gFrameCount"] = mFrameCount;

        // Draw probe visualization mesh with one instance per probe.
        // Note we can't use Scene::render() as it only draws a single instance.
        uint32_t probeCount = mpVolume->getProbeCount();
        const auto& mesh = mpSphereScene->getMesh(MeshID{ 0 });
        FALCOR_ASSERT(mesh.indexCount > 0);
        // TODO: We're drawing more instances here than the VAO is setup for so we will get validation errors.
        pRenderContext->drawIndexedInstanced(mVisualizeProbes.pState.get(), mVisualizeProbes.pVars.get(), mesh.indexCount, probeCount, mesh.ibOffset, mesh.vbOffset, 0);
    }
}

void RTXGIPass::computeIndirectPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE("computeIndirectPass");

    mpComputeIndirectPass->addDefine("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    mpComputeIndirectPass->getProgram()->addDefines(getValidResourceDefines(mInputChannels, renderData));

    if (!mpComputeIndirectPass->hasVars()) mpComputeIndirectPass->setVars(nullptr); // Re-create vars

    auto camera = mpScene->getCamera();
    if (camera->getApertureRadius() > 0.f)
    {
        logWarning("Depth-of-field is enabled but RTXGIPass assumes a pinhole camera. Expect incorrect shading.");
    }

    // Bind the resources.
    auto var = mpComputeIndirectPass->getRootVar();
    bindPassIO(var, renderData);

    var["gScene"] = mpScene->getParameterBlock(); // Bind the scene manually since it's a fullscreen compute pass.
    var["CB"]["gFrameDim"] = mFrameDim;
    var["CB"]["gFrameCount"] = mFrameCount;

    mpVolume->setShaderData(var);

    // Run the pass
    mpComputeIndirectPass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void RTXGIPass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enable RTXGI", mEnablePass);
    if (!mEnablePass) return;

    if (mpVolume == nullptr) return;

    if (auto rtxgiGroup = widget.group("RTXGI", true))
    {
        mpVolume->renderUI(rtxgiGroup);
    }

    if (auto visualizerGroup = widget.group("Visualizer", true))
    {
        renderVisualizerUI(visualizerGroup);
    }
}

void RTXGIPass::renderVisualizerUI(Gui::Widgets& widget)
{
    FALCOR_ASSERT(mpVolume);

    widget.checkbox("Enable", mVisualizerOptions.enableVisualizer);
    if (!mVisualizerOptions.enableVisualizer) return;

    widget.dummy("#spacing0", { 1, 8 });
    widget.text("Scene visualization mode:");
    widget.tooltip("Selects how the scene is drawn as background to the probe visualization:\n\n"
        "Indirect diffuse\n"
        "    the scene is visualized using the indirect diffuse lighting\n"
        "    obtained from evaluating the irradiance probes.\n\n"
        "Direct diffuse\n"
        "    the scene is visualized using the same direct illumination\n"
        "    solution as is used to feed the probe radiance updates.\n\n"
        "Radiance texture\n"
        "    the probe update rays radiance texture. This is the randomly sampled\n"
        "    lighting we feed back to RTXGI for it to update the irradiance probes.\n\n"
        "Albedo\n"
        "    the perfectly diffuse albedo (total diffuse reflectivity).\n\n"
        "Normals\n"
        "    the shading normals.");

    widget.dropdown("#SceneMode", kVisualizerSceneModeList, (uint32_t&)mVisualizerOptions.sceneMode);

    if (mVisualizerOptions.sceneMode != VisualizerSceneMode::RadianceTexture)
    {
        widget.dummy("#spacing0", { 1, 8 });
        widget.text("Probe visualization mode:");
        widget.tooltip("Selects how the probes are drawn in the probe visualization.");

        widget.dropdown("#ProbeMode", kVisualizerProbeModeList, (uint32_t&)mVisualizerOptions.probeMode);
        widget.var("Probe radius", mVisualizerOptions.probeRadius, 0.f);
        widget.checkbox("Show probe states", mVisualizerOptions.showProbeStates);
        widget.tooltip(
            "Green = Active probe\n"
            "Red = Inactive probe");
        widget.checkbox("Highlight probe", mVisualizerOptions.highlightProbe);
        widget.tooltip("The probe with the selected index is flashing");
        if (mVisualizerOptions.highlightProbe)
        {
            widget.var("Probe index", mVisualizerOptions.probeIndex, 0u, mpVolume->getProbeCount() - 1);
        }
    }
}
