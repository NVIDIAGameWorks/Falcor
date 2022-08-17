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
#include "WhittedRayTracer.h"
#include "RenderGraph/RenderPassLibrary.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

const RenderPass::Info WhittedRayTracer::kInfo { "WhittedRayTracer", "Simple Whitted ray tracer." };

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

namespace
{
    const char kShaderFile[] = "RenderPasses/WhittedRayTracer/WhittedRayTracer.rt.slang";

    const char kMaxBounces[] = "maxBounces";
    const char kTexLODMode[] = "texLODMode";
    const char kRayConeMode[] = "rayConeMode";
    const char kRayConeFilterMode[] = "rayConeFilterMode";
    const char kRayDiffFilterMode[] = "rayDiffFilterMode";
    const char kUseRoughnessToVariance[] = "useRoughnessToVariance";

    const Gui::DropdownList kTexLODModeList =
    {
        { uint32_t(TexLODMode::Mip0), "Mip0" },
        { uint32_t(TexLODMode::RayCones), "Ray cones" },
        { uint32_t(TexLODMode::RayDiffs), "Ray diffs" },
    };

    const Gui::DropdownList kRayFootprintFilterModeList =
    {
        { uint32_t(RayFootprintFilterMode::Isotropic), "Isotropic" },
        { uint32_t(RayFootprintFilterMode::Anisotropic), "Anisotropic" },
        { uint32_t(RayFootprintFilterMode::AnisotropicWhenRefraction), "Anisotropic only for refraction" },
    };

    const Gui::DropdownList kRayConeModeList =
    {
        { uint32_t(RayConeMode::Combo), "Combo" },
        { uint32_t(RayConeMode::Unified), "Unified" },
    };

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 164;
    const uint32_t kMaxAttributeSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 2;

    const ChannelList kOutputChannels =
    {
        { "color",          "gOutputColor",               "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    };

    const ChannelList kInputChannels =
    {
        { "posW",           "gWorldPosition",             "World-space position (xyz) and foreground flag (w)"       },
        { "normalW",        "gWorldShadingNormal",        "World-space shading normal (xyz)"                         },
        { "tangentW",       "gWorldShadingTangent",       "World-space shading tangent (xyz) and sign (w)"           },
        { "faceNormalW",    "gWorldFaceNormal",           "Face normal in world space (xyz)",                        },
        { "texC",           "gTextureCoord",              "Texture coordinate",                                      },
        { "texGrads",       "gTextureGrads",              "Texture gradients", true /* optional */                   },
        { "mtlData",        "gMaterialData",              "Material data"                                            },
        { "vbuffer",        "gVBuffer",                   "V-buffer buffer in packed format"                         },
    };
};

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerPass(WhittedRayTracer::kInfo, WhittedRayTracer::create);
    ScriptBindings::registerBinding(WhittedRayTracer::registerBindings);
}

void WhittedRayTracer::registerBindings(pybind11::module& m)
{
    pybind11::class_<WhittedRayTracer, RenderPass, WhittedRayTracer::SharedPtr> pass(m, "WhittedRayTracer");

    pybind11::enum_<RayConeMode> rayConeMode(m, "RayConeMode");
    rayConeMode.value("Combo", RayConeMode::Combo);
    rayConeMode.value("Unified", RayConeMode::Unified);

    pybind11::enum_<RayFootprintFilterMode> rayConeFilterMode(m, "RayFootprintFilterMode");
    rayConeFilterMode.value("Isotropic", RayFootprintFilterMode::Isotropic);
    rayConeFilterMode.value("Anisotropic", RayFootprintFilterMode::Anisotropic);
    rayConeFilterMode.value("AnisotropicWhenRefraction", RayFootprintFilterMode::AnisotropicWhenRefraction);
}

WhittedRayTracer::SharedPtr WhittedRayTracer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new WhittedRayTracer(dict));
}

WhittedRayTracer::WhittedRayTracer(const Dictionary& dict)
    : RenderPass(kInfo)
{
    // Parse dictionary.
    for (const auto& [key, value] : dict)
    {
        if (key == kMaxBounces) mMaxBounces = (uint32_t)value;
        else if (key == kTexLODMode)  mTexLODMode = value;
        else if (key == kRayConeMode) mRayConeMode = value;
        else if (key == kRayConeFilterMode) mRayConeFilterMode = value;
        else if (key == kRayDiffFilterMode) mRayDiffFilterMode = value;
        else if (key == kUseRoughnessToVariance)  mUseRoughnessToVariance = value;
        else logWarning("Unknown field '{}' in a WhittedRayTracer dictionary.", key);
    }

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);

}

Dictionary WhittedRayTracer::getScriptingDictionary()
{
    Dictionary d;
    d[kMaxBounces] = mMaxBounces;
    d[kTexLODMode] = mTexLODMode;
    d[kRayConeMode] = mRayConeMode;
    d[kRayConeFilterMode] = mRayConeFilterMode;
    d[kRayDiffFilterMode] = mRayDiffFilterMode;
    d[kUseRoughnessToVariance] = mUseRoughnessToVariance;
    return d;
}

RenderPassReflection WhittedRayTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void WhittedRayTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst) pRenderContext->clearTexture(pDst);
        }
        return;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        throw RuntimeError("WhittedRayTracer: This render pass does not support scene geometry changes.");
    }

    setStaticParams(mTracer.pProgram.get());

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars) prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    // Set up screen space pixel angle for texture LOD using ray cones
    var["CB"]["gScreenSpacePixelSpreadAngle"] = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(targetDim.y);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : kInputChannels) bind(channel);
    for (auto channel : kOutputChannels) bind(channel);

    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));

    mFrameCount++;
}

void WhittedRayTracer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 10u);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    uint32_t modeIndex = static_cast<uint32_t>(mTexLODMode);
    if (widget.dropdown("Texture LOD mode", kTexLODModeList, modeIndex))
    {
        setTexLODMode(TexLODMode(modeIndex));
        dirty = true;
    }
    widget.tooltip("The texture level-of-detail mode to use.");
    if (mTexLODMode == TexLODMode::RayCones)
    {
        uint32_t modeIndex = static_cast<uint32_t>(mRayConeMode);
        if (widget.dropdown("Ray cone mode", kRayConeModeList, modeIndex))
        {
            setRayConeMode(RayConeMode(modeIndex));
            dirty = true;
        }
        widget.tooltip("The variant of ray cones to use.");

        uint32_t rayConeFilterModeIndex = static_cast<uint32_t>(mRayConeFilterMode);
        if (widget.dropdown("Ray cone filter mode", kRayFootprintFilterModeList, rayConeFilterModeIndex))
        {
            setRayConeFilterMode(RayFootprintFilterMode(rayConeFilterModeIndex));
            dirty = true;
        }
        widget.tooltip("What type of ray cone filter method to use beyond the first hit");

        dirty |= widget.checkbox("Use BDSF roughness", mUseRoughnessToVariance);
        widget.tooltip("Grow ray cones based on BDSF roughness.");

        dirty |= widget.checkbox("Visualize surface spread", mVisualizeSurfaceSpread);
        widget.tooltip("Visualize the surface spread angle for the ray cones methods times 10.");
    }

    if (mTexLODMode == TexLODMode::RayDiffs)
    {
        uint32_t rayDiffFilterModeIndex = static_cast<uint32_t>(mRayDiffFilterMode);
        if (widget.dropdown("Ray diff filter mode", kRayFootprintFilterModeList, rayDiffFilterModeIndex))
        {
            setRayDiffFilterMode(RayFootprintFilterMode(rayDiffFilterModeIndex));
            dirty = true;
        }
        widget.tooltip("What type of ray diff filter method to use beyond the first hit");
    }
    dirty |= widget.checkbox("Use Fresnel As BRDF", mUseFresnelAsBRDF);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void WhittedRayTracer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        if (mpScene->hasProceduralGeometry())
        {
            logWarning("WhittedRayTracer: This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        // Create ray tracing program.
        RtProgram::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(kMaxAttributeSizeBytes);
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));
        sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("scatterClosestHit", "scatterAnyHit"));
        sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowAnyHit"));

        mTracer.pProgram = RtProgram::create(desc, mpScene->getSceneDefines());
    }
}

void WhittedRayTracer::prepareVars()
{
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->setShaderData(var);
}

void WhittedRayTracer::setStaticParams(RtProgram* pProgram) const
{
    Program::DefineList defines;
    defines.add("MAX_BOUNCES", std::to_string(mMaxBounces));
    defines.add("TEX_LOD_MODE", std::to_string(static_cast<uint32_t>(mTexLODMode)));
    defines.add("RAY_CONE_MODE", std::to_string(static_cast<uint32_t>(mRayConeMode)));
    defines.add("VISUALIZE_SURFACE_SPREAD", mVisualizeSurfaceSpread ? "1" : "0");
    defines.add("RAY_CONE_FILTER_MODE", std::to_string(static_cast<uint32_t>(mRayConeFilterMode)));
    defines.add("RAY_DIFF_FILTER_MODE", std::to_string(static_cast<uint32_t>(mRayDiffFilterMode)));
    defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    defines.add("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    defines.add("USE_ROUGHNESS_TO_VARIANCE", mUseRoughnessToVariance ? "1" : "0");
    defines.add("USE_FRESNEL_AS_BRDF", mUseFresnelAsBRDF ? "1" : "0");
    pProgram->addDefines(defines);
}
