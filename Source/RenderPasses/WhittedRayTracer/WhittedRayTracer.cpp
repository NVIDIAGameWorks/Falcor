/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "RenderGraph/RenderPassHelpers.h"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

namespace
{
    const char kShaderFile[] = "RenderPasses/WhittedRayTracer/WhittedRayTracer.rt.slang";

    const char kTextureLODMode[] = "mode";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 164;
    const uint32_t kMaxAttributesSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 2;

    const ChannelList kOutputChannels =
    {
        { "color",          "gOutputColor",               "Output color (sum of direct and indirect)"                },
    };
};

static void regTextureLOD(pybind11::module& m)
{
    pybind11::class_<WhittedRayTracer, RenderPass, WhittedRayTracer::SharedPtr> pass(m, "WhittedRayTracer");
    pass.def_property(kTextureLODMode, &WhittedRayTracer::getTexLODMode, &WhittedRayTracer::setTexLODMode);

    pybind11::enum_<TexLODMode> texLODMode(m, "TextureLODMode");
    texLODMode.value("Mip0", TexLODMode::Mip0);
    texLODMode.value("RayCones", TexLODMode::RayCones);
    texLODMode.value("RayDiffsIsotropic", TexLODMode::RayDiffsIsotropic);
    texLODMode.value("RayDiffsAnisotropic", TexLODMode::RayDiffsAnisotropic);
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerClass("WhittedRayTracer", "Simple Whitted ray tracer", WhittedRayTracer::create);
    ScriptBindings::registerBinding(regTextureLOD);
}

WhittedRayTracer::SharedPtr WhittedRayTracer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new WhittedRayTracer(dict));
}

WhittedRayTracer::WhittedRayTracer(const Dictionary& dict)
{
    // Deserialize pass from dictionary.
    serializePass<true>(dict);

    // Create ray tracing program.
    RtProgram::Desc progDesc;
    progDesc.addShaderLibrary(kShaderFile).setRayGen("rayGen");
    progDesc.addHitGroup(0, "scatterClosestHit", "scatterAnyHit").addMiss(0, "scatterMiss");
    progDesc.addHitGroup(1, "", "shadowAnyHit").addMiss(1, "shadowMiss");
    progDesc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    mTracer.pProgram = RtProgram::create(progDesc, kMaxPayloadSizeBytes, kMaxAttributesSizeBytes);
    assert(mTracer.pProgram);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
    assert(mpSampleGenerator);

    mTexLODModes =
    {
        { uint(TexLODMode::Mip0), "Mip0" },
        { uint(TexLODMode::RayCones), "Ray cones" },
        { uint(TexLODMode::RayDiffsIsotropic), "Ray diffs (isotropic)" },
        { uint(TexLODMode::RayDiffsAnisotropic), "Ray diffs (anisotropic)" },
    };

    // Adjust input channels and dropdown to whether we use a rasterized G-buffer or a ray traced G-buffer
    if (mUsingRasterizedGBuffer)
    {
        mInputChannels =
        {
            { "posW",             "gWorldPosition",             "World-space position (xyz) and foreground flag (w)"            },
            { "normalW",          "gWorldShadingNormal",        "World-space shading normal (xyz)"                              },
            { "tangentW",         "gWorldShadingTangent",       "World-space shading tangent (xyz) and sign (w)", true /* optional */ },
            { "faceNormalW",      "gWorldFaceNormal",           "Face normal in world space (xyz)",                             },
            { "mtlDiffOpacity",   "gMaterialDiffuseOpacity",    "Material diffuse color (xyz) and opacity (w)"                  },
            { "mtlSpecRough",     "gMaterialSpecularRoughness", "Material specular color (xyz) and roughness (w)"               },
            { "mtlEmissive",      "gMaterialEmissive",          "Material emissive color (xyz)"                                 },
            { "mtlParams",        "gMaterialExtraParams",       "Material parameters (IoR, flags etc)"                          },
            { "surfSpreadAngle",  "gSurfaceSpreadAngle",        "surface spread angle (texlod)", true, ResourceFormat::R16Float },
            { "vbuffer",          "gVBuffer",                   "Visibility buffer in packed 64-bit format", true, ResourceFormat::RG32Uint },
        };
        mRayConeModes =
        {
            { uint(RayConeMode::RayTracingGems1), "Ray Tracing Gems (G-buffer)" },
            { uint(RayConeMode::Combo), "Combo" },
            { uint(RayConeMode::Unified), "Unified" },
        };
    }
    else
    {
        mInputChannels =
        {
            { "posW",             "gWorldPosition",             "World-space position (xyz) and foreground flag (w)"            },
            { "normalW",          "gWorldShadingNormal",        "World-space shading normal (xyz)"                              },
            { "tangentW",         "gWorldShadingTangent",       "World-space shading tangent (xyz) and sign (w)", true /* optional */ },
            { "faceNormalW",      "gWorldFaceNormal",           "Face normal in world space (xyz)",                             },
            { "mtlDiffOpacity",   "gMaterialDiffuseOpacity",    "Material diffuse color (xyz) and opacity (w)"                  },
            { "mtlSpecRough",     "gMaterialSpecularRoughness", "Material specular color (xyz) and roughness (w)"               },
            { "mtlEmissive",      "gMaterialEmissive",          "Material emissive color (xyz)"                                 },
            { "mtlParams",        "gMaterialExtraParams",       "Material parameters (IoR, flags etc)"                          },
            { "vbuffer",          "gVBuffer",                   "Visibility buffer in packed 64-bit format", true, ResourceFormat::RG32Uint },
        };
        mRayConeModes =
        {
            { uint(RayConeMode::Combo), "Combo" },
            { uint(RayConeMode::Unified), "Unified" },
        };
    }
}

Dictionary WhittedRayTracer::getScriptingDictionary()
{
    Dictionary dict;
    serializePass<false>(dict);
    return dict;
}

RenderPassReflection WhittedRayTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, mInputChannels);
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
            Texture* pDst = renderData[it.name]->asTexture().get();
            if (pDst) pRenderContext->clearTexture(pDst);
        }
        return;
    }

    setStaticParams(mTracer.pProgram.get());

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(mInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars) prepareVars();
    assert(mTracer.pVars);

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);

    // Set constants.
    auto pVars = mTracer.pVars;
    pVars["CB"]["gFrameCount"] = mFrameCount;
    pVars["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    // Set up screen space pixel angle for texture LOD using ray cones
    pVars["CB"]["gScreenSpacePixelSpreadAngle"] = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(targetDim.y);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            auto pGlobalVars = mTracer.pVars;
            pGlobalVars[desc.texname] = renderData[desc.name]->asTexture();
        }
    };
    for (auto channel : mInputChannels) bind(channel);
    for (auto channel : kOutputChannels) bind(channel);

    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));

    mFrameCount++;
}

void WhittedRayTracer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 5u);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    std::string text = std::string("Whitted Ray Tracer with ") + (mUsingRasterizedGBuffer ? std::string("rasterized G-buffer") : std::string("ray traced G-buffer"));
    widget.text(text);
    uint32_t modeIndex = static_cast<uint32_t>(mTexLODMode);
    if (widget.dropdown("Texture LOD mode", mTexLODModes, modeIndex))
    {
        setTexLODMode(TexLODMode(modeIndex));
        dirty = true;
    }
    widget.tooltip("The texture level-of-detail mode to use.");
    if (mTexLODMode == TexLODMode::RayCones)
    {
        uint32_t modeIndex = static_cast<uint32_t>(mRayConeMode);
        if (widget.dropdown("Ray cone mode", mRayConeModes, modeIndex))
        {
            setRayConeMode(RayConeMode(modeIndex));
            dirty = true;
        }
        widget.tooltip("The variant of ray cones to use.");

        dirty |= widget.checkbox("Use BDSF roughness", mUseRoughnessToVariance);
        widget.tooltip("Grow ray cones based on BDSF roughness.");

        dirty |= widget.checkbox("Visualize surface spread", mVisualizeSurfaceSpread);
        widget.tooltip("Visualize the surface spread angle for the ray cones methods times 10.");
    }

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
    // After changing scene, the program vars should to be recreated.
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (pScene)
    {
        mTracer.pProgram->addDefines(pScene->getSceneDefines());
    }
}

void WhittedRayTracer::prepareVars()
{
    assert(mpScene);
    assert(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());

    // Create program variables for the current program/scene.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mpScene);
    assert(mTracer.pVars);

    // Bind utility classes into shared data.
    auto pGlobalVars = mTracer.pVars->getRootVar();
    bool success = mpSampleGenerator->setShaderData(pGlobalVars);
    if (!success) throw std::exception("Failed to bind sample generator");
}

void WhittedRayTracer::setStaticParams(RtProgram* pProgram) const
{
    Program::DefineList defines;
    defines.add("MAX_BOUNCES", std::to_string(mMaxBounces));
    defines.add("TEXLOD_MODE", std::to_string(static_cast<uint32_t>(mTexLODMode)));
    defines.add("RAY_CONE_MODE", std::to_string(static_cast<uint32_t>(mRayConeMode)));
    defines.add("VISUALIZE_SURFACE_SPREAD", mVisualizeSurfaceSpread ? "1" : "0");
    defines.add("USE_RASTERIZED_GBUFFER", mUsingRasterizedGBuffer ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    defines.add("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    defines.add("USE_ROUGHNESS_TO_VARIANCE", mUseRoughnessToVariance ? "1" : "0");
    pProgram->addDefines(defines);
}
