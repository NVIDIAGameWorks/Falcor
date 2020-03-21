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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
#include "MinimalPathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("MinimalPathTracer", "Minimal path tracer", MinimalPathTracer::create);
}

namespace
{
    const char kShaderFile[] = "RenderPasses/MinimalPathTracer/MinimalPathTracer.rt.slang";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 80u;
    const uint32_t kMaxAttributesSizeBytes = 8u;
    const uint32_t kMaxRecursionDepth = 2u;

    const char kViewDirInput[] = "viewW";

    const ChannelList kInputChannels =
    {
        { "posW",           "gWorldPosition",             "World-space position (xyz) and foreground flag (w)"       },
        { "normalW",        "gWorldShadingNormal",        "World-space shading normal (xyz)"                         },
        { "bitangentW",     "gWorldShadingBitangent",     "World-space shading bitangent (xyz)", true /* optional */ },
        { "faceNormalW",    "gWorldFaceNormal",           "Face normal in world space (xyz)",                        },
        { kViewDirInput,    "gWorldView",                 "World-space view direction (xyz)", true /* optional */    },
        { "mtlDiffOpacity", "gMaterialDiffuseOpacity",    "Material diffuse color (xyz) and opacity (w)"             },
        { "mtlSpecRough",   "gMaterialSpecularRoughness", "Material specular color (xyz) and roughness (w)"          },
        { "mtlEmissive",    "gMaterialEmissive",          "Material emissive color (xyz)"                            },
        { "mtlParams",      "gMaterialExtraParams",       "Material parameters (IoR, flags etc)"                     },
    };

    const ChannelList kOutputChannels =
    {
        { "color",          "gOutputColor",               "Output color (sum of direct and indirect)"                },
    };
};

MinimalPathTracer::SharedPtr MinimalPathTracer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new MinimalPathTracer(dict));
}

MinimalPathTracer::MinimalPathTracer(const Dictionary& dict)
{
    // Deserialize pass from dictionary.
    serializePass<true>(dict);

    // Create ray tracing program.
    RtProgram::Desc progDesc;
    progDesc.addShaderLibrary(kShaderFile).setRayGen("rayGen");
    progDesc.addHitGroup(0, "scatterClosestHit", "scatterAnyHit").addMiss(0, "scatterMiss");
    progDesc.addHitGroup(1, "", "shadowAnyHit").addMiss(1, "shadowMiss");
    progDesc.addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    progDesc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    mTracer.pProgram = RtProgram::create(progDesc, kMaxPayloadSizeBytes, kMaxAttributesSizeBytes);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
    assert(mpSampleGenerator);
}

Dictionary MinimalPathTracer::getScriptingDictionary()
{
    Dictionary dict;
    serializePass<false>(dict);
    return dict;
}

RenderPassReflection MinimalPathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void MinimalPathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    Dictionary& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto prevFlags = (Falcor::RenderPassRefreshFlags)(dict.keyExists(kRenderPassRefreshFlags) ? dict[Falcor::kRenderPassRefreshFlags] : 0u);
        dict[Falcor::kRenderPassRefreshFlags] = (uint32_t)(prevFlags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged);
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

    // Configure depth-of-field.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kViewDirInput] == nullptr)
    {
        logWarning("Depth-of-field requires the '" + std::string(kViewDirInput) + "' input. Expect incorrect shading.");
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mUseAnalyticLights ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mUseEmissiveLights ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", (mpEnvProbe && mUseEnvLight) ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", (mpEnvProbe && mUseEnvBackground) ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars) prepareVars();
    assert(mTracer.pVars);

    // Set constants.
    auto pVars = mTracer.pVars;
    pVars["CB"]["gFrameCount"] = mFrameCount;
    pVars["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            auto pGlobalVars = mTracer.pVars;
            pGlobalVars[desc.texname] = renderData[desc.name]->asTexture();
        }
    };
    for (auto channel : kInputChannels) bind(channel);
    for (auto channel : kOutputChannels) bind(channel);

    // Get dimensions of ray dispatch.
    const uvec2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);

    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uvec3(targetDim, 1));

    mFrameCount++;
}

void MinimalPathTracer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u<<16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    // Lighting controls.
    auto lightsGroup = Gui::Group(widget, "Lights", true);
    if (lightsGroup.open())
    {
        dirty |= lightsGroup.checkbox("Use analytic lights", mUseAnalyticLights);
        lightsGroup.tooltip("This enables Falcor's built-in analytic lights.\nThese are specified in the scene description (.fscene).", true);
        dirty |= lightsGroup.checkbox("Use emissive lights", mUseEmissiveLights);
        lightsGroup.tooltip("This enables using emissive triangles as light sources.", true);
        dirty |= lightsGroup.checkbox("Use env map as light", mUseEnvLight);
        lightsGroup.tooltip("This enables using the environment map as a distant light source", true);
        dirty |= lightsGroup.checkbox("Use env map as background", mUseEnvBackground);
        lightsGroup.text(("Env map: " + (mpEnvProbe ? mEnvProbeFilename : "N/A")).c_str());

        lightsGroup.release();
    }

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void MinimalPathTracer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the program vars should to be recreated.
    mTracer.pVars = nullptr;
    mpEnvProbe = nullptr;
    mEnvProbeFilename = "";
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (pScene)
    {
        mTracer.pProgram->addDefines(pScene->getSceneDefines());

        // Load environment map if scene uses one.
        // We're getting the file name from the scene's LightProbe because that was used in the fscene files.
        // TODO: Switch to use Scene::getEnvironmentMap() when the assets have been updated.
        auto pLightProbe = mpScene->getLightProbe();
        if (pLightProbe != nullptr)
        {
            std::string fn = pLightProbe->getOrigTexture()->getSourceFilename();
            mpEnvProbe = EnvProbe::create(pRenderContext, fn);
            mEnvProbeFilename = mpEnvProbe ? getFilenameFromPath(mpEnvProbe->getEnvMap()->getSourceFilename()) : "";
        }
    }
}

void MinimalPathTracer::prepareVars()
{
    assert(mpScene);
    assert(mTracer.pProgram);

    // Configure program.
    mpSampleGenerator->prepareProgram(mTracer.pProgram.get());

    // Create program variables for the current program/scene.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mpScene);

    // Bind utility classes into shared data.
    auto pGlobalVars = mTracer.pVars->getRootVar();
    bool success = mpSampleGenerator->setShaderData(pGlobalVars);
    if (!success) throw std::exception("Failed to bind sample generator");

    // Bind the light probe if one is loaded.
    if (mpEnvProbe)
    {
        bool success = mpEnvProbe->setShaderData(pGlobalVars["CB"]["gEnvProbe"]);
        if (!success) throw std::exception("Failed to bind environment map");
    }
}
