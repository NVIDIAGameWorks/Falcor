/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
    const uint32_t kMaxAttributeSizeBytes = 8u;
    const uint32_t kMaxRecursionDepth = 2u;

    const char kViewDirInput[] = "viewW";

    const ChannelList kInputChannels =
    {
        { "posW",           "gWorldPosition",             "World-space position (xyz) and foreground flag (w)"       },
        { "normalW",        "gWorldShadingNormal",        "World-space shading normal (xyz)"                         },
        { "tangentW",       "gWorldShadingTangent",       "World-space shading tangent (xyz) and sign (w)", true /* optional */ },
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

    const char kMaxBounces[] = "maxBounces";
    const char kComputeDirect[] = "computeDirect";
}

MinimalPathTracer::SharedPtr MinimalPathTracer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new MinimalPathTracer(dict));
}

MinimalPathTracer::MinimalPathTracer(const Dictionary& dict)
{
    parseDictionary(dict);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
    assert(mpSampleGenerator);
}

void MinimalPathTracer::parseDictionary(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kMaxBounces) mMaxBounces = value;
        else if (key == kComputeDirect) mComputeDirect = value;
        else logWarning("Unknown field '" + key + "' in MinimalPathTracer dictionary");
    }
}

Dictionary MinimalPathTracer::getScriptingDictionary()
{
    Dictionary d;
    d[kMaxBounces] = mMaxBounces;
    d[kComputeDirect] = mComputeDirect;
    return d;
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

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        throw std::runtime_error("This render pass does not support scene geometry changes. Aborting.");
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
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
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars) prepareVars();
    assert(mTracer.pVars);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData[desc.name]->asTexture();
        }
    };
    for (auto channel : kInputChannels) bind(channel);
    for (auto channel : kOutputChannels) bind(channel);

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);

    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));

    mFrameCount++;
}

void MinimalPathTracer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u<<16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

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
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        if (mpScene->hasGeometryType(Scene::GeometryType::Procedural))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        // Create ray tracing program.
        RtProgram::Desc desc;
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(kMaxAttributeSizeBytes);
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
        desc.addDefines(mpScene->getSceneDefines());

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));
        sbt->setHitGroupByType(0, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("scatterClosestHit", "scatterAnyHit"));
        sbt->setHitGroupByType(1, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("", "shadowAnyHit"));

        mTracer.pProgram = RtProgram::create(desc);
    }
}

void MinimalPathTracer::prepareVars()
{
    assert(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    bool success = mpSampleGenerator->setShaderData(var);
    if (!success) throw std::exception("Failed to bind sample generator");
}
