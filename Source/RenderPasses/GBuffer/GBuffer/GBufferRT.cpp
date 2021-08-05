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
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "GBufferRT.h"

const char* GBufferRT::kDesc = "Ray traced G-buffer generation pass";

namespace
{
    const std::string kProgramRaytraceFile = "RenderPasses/GBuffer/GBuffer/GBufferRT.rt.slang";
    const std::string kProgramComputeFile = "RenderPasses/GBuffer/GBuffer/GBufferRT.cs.slang";

    // Scripting options.
    const char kUseTraceRayInline[] = "useTraceRayInline";

    // Ray tracing settings that affect the traversal stack size. Set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 4;
    const uint32_t kMaxRecursionDepth = 1;

    // Scripting options
    const std::string kLODMode = "texLOD";

    const Falcor::Gui::DropdownList kLODModeList =
    {
        { (uint32_t)TexLODMode::Mip0, "Mip0" },
        { (uint32_t)TexLODMode::RayDiffs, "Ray Diffs" },
        { (uint32_t)TexLODMode::RayCones, "Ray Cones" },
    };

    // Additional output channels.
    const std::string kVBufferName = "vbuffer";
    const ChannelList kGBufferExtraChannels =
    {
        { kVBufferName,     "gVBuffer",             "Visibility buffer",                    true /* optional */, ResourceFormat::Unknown /* set at runtime */ },
        { "depth",          "gDepth",               "Depth buffer (NDC)",                   true /* optional */, ResourceFormat::R32Float       },
        { "linearZ",        "gLinearZ",             "Linear Z and slope",                   true /* optional */, ResourceFormat::RG32Float      },
        { "mvec",           "gMotionVectors",       "Motion vectors",                       true /* optional */, ResourceFormat::RG32Float      },
        { "mvecW",          "gMotionVectorsW",      "Motion vectors in world space",        true /* optional */, ResourceFormat::RGBA16Float    },
        { "normWRoughness", "gNormalWRoughness",    "Normal in world space and roughness",  true /* optional */, ResourceFormat::RGBA8Unorm     },
        { "roughness",      "gRoughness",           "Roughness",                            true /* optional */, ResourceFormat::RGBA8Unorm     },
        { "metallic",       "gMetallic",            "Metallic",                             true /* optional */, ResourceFormat::RGBA8Unorm     },
        { "faceNormalW",    "gFaceNormalW",         "Face normal in world space",           true /* optional */, ResourceFormat::RGBA32Float    },
        { "viewW",          "gViewW",               "View direction in world space",        true /* optional */, ResourceFormat::RGBA32Float    }, // TODO: Switch to packed 2x16-bit snorm format.
        { "time",           "gTime",                "Per-pixel execution time",             true /* optional */, ResourceFormat::R32Uint        },
        { "disocclusion",   "gDisocclusion",        "Disocclusion mask",                    true /* optional */, ResourceFormat::R32Float       },
    };
};

GBufferRT::SharedPtr GBufferRT::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new GBufferRT(dict));
}

RenderPassReflection GBufferRT::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Add all outputs as UAVs.
    addRenderPassOutputs(reflector, kGBufferChannels);
    addRenderPassOutputs(reflector, kGBufferExtraChannels);
    reflector.getField(kVBufferName)->format(mVBufferFormat);

    return reflector;
}

void GBufferRT::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    GBuffer::execute(pRenderContext, renderData);

    // If there is no scene, clear the output and return.
    if (mpScene == nullptr)
    {
        auto clear = [&](const ChannelDesc& channel)
        {
            auto pTex = renderData[channel.name]->asTexture();
            if (pTex) pRenderContext->clearUAV(pTex->getUAV().get(), float4(0.f));
        };
        for (const auto& channel : kGBufferChannels) clear(channel);
        for (const auto& channel : kGBufferExtraChannels) clear(channel);
        return;
    }

    // Check for scene geometry changes.
    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        recreatePrograms();
    }

    // Configure depth-of-field.
    // When DOF is enabled, two PRNG dimensions are used. Pass this info to subsequent passes via the dictionary.
    mUseDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (mUseDOF) renderData.getDictionary()[Falcor::kRenderPassPRNGDimension] = mUseDOF ? 2u : 0u;

    if (mLODMode == TexLODMode::RayDiffs)
    {
        // TODO: Remove this warning when the TexLOD code has been fixed.
        // logWarning("GBufferRT::execute() - Ray differentials are not tested for instance transforms that flip the coordinate system handedness. The results may be incorrect.");
    }

    mUseTraceRayInline ? executeCompute(pRenderContext, renderData) : executeRaytrace(pRenderContext, renderData);

    mFrameCount++;
}

void GBufferRT::renderUI(Gui::Widgets& widget)
{
    // Render the base class UI first.
    GBuffer::renderUI(widget);

    // Ray tracing specific options.
    if (widget.dropdown("LOD Mode", kLODModeList, reinterpret_cast<uint32_t&>(mLODMode)))
    {
        mOptionsChanged = true;
    }

    if (widget.checkbox("Use TraceRayInline", mUseTraceRayInline))
    {
        mOptionsChanged = true;
    }
}

Dictionary GBufferRT::getScriptingDictionary()
{
    Dictionary dict = GBuffer::getScriptingDictionary();
    dict[kLODMode] = mLODMode;
    dict[kUseTraceRayInline] = mUseTraceRayInline;
    return dict;
}

void GBufferRT::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    GBuffer::setScene(pRenderContext, pScene);

    recreatePrograms();
}

void GBufferRT::recreatePrograms()
{
    mRaytrace.pProgram = nullptr;
    mRaytrace.pVars = nullptr;
    mpComputePass = nullptr;
}

void GBufferRT::executeRaytrace(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mRaytrace.pProgram || !mRaytrace.pVars)
    {
        // Create ray tracing program.
        RtProgram::Desc desc;
        desc.addShaderLibrary(kProgramRaytraceFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
        desc.addDefines(mpScene->getSceneDefines());
        desc.addDefines(mpSampleGenerator->getDefines());
        desc.addDefines(getShaderDefines(renderData));

        RtBindingTable::SharedPtr sbt = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("miss"));
        sbt->setHitGroupByType(0, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("closestHit", "anyHit"));

        // Add hit group with intersection shader for displaced meshes.
        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroupByType(0, mpScene, Scene::GeometryType::DisplacedTriangleMesh, desc.addHitGroup("displacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection"));
        }

        // Add hit group with intersection shader for curves (represented as linear swept spheres).
        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            sbt->setHitGroupByType(0, mpScene, Scene::GeometryType::Curve, desc.addHitGroup("curveClosestHit", "", "curveIntersection"));
        }

        // Add hit groups for for other procedural primitives here.

        mRaytrace.pProgram = RtProgram::create(desc);
        mRaytrace.pVars = RtProgramVars::create(mRaytrace.pProgram, sbt);

        // Bind static resources.
        ShaderVar var = mRaytrace.pVars->getRootVar();
        if (!mpSampleGenerator->setShaderData(var)) throw std::exception("Failed to bind sample generator");
    }

    mRaytrace.pProgram->addDefines(getShaderDefines(renderData));

    ShaderVar var = mRaytrace.pVars->getRootVar();
    setShaderData(var, renderData);

    // Dispatch the rays.
    mpScene->raytrace(pRenderContext, mRaytrace.pProgram.get(), mRaytrace.pVars, uint3(mFrameDim, 1));
}

void GBufferRT::executeCompute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw std::exception("Raytracing Tier 1.1 is not supported by the current device");
    }

    // Create compute pass.
    if (!mpComputePass)
    {
    	Program::Desc desc;
    	desc.addShaderLibrary(kProgramComputeFile).csEntry("main").setShaderModel("6_5");

        Program::DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(getShaderDefines(renderData));

    	mpComputePass = ComputePass::create(desc, defines, true);

        // Bind static resources
        ShaderVar var = mpComputePass->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);
        if (!mpSampleGenerator->setShaderData(var)) throw std::exception("Failed to bind sample generator");
    }

    mpComputePass->getProgram()->addDefines(getShaderDefines(renderData));

    ShaderVar var = mpComputePass->getRootVar();
    setShaderData(var, renderData);

    mpComputePass->execute(pRenderContext, uint3(mFrameDim, 1));
}

Program::DefineList GBufferRT::getShaderDefines(const RenderData& renderData) const
{
    Program::DefineList defines;
    defines.add("USE_DEPTH_OF_FIELD", mUseDOF ? "1" : "0");
    defines.add("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");
    defines.add("LOD_MODE", std::to_string((uint32_t)mLODMode));
    defines.add("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");

    // Setup ray flags.
    uint32_t rayFlags = D3D12_RAY_FLAG_NONE;
    if (mForceCullMode && mCullMode == RasterizerState::CullMode::Front) rayFlags = D3D12_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;
    else if (mForceCullMode && mCullMode == RasterizerState::CullMode::Back) rayFlags = D3D12_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    defines.add("RAY_FLAGS", std::to_string(rayFlags));

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    defines.add(getValidResourceDefines(kGBufferExtraChannels, renderData));
    return defines;
}

void GBufferRT::setShaderData(const ShaderVar& var, const RenderData& renderData)
{
    var["gGBufferRT"]["frameDim"] = mFrameDim;
    var["gGBufferRT"]["invFrameDim"] = mInvFrameDim;
    var["gGBufferRT"]["frameCount"] = mFrameCount;
    var["gGBufferRT"]["screenSpacePixelSpreadAngle"] = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(mFrameDim.y);

    // Bind output channels as UAV buffers.
    auto bind = [&](const ChannelDesc& channel)
    {
        Texture::SharedPtr pTex = renderData[channel.name]->asTexture();
        var[channel.texname] = pTex;
    };
    for (const auto& channel : kGBufferChannels) bind(channel);
    for (const auto& channel : kGBufferExtraChannels) bind(channel);
}

GBufferRT::GBufferRT(const Dictionary& dict)
    : GBuffer()
{
    parseDictionary(dict);

    // Create random engine
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_DEFAULT);
}

void GBufferRT::parseDictionary(const Dictionary& dict)
{
    GBuffer::parseDictionary(dict);

    for (const auto& [key, value] : dict)
    {
        if (key == kLODMode) mLODMode = value;
        if (key == kUseTraceRayInline) mUseTraceRayInline = value;
        // TODO: Check for unparsed fields, including those parsed in base classes.
    }
}
