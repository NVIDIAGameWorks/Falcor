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
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "GBufferRT.h"

const char* GBufferRT::kDesc = "Ray traced G-buffer generation pass";

namespace
{
    const std::string kProgramFile = "RenderPasses/GBuffer/GBuffer/GBufferRT.rt.slang";

    // Ray tracing settings that affect the traversal stack size. Set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 4;
    const uint32_t kMaxAttributesSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;

    // Scripting options
    const std::string kLOD = "texLOD";

    const Falcor::Gui::DropdownList kLODModeList =
    {
        { (uint32_t)GBufferRT::LODMode::UseMip0, "Mip0" },
        { (uint32_t)GBufferRT::LODMode::RayDifferentials, "Ray Diff" },
        { (uint32_t)GBufferRT::LODMode::RayCones, "Ray Cones" },
    };

    // Additional output channels.
    const ChannelList kGBufferExtraChannels =
    {
        { "vbuffer",        "gVBuffer",         "Visibility buffer",                true /* optional */, ResourceFormat::RG32Uint    },
        { "mvec",           "gMotionVectors",   "Motion vectors",                   true /* optional */, ResourceFormat::RG32Float   },
        { "faceNormalW",    "gFaceNormalW",     "Face normal in world space",       true /* optional */, ResourceFormat::RGBA32Float },
        { "viewW",          "gViewW",           "View direction in world space",    true /* optional */, ResourceFormat::RGBA32Float }, // TODO: Switch to packed 2x16-bit snorm format.
        { "time",           "gTime",            "Per-pixel execution time",         true /* optional */, ResourceFormat::R32Uint     },
    };
};

void GBufferRT::registerBindings(pybind11::module& m)
{
    pybind11::enum_<GBufferRT::LODMode> lodMode(m, "LODMode");
    lodMode.value("UseMip0", GBufferRT::LODMode::UseMip0);
    lodMode.value("RayDifferentials", GBufferRT::LODMode::RayDifferentials);
    lodMode.value("RayCones", GBufferRT::LODMode::RayCones);
}

RenderPassReflection GBufferRT::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Add all outputs as UAVs.
    addRenderPassOutputs(reflector, kGBufferChannels);
    addRenderPassOutputs(reflector, kGBufferExtraChannels);

    return reflector;
}

void GBufferRT::parseDictionary(const Dictionary& dict)
{
    // Call the base class first.
    GBuffer::parseDictionary(dict);

    for (const auto& [key, value] : dict)
    {
        if (key == kLOD) mLODMode = value;
        // TODO: Check for unparsed fields, including those parsed in base classes.
    }
}

GBufferRT::SharedPtr GBufferRT::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new GBufferRT(dict));
}

Dictionary GBufferRT::getScriptingDictionary()
{
    Dictionary dict = GBuffer::getScriptingDictionary();
    dict[kLOD] = mLODMode;
    return dict;
}

GBufferRT::GBufferRT(const Dictionary& dict)
    : GBuffer()
{
    parseDictionary(dict);

    // Create random engine
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_DEFAULT);

    // Create ray tracing program
    RtProgram::Desc desc;
    desc.addShaderLibrary(kProgramFile).setRayGen("rayGen");
    desc.addHitGroup(0, "closestHit", "anyHit").addMiss(0, "miss");
    desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    desc.addDefines(mpSampleGenerator->getDefines());
    mRaytrace.pProgram = RtProgram::create(desc, kMaxPayloadSizeBytes, kMaxAttributesSizeBytes);

    // Set default cull mode
    setCullMode(mCullMode);
}

void GBufferRT::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    GBuffer::setScene(pRenderContext, pScene);

    mRaytrace.pVars = nullptr;

    if (pScene)
    {
        mRaytrace.pProgram->addDefines(pScene->getSceneDefines());
    }
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

    // Configure depth-of-field.
    // When DOF is enabled, two PRNG dimensions are used. Pass this info to subsequent passes via the dictionary.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF) renderData.getDictionary()[Falcor::kRenderPassPRNGDimension] = useDOF ? 2u : 0u;

    // Set program defines.
    mRaytrace.pProgram->addDefine("USE_DEPTH_OF_FIELD", useDOF ? "1" : "0");
    mRaytrace.pProgram->addDefine("USE_RAY_DIFFERENTIALS", mLODMode == LODMode::RayDifferentials ? "1" : "0");
    mRaytrace.pProgram->addDefine("USE_RAY_CONES", mLODMode == LODMode::RayCones ? "1" : "0");
    mRaytrace.pProgram->addDefine("DISABLE_ALPHA_TEST", mDisableAlphaTest ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mRaytrace.pProgram->addDefines(getValidResourceDefines(kGBufferExtraChannels, renderData));

    // Create program vars.
    if (!mRaytrace.pVars)
    {
        mRaytrace.pVars = RtProgramVars::create(mRaytrace.pProgram, mpScene);
    }

    // Setup ray flags.
    if (mForceCullMode && mCullMode == RasterizerState::CullMode::Front) mGBufferParams.rayFlags = D3D12_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;
    else if (mForceCullMode && mCullMode == RasterizerState::CullMode::Back) mGBufferParams.rayFlags = D3D12_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    else mGBufferParams.rayFlags = D3D12_RAY_FLAG_NONE;

    if (mLODMode == LODMode::RayDifferentials)
    {
        // TODO: Remove this warning when the TexLOD code has been fixed.
        logWarning("GBufferRT::execute() - Ray differentials are not tested for instance transforms that flip the coordinate system handedness. The results may be incorrect.");
    }

    mGBufferParams.screenSpacePixelSpreadAngle = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(uint32_t(mGBufferParams.frameSize.y));

    ShaderVar pGlobalVars = mRaytrace.pVars->getRootVar();
    pGlobalVars["PerFrameCB"]["gParams"].setBlob(mGBufferParams);

    bool success = mpSampleGenerator->setShaderData(pGlobalVars);
    if (!success) throw std::exception("Failed to bind sample generator");

    // Bind output channels as UAV buffers.
    // TODO: Check if we write all pixels for a buffer, if so remove clear
    auto bind = [&](const ChannelDesc& channel)
    {
        Texture::SharedPtr pTex = renderData[channel.name]->asTexture();
        if (pTex) pRenderContext->clearUAV(pTex->getUAV().get(), float4(0, 0, 0, 0));
        pGlobalVars[channel.texname] = pTex;
    };
    for (const auto& channel : kGBufferChannels) bind(channel);
    for (const auto& channel : kGBufferExtraChannels) bind(channel);

    // Launch the rays.
    uint3 targetDim = uint3((int)mGBufferParams.frameSize.x, (int)mGBufferParams.frameSize.y, 1u);
    mpScene->raytrace(pRenderContext, mRaytrace.pProgram.get(), mRaytrace.pVars, targetDim);

    mGBufferParams.frameCount++;
}

void GBufferRT::renderUI(Gui::Widgets& widget)
{
    // Render the base class UI first.
    GBuffer::renderUI(widget);

    // Ray tracing specific options.
    uint32_t lodMode = (uint32_t)mLODMode;
    if (widget.dropdown("LOD Mode", kLODModeList, lodMode))
    {
        mLODMode = (LODMode)lodMode;
        mOptionsChanged = true;
    }
}
