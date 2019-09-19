/***************************************************************************
 # Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 #/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "GBufferRT.h"

namespace
{
    const char* kFileRayTrace = "RenderPasses/GBuffer/RaytracePrimary.rt.slang";

    const char* kEntryPointRayGen = "rayGen";
    const char* kEntryPointMiss0 = "primaryMiss";
    const char* kEntryPrimaryAnyHit = "primaryAnyHit";
    const char* kEntryPrimaryClosestHit = "primaryClosestHit";

    // Serialized parameters
    const std::string kLOD = "texLOD";

    const Falcor::Gui::DropdownList kLODModeList =
    {
        { (uint32_t)GBufferRT::LODMode::UseMip0, "Mip0" },
        { (uint32_t)GBufferRT::LODMode::RayDifferentials, "Ray Diff" },
        //{ (uint32_t)GBufferRT::LODMode::TexLODCone, "Tex LOD cone" }, // Not implemented
    };

    // Additional output channels.
    const ChannelList kGBufferRTChannels =
    {
        { "faceNormalW", "gFaceNormalW", "Face normal in world space",       true /* optional */, ResourceFormat::RGBA32Float },
        { "viewW",       "gViewW",       "View direction in world space",    true /* optional */, ResourceFormat::RGBA32Float }, // TODO: Switch to packed 2x16-bit snorm format.
        { "visBuffer",   "gVisBuffer",   "Visibility buffer",                true /* optional */, ResourceFormat::RGBA32Uint },
    };
};

RenderPassReflection GBufferRT::reflect(const CompileData& compileData)
{
    RenderPassReflection r;

    // Add all outputs as UAVs.
    auto addOutput = [&](const ChannelDesc& output)
    {
        auto& f = r.addOutput(output.name, output.desc).format(output.format).bindFlags(Resource::BindFlags::UnorderedAccess);
        if (output.optional) f.flags(RenderPassReflection::Field::Flags::Optional);
    };
    for (auto it : kGBufferChannels) addOutput(it);
    for (auto it : kGBufferRTChannels) addOutput(it);

    return r;
}

bool GBufferRT::parseDictionary(const Dictionary& dict)
{
    // Call the base class first.
    if (!GBuffer::parseDictionary(dict)) return false;

    for (const auto& v : dict)
    {
        if (v.key() == kLOD) mLODMode = v.val();
        else
        {
            // TODO: This incorrectly logs warnings about unknown fields because some fields are parsed in the base class. Should be fixed somehow.
            logWarning("Unknown field `" + v.key() + "` in a GBufferRT dictionary");
        }
    }
    return true;
}

GBufferRT::SharedPtr GBufferRT::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new GBufferRT);
    return pPass->parseDictionary(dict) ? pPass : nullptr;
}

Dictionary GBufferRT::getScriptingDictionary()
{
    Dictionary dict = GBuffer::getScriptingDictionary();
    dict[kLOD] = mLODMode;
    return dict;
}

GBufferRT::GBufferRT() : GBuffer()
{
    // Create ray tracing program
    RtProgram::Desc progDesc;
    progDesc.addShaderLibrary(kFileRayTrace).setRayGen("rayGen");
    progDesc.addHitGroup(0, "primaryClosestHit", "primaryAnyHit").addMiss(0, "primaryMiss");
    mRaytrace.pProgram = RtProgram::create(progDesc);
    if (!mRaytrace.pProgram) throw std::exception("Failed to create program");

    // Initialize ray tracing state
    mRaytrace.pState = RtState::create();
    mRaytrace.pState->setMaxTraceRecursionDepth(1);     // Max trace depth 1 allows TraceRay to be called from RGS, but no secondary rays.
    mRaytrace.pState->setProgram(mRaytrace.pProgram);

    // Create random engine
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
    if (!mpSampleGenerator) throw std::exception("Failed to create sample generator");
    mpSampleGenerator->prepareProgram(mRaytrace.pProgram.get());

    // Set default cull mode
    setCullMode(mCullMode);
}

void GBufferRT::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    GBuffer::setScene(pRenderContext, pScene);

    assert(pScene);
    mRaytrace.pProgram->addDefines(pScene->getSceneDefines());
    mRaytrace.pVars = RtProgramVars::create(mRaytrace.pProgram, pScene);
    if (!mRaytrace.pVars) throw std::exception("Failed to create program vars");
}

void GBufferRT::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    Dictionary& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto prevFlags = (Falcor::RenderPassRefreshFlags)(dict.keyExists(kRenderPassRefreshFlags) ? dict[Falcor::kRenderPassRefreshFlags] : 0u);
        dict[Falcor::kRenderPassRefreshFlags] = (uint32_t)(prevFlags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged);
        mOptionsChanged = false;
    }

    if (mpScene == nullptr)
    {
        logWarning("GBufferRT::execute() - No scene available");
        return;
    }

    // Setup ray flags.
    if (mForceCullMode && mCullMode == RasterizerState::CullMode::Front) mGBufferParams.rayFlags = D3D12_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;
    else if (mForceCullMode && mCullMode == RasterizerState::CullMode::Back) mGBufferParams.rayFlags = D3D12_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    else mGBufferParams.rayFlags = D3D12_RAY_FLAG_NONE;

    // Configure depth-of-field.
    // When DOF is enabled, two PRNG dimensions are used. Pass this info to subsequent passes via the dictionary.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF) dict[Falcor::kRenderPassPRNGDimension] = useDOF ? 2u : 0u;

    mRaytrace.pProgram->addDefine("USE_DEPTH_OF_FIELD", useDOF ? "1" : "0");
    mRaytrace.pProgram->addDefine("USE_RAY_DIFFERENTIALS", mLODMode == LODMode::RayDifferentials ? "1" : "0");

    if (mLODMode == LODMode::RayDifferentials)
    {
        logWarning("GBufferRT::execute() - Ray differentials are not tested for instance transforms that flip the coordinate system handedness. The results may be incorrect.");
    }
    
    GraphicsVars::SharedPtr pGlobalVars = mRaytrace.pVars->getGlobalVars();
    pGlobalVars["PerFrameCB"]["gParams"].setBlob(mGBufferParams);

    bool success = mpSampleGenerator->setIntoProgramVars(pGlobalVars.get());
    if (!success) throw std::exception("Failed to bind sample generator");

    // Bind outputs.
    auto bind = [&](const ChannelDesc& output)
    {
        Texture::SharedPtr pTex = renderData[output.name]->asTexture();
        if (pTex) pRenderContext->clearUAV(pTex->getUAV().get(), glm::vec4(0, 0, 0, 0));
        pGlobalVars[output.texname] = pTex;
    };
    for (auto it : kGBufferChannels) bind(it);
    for (auto it : kGBufferRTChannels) bind(it);

    // Launch the rays.
    uvec3 targetDim = uvec3((int)mGBufferParams.frameSize.x, (int)mGBufferParams.frameSize.y, 1u);
    mpScene->raytrace(pRenderContext, mRaytrace.pState, mRaytrace.pVars, targetDim);

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

    widget.tooltip("Enables adjustment of the shading normals to reduce the risk of black pixels due to back-facing vectors.", true);
}
