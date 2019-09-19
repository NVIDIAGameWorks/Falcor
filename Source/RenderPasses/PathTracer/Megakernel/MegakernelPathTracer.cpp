/***************************************************************************
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
#include "MegakernelPathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include <sstream>

namespace
{
    const char kShaderFile[] = "RenderPasses/PathTracer/Megakernel/PathTracer.rt.slang";
    const char kParameterBlockName[] = "gData";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 96;
    const uint32_t kMaxAttributesSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;
};

const char* MegakernelPathTracer::sDesc = "Megakernel path tracer";

MegakernelPathTracer::SharedPtr MegakernelPathTracer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new MegakernelPathTracer);
    return pPass->init(dict) ? pPass : nullptr;
}

bool MegakernelPathTracer::init(const Dictionary& dict)
{
    // Call the base class first.
    if (!PathTracer::init(dict)) return false;

    // Create ray tracing program.
    RtProgram::Desc progDesc;
    progDesc.addShaderLibrary(kShaderFile).setRayGen("rayGen");
    progDesc.addHitGroup(kRayTypeScatter, "scatterClosestHit", "scatterAnyHit").addMiss(kRayTypeScatter, "scatterMiss");
    progDesc.addHitGroup(kRayTypeShadow, "", "shadowAnyHit").addMiss(kRayTypeShadow, "shadowMiss");
    progDesc.addDefine("MAX_BOUNCES", std::to_string(mSharedParams.maxBounces));
    progDesc.addDefine("SAMPLES_PER_PIXEL", std::to_string(mSharedParams.samplesPerPixel));
    mTracer.pProgram = RtProgram::create(progDesc, kMaxPayloadSizeBytes, kMaxAttributesSizeBytes);
    if (!mTracer.pProgram) return false;

    // Setup ray tracing state.
    mTracer.pState = RtState::create();
    assert(mTracer.pState);
    mTracer.pState->setMaxTraceRecursionDepth(kMaxRecursionDepth);
    mTracer.pState->setProgram(mTracer.pProgram);

    return true;
}

void MegakernelPathTracer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    PathTracer::setScene(pRenderContext, pScene);

    assert(pScene);
    mTracer.pProgram->addDefines(pScene->getSceneDefines());
}

void MegakernelPathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // TODO: Remove this check when the code has been generalized.
    if (mSharedParams.lightSamplesPerVertex != 1)
    {
        logError("MegakernelPathTracer currently requires 1 light sample per path vertex. Resetting to one.");
        mSharedParams.lightSamplesPerVertex = 1;
    }

    // Call shared pre-render code.
    if (!beginFrame(pRenderContext, renderData)) return;

    // Set compile-time constants.
    RtProgram::SharedPtr pProgram = mTracer.pProgram;
    setStaticParams(pProgram.get());

    // For optional channels, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    Program::DefineList defines;
    auto prepare = [&](const ChannelDesc& desc)
    {
        if (desc.optional && !desc.texname.empty())
        {
            std::string define = "is_valid_" + std::string(desc.texname);
            defines.add(define, renderData[desc.name] != nullptr ? "1" : "0");
        }
    };
    for (auto channel : kInputChannels) prepare(channel);
    for (auto channel : kOutputChannels) prepare(channel);
    pProgram->addDefines(defines);

    if (mUseEmissiveSampler)
    {
        // Specialize program for the current emissive light sampler options.
        assert(mpEmissiveSampler);
        if (mpEmissiveSampler->prepareProgram(pProgram.get())) mTracer.pVars = nullptr;
    }

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars) prepareVars();
    assert(mTracer.pVars);

    // Set shared data into parameter block.
    setTracerData(renderData);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            auto pGlobalVars = mTracer.pVars->getGlobalVars();
            pGlobalVars[desc.texname] = renderData[desc.name]->asTexture();
        }
    };
    for (auto channel : kInputChannels) bind(channel);
    for (auto channel : kOutputChannels) bind(channel);

    // Get dimensions of ray dispatch.
    const uvec2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);

    mPixelDebugger->begin(pRenderContext, pProgram, mTracer.pVars, targetDim);
    mStatsLogger->begin(pRenderContext, pProgram, mTracer.pVars, targetDim);

    // Spawn the rays.
    {
        PROFILE("MegakernelPathTracer::execute()_RayTrace");
        mpScene->raytrace(pRenderContext, mTracer.pState, mTracer.pVars, uvec3(targetDim, 1));
    }

    mStatsLogger->end(pRenderContext);
    mPixelDebugger->end(pRenderContext);

    // Call shared post-render code.
    endFrame(pRenderContext, renderData);
}

void MegakernelPathTracer::prepareVars()
{
    assert(mpScene);
    assert(mTracer.pProgram);

    // Configure program.
    mpSampleGenerator->prepareProgram(mTracer.pProgram.get());

    // Create program variables for the current program/scene.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mpScene);
    if (!mTracer.pVars) throw std::exception("Failed to create shader variables");

    // Bind utility classes into shared data.
    auto pGlobalVars = mTracer.pVars->getGlobalVars();
    bool success = mpSampleGenerator->setIntoProgramVars(pGlobalVars.get());
    if (!success) throw std::exception("Failed to bind sample generator");

    // Create parameter block for shared data.
    ProgramReflection::SharedConstPtr pReflection = mTracer.pProgram->getGlobalReflector();
    ParameterBlockReflection::SharedConstPtr pBlockReflection = pReflection->getParameterBlock(kParameterBlockName);
    assert(pBlockReflection);
    mTracer.pParameterBlock = ParameterBlock::create(pBlockReflection, true);
    assert(mTracer.pParameterBlock);

    // Bind static resources to the parameter block here. No need to rebind them every frame if they don't change.
    // Bind the light probe if one is loaded.
    if (mpEnvProbe)
    {
        bool success = mpEnvProbe->setIntoParameterBlock(mTracer.pParameterBlock.get(), "envProbe");
        if (!success) throw std::exception("Failed to bind environment map");
    }

    // Bind the parameter block to the global program variables.
    mTracer.pVars->getGlobalVars()->setParameterBlock(kParameterBlockName, mTracer.pParameterBlock);
}

void MegakernelPathTracer::setTracerData(const RenderData& renderData)
{
    auto pBlock = mTracer.pParameterBlock;
    assert(pBlock);

    // Upload parameters struct.
    pBlock->getDefaultConstantBuffer()["params"].setBlob(mSharedParams);

    // Bind emissive light sampler.
    if (mUseEmissiveSampler)
    {
        assert(mpEmissiveSampler);
        bool success = mpEmissiveSampler->setIntoParameterBlock(pBlock, "emissiveSampler");
        if (!success) throw std::exception("Failed to bind emissive light sampler");
    }
}
