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
#include "VBufferRT.h"
#include "RenderPasses/Shared/HitInfo.h"
#include "RenderGraph/RenderPassStandardFlags.h"

const char* VBufferRT::kDesc = "Ray traced V-buffer generation pass";

namespace
{
    const std::string kProgramFile = "RenderPasses/GBuffer/VBuffer/VBufferRT.rt.slang";

    // Ray tracing settings that affect the traversal stack size. Set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 4; // TODO: The shader doesn't need a payload, set this to zero if it's possible to pass a null payload to TraceRay()
    const uint32_t kMaxAttributesSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;

    const std::string kOutputName = "vbuffer";
    const std::string kOutputDesc = "V-buffer packed into 64 bits (indices + barys)";
};

RenderPassReflection VBufferRT::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addOutput(kOutputName, kOutputDesc).bindFlags(Resource::BindFlags::UnorderedAccess).format(ResourceFormat::RG32Uint);

    return reflector;
}

VBufferRT::SharedPtr VBufferRT::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new VBufferRT(dict));
}

VBufferRT::VBufferRT(const Dictionary& dict)
    : GBufferBase()
{
    parseDictionary(dict);

    // Create ray tracing program
    RtProgram::Desc desc;
    desc.addShaderLibrary(kProgramFile).setRayGen("rayGen");
    desc.addHitGroup(0, "closestHit", "anyHit").addMiss(0, "miss");
    desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    mRaytrace.pProgram = RtProgram::create(desc, kMaxPayloadSizeBytes, kMaxAttributesSizeBytes);

    // Create sample generator
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_DEFAULT);
    mpSampleGenerator->prepareProgram(mRaytrace.pProgram.get());
}

void VBufferRT::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    GBufferBase::setScene(pRenderContext, pScene);

    mFrameCount = 0;
    mRaytrace.pVars = nullptr;

    if (pScene)
    {
        mRaytrace.pProgram->addDefines(pScene->getSceneDefines());
        mRaytrace.pProgram->addDefines(HitInfo::getDefines(pScene));
    }
}

void VBufferRT::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    Dictionary& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto prevFlags = (Falcor::RenderPassRefreshFlags)(dict.keyExists(kRenderPassRefreshFlags) ? dict[Falcor::kRenderPassRefreshFlags] : 0u);
        dict[Falcor::kRenderPassRefreshFlags] = (uint32_t)(prevFlags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged);
        mOptionsChanged = false;
    }

    // If there is no scene, clear the output and return.
    if (mpScene == nullptr)
    {
        auto pOutput = renderData[kOutputName]->asTexture();
        pRenderContext->clearUAV(pOutput->getUAV().get(), glm::uvec4(kInvalidIndex));
        return;
    }

    // Configure depth-of-field.
    // When DOF is enabled, two PRNG dimensions are used. Pass this info to subsequent passes via the dictionary.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF) dict[Falcor::kRenderPassPRNGDimension] = useDOF ? 2u : 0u;

    // Set program defines.
    mRaytrace.pProgram->addDefine("USE_DEPTH_OF_FIELD", useDOF ? "1" : "0");
    mRaytrace.pProgram->addDefine("DISABLE_ALPHA_TEST", mDisableAlphaTest ? "1" : "0");

    // Create program vars.
    if (!mRaytrace.pVars)
    {
        mRaytrace.pVars = RtProgramVars::create(mRaytrace.pProgram, mpScene);

        // Bind static resources
        ShaderVar var = mRaytrace.pVars->getRootVar();
        if (!mpSampleGenerator->setShaderData(var)) throw std::exception("Failed to bind sample generator");
    }

    // Bind resources.
    ShaderVar var = mRaytrace.pVars->getRootVar();
    var["PerFrameCB"]["frameCount"] = mFrameCount++;
    var["gVBuffer"] = renderData[kOutputName]->asTexture();

    // Dispatch the rays.
    mpScene->raytrace(pRenderContext, mRaytrace.pProgram.get(), mRaytrace.pVars, uvec3(mFrameDim, 1));
}
