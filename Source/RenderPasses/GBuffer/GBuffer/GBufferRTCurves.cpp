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
#include "GBufferRTCurves.h"

const char* GBufferRTCurves::kDesc = "Ray traced G-buffer generation pass that supports curves";

namespace
{
    const std::string kProgramFile = "RenderPasses/GBuffer/GBuffer/GBufferRT.rt.slang";

    // Ray tracing settings that affect the traversal stack size. Set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 4;
    const uint32_t kMaxAttributesSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;
};

GBufferRTCurves::SharedPtr GBufferRTCurves::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new GBufferRTCurves(dict));
}

GBufferRTCurves::GBufferRTCurves(const Dictionary& dict)
    : GBufferRT()
{
    parseDictionary(dict);

    // Create random engine
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_DEFAULT);

    // Create ray tracing program
    RtProgram::Desc desc;
    desc.addShaderLibrary(kProgramFile).setRayGen("rayGen");
    desc.addHitGroup(0, "closestHit", "anyHit").addMiss(0, "miss");

    // Add intersection shaders for custom primitives
    // Now we only support curve primitives (represented as linear swept spheres)
    desc.addIntersection(0, "linearSweptSphereIntersection");
    desc.addAABBHitGroup(0, "curveClosestHit", "");

    desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    desc.addDefines(mpSampleGenerator->getDefines());
    desc.addDefine("USE_CURVES", "1");
    mRaytrace.pProgram = RtProgram::create(desc, kMaxPayloadSizeBytes, kMaxAttributesSizeBytes);

    // Set default cull mode
    setCullMode(mCullMode);
}
