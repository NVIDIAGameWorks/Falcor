/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "RtProgram.h"
#include "ProgramManager.h"
#include "ProgramVars.h"

#include <slang.h>

namespace Falcor
{
void RtProgram::Desc::init()
{
    mBaseDesc.setShaderModel("6_5");
}

RtProgram::ShaderID RtProgram::Desc::addRayGen(
    const std::string& raygen,
    const TypeConformanceList& typeConformances,
    const std::string& entryPointNameSuffix
)
{
    checkArgument(!raygen.empty(), "'raygen' entry point name must not be empty");

    mBaseDesc.beginEntryPointGroup(entryPointNameSuffix);
    mBaseDesc.entryPoint(ShaderType::RayGeneration, raygen);
    mBaseDesc.addTypeConformancesToGroup(typeConformances);

    mRayGenCount++;
    return {mBaseDesc.mActiveGroup};
}

RtProgram::ShaderID RtProgram::Desc::addMiss(
    const std::string& miss,
    const TypeConformanceList& typeConformances,
    const std::string& entryPointNameSuffix
)
{
    checkArgument(!miss.empty(), "'miss' entry point name must not be empty");

    mBaseDesc.beginEntryPointGroup(entryPointNameSuffix);
    mBaseDesc.entryPoint(ShaderType::Miss, miss);
    mBaseDesc.addTypeConformancesToGroup(typeConformances);

    return {mBaseDesc.mActiveGroup};
}

RtProgram::ShaderID RtProgram::Desc::addHitGroup(
    const std::string& closestHit,
    const std::string& anyHit,
    const std::string& intersection,
    const TypeConformanceList& typeConformances,
    const std::string& entryPointNameSuffix
)
{
    checkArgument(
        !(closestHit.empty() && anyHit.empty() && intersection.empty()),
        "At least one of 'closestHit', 'anyHit' or 'intersection' entry point names must not be empty"
    );

    mBaseDesc.beginEntryPointGroup(entryPointNameSuffix);
    mBaseDesc.addTypeConformancesToGroup(typeConformances);
    if (!closestHit.empty())
    {
        mBaseDesc.entryPoint(ShaderType::ClosestHit, closestHit);
    }
    if (!anyHit.empty())
    {
        mBaseDesc.entryPoint(ShaderType::AnyHit, anyHit);
    }
    if (!intersection.empty())
    {
        mBaseDesc.entryPoint(ShaderType::Intersection, intersection);
    }

    return {mBaseDesc.mActiveGroup};
}

RtProgram::SharedPtr RtProgram::create(std::shared_ptr<Device> pDevice, Desc desc, const DefineList& programDefines)
{
    auto pProgram = SharedPtr(new RtProgram(pDevice, desc, programDefines));
    pDevice->getProgramManager()->registerProgramForReload(pProgram);
    return pProgram;
}

RtProgram::RtProgram(std::shared_ptr<Device> pDevice, const RtProgram::Desc& desc, const DefineList& programDefines)
    : Program(std::move(pDevice), desc.mBaseDesc, programDefines), mRtDesc(desc)
{
    if (desc.mRayGenCount == 0)
    {
        throw ArgumentError("Can't create an RtProgram without a ray generation shader");
    }
    if (desc.mMaxTraceRecursionDepth == -1)
    {
        throw ArgumentError("Can't create an RtProgram without specifying maximum trace recursion depth");
    }
    if (desc.mMaxPayloadSize == -1)
    {
        throw ArgumentError("Can't create an RtProgram without specifying maximum ray payload size");
    }
}

RtStateObject::SharedPtr RtProgram::getRtso(RtProgramVars* pVars)
{
    auto pProgramVersion = getActiveVersion();
    auto pProgramKernels = pProgramVersion->getKernels(mpDevice.get(), pVars);

    mRtsoGraph.walk((void*)pProgramKernels.get());

    RtStateObject::SharedPtr pRtso = mRtsoGraph.getCurrentNode();

    if (pRtso == nullptr)
    {
        RtStateObject::Desc desc;
        desc.setKernels(pProgramKernels);
        desc.setMaxTraceRecursionDepth(mRtDesc.mMaxTraceRecursionDepth);
        desc.setPipelineFlags(mRtDesc.mPipelineFlags);

        StateGraph::CompareFunc cmpFunc = [&desc](RtStateObject::SharedPtr pRtso) -> bool { return pRtso && (desc == pRtso->getDesc()); };

        if (mRtsoGraph.scanForMatchingNode(cmpFunc))
        {
            pRtso = mRtsoGraph.getCurrentNode();
        }
        else
        {
            pRtso = RtStateObject::create(mpDevice.get(), desc);
            mRtsoGraph.setCurrentNodeData(pRtso);
        }
    }

    return pRtso;
}
} // namespace Falcor
