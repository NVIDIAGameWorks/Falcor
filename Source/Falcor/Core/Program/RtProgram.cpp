/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "ProgramVars.h"

#include <slang.h>

namespace Falcor
{
    void RtProgram::Desc::init()
    {
        mBaseDesc.setShaderModel("6_5");
    }

    RtProgram::ShaderID RtProgram::Desc::addRayGen(const std::string& raygen, const TypeConformanceList& typeConformances, const std::string& entryPointNameSuffix)
    {
        mBaseDesc.beginEntryPointGroup(entryPointNameSuffix);
        mBaseDesc.entryPoint(ShaderType::RayGeneration, raygen);
        mBaseDesc.addTypeConformancesToGroup(typeConformances);

        mRayGenCount++;
        return { mBaseDesc.mActiveGroup };
    }

    RtProgram::ShaderID RtProgram::Desc::addMiss(const std::string& miss, const TypeConformanceList& typeConformances, const std::string& entryPointNameSuffix)
    {
        mBaseDesc.beginEntryPointGroup(entryPointNameSuffix);
        mBaseDesc.entryPoint(ShaderType::Miss, miss);
        mBaseDesc.addTypeConformancesToGroup(typeConformances);

        return { mBaseDesc.mActiveGroup };
    }

    RtProgram::ShaderID RtProgram::Desc::addHitGroup(const std::string& closestHit, const std::string& anyHit, const std::string& intersection, const TypeConformanceList& typeConformances, const std::string& entryPointNameSuffix)
    {
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

        return { mBaseDesc.mActiveGroup };
    }

    RtProgram::SharedPtr RtProgram::create(Desc desc, const DefineList& programDefines)
    {
        auto pProg = SharedPtr(new RtProgram(desc, programDefines));
        registerProgramForReload(pProg);
        return pProg;
    }

    RtProgram::RtProgram(const RtProgram::Desc& desc, const DefineList& programDefines)
        : Program(desc.mBaseDesc, programDefines)
        , mRtDesc(desc)
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

    static uint64_t sHitGroupID = 0;

    EntryPointGroupKernels::SharedPtr RtProgram::createEntryPointGroupKernels(
        const std::vector<Shader::SharedPtr>& shaders,
        EntryPointBaseReflection::SharedPtr const& pReflector) const
    {
        FALCOR_ASSERT(shaders.size() != 0);

        if (
#ifdef FALCOR_D3D12
            pReflector->getD3D12DescriptorSetCount() > 0 ||
#endif
            pReflector->getResourceRangeCount() > 0 ||
            pReflector->getRootDescriptorRangeCount() > 0 ||
            pReflector->getParameterBlockSubObjectRangeCount() > 0)
        {
            throw RuntimeError("Local root signatures are not supported for raytracing entry points.");
        }

        switch (shaders[0]->getType())
        {
        case ShaderType::AnyHit:
        case ShaderType::ClosestHit:
        case ShaderType::Intersection:
            {
                std::string exportName = "HitGroup" + std::to_string(sHitGroupID++);
                return RtEntryPointGroupKernels::create(RtEntryPointGroupKernels::Type::RtHitGroup, shaders, exportName, mRtDesc.mMaxPayloadSize, mRtDesc.mMaxAttributeSize);
            }

        default:
            return RtEntryPointGroupKernels::create(RtEntryPointGroupKernels::Type::RtSingleShader, shaders, shaders[0]->getEntryPoint(), mRtDesc.mMaxPayloadSize, mRtDesc.mMaxAttributeSize);
        }
    }

    RtStateObject::SharedPtr RtProgram::getRtso(RtProgramVars* pVars)
    {
        auto pProgramVersion = getActiveVersion();
        auto pProgramKernels = pProgramVersion->getKernels(pVars);

        mRtsoGraph.walk((void*) pProgramKernels.get());

        RtStateObject::SharedPtr pRtso = mRtsoGraph.getCurrentNode();

        if (pRtso == nullptr)
        {
            RtStateObject::Desc desc;
            desc.setKernels(pProgramKernels);
            desc.setMaxTraceRecursionDepth(mRtDesc.mMaxTraceRecursionDepth);
            desc.setPipelineFlags(mRtDesc.mPipelineFlags);

            StateGraph::CompareFunc cmpFunc = [&desc](RtStateObject::SharedPtr pRtso) -> bool
            {
                return pRtso && (desc == pRtso->getDesc());
            };

            if (mRtsoGraph.scanForMatchingNode(cmpFunc))
            {
                pRtso = mRtsoGraph.getCurrentNode();
            }
            else
            {
                pRtso = RtStateObject::create(desc);
                mRtsoGraph.setCurrentNodeData(pRtso);
            }
        }

        return pRtso;
    }
}
