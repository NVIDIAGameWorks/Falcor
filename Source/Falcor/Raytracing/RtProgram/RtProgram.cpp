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
#include "stdafx.h"
#include "RtProgram.h"

#include "Raytracing/RtProgramVars.h"

#include <slang/slang.h>

namespace Falcor
{
    void RtProgram::Desc::init()
    {
        mBaseDesc.setShaderModel("6_2");
    }

    RtProgram::Desc& RtProgram::Desc::addShaderLibrary(const std::string& filename)
    {
        mBaseDesc.addShaderLibrary(filename);
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::setRayGen(const std::string& raygen)
    {
        return addRayGen(raygen);
    }

    RtProgram::Desc& RtProgram::Desc::addRayGen(const std::string& raygen)
    {
        mBaseDesc.beginEntryPointGroup();
        mBaseDesc.entryPoint(ShaderType::RayGeneration, raygen);

        DescExtra::GroupInfo info = { mBaseDesc.mActiveGroup };
        mRayGenEntryPoints.push_back(info);
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addMiss(uint32_t missIndex, const std::string& miss)
    {
        if(missIndex >= mMissEntryPoints.size())
        {
            mMissEntryPoints.resize(missIndex+1);
        }
        else if(mMissEntryPoints[missIndex].groupIndex >= 0)
        {
            logError("already have a miss shader at that index");
        }

        auto entryPointIndex = int32_t(mBaseDesc.mEntryPoints.size());
        mBaseDesc.beginEntryPointGroup();
        mBaseDesc.entryPoint(ShaderType::Miss, miss);

        DescExtra::GroupInfo info = { mBaseDesc.mActiveGroup };
        mMissEntryPoints[missIndex] = info;
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addHitGroup(uint32_t hitIndex, const std::string& closestHit, const std::string& anyHit, const std::string& intersection /* = "" */)
    {
        if(hitIndex >= mHitGroups.size())
        {
            mHitGroups.resize(hitIndex+1);
        }
        else if(mHitGroups[hitIndex].groupIndex >= 0)
        {
            logError("already have a hit group at that index");
        }

        auto groupIndex = int32_t(mBaseDesc.mGroups.size());
        mBaseDesc.beginEntryPointGroup();
        if(closestHit.length())
        {
            mBaseDesc.entryPoint(ShaderType::ClosestHit, closestHit);
        }
        if(anyHit.length())
        {
            mBaseDesc.entryPoint(ShaderType::AnyHit, anyHit);
        }
        if(intersection.length())
        {
            mBaseDesc.entryPoint(ShaderType::Intersection, intersection);
        }

        DescExtra::GroupInfo info = { mBaseDesc.mActiveGroup };
        mHitGroups[hitIndex] = info;
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addDefine(const std::string& name, const std::string& value)
    {
        mDefineList.add(name, value);
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addDefines(const DefineList& defines)
    {
        for (auto it : defines) addDefine(it.first, it.second);
        return *this;
    }

    RtProgram::SharedPtr RtProgram::create(const Desc& desc, uint32_t maxPayloadSize, uint32_t maxAttributesSize)
    {
        size_t rayGenCount = desc.mRayGenEntryPoints.size();
        if (rayGenCount == 0)
        {
            throw std::exception("Can't create an RtProgram without a ray generation shader");
        }
        else if(rayGenCount > 1)
        {
            throw std::exception("Can't create an RtProgram with more than one ray generation shader");
        }

        SharedPtr pProg = SharedPtr(new RtProgram(desc, maxPayloadSize, maxAttributesSize));
        pProg->init(desc);
        pProg->addDefine("_MS_DISABLE_ALPHA_TEST");
        pProg->addDefine("_DEFAULT_ALPHA_TEST");

        return pProg;
    }

    void RtProgram::init(const RtProgram::Desc& desc)
    {
        Program::init(desc.mBaseDesc, desc.mDefineList);
        mDescExtra = desc;
    }

    RtProgram::RtProgram(const Desc& desc, uint32_t maxPayloadSize, uint32_t maxAttributesSize)
        : Program()
        , mMaxPayloadSize(maxPayloadSize)
        , mMaxAttributesSize(maxAttributesSize)
    {
    }

    static uint64_t sHitGroupID = 0;

    EntryPointGroupKernels::SharedPtr RtProgram::createEntryPointGroupKernels(
        const std::vector<Shader::SharedPtr>& shaders,
        EntryPointBaseReflection::SharedPtr const& pReflector) const
    {
        assert(shaders.size() != 0);

        auto localRootSignature = RootSignature::createLocal(pReflector.get());

        switch( shaders[0]->getType() )
        {
        case ShaderType::AnyHit:
        case ShaderType::ClosestHit:
        case ShaderType::Intersection:
            {
                std::string exportName = "HitGroup" + std::to_string(sHitGroupID++);
                return RtEntryPointGroupKernels::create(RtEntryPointGroupKernels::Type::RtHitGroup, shaders, exportName, localRootSignature, mMaxPayloadSize, mMaxAttributesSize);
            }

        default:
            return RtEntryPointGroupKernels::create(RtEntryPointGroupKernels::Type::RtSingleShader, shaders, shaders[0]->getEntryPoint(), localRootSignature, mMaxPayloadSize, mMaxAttributesSize);
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
            desc.setMaxTraceRecursionDepth(mDescExtra.mMaxTraceRecursionDepth);
            desc.setGlobalRootSignature(pProgramKernels->getRootSignature());

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

    void RtProgram::setScene(const Scene::SharedPtr& pScene)
    {
        if (mpScene == pScene) return;
        mpScene = pScene;
    }
}
