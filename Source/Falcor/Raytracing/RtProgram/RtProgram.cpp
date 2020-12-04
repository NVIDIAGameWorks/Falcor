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

        mBaseDesc.beginEntryPointGroup();
        mBaseDesc.entryPoint(ShaderType::Miss, miss);

        DescExtra::GroupInfo info = { mBaseDesc.mActiveGroup };
        mMissEntryPoints[missIndex] = info;
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addHitGroup(uint32_t hitIndex, const std::string& closestHit, const std::string& anyHit)
    {
        if (hitIndex >= mHitGroups.size())
        {
            mHitGroups.resize(hitIndex + 1);
        }
        else if (mHitGroups[hitIndex].groupIndex >= 0)
        {
            logError("already have a hit group at that index");
        }

        mBaseDesc.beginEntryPointGroup();
        if (closestHit.length())
        {
            mBaseDesc.entryPoint(ShaderType::ClosestHit, closestHit);
        }
        if (anyHit.length())
        {
            mBaseDesc.entryPoint(ShaderType::AnyHit, anyHit);
        }

        DescExtra::GroupInfo info = { mBaseDesc.mActiveGroup };
        mHitGroups[hitIndex] = info;
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addAABBHitGroup(uint32_t hitIndex, const std::string& closestHit, const std::string& anyHit /*= ""*/)
    {
        if (hitIndex >= mAABBHitGroupEntryPoints.size())
        {
            mAABBHitGroupEntryPoints.resize(hitIndex + 1);
        }
        else
        {
            auto& group = mAABBHitGroupEntryPoints[hitIndex];
            if (group.closestHit != uint32_t(-1) || group.anyHit != uint32_t(-1))
            {
                throw std::exception(("There is already an AABB hit group defined at index " + std::to_string(hitIndex)).c_str());
            }
        }

        auto& group = mAABBHitGroupEntryPoints[hitIndex];

        if (!closestHit.empty())
        {
            group.closestHit = mBaseDesc.declareEntryPoint(ShaderType::ClosestHit, closestHit);
        }

        if (!anyHit.empty())
        {
            group.anyHit = mBaseDesc.declareEntryPoint(ShaderType::AnyHit, anyHit);
        }

        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addIntersection(uint32_t typeIndex, const std::string& intersection)
    {
        if (typeIndex >= mIntersectionEntryPoints.size())
        {
            mIntersectionEntryPoints.resize(typeIndex + 1);
        }
        else if (mIntersectionEntryPoints[typeIndex] != uint32_t(-1))
        {
            throw std::exception(("There is already an intersection shader defined at primitive type index " + std::to_string(typeIndex)).c_str());
        }

        assert(!intersection.empty());
        mIntersectionEntryPoints[typeIndex] = mBaseDesc.declareEntryPoint(ShaderType::Intersection, intersection);

        return *this;
    }

    void RtProgram::Desc::resolveAABBHitGroups()
    {
        // Every intersection shader defines a custom primitive type, so we need permutations where each CHS/AHS is paired
        // with each intersection shader. This is required by state object creation time so we need to generate all groups up front.
        for (auto& intersection : mIntersectionEntryPoints)
        {
            for (auto& hitGroup : mAABBHitGroupEntryPoints)
            {
                auto& closestHit = hitGroup.closestHit;
                auto& anyHit = hitGroup.anyHit;

                // Save index of the group being added
                DescExtra::GroupInfo groupInfo;
                groupInfo.groupIndex = (int32_t)mBaseDesc.mGroups.size();

                // Add entry point group containing each shader in the hit group
                Program::Desc::EntryPointGroup group;
                if (closestHit != uint32_t(-1)) group.entryPoints.push_back(closestHit);
                if (anyHit != uint32_t(-1)) group.entryPoints.push_back(anyHit);
                group.entryPoints.push_back(intersection); // TODO: This has to go last. Why?
                mBaseDesc.mGroups.push_back(group);

                // Save association of hit group index -> program entry point group
                mAABBHitGroups.push_back(groupInfo);
            }
        }
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

    RtProgram::SharedPtr RtProgram::create(Desc desc, uint32_t maxPayloadSize, uint32_t maxAttributesSize)
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

        if (!desc.mAABBHitGroupEntryPoints.empty() && (desc.mHitGroups.size() != desc.mAABBHitGroupEntryPoints.size()))
        {
            logWarning("There are not corresponding hit shaders for each ray type defined for custom primitives.");
        }

        // Both intersection shaders and hit groups must be defined for custom primitives for it to be valid/complete
        if (!desc.mIntersectionEntryPoints.empty() && !desc.mAABBHitGroupEntryPoints.empty())
        {
            desc.resolveAABBHitGroups();
        }

        SharedPtr pProg = SharedPtr(new RtProgram(maxPayloadSize, maxAttributesSize));
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

    RtProgram::RtProgram(uint32_t maxPayloadSize, uint32_t maxAttributesSize)
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
