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
#include "ProgramVars.h"
#include "GraphicsProgram.h"
#include "ComputeProgram.h"
#include "Core/API/ComputeContext.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"

#include <slang.h>

#include <set>

namespace Falcor
{
    void ProgramVars::addSimpleEntryPointGroups()
    {
#ifdef FALCOR_D3D12
        auto& entryPointGroups = mpReflector->getEntryPointGroups();
        auto groupCount = entryPointGroups.size();
        for( size_t gg = 0; gg < groupCount; ++gg )
        {
            auto pGroup = entryPointGroups[gg];
            auto pGroupVars = EntryPointGroupVars::create(pGroup, uint32_t(gg));
            mpEntryPointGroupVars.push_back(pGroupVars);
        }
#endif
    }

    GraphicsVars::GraphicsVars(const ProgramReflection::SharedConstPtr& pReflector)
        : ProgramVars(pReflector)
    {
        addSimpleEntryPointGroups();
    }

    GraphicsVars::SharedPtr GraphicsVars::create(const ProgramReflection::SharedConstPtr& pReflector)
    {
        if (pReflector == nullptr) throw ArgumentError("Can't create a GraphicsVars object without a program reflector");
        return SharedPtr(new GraphicsVars(pReflector));
    }

    GraphicsVars::SharedPtr GraphicsVars::create(const GraphicsProgram* pProg)
    {
        if (pProg == nullptr) throw ArgumentError("Can't create a GraphicsVars object without a program");
        return create(pProg->getReflector());
    }

    ComputeVars::SharedPtr ComputeVars::create(const ProgramReflection::SharedConstPtr& pReflector)
    {
        if (pReflector == nullptr) throw ArgumentError("Can't create a ComputeVars object without a program reflector");
        return SharedPtr(new ComputeVars(pReflector));
    }

    ComputeVars::SharedPtr ComputeVars::create(const ComputeProgram* pProg)
    {
        if (pProg == nullptr) throw ArgumentError("Can't create a ComputeVars object without a program");
        return create(pProg->getReflector());
    }

    ComputeVars::ComputeVars(const ProgramReflection::SharedConstPtr& pReflector)
        : ProgramVars(pReflector)
    {
        addSimpleEntryPointGroups();
    }


    RtProgramVars::RtProgramVars(const RtProgram::SharedPtr& pProgram, const RtBindingTable::SharedPtr& pBindingTable)
        : ProgramVars(pProgram->getReflector())
    {
        if (pProgram == nullptr)
        {
            throw ArgumentError("RtProgramVars must have a raytracing program attached to it");
        }
        if (pBindingTable == nullptr || !pBindingTable->getRayGen().isValid())
        {
            throw ArgumentError("RtProgramVars must have a raygen program attached to it");
        }

        init(pBindingTable);
    }

    RtProgramVars::SharedPtr RtProgramVars::create(const RtProgram::SharedPtr& pProgram, const RtBindingTable::SharedPtr& pBindingTable)
    {
        return SharedPtr(new RtProgramVars(pProgram, pBindingTable));
    }

    void RtProgramVars::init(const RtBindingTable::SharedPtr& pBindingTable)
    {
        mRayTypeCount = pBindingTable->getRayTypeCount();
        mGeometryCount = pBindingTable->getGeometryCount();

        // We must create sub-shader-objects for all the entry point
        // groups that are used by the supplied binding table.
        //
        FALCOR_ASSERT(mpProgramVersion);
        FALCOR_ASSERT(dynamic_cast<RtProgram*>(mpProgramVersion->getProgram().get()));
        auto pProgram = static_cast<RtProgram*>(mpProgramVersion->getProgram().get());
        auto pReflector = mpProgramVersion->getReflector();

        auto& rtDesc = pProgram->getRtDesc();
        std::set<int32_t> entryPointGroupIndices;

        // Ray generation and miss programs are easy: we just allocate space
        // for one parameter block per entry-point of the given type in the binding table.
        //
        const auto& info = pBindingTable->getRayGen();
        FALCOR_ASSERT(info.isValid());
        mRayGenVars.resize(1);
#if defined(FALCOR_D3D12)
        mRayGenVars[0].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
#elif defined(FALCOR_GFX)
        mRayGenVars[0].entryPointGroupIndex = info.groupIndex;
#endif
        entryPointGroupIndices.insert(info.groupIndex);

        uint32_t missCount = pBindingTable->getMissCount();
        mMissVars.resize(missCount);

        for (uint32_t i = 0; i < missCount; ++i)
        {
            const auto& info = pBindingTable->getMiss(i);
            if (!info.isValid())
            {
                logWarning("Raytracing binding table has no shader at miss index {}. Is that intentional?", i);
                continue;
            }

#if defined(FALCOR_D3D12)
            mMissVars[i].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
#elif defined(FALCOR_GFX)
            mMissVars[i].entryPointGroupIndex = info.groupIndex;
#endif
            entryPointGroupIndices.insert(info.groupIndex);
        }

        // Hit groups are more complicated than ray generation and miss shaders.
        // We typically want a distinct parameter block per declared hit group
        // and per geometry in the scene.
        //
        // We need to take this extra complexity into account when allocating
        // space for the hit group parameter blocks.
        //
        uint32_t hitCount = mRayTypeCount * mGeometryCount;
        mHitVars.resize(hitCount);

        for (uint32_t rayType = 0; rayType < mRayTypeCount; rayType++)
        {
            for (uint32_t geometryID = 0; geometryID < mGeometryCount; geometryID++)
            {
                const auto& info = pBindingTable->getHitGroup(rayType, geometryID);
                if (!info.isValid()) continue;

#if defined(FALCOR_D3D12)
                mHitVars[mRayTypeCount * geometryID + rayType].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
#elif defined(FALCOR_GFX)
                mHitVars[mRayTypeCount * geometryID + rayType].entryPointGroupIndex = info.groupIndex;
#endif
                entryPointGroupIndices.insert(info.groupIndex);
            }
        }

        mUniqueEntryPointGroupIndices.assign(entryPointGroupIndices.begin(), entryPointGroupIndices.end());
        FALCOR_ASSERT(!mUniqueEntryPointGroupIndices.empty());

        // Build list of vars for all entry point groups.
        // Note that there may be nullptr entries, as not all hit groups need to be assigned.
        FALCOR_ASSERT(mRayGenVars.size() == 1);
#if defined(FALCOR_D3D12)
        mpEntryPointGroupVars.push_back(mRayGenVars[0].pVars);
        for (auto entryPointGroupInfo : mMissVars)
        {
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
        }
        for (auto entryPointGroupInfo : mHitVars)
        {
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
        }
#endif
    }

    RtEntryPointGroupKernels* RtProgramVars::getUniqueRtEntryPointGroupKernels(const ProgramKernels::SharedConstPtr& pKernels, int32_t uniqueEntryPointGroupIndex)
    {
        if (uniqueEntryPointGroupIndex < 0) return nullptr;
        auto pEntryPointGroup = pKernels->getUniqueEntryPointGroup(uniqueEntryPointGroupIndex);
        FALCOR_ASSERT(dynamic_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get()));
        return static_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get());
    }
}
