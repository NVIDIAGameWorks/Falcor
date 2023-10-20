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
#include "ProgramVars.h"
#include "Program.h"
#include "Core/API/Device.h"
#include "Core/API/ComputeContext.h"
#include "Core/API/RenderContext.h"
#include "Core/API/GFXAPI.h"
#include "Utils/Logger.h"

#include <slang.h>

#include <set>

namespace Falcor
{

ProgramVars::ProgramVars(ref<Device> pDevice, const ref<const ProgramReflection>& pReflector)
    : ParameterBlock(pDevice, pReflector), mpReflector(pReflector)
{
    FALCOR_ASSERT(pReflector);
}

ref<ProgramVars> ProgramVars::create(ref<Device> pDevice, const ref<const ProgramReflection>& pReflector)
{
    FALCOR_CHECK(pReflector, "Can't create a ProgramVars object without a program reflector");
    return ref<ProgramVars>(new ProgramVars(pDevice, pReflector));
}

ref<ProgramVars> ProgramVars::create(ref<Device> pDevice, const Program* pProg)
{
    FALCOR_CHECK(pProg, "Can't create a ProgramVars object without a program");
    return create(pDevice, pProg->getReflector());
}

RtProgramVars::RtProgramVars(ref<Device> pDevice, const ref<Program>& pProgram, const ref<RtBindingTable>& pBindingTable)
    : ProgramVars(pDevice, pProgram->getReflector()), mpShaderTable(pDevice)
{
    FALCOR_CHECK(pProgram, "RtProgramVars must have a raytracing program attached to it");
    FALCOR_CHECK(pBindingTable && pBindingTable->getRayGen().isValid(), "RtProgramVars must have a raygen program attached to it");

    init(pBindingTable);
}

ref<RtProgramVars> RtProgramVars::create(ref<Device> pDevice, const ref<Program>& pProgram, const ref<RtBindingTable>& pBindingTable)
{
    return ref<RtProgramVars>(new RtProgramVars(pDevice, pProgram, pBindingTable));
}

void RtProgramVars::init(const ref<RtBindingTable>& pBindingTable)
{
    mRayTypeCount = pBindingTable->getRayTypeCount();
    mGeometryCount = pBindingTable->getGeometryCount();

    // We must create sub-shader-objects for all the entry point
    // groups that are used by the supplied binding table.
    //
    FALCOR_ASSERT(mpProgramVersion);
    auto pProgram = dynamic_cast<Program*>(mpProgramVersion->getProgram());
    FALCOR_ASSERT(pProgram);
    auto pReflector = mpProgramVersion->getReflector();

    std::set<int32_t> entryPointGroupIndices;

    // Ray generation and miss programs are easy: we just allocate space
    // for one parameter block per entry-point of the given type in the binding table.
    //
    const auto& rayGenInfo = pBindingTable->getRayGen();
    FALCOR_ASSERT(rayGenInfo.isValid());
    mRayGenVars.resize(1);
    mRayGenVars[0].entryPointGroupIndex = rayGenInfo.groupIndex;
    entryPointGroupIndices.insert(rayGenInfo.groupIndex);

    uint32_t missCount = pBindingTable->getMissCount();
    mMissVars.resize(missCount);

    for (uint32_t i = 0; i < missCount; ++i)
    {
        const auto& missInfo = pBindingTable->getMiss(i);
        if (!missInfo.isValid())
        {
            logWarning("Raytracing binding table has no shader at miss index {}. Is that intentional?", i);
            continue;
        }

        mMissVars[i].entryPointGroupIndex = missInfo.groupIndex;

        entryPointGroupIndices.insert(missInfo.groupIndex);
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
            const auto& hitGroupInfo = pBindingTable->getHitGroup(rayType, geometryID);
            if (!hitGroupInfo.isValid())
                continue;

            mHitVars[mRayTypeCount * geometryID + rayType].entryPointGroupIndex = hitGroupInfo.groupIndex;

            entryPointGroupIndices.insert(hitGroupInfo.groupIndex);
        }
    }

    mUniqueEntryPointGroupIndices.assign(entryPointGroupIndices.begin(), entryPointGroupIndices.end());
    FALCOR_ASSERT(!mUniqueEntryPointGroupIndices.empty());

    // Build list of vars for all entry point groups.
    // Note that there may be nullptr entries, as not all hit groups need to be assigned.
    FALCOR_ASSERT(mRayGenVars.size() == 1);
}

bool RtProgramVars::prepareShaderTable(RenderContext* pCtx, RtStateObject* pRtso)
{
    auto& pKernels = pRtso->getKernels();

    bool needShaderTableUpdate = false;
    if (!mpShaderTable)
    {
        needShaderTableUpdate = true;
    }

    if (!needShaderTableUpdate)
    {
        if (pRtso != mpCurrentRtStateObject)
        {
            needShaderTableUpdate = true;
        }
    }

    if (needShaderTableUpdate)
    {
        auto getShaderNames = [&](VarsVector& varsVec, std::vector<const char*>& shaderNames)
        {
            for (uint32_t i = 0; i < (uint32_t)varsVec.size(); i++)
            {
                auto& varsInfo = varsVec[i];

                auto uniqueGroupIndex = varsInfo.entryPointGroupIndex;

                auto pGroupKernels = uniqueGroupIndex >= 0 ? pKernels->getUniqueEntryPointGroup(uniqueGroupIndex) : nullptr;
                if (!pGroupKernels)
                {
                    shaderNames.push_back(nullptr);
                    continue;
                }

                shaderNames.push_back(static_cast<const char*>(pRtso->getShaderIdentifier(uniqueGroupIndex)));
            }
        };

        std::vector<const char*> rayGenShaders;
        getShaderNames(mRayGenVars, rayGenShaders);

        std::vector<const char*> missShaders;
        getShaderNames(mMissVars, missShaders);

        std::vector<const char*> hitgroupShaders;
        getShaderNames(mHitVars, hitgroupShaders);

        gfx::IShaderTable::Desc desc = {};
        desc.rayGenShaderCount = (uint32_t)rayGenShaders.size();
        desc.rayGenShaderEntryPointNames = rayGenShaders.data();
        desc.missShaderCount = (uint32_t)missShaders.size();
        desc.missShaderEntryPointNames = missShaders.data();
        desc.hitGroupCount = (uint32_t)hitgroupShaders.size();
        desc.hitGroupNames = hitgroupShaders.data();
        desc.program = pRtso->getKernels()->getGfxProgram();
        if (SLANG_FAILED(mpDevice->getGfxDevice()->createShaderTable(desc, mpShaderTable.writeRef())))
            return false;
        mpCurrentRtStateObject = pRtso;
    }

    return true;
}
} // namespace Falcor
