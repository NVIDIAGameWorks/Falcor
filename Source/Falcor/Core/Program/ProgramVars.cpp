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
#include "GraphicsProgram.h"
#include "ComputeProgram.h"
#include "Core/API/Device.h"
#include "Core/API/ComputeContext.h"
#include "Core/API/RenderContext.h"
#include "Core/API/GFXAPI.h"
#include "Utils/Logger.h"

#include <slang.h>

#include <set>

namespace Falcor
{

ProgramVars::ProgramVars(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector)
    : ParameterBlock(std::move(pDevice), pReflector), mpReflector(pReflector)
{
    FALCOR_ASSERT(pReflector);
}

GraphicsVars::GraphicsVars(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector)
    : ProgramVars(std::move(pDevice), pReflector)
{}

GraphicsVars::SharedPtr GraphicsVars::create(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector)
{
    if (pReflector == nullptr)
        throw ArgumentError("Can't create a GraphicsVars object without a program reflector");
    return SharedPtr(new GraphicsVars(std::move(pDevice), pReflector));
}

GraphicsVars::SharedPtr GraphicsVars::create(std::shared_ptr<Device> pDevice, const GraphicsProgram* pProg)
{
    if (pProg == nullptr)
        throw ArgumentError("Can't create a GraphicsVars object without a program");
    return create(std::move(pDevice), pProg->getReflector());
}

ComputeVars::SharedPtr ComputeVars::create(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector)
{
    if (pReflector == nullptr)
        throw ArgumentError("Can't create a ComputeVars object without a program reflector");
    return SharedPtr(new ComputeVars(std::move(pDevice), pReflector));
}

ComputeVars::SharedPtr ComputeVars::create(std::shared_ptr<Device> pDevice, const ComputeProgram* pProg)
{
    if (pProg == nullptr)
        throw ArgumentError("Can't create a ComputeVars object without a program");
    return create(std::move(pDevice), pProg->getReflector());
}

ComputeVars::ComputeVars(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector)
    : ProgramVars(pDevice, pReflector)
{}

void ComputeVars::dispatchCompute(ComputeContext* pContext, uint3 const& threadGroupCount)
{
    auto pProgram = std::dynamic_pointer_cast<ComputeProgram>(getReflection()->getProgramVersion()->getProgram());
    FALCOR_ASSERT(pProgram);
    pProgram->dispatchCompute(pContext, this, threadGroupCount);
}

RtProgramVars::RtProgramVars(
    std::shared_ptr<Device> pDevice,
    const RtProgram::SharedPtr& pProgram,
    const RtBindingTable::SharedPtr& pBindingTable
)
    : ProgramVars(pDevice, pProgram->getReflector()), mpShaderTable(pDevice)
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

RtProgramVars::SharedPtr RtProgramVars::create(
    std::shared_ptr<Device> pDevice,
    const RtProgram::SharedPtr& pProgram,
    const RtBindingTable::SharedPtr& pBindingTable
)
{
    return SharedPtr(new RtProgramVars(pDevice, pProgram, pBindingTable));
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
    mRayGenVars[0].entryPointGroupIndex = info.groupIndex;
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

        mMissVars[i].entryPointGroupIndex = info.groupIndex;

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
            if (!info.isValid())
                continue;

            mHitVars[mRayTypeCount * geometryID + rayType].entryPointGroupIndex = info.groupIndex;

            entryPointGroupIndices.insert(info.groupIndex);
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
