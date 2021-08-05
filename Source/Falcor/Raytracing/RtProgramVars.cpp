/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "RtProgramVars.h"

namespace Falcor
{
    RtProgramVars::RtProgramVars(const RtProgram::SharedPtr& pProgram, const RtBindingTable::SharedPtr& pBindingTable)
        : ProgramVars(pProgram->getReflector())
    {
        if (pProgram == nullptr)
        {
            throw std::exception("RtProgramVars must have a raytracing program attached to it");
        }
        if (pBindingTable == nullptr || !pBindingTable->getRayGen().isValid())
        {
            throw std::exception("RtProgramVars must have a raygen program attached to it");
        }

        mpRtVarsHelper = RtVarsContext::create();
        assert(mpRtVarsHelper);
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
        assert(mpProgramVersion);
        assert(dynamic_cast<RtProgram*>(mpProgramVersion->getProgram().get()));
        auto pProgram = static_cast<RtProgram*>(mpProgramVersion->getProgram().get());
        auto pReflector = mpProgramVersion->getReflector();

        auto& rtDesc = pProgram->getRtDesc();
        std::set<int32_t> entryPointGroupIndices;

        // Ray generation and miss programs are easy: we just allocate space
        // for one parameter block per entry-point of the given type in the binding table.
        //
        const auto& info = pBindingTable->getRayGen();
        assert(info.isValid());
        mRayGenVars.resize(1);
        mRayGenVars[0].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
        entryPointGroupIndices.insert(info.groupIndex);

        uint32_t missCount = pBindingTable->getMissCount();
        mMissVars.resize(missCount);

        for (uint32_t i = 0; i < missCount; ++i)
        {
            const auto& info = pBindingTable->getMiss(i);
            if (!info.isValid())
            {
                logWarning("Raytracing binding table has no shader at miss index " + std::to_string(i) + ". Is that intentional?");
                continue;
            }

            mMissVars[i].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
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

                mHitVars[mRayTypeCount * geometryID + rayType].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
                entryPointGroupIndices.insert(info.groupIndex);
            }
        }

        mUniqueEntryPointGroupIndices.assign(entryPointGroupIndices.begin(), entryPointGroupIndices.end());
        assert(!mUniqueEntryPointGroupIndices.empty());

        // Build list of vars for all entry point groups.
        // Note that there may be nullptr entries, as not all hit groups need to be assigned.
        assert(mRayGenVars.size() == 1);
        mpEntryPointGroupVars.push_back(mRayGenVars[0].pVars);
        for (auto entryPointGroupInfo : mMissVars)
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
        for (auto entryPointGroupInfo : mHitVars)
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
    }

    static bool applyRtProgramVars(
        uint8_t* pRecord,
        const RtEntryPointGroupKernels* pKernels,
        uint32_t uniqueEntryPointGroupIndex,
        const RtStateObject* pRtso,
        ParameterBlock* pVars,
        RtVarsContext* pContext)
    {
        assert(pKernels);

        auto pShaderIdentifier = pRtso->getShaderIdentifier(uniqueEntryPointGroupIndex);
        memcpy(pRecord, pShaderIdentifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        pRecord += D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

        auto pLocalRootSignature = pKernels->getLocalRootSignature();
        pContext->getRtVarsCmdList()->setRootParams(pLocalRootSignature, pRecord);

        return applyProgramVarsCommon<true>(pVars, pContext, true, pLocalRootSignature.get());
    }

    static RtEntryPointGroupKernels* getUniqueRtEntryPointGroupKernels(const ProgramKernels::SharedConstPtr& pKernels, uint32_t uniqueEntryPointGroupIndex)
    {
        if (uniqueEntryPointGroupIndex < 0) return nullptr;
        auto pEntryPointGroup = pKernels->getUniqueEntryPointGroup(uniqueEntryPointGroupIndex);
        assert(dynamic_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get()));
        return static_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get());
    }

    bool RtProgramVars::applyVarsToTable(ShaderTable::SubTableType type, uint32_t tableOffset, VarsVector& varsVec, const RtStateObject* pRtso)
    {
        auto& pKernels = pRtso->getKernels();
        
        for (uint32_t i = 0; i < (uint32_t)varsVec.size(); i++)
        {
            auto& varsInfo = varsVec[i];
            auto pBlock = varsInfo.pVars.get();
            if (!pBlock) continue;

            auto uniqueGroupIndex = pBlock->getGroupIndexInProgram();

            auto pGroupKernels = getUniqueRtEntryPointGroupKernels(pKernels, uniqueGroupIndex);
            if (!pGroupKernels) continue;

            // If we get here the shader record is used and will be assigned a valid shader identifier.
            // Shader records that are skipped above are left in their default state: initialized to zeros,
            // indicating that no shader should be executed and the local root signature will not be accessed.

            uint8_t* pRecord = mpShaderTable->getRecordPtr(type, tableOffset + i);

            if (!applyRtProgramVars(pRecord, pGroupKernels, uniqueGroupIndex, pRtso, pBlock, mpRtVarsHelper.get()))
            {
                return false;
            }
            varsInfo.lastObservedChangeEpoch = getEpochOfLastChange(pBlock);
        }

        return true;
    }

    bool RtProgramVars::apply(RenderContext* pCtx, RtStateObject* pRtso)
    {
        auto& pKernels = pRtso->getKernels();

        bool needShaderTableUpdate = false;
        if (!mpShaderTable)
        {
            mpShaderTable = ShaderTable::create();
            needShaderTableUpdate = true;
        }

        if (!needShaderTableUpdate)
        {
            if (pRtso != mpShaderTable->getRtso())
            {
                needShaderTableUpdate = true;
            }
        }

        if (!needShaderTableUpdate)
        {
            // We need to check if anything has changed that would require the shader
            // table to be rebuilt.
            assert(mRayGenVars.size() == 1);
            auto& varsInfo = mRayGenVars[0];
            auto pBlock = varsInfo.pVars.get();

            auto changeEpoch = computeEpochOfLastChange(pBlock);

            if (changeEpoch != varsInfo.lastObservedChangeEpoch)
            {
                needShaderTableUpdate = true;
            }
        }

        if (needShaderTableUpdate)
        {
            mpShaderTable->update(pCtx, pRtso, this);

            // We will iterate over the sub-tables (ray-gen, hit, miss)
            // in a specific order that matches the way that we have
            // enumerated the entry-point-group "instances" for indexing
            // in other parts of the code.
            if (!applyVarsToTable(ShaderTable::SubTableType::RayGen, 0, mRayGenVars, pRtso)) return false;
            if (!applyVarsToTable(ShaderTable::SubTableType::Miss, 0, mMissVars, pRtso)) return false;
            if (!applyVarsToTable(ShaderTable::SubTableType::Hit, 0, mHitVars, pRtso)) return false;

            mpShaderTable->flushBuffer(pCtx);
        }

        if (!applyProgramVarsCommon<false>(this, pCtx, true, pRtso->getGlobalRootSignature().get()))
        {
            return false;
        }

        return true;
    }
}
