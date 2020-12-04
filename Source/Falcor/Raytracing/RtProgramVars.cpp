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
#include "RtProgramVars.h"

namespace Falcor
{
    static bool checkParams(RtProgram::SharedPtr pProgram, Scene::SharedPtr pScene)
    {
        if (pScene == nullptr)
        {
            logError("RtProgramVars must have a scene attached to it");
            return false;
        }

        if (pProgram == nullptr || pProgram->getRayGenProgramCount() == 0)
        {
            logError("RtProgramVars must have a ray-gen program attached to it");
            return false;
        }
        return true;
    }
    
    RtProgramVars::RtProgramVars(const RtProgram::SharedPtr& pProgram, const Scene::SharedPtr& pScene)
        : ProgramVars(pProgram->getReflector())
        , mpScene(pScene)
    {
        if (checkParams(pProgram, pScene) == false)
        {
            throw std::exception("Failed to create RtProgramVars object");
        }
        mpRtVarsHelper = RtVarsContext::create();
        assert(mpRtVarsHelper);
        init();
    }

    RtProgramVars::SharedPtr RtProgramVars::create(const RtProgram::SharedPtr& pProgram, const Scene::SharedPtr& pScene)
    {
        return SharedPtr(new RtProgramVars(pProgram, pScene));
    }

    void RtProgramVars::init()
    {
        // We must create sub-shader-objects for all the entry point
        // groups that are required by the scene.
        //
        assert(mpProgramVersion);
        auto pProgram = (RtProgram*) mpProgramVersion->getProgram().get();
        auto pReflector = mpProgramVersion->getReflector();

        auto& descExtra = pProgram->getDescExtra();

        // Ray generation and miss programs are easy: we just allocate space
        // for one parameter block per entry-point of the given type in
        // the original `RtProgram::Desc`.
        //
        // TODO: We could easily support multiple "instances" of ray generation
        // programs without requiring the SBT for miss or hit shaders to be
        // rebuild on parameter changes. It might make sense for ray-gen programs
        // to be more flexibly allocated.
        //

        uint32_t rayGenProgCount = uint32_t(descExtra.mRayGenEntryPoints.size());
        mRayGenVars.resize(rayGenProgCount);
        for (uint32_t i = 0; i < rayGenProgCount; ++i)
        {
            auto& info = descExtra.mRayGenEntryPoints[i];
            if (info.groupIndex < 0) continue;

            mRayGenVars[i].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
        }

        uint32_t missProgCount  = uint32_t(descExtra.mMissEntryPoints.size());
        mMissVars.resize(missProgCount);
        for (uint32_t i = 0; i < missProgCount; ++i)
        {
            auto& info = descExtra.mMissEntryPoints[i];
            if (info.groupIndex < 0) continue;

            mMissVars[i].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
        }

        // Hit groups are more complicated than ray generation and miss shaders.
        // We typically want a distinct parameter block per declared hit group
        // and per mesh in the scene (and sometimes even per mesh instance).
        //
        // We need to take this extra complexity into account when allocating
        // space for the hit group parameter blocks.
        //
        uint32_t descHitGroupCount = uint32_t(descExtra.mHitGroups.size());
        uint32_t blockCountPerHitGroup = mpScene->getMeshCount();
        uint32_t totalHitBlockCount = descHitGroupCount * blockCountPerHitGroup;
        mDescHitGroupCount = descHitGroupCount;

        mHitVars.resize(totalHitBlockCount);
        for (uint32_t i = 0; i < descHitGroupCount; ++i)
        {
            auto& info = descExtra.mHitGroups[i];
            if (info.groupIndex < 0) continue;

            for (uint32_t j = 0; j < blockCountPerHitGroup; ++j)
            {
                mHitVars[j * descHitGroupCount + i].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
            }
        }

        // Hit Groups for procedural primitives are different than for triangles.
        //
        // There must be a set of vars for every geometry defined in the BLAS (i.e. every prim added to the scene).
        // All intersection shader x hit-shader permutations are already generated in Program creation, so we look up entry points based on each
        // each geometry's type index.
        //
        // Hit groups in the program are ordered in the following way:
        //
        // [  Intersection Shader 0   |  Intersection Shader 1   | ... |  Intersection Shader N   ]
        // [          with            |           with           |     |          with            ]
        // [ Ray0 | Ray1 | ... | RayN | Ray0 | Ray1 | ... | RayN | ... | Ray0 | Ray1 | ... | RayN ]
        //
        // So the index of any specific hit group is calculated using: (IntersectionShaderIdx * RayCount + RayIdx)
        //
        // For each primitive, the hit groups for the corresponding intersection shader are looked up and appended to the vars.
        //
        uint32_t intersectionShaderCount = (uint32_t)descExtra.mIntersectionEntryPoints.size();
        uint32_t proceduralPrimHitVarCount = mpScene->getProceduralPrimitiveCount() * descHitGroupCount; // Total Var Count = Prim Count * Ray Count
        mAABBHitVars.resize(proceduralPrimHitVarCount);
        uint32_t currAABBHitVar = 0;
        for (uint32_t i = 0; i < mpScene->getProceduralPrimitiveCount(); i++)
        {
            uint32_t intersectionShaderId = mpScene->getProceduralPrimitive(i).typeID;
            assert(intersectionShaderId < intersectionShaderCount);

            // For this primitive's intersection shader group/type, get the hit vars for each ray type
            for (uint32_t j = 0; j < descHitGroupCount; j++)
            {
                auto& info = descExtra.mAABBHitGroups[intersectionShaderId * descHitGroupCount + j];
                if (info.groupIndex < 0) continue;

                mAABBHitVars[currAABBHitVar++].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
            }
        }

        for (auto entryPointGroupInfo : mRayGenVars)
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
        for (auto entryPointGroupInfo : mMissVars)
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
        for (auto entryPointGroupInfo : mHitVars)
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
        for (auto entryPointGroupInfo : mAABBHitVars)
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
    }

    bool applyRtProgramVars(
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

            auto uniqueGroupIndex = pBlock->getGroupIndexInProgram();

            auto pGroupKernels = getUniqueRtEntryPointGroupKernels(pKernels, uniqueGroupIndex);
            if (!pGroupKernels) continue;

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
        auto pKernels = pRtso->getKernels();
        auto pProgram = static_cast<RtProgram*>(pKernels->getProgramVersion()->getProgram().get());

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
            uint32_t rayGenCount = getRayGenVarsCount();
            for (uint32_t r = 0; r < rayGenCount; r++)
            {
                auto& varsInfo = mRayGenVars[r];
                auto pBlock = varsInfo.pVars.get();

                auto changeEpoch = computeEpochOfLastChange(pBlock);

                if (changeEpoch != varsInfo.lastObservedChangeEpoch)
                {
                    needShaderTableUpdate = true;
                }
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
            if (!applyVarsToTable(ShaderTable::SubTableType::Hit, (uint32_t)mHitVars.size(), mAABBHitVars, pRtso)) return false;

            mpShaderTable->flushBuffer(pCtx);
        }

        if (!applyProgramVarsCommon<false>(this, pCtx, true, pRtso->getGlobalRootSignature().get()))
        {
            return false;
        }

        return true;
    }
}
