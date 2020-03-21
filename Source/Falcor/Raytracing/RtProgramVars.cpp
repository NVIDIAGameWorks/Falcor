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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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

#include "Core/API/Device.h"
//#include "RtStateObject.h"

#include "RtProgramVarsHelper.h"

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
    
    RtProgramVars::RtProgramVars(
        const RtProgram::SharedPtr& pProgram,
        const Scene::SharedPtr& pScene)
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
        uint32_t rayGenProgCount    = uint32_t(descExtra.mRayGenEntryPoints.size());
        uint32_t missProgCount      = uint32_t(descExtra.mMissEntryPoints.size());

        mRayGenVars.resize(rayGenProgCount);
        mMissVars.resize(missProgCount);

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

        mHitVars.resize(totalHitBlockCount);
        mDescHitGroupCount = descHitGroupCount;

        for(uint32_t i = 0; i < rayGenProgCount; ++i)
        {
            auto& info = descExtra.mRayGenEntryPoints[i];
            if(info.groupIndex < 0) continue;

            mRayGenVars[i].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
        }

        for(uint32_t i = 0; i < descHitGroupCount; ++i)
        {
            auto& info = descExtra.mHitGroups[i];
            if(info.groupIndex < 0) continue;

            for(uint32_t j = 0; j < blockCountPerHitGroup; ++j)
            {
                mHitVars[j*descHitGroupCount + i].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
            }
        }

        for(uint32_t i = 0; i < missProgCount; ++i)
        {
            auto& info = descExtra.mMissEntryPoints[i];
            if(info.groupIndex < 0) continue;

            mMissVars[i].pVars = EntryPointGroupVars::create(pReflector->getEntryPointGroup(info.groupIndex), info.groupIndex);
        }

        for(auto entryPointGroupInfo : mRayGenVars)
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
        for(auto entryPointGroupInfo : mHitVars)
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
        for(auto entryPointGroupInfo : mMissVars)
            mpEntryPointGroupVars.push_back(entryPointGroupInfo.pVars);
    }

    bool applyRtProgramVars(
        uint8_t*                        pRecord,
        const RtEntryPointGroupKernels* pKernels,
        uint32_t                        uniqueEntryPointGroupIndex,
        const RtStateObject*            pRtso,
        ParameterBlock*                 pVars,
        RtVarsContext*                  pContext
        )
    {
        assert(pKernels);

        auto pShaderIdentifier = pRtso->getShaderIdentifier(uniqueEntryPointGroupIndex);
        memcpy(pRecord, pShaderIdentifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        pRecord += D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

        auto pLocalRootSignature = pKernels->getLocalRootSignature();

        pContext->getRtVarsCmdList()->setRootParams(pLocalRootSignature, pRecord);
        return applyProgramVarsCommon<true>(pVars, pContext, true, pLocalRootSignature.get());

        return true;
    }

    static RtEntryPointGroupKernels* getUniqueRtEntryPointGroupKernels(
        const ProgramKernels::SharedConstPtr&   pKernels,
        uint32_t                                uniqueEntryPointGroupIndex)
    {
        if(uniqueEntryPointGroupIndex < 0) return nullptr;
        auto pEntryPointGroup = pKernels->getUniqueEntryPointGroup(uniqueEntryPointGroupIndex);
        assert(dynamic_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get()));
        return static_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get());
    }

    bool RtProgramVars::apply(
        RenderContext*  pCtx,
        RtStateObject*  pRtso)
    {
        auto pKernels = pRtso->getKernels();
        auto pProgram = static_cast<RtProgram*>(pKernels->getProgramVersion()->getProgram().get());

        bool needShaderTableUpdate = false;
        if(!mpShaderTable)
        {
            mpShaderTable = ShaderTable::create();
            needShaderTableUpdate = true;
        }

        if( !needShaderTableUpdate )
        {
            if( pRtso != mpShaderTable->getRtso() )
            {
                needShaderTableUpdate = true;
            }
        }

        if( !needShaderTableUpdate )
        {
            // We need to check if anything has changed that would require the shader
            // table to be rebuilt.
            //
            uint32_t rayGenCount = getRayGenVarsCount();
            for (uint32_t r = 0; r < rayGenCount; r++)
            {
                auto& varsInfo = mRayGenVars[r];
                auto pBlock = varsInfo.pVars.get();

                auto changeEpoch = computeEpochOfLastChange(pBlock);

                if(changeEpoch != varsInfo.lastObservedChangeEpoch)
                {
                    needShaderTableUpdate = true;
                }
            }
        }

        if(needShaderTableUpdate)
        {
            mpShaderTable->update(pCtx, pRtso, this, mpScene.get());

            // We will iterate over the sub-tables (ray-gen, hit, miss)
            // in a specific order that matches the way that we have
            // enumerated the entry-point-group "instances" for indexing
            // in other parts of the code.
            //

            uint32_t rayGenCount = getRayGenVarsCount();
            for (uint32_t r = 0; r < rayGenCount; r++)
            {
                auto& varsInfo = mRayGenVars[r];
                auto pBlock = varsInfo.pVars.get();

                auto uniqueGroupIndex = pBlock->getGroupIndexInProgram();

                auto pGroupKernels = getUniqueRtEntryPointGroupKernels(pKernels, uniqueGroupIndex);
                if(!pGroupKernels) { continue; }

                uint8_t* pRecord = mpShaderTable->getRecordPtr(ShaderTable::SubTableType::RayGen, r);

                if (!applyRtProgramVars(pRecord, pGroupKernels, uniqueGroupIndex, pRtso, pBlock, mpRtVarsHelper.get()))
                {
                    return false;
                }
                varsInfo.lastObservedChangeEpoch = getEpochOfLastChange(pBlock);
            }


            // Loop over the rays
            uint32_t hitCount = getTotalHitVarsCount();
            for (uint32_t h = 0; h < hitCount; h++)
            {
                auto& varsInfo = mHitVars[h];
                auto pBlock = varsInfo.pVars.get();

                auto uniqueGroupIndex = pBlock->getGroupIndexInProgram();

                auto pGroupKernels = getUniqueRtEntryPointGroupKernels(pKernels, uniqueGroupIndex);
                if(!pGroupKernels) { continue; }

                uint8_t* pRecord = mpShaderTable->getRecordPtr(ShaderTable::SubTableType::Hit, h);

                if (!applyRtProgramVars(pRecord, pGroupKernels, uniqueGroupIndex, pRtso, pBlock, mpRtVarsHelper.get()))
                {
                    return false;
                }
                varsInfo.lastObservedChangeEpoch = getEpochOfLastChange(pBlock);
            }

            uint32_t missCount = pProgram->getMissProgramCount();
            for (uint32_t m = 0; m < missCount; m++)
            {
                auto& varsInfo = mMissVars[m];
                auto pBlock = varsInfo.pVars.get();

                auto uniqueGroupIndex = pBlock->getGroupIndexInProgram();

                auto pGroupKernels = getUniqueRtEntryPointGroupKernels(pKernels, uniqueGroupIndex);
                if(!pGroupKernels) { continue; }

                uint8_t* pRecord = mpShaderTable->getRecordPtr(ShaderTable::SubTableType::Miss, m);

                if (!applyRtProgramVars(pRecord, pGroupKernels, uniqueGroupIndex, pRtso, pBlock, mpRtVarsHelper.get()))
                {
                    return false;
                }
                varsInfo.lastObservedChangeEpoch = getEpochOfLastChange(pBlock);
            }

            mpShaderTable->flushBuffer(pCtx);
        }

        if( !applyProgramVarsCommon<false>(this, pCtx, true, pRtso->getGlobalRootSignature().get()) )
        {
            return false;
        }

        return true;
    }
}
