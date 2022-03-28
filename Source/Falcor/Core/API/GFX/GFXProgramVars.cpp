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
#include "stdafx.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Program/GraphicsProgram.h"
#include "Core/Program/ComputeProgram.h"
#include "Core/API/ComputeContext.h"
#include "Core/API/RenderContext.h"
#include "GFXLowLevelContextApiData.h"

namespace Falcor
{
    ProgramVars::ProgramVars(
        const ProgramReflection::SharedConstPtr& pReflector)
        : ParameterBlock(pReflector)
        , mpReflector(pReflector)
    {
        FALCOR_ASSERT(pReflector);
    }

    void ComputeVars::dispatchCompute(ComputeContext* pContext, uint3 const& threadGroupCount)
    {
        auto pProgram = std::dynamic_pointer_cast<ComputeProgram>(getReflection()->getProgramVersion()->getProgram());
        FALCOR_ASSERT(pProgram);
        pProgram->dispatchCompute(pContext, this, threadGroupCount);
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
            auto getShaderNames = [&](VarsVector& varsVec, std::vector<const char*>& shaderNames, std::vector<gfx::IShaderTable::ShaderRecordOverwrite>* overwrites)
            {
                for (uint32_t i = 0; i < (uint32_t)varsVec.size(); i++)
                {
                    auto& varsInfo = varsVec[i];

                    auto uniqueGroupIndex = varsInfo.entryPointGroupIndex;

                    auto pGroupKernels = getUniqueRtEntryPointGroupKernels(pKernels, uniqueGroupIndex);
                    if (!pGroupKernels)
                    {
                        shaderNames.push_back(nullptr);
                        if (overwrites)
                        {
                            overwrites->push_back(gfx::IShaderTable::ShaderRecordOverwrite{});
                        }
                        continue;
                    }

                    shaderNames.push_back(static_cast<const char*>(pRtso->getShaderIdentifier(uniqueGroupIndex)));
                    
                }
            };

            std::vector<const char*> rayGenShaders;
            getShaderNames(mRayGenVars, rayGenShaders, nullptr);

            std::vector<const char*> missShaders;
            getShaderNames(mMissVars, missShaders, nullptr);

            std::vector<const char*> hitgroupShaders;
            std::vector<gfx::IShaderTable::ShaderRecordOverwrite> hitGroupRecordOverwrites;
            getShaderNames(mHitVars, hitgroupShaders, &hitGroupRecordOverwrites);

            gfx::IShaderTable::Desc desc = {};
            desc.rayGenShaderCount = (uint32_t)rayGenShaders.size();
            desc.rayGenShaderEntryPointNames = rayGenShaders.data();
            desc.missShaderCount = (uint32_t)missShaders.size();
            desc.missShaderEntryPointNames = missShaders.data();
            desc.hitGroupCount = (uint32_t)hitgroupShaders.size();
            desc.hitGroupNames = hitgroupShaders.data();
            desc.hitGroupRecordOverwrites = hitGroupRecordOverwrites.data();
            FALCOR_ASSERT(hitGroupRecordOverwrites.size() == desc.hitGroupCount);
            desc.program = pRtso->getKernels()->getApiHandle();
            if (SLANG_FAILED(gpDevice->getApiHandle()->createShaderTable(desc, mpShaderTable.writeRef())))
                return false;
            mpCurrentRtStateObject = pRtso;
        }

        return true;
    }
}
