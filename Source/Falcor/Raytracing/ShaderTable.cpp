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
#include "ShaderTable.h"
#include "RtProgram/RtProgram.h"
#include "RtProgramVars.h"

namespace Falcor
{
    ShaderTable::SharedPtr ShaderTable::create()
    {
        return SharedPtr(new ShaderTable());
    }

    uint8_t* ShaderTable::getRecordPtr(SubTableType type, uint32_t index)
    {
        auto info = getSubTableInfo(type);
        assert(index < info.recordCount);
        return &mData[0] + info.offset + index * info.recordSize;
    }

    static RtEntryPointGroupKernels* getUniqueRtEntryPointGroup(const ProgramKernels::SharedConstPtr& pKernels, int32_t index)
    {
        if (index < 0) return nullptr;
        auto pEntryPointGroup = pKernels->getUniqueEntryPointGroup(index);
        assert(dynamic_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get()));
        return static_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get());
    }

    void ShaderTable::update(RenderContext* pCtx, RtStateObject* pRtso, RtProgramVars const* pVars)
    {
        mpRtso = pRtso;

        auto pKernels = pRtso->getKernels();
        auto pProgram = static_cast<RtProgram*>(pKernels->getProgramVersion()->getProgram().get());

        for (uint32_t i = 0; i < uint32_t(SubTableType::Count); ++i)
        {
            mSubTables[i].offset = 0;
            mSubTables[i].recordCount = 0;
            mSubTables[i].recordSize = 0;
        }

        mSubTables[uint32_t(SubTableType::RayGen)].recordCount = pVars->getRayGenVarsCount();
        mSubTables[uint32_t(SubTableType::Miss)].recordCount = pVars->getMissVarsCount();
        mSubTables[uint32_t(SubTableType::Hit)].recordCount = pVars->getTotalHitVarsCount() + pVars->getAABBHitVarsCount();

        for (auto pUniqueEntryPointGroup : pKernels->getUniqueEntryPointGroups())
        {
            auto pEntryPointGroup = static_cast<RtEntryPointGroupKernels*>(pUniqueEntryPointGroup.get());

            SubTableType subTableType = SubTableType::Count;
            switch( pEntryPointGroup->getShaderByIndex(0)->getType() )
            {
            case ShaderType::AnyHit:
            case ShaderType::ClosestHit:
            case ShaderType::Intersection:
                subTableType = SubTableType::Hit;
                break;

            case ShaderType::RayGeneration:
                subTableType = SubTableType::RayGen;
                break;

            case ShaderType::Miss:
                subTableType = SubTableType::Miss;
                break;

            default:
                should_not_get_here();
                break;
            }

            auto& info = mSubTables[uint32_t(subTableType)];
            info.recordSize = std::max(pEntryPointGroup->getLocalRootSignature()->getSizeInBytes(), info.recordSize);
        }

        uint32_t subTableOffset = 0;
        for (uint32_t i = 0; i < uint32_t(SubTableType::Count); ++i)
        {
            auto& info = mSubTables[i];

            info.offset = subTableOffset;
            info.recordSize += D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
            info.recordSize = align_to(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, info.recordSize);

            subTableOffset += info.recordCount * info.recordSize;
            subTableOffset = align_to(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT, subTableOffset);
        }

        uint32_t shaderTableBufferSize = subTableOffset;

        mData.resize(shaderTableBufferSize);

        // Create a buffer
        if (!mpBuffer || mpBuffer->getSize() < shaderTableBufferSize)
        {
            mpBuffer = Buffer::create(shaderTableBufferSize, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None);
        }

        pCtx->updateBuffer(mpBuffer.get(), mData.data());
    }

    void ShaderTable::flushBuffer(RenderContext* pCtx)
    {
        pCtx->updateBuffer(mpBuffer.get(), mData.data());
    }
}
