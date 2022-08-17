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
#include "Core/API/ShaderTable.h"
#include "Core/API/Device.h"
#include "Core/API/D3D12/D3D12API.h"
#include "Core/Program/ProgramVars.h"
#include "Utils/Math/Common.h"

namespace Falcor
{
    ShaderTable::SharedPtr ShaderTable::create()
    {
        return SharedPtr(new ShaderTable());
    }

    ShaderTable::ShaderTable()
    {
        apiInit();
    }

    void ShaderTable::apiInit()
    {
        mShaderIdentifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        mShaderRecordAlignment = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
        mShaderTableAlignment = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    }

    uint8_t* ShaderTable::getRecordPtr(SubTableType type, uint32_t index)
    {
        auto info = getSubTableInfo(type);
        FALCOR_ASSERT(index < info.recordCount);
        return mData.data() + info.offset + index * info.recordSize;
    }

    static RtEntryPointGroupKernels* getUniqueRtEntryPointGroup(const ProgramKernels::SharedConstPtr& pKernels, int32_t index)
    {
        if (index < 0) return nullptr;
        auto pEntryPointGroup = pKernels->getUniqueEntryPointGroup(index);
        FALCOR_ASSERT(dynamic_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get()));
        return static_cast<RtEntryPointGroupKernels*>(pEntryPointGroup.get());
    }

    void ShaderTable::update(RenderContext* pCtx, RtStateObject* pRtso, RtProgramVars const* pVars)
    {
        mpRtso = pRtso;

        auto& pKernels = pRtso->getKernels();

        for (uint32_t i = 0; i < uint32_t(SubTableType::Count); ++i)
        {
            mSubTables[i] = {};
        }

        mSubTables[uint32_t(SubTableType::RayGen)].recordCount = 1;
        mSubTables[uint32_t(SubTableType::Miss)].recordCount = pVars->getMissVarsCount();
        mSubTables[uint32_t(SubTableType::Hit)].recordCount = pVars->getTotalHitVarsCount();

        // Iterate over the entry points used by RtProgramVars to compute the
        // maximum shader table record size for each sub-table.
        for (auto index : pVars->getUniqueEntryPointGroupIndices())
        {
            auto pEntryPointGroup = getUniqueRtEntryPointGroup(pKernels, index);

            SubTableType subTableType = SubTableType::Count;
            switch (pEntryPointGroup->getShaderByIndex(0)->getType())
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
                FALCOR_UNREACHABLE();
                break;
            }

            // Nothing done here because we currently don't support additional shader record data.
        }

        uint32_t subTableOffset = 0;
        for (uint32_t i = 0; i < uint32_t(SubTableType::Count); ++i)
        {
            auto& info = mSubTables[i];

            info.offset = subTableOffset;
            info.recordSize += mShaderIdentifierSize;
            info.recordSize = align_to(mShaderRecordAlignment, info.recordSize);

            subTableOffset += info.recordCount * info.recordSize;
            subTableOffset = align_to(mShaderTableAlignment, subTableOffset);
        }

        uint32_t shaderTableBufferSize = subTableOffset;

        // Reallocate CPU buffer for shader table.
        // Make sure it's zero initialized as there may be unused miss/hit entries.
        if (shaderTableBufferSize != mData.size()) mData.resize(shaderTableBufferSize, 0);
        else std::fill(mData.begin(), mData.end(), 0);

        // Create GPU buffer.
        if (!mpBuffer || mpBuffer->getSize() < shaderTableBufferSize)
        {
            mpBuffer = Buffer::create(shaderTableBufferSize, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None);
        }
    }

    void ShaderTable::flushBuffer(RenderContext* pCtx)
    {
        pCtx->updateBuffer(mpBuffer.get(), mData.data());
    }
}
