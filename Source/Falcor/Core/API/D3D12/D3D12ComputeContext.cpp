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
#include "Core/API/ComputeContext.h"
#include "glm/gtc/type_ptr.hpp"
#include "Core/API/Device.h"

namespace Falcor
{
    namespace
    {
        struct ComputeContextApiData
        {
            size_t refCount = 0;
            CommandSignatureHandle pDispatchCommandSig = nullptr;
            static void init();
            static void release();
        };

        ComputeContextApiData sApiData;

        void ComputeContextApiData::init()
        {
            if (!sApiData.pDispatchCommandSig)
            {
                D3D12_COMMAND_SIGNATURE_DESC sigDesc;
                sigDesc.NumArgumentDescs = 1;
                sigDesc.NodeMask = 0;
                D3D12_INDIRECT_ARGUMENT_DESC argDesc;
                sigDesc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
                argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
                sigDesc.pArgumentDescs = &argDesc;
                gpDevice->getApiHandle()->CreateCommandSignature(&sigDesc, nullptr, IID_PPV_ARGS(&sApiData.pDispatchCommandSig));
            }
            sApiData.refCount++;
        }

        void ComputeContextApiData::release()
        {
            sApiData.refCount--;
            if (sApiData.refCount == 0) sApiData = {};
        }
    }

    ComputeContext::ComputeContext(LowLevelContextData::CommandQueueType type, CommandQueueHandle queue)
        : CopyContext(type, queue)
    {
        assert(queue);
        ComputeContextApiData::init();
    }

    ComputeContext::~ComputeContext()
    {
        ComputeContextApiData::release();
    }

    bool ComputeContext::prepareForDispatch(ComputeState* pState, ComputeVars* pVars)
    {
        assert(pState);

        auto pCSO = pState->getCSO(pVars);

        // Apply the vars. Must be first because applyComputeVars() might cause a flush
        if (pVars)
        {
            if (applyComputeVars(pVars, pCSO->getDesc().getProgramKernels()->getRootSignature().get()) == false) return false;
        }
        else mpLowLevelData->getCommandList()->SetComputeRootSignature(RootSignature::getEmpty()->getApiHandle());

        mpLastBoundComputeVars = pVars;
        mpLowLevelData->getCommandList()->SetPipelineState(pCSO->getApiHandle());
        mCommandsPending = true;
        return true;
    }

    void ComputeContext::dispatch(ComputeState* pState, ComputeVars* pVars, const uint3& dispatchSize)
    {
        // Check dispatch dimensions. TODO: Should be moved into Falcor.
        if (dispatchSize.x > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            dispatchSize.y > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            dispatchSize.z > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
        {
            logError("ComputePass::execute() - Dispatch dimension exceeds maximum. Skipping.");
            return;
        }

        if (prepareForDispatch(pState, pVars) == false) return;
        mpLowLevelData->getCommandList()->Dispatch(dispatchSize.x, dispatchSize.y, dispatchSize.z);
    }


    template<typename ClearType>
    void clearUavCommon(ComputeContext* pContext, const UnorderedAccessView* pUav, const ClearType& clear, ID3D12GraphicsCommandList* pList)
    {
        pContext->resourceBarrier(pUav->getResource(), Resource::State::UnorderedAccess);
        UavHandle uav = pUav->getApiHandle();
        if (typeid(ClearType) == typeid(float4))
        {
            pList->ClearUnorderedAccessViewFloat(uav->getGpuHandle(0), uav->getCpuHandle(0), pUav->getResource()->getApiHandle(), (float*)value_ptr(clear), 0, nullptr);
        }
        else if (typeid(ClearType) == typeid(uint4))
        {
            pList->ClearUnorderedAccessViewUint(uav->getGpuHandle(0), uav->getCpuHandle(0), pUav->getResource()->getApiHandle(), (uint32_t*)value_ptr(clear), 0, nullptr);
        }
        else
        {
            should_not_get_here();
        }
    }

    void ComputeContext::clearUAV(const UnorderedAccessView* pUav, const float4& value)
    {
        clearUavCommon(this, pUav, value, mpLowLevelData->getCommandList().GetInterfacePtr());
        mCommandsPending = true;
    }

    void ComputeContext::clearUAV(const UnorderedAccessView* pUav, const uint4& value)
    {
        clearUavCommon(this, pUav, value, mpLowLevelData->getCommandList().GetInterfacePtr());
        mCommandsPending = true;
    }

    void ComputeContext::clearUAVCounter(const Buffer::SharedPtr& pBuffer, uint32_t value)
    {
        if (pBuffer->getUAVCounter())
        {
            clearUAV(pBuffer->getUAVCounter()->getUAV().get(), uint4(value));
        }
    }

    void ComputeContext::dispatchIndirect(ComputeState* pState, ComputeVars* pVars, const Buffer* pArgBuffer, uint64_t argBufferOffset)
    {
        if (prepareForDispatch(pState, pVars) == false) return;
        resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
        mpLowLevelData->getCommandList()->ExecuteIndirect(sApiData.pDispatchCommandSig, 1, pArgBuffer->getApiHandle(), argBufferOffset, nullptr, 0);
    }
}
