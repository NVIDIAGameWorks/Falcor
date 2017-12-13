/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#include "Framework.h"
#include "API/ComputeContext.h"
#include "glm/gtc/type_ptr.hpp"
#include "API/Device.h"
#include "API/DescriptorSet.h"

namespace Falcor
{
    void ComputeContext::prepareForDispatch()
    {
        assert(mpComputeState);

        // Apply the vars. Must be first because applyComputeVars() might cause a flush        
        if (mpComputeVars)
        {
            applyComputeVars();
        }
        else
        {
            mpLowLevelData->getCommandList()->SetComputeRootSignature(RootSignature::getEmpty()->getApiHandle());
        }
        mBindComputeRootSig = false;
        mpLowLevelData->getCommandList()->SetPipelineState(mpComputeState->getCSO(mpComputeVars.get())->getApiHandle());
        mCommandsPending = true;
    }

    void ComputeContext::dispatch(uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ)
    {
        prepareForDispatch();
        mpLowLevelData->getCommandList()->Dispatch(groupSizeX, groupSizeY, groupSizeZ);
    }


    template<typename ClearType>
    void clearUavCommon(ComputeContext* pContext, const UnorderedAccessView* pUav, const ClearType& clear, ID3D12GraphicsCommandList* pList)
    {
        pContext->resourceBarrier(pUav->getResource(), Resource::State::UnorderedAccess);
        UavHandle uav = pUav->getApiHandle();
        if (typeid(ClearType) == typeid(vec4))
        {
            pList->ClearUnorderedAccessViewFloat(uav->getGpuHandle(0), uav->getCpuHandle(0), pUav->getResource()->getApiHandle(), (float*)value_ptr(clear), 0, nullptr);
        }
        else if (typeid(ClearType) == typeid(uvec4))
        {
            pList->ClearUnorderedAccessViewUint(uav->getGpuHandle(0), uav->getCpuHandle(0), pUav->getResource()->getApiHandle(), (uint32_t*)value_ptr(clear), 0, nullptr);
        }
        else
        {
            should_not_get_here();
        }
    }

    void ComputeContext::clearUAV(const UnorderedAccessView* pUav, const vec4& value)
    {
        clearUavCommon(this, pUav, value, mpLowLevelData->getCommandList().GetInterfacePtr());
        mCommandsPending = true;
    }

    void ComputeContext::clearUAV(const UnorderedAccessView* pUav, const uvec4& value)
    {
        clearUavCommon(this, pUav, value, mpLowLevelData->getCommandList().GetInterfacePtr());
        mCommandsPending = true;
    }

    void ComputeContext::clearUAVCounter(const StructuredBuffer::SharedPtr& pBuffer, uint32_t value)
    {
        if (pBuffer->hasUAVCounter())
        {
            clearUAV(pBuffer->getUAVCounter()->getUAV().get(), uvec4(value));
        }
    }

    void ComputeContext::initDispatchCommandSignature()
    {
        D3D12_COMMAND_SIGNATURE_DESC sigDesc;
        sigDesc.NumArgumentDescs = 1;
        sigDesc.NodeMask = 0;
        D3D12_INDIRECT_ARGUMENT_DESC argDesc;        
        sigDesc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        sigDesc.pArgumentDescs = &argDesc;
        gpDevice->getApiHandle()->CreateCommandSignature(&sigDesc, nullptr, IID_PPV_ARGS(&spDispatchCommandSig));
    }

    void ComputeContext::dispatchIndirect(const Buffer* argBuffer, uint64_t argBufferOffset)
    {
        prepareForDispatch();
        resourceBarrier(argBuffer, Resource::State::IndirectArg);
        mpLowLevelData->getCommandList()->ExecuteIndirect(spDispatchCommandSig, 1, argBuffer->getApiHandle(), argBufferOffset, nullptr, 0);
    }
}
