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
#include "Core/API/ComputeContext.h"
#include "Core/API/Device.h"
#include "Core/API/D3D12/D3D12API.h"
#include "Core/State/ComputeState.h"
#include "Core/Program/ProgramVars.h"
#include <glm/gtc/type_ptr.hpp>

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
        FALCOR_ASSERT(queue);
        ComputeContextApiData::init();
    }

    ComputeContext::~ComputeContext()
    {
        ComputeContextApiData::release();
    }

    void ComputeContext::applyComputeVars(ComputeVars* pVars, const ProgramKernels* pProgramKernels)
    {
        bool varsChanged = (pVars != mpLastBoundComputeVars);

        // FIXME TODO Temporary workaround
        varsChanged = true;

        if (pVars->apply(this, varsChanged, pProgramKernels) == false)
        {
            logWarning("ComputeContext::applyComputeVars() - applying ComputeVars failed, most likely because we ran out of descriptors. Flushing the GPU and retrying");
            flush(true);
            if (!pVars->apply(this, varsChanged, pProgramKernels))
            {
                throw RuntimeError("ComputeVars::applyComputeVars() - applying ComputeVars failed, most likely because we ran out of descriptors");
            }
        }
    }

    void ComputeContext::prepareForDispatch(ComputeState* pState, ComputeVars* pVars)
    {
        FALCOR_ASSERT(pState);

        auto pCSO = pState->getCSO(pVars);

        // Apply the vars. Must be first because applyComputeVars() might cause a flush
        if (pVars)
        {
            applyComputeVars(pVars, pCSO->getDesc().getProgramKernels().get());
        }
        else
        {
            mpLowLevelData->getCommandList()->SetComputeRootSignature(D3D12RootSignature::getEmpty()->getApiHandle());
        }

        mpLastBoundComputeVars = pVars;
        mpLowLevelData->getCommandList()->SetPipelineState(pCSO->getApiHandle());
        mCommandsPending = true;
    }

    void ComputeContext::dispatch(ComputeState* pState, ComputeVars* pVars, const uint3& dispatchSize)
    {
        // Check dispatch dimensions. TODO: Should be moved into Falcor.
        if (dispatchSize.x > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            dispatchSize.y > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            dispatchSize.z > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
        {
            throw RuntimeError("ComputePass::execute() - Dispatch dimension exceeds maximum.");
        }

        prepareForDispatch(pState, pVars);
        mpLowLevelData->getCommandList()->Dispatch(dispatchSize.x, dispatchSize.y, dispatchSize.z);
    }

    D3D12_DESCRIPTOR_HEAP_TYPE falcorToDxDescType(D3D12DescriptorPool::Type t); // Defined in D3D12DescriptorPool.cpp

    template<typename ClearType>
    void clearUavCommon(ComputeContext* pContext, const UnorderedAccessView* pUav, const ClearType& clear, ID3D12GraphicsCommandList* pList)
    {
        auto pResource = pUav->getResource();
        pContext->resourceBarrier(pResource.get(), Resource::State::UnorderedAccess);
        UavHandle uav = pUav->getApiHandle();

        // ClearUnorderedAccessView* requires both CPU and GPU handles to descriptors
        // created on two different heaps: one shader visible and one that is not shader visible.
        // The supplied UAV is created on a CPU descriptor heap. We'll copy it here to a transient GPU descriptor.
        // This is a special-case for UAV clears. Other D3D12 operations copy the CPU descriptor into the command list.

        D3D12DescriptorSet::Layout layout;
        layout.addRange(ShaderResourceType::TextureUav, 0, 1);
        auto pSet = D3D12DescriptorSet::create(gpDevice->getD3D12GpuDescriptorPool(), layout);
        auto dstHandle = pSet->getCpuHandle(0);

        D3D12DescriptorSet::CpuHandle cpuHandle = uav->getCpuHandle(0);
        D3D12DescriptorSet::GpuHandle gpuHandle = pSet->getGpuHandle(0);
        gpDevice->getApiHandle()->CopyDescriptorsSimple(1, dstHandle, cpuHandle, falcorToDxDescType(pSet->getRange(0).type));

        if (typeid(ClearType) == typeid(float4))
        {
            pList->ClearUnorderedAccessViewFloat(gpuHandle, cpuHandle, pResource->getApiHandle(), (float*)value_ptr(clear), 0, nullptr);
        }
        else if (typeid(ClearType) == typeid(uint4))
        {
            pList->ClearUnorderedAccessViewUint(gpuHandle, cpuHandle, pResource->getApiHandle(), (uint32_t*)value_ptr(clear), 0, nullptr);
        }
        else
        {
            FALCOR_UNREACHABLE();
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
        prepareForDispatch(pState, pVars);
        resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
        mpLowLevelData->getCommandList()->ExecuteIndirect(sApiData.pDispatchCommandSig, 1, pArgBuffer->getApiHandle(), argBufferOffset, nullptr, 0);
    }
}
