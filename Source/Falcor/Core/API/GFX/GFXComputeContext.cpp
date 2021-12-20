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
#include "Core/API/ComputeContext.h"
#include "GFXLowLevelContextApiData.h"

namespace Falcor
{
    ComputeContext::ComputeContext(LowLevelContextData::CommandQueueType type, CommandQueueHandle queue)
        : CopyContext(type, queue)
    {
        assert(queue);
    }

    ComputeContext::~ComputeContext()
    {
    }

    void ComputeContext::dispatch(ComputeState* pState, ComputeVars* pVars, const uint3& dispatchSize)
    {
        auto computeEncoder = mpLowLevelData->getApiData()->getComputeCommandEncoder();
        auto rootObject = computeEncoder->bindPipeline(pState->getCSO(pVars)->getApiHandle());
        rootObject->copyFrom(pVars->getShaderObject());
        computeEncoder->dispatchCompute((int)dispatchSize.x, (int)dispatchSize.y, (int)dispatchSize.z);
    }

    void ComputeContext::clearUAV(const UnorderedAccessView* pUav, const float4& value)
    {
    }

    void ComputeContext::clearUAV(const UnorderedAccessView* pUav, const uint4& value)
    {
    }

    void ComputeContext::clearUAVCounter(const Buffer::SharedPtr& pBuffer, uint32_t value)
    {
    }

    void ComputeContext::dispatchIndirect(ComputeState* pState, ComputeVars* pVars, const Buffer* pArgBuffer, uint64_t argBufferOffset)
    {
    }
}
