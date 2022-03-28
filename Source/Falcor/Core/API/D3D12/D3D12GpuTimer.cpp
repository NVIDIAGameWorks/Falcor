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
#include "Core/API/GpuTimer.h"

namespace Falcor
{
    void GpuTimer::apiBegin()
    {
        mpLowLevelData->getCommandList()->EndQuery(spHeap.lock()->getApiHandle(), D3D12_QUERY_TYPE_TIMESTAMP, mStart);
    }

    void GpuTimer::apiEnd()
    {
        mpLowLevelData->getCommandList()->EndQuery(spHeap.lock()->getApiHandle(), D3D12_QUERY_TYPE_TIMESTAMP, mEnd);
    }

    void GpuTimer::apiResolve()
    {
        // TODO: The code here is inefficient as it resolves each timer individually.
        // This should be batched across all active timers and results copied into a single staging buffer once per frame instead.

        // Resolve timestamps into buffer.
        mpLowLevelData->getCommandList()->ResolveQueryData(spHeap.lock()->getApiHandle(), D3D12_QUERY_TYPE_TIMESTAMP, mStart, 2, mpResolveBuffer->getApiHandle(), 0);

        // Copy resolved timestamps to staging buffer for readback. This inserts the necessary barriers.
        gpDevice->getRenderContext()->copyResource(mpResolveStagingBuffer.get(), mpResolveBuffer.get());
    }

    void GpuTimer::apiReadback(uint64_t result[2])
    {
        uint64_t* pRes = (uint64_t*)mpResolveStagingBuffer->map(Buffer::MapType::Read);
        result[0] = pRes[0];
        result[1] = pRes[1];
        mpResolveStagingBuffer->unmap();
    }
}
