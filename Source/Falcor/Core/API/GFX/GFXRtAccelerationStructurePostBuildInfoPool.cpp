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
#include "GFXRtAccelerationStructure.h"
#include "Core/API/RtAccelerationStructurePostBuildInfoPool.h"
#include "Core/API/Device.h"
#include "Core/API/CopyContext.h"
#include "Core/API/GFX/GFXAPI.h"

namespace Falcor
{
    RtAccelerationStructurePostBuildInfoPool::RtAccelerationStructurePostBuildInfoPool(const Desc& desc)
        : mDesc(desc)
    {
        gfx::IQueryPool::Desc queryPoolDesc = {};
        queryPoolDesc.count = desc.elementCount;
        queryPoolDesc.type = getGFXAccelerationStructurePostBuildQueryType(desc.queryType);
        FALCOR_GFX_CALL(gpDevice->getApiHandle()->createQueryPool(queryPoolDesc, mpGFXQueryPool.writeRef()));
    }

    RtAccelerationStructurePostBuildInfoPool::~RtAccelerationStructurePostBuildInfoPool()
    {
    }

    uint64_t RtAccelerationStructurePostBuildInfoPool::getElement(CopyContext* pContext, uint32_t index)
    {
        if (mNeedFlush)
        {
            pContext->flush(true);
            mNeedFlush = false;
        }
        uint64_t result = 0;
        FALCOR_GFX_CALL(mpGFXQueryPool->getResult(index, 1, &result));
        return result;
    }

    void RtAccelerationStructurePostBuildInfoPool::reset(CopyContext* pContext)
    {
        FALCOR_GFX_CALL(mpGFXQueryPool->reset());
        mNeedFlush = true;
    }
}
