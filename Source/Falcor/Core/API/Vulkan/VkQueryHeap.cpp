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
#include "API/Device.h"
#include "API/QueryHeap.h"

namespace Falcor
{
    static VkQueryType getVkPoolType(QueryHeap::Type t)
    {
        switch (t)
        {
        case QueryHeap::Type::Timestamp:
            return VK_QUERY_TYPE_TIMESTAMP;
        case QueryHeap::Type::Occlusion:
            return VK_QUERY_TYPE_OCCLUSION;
        case QueryHeap::Type::PipelineStats:
            return VK_QUERY_TYPE_PIPELINE_STATISTICS;
        default:
            should_not_get_here();
            return VK_QUERY_TYPE_MAX_ENUM;
        }
    }

    QueryHeap::QueryHeap(Type type, uint32_t count) : mType(type), mCount(count)
    {
        VkQueryPoolCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryCount = count;
        info.queryType = getVkPoolType(type);
        info.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_FLAG_BITS_MAX_ENUM;
        VkQueryPool pool;
        vk_call(vkCreateQueryPool(gpDevice->getApiHandle(), &info, nullptr, &pool));
        mApiHandle = ApiHandle::create(pool);
    }
}