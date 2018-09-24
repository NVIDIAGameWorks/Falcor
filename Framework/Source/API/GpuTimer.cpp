/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "Device.h"
#include "GpuTimer.h"

namespace Falcor
{
    GpuTimer::SharedPtr GpuTimer::create()
    {
        return SharedPtr(new GpuTimer());
    }

    GpuTimer::GpuTimer()
    {
#ifdef FALCOR_D3D12
        mpResolveBuffer = Buffer::create(sizeof(uint64_t) * 2, Buffer::BindFlags::None, Buffer::CpuAccess::Read, nullptr);
#endif
        auto& pHeap = gpDevice->getTimestampQueryHeap();
        mStart = pHeap->allocate();
        mEnd = pHeap->allocate();
        assert(mEnd == (mStart + 1));
        mpHeap = pHeap->getApiHandle();
        mpLowLevelData = gpDevice->getRenderContext()->getLowLevelData();
    }

    GpuTimer::~GpuTimer()
    {
        auto& pHeap = gpDevice->getTimestampQueryHeap();
        pHeap->release(mStart);
        pHeap->release(mEnd);
    }

    void GpuTimer::begin()
    {
        if (mStatus == Status::Begin)
        {
            logWarning("GpuTimer::begin() was followed by another call to GpuTimer::begin() without a GpuTimer::end() in-between. Ignoring call.");
            return;
        }

        if (mStatus == Status::End)
        {
            logWarning("GpuTimer::begin() was followed by a call to GpuTimer::end() without querying the data first. The previous results will be discarded.");
        }
        mStatus = Status::Begin;
        apiBegin();
    }

    void GpuTimer::end()
    {
        if (mStatus != Status::Begin)
        {
            logWarning("GpuTimer::end() was called without a preciding GpuTimer::begin(). Ignoring call.");
            return;
        }
        mStatus = Status::End;
        apiEnd();
    }


    double GpuTimer::getElapsedTime()
    {
        if (mStatus == Status::Begin)
        {
            logWarning("GpuTimer::getElapsedTime() was called but the GpuTimer::end() wasn't called. No data to fetch.");
            return 0;
        }
        else if (mStatus == Status::End)
        {
            uint64_t result[2];
            apiResolve(result);

            double start = (double)result[0];
            double end = (double)result[1];
            double range = end - start;
            mElapsedTime = range * gpDevice->getGpuTimestampFrequency();
            mStatus = Status::Idle;
        }
        assert(mStatus == Status::Idle);
        return mElapsedTime;
    }
}
