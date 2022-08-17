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
#pragma once
#include "Handles.h"
#include "Shared/D3D12Handles.h"
#include "Core/Macros.h"
#include <memory>
#include <optional>

namespace Falcor
{
    struct FenceApiData;

    /** This class can be used to synchronize GPU and CPU execution
        It's value monotonically increasing - every time a signal is sent, it will change the value first
    */
    class FALCOR_API GpuFence
    {
    public:
        using SharedPtr = std::shared_ptr<GpuFence>;
        using SharedConstPtr = std::shared_ptr<const GpuFence>;
        using ApiHandle = FenceHandle;
        ~GpuFence();

        /** Create a new GPU fence.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(bool shared = false);

        /** Get the internal API handle
        */
        const ApiHandle& getApiHandle() const { return mApiHandle; }

        /** Get the internal D3D12 handle. Available only when D3D12 is the underlying API.
        */
        const D3D12FenceHandle& getD3D12Handle() const;

        /** Get the last value the GPU has signaled
        */
        uint64_t getGpuValue() const;

        /** Sets the current GPU value to a specific value (signals the fence from the CPU side).
        */
        void setGpuValue(uint64_t val);

        /** Get the current CPU value
        */
        uint64_t getCpuValue() const { return mCpuValue; }

        /** Tell the GPU to wait until the fence reaches the last GPU-value signaled (which is (mCpuValue - 1))
        */
        void syncGpu(CommandQueueHandle pQueue);

        /** Tell the CPU to wait until the fence reaches the current value
        */
        void syncCpu(std::optional<uint64_t> val = {});

        /** Insert a signal command into the command queue. This will increase the internal value
        */
        uint64_t gpuSignal(CommandQueueHandle pQueue);

        /** Must be called to obtain the current value for use in an external semaphore.
        */
        uint64_t externalSignal() { return mCpuValue++; }

        /** Creates a shared fence API handle.
        */
        SharedResourceApiHandle getSharedApiHandle() const;
    private:
        GpuFence();
        uint64_t mCpuValue;

        ApiHandle mApiHandle;
        std::unique_ptr<FenceApiData> mpApiData;
        mutable SharedResourceApiHandle mSharedApiHandle = 0;
    };
}
