/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "fwd.h"
#include "QueryHeap.h"
#include "Buffer.h"
#include "Core/Macros.h"
#include "Core/Object.h"

namespace Falcor
{
/**
 * Abstracts GPU timer queries.
 * This class provides mechanism to get elapsed time in milliseconds between a pair of begin()/end() calls.
 */
class FALCOR_API GpuTimer : public Object
{
    FALCOR_OBJECT(GpuTimer)
public:
    /**
     * Create a new timer object.
     * @return A new object, or throws an exception if creation failed.
     */
    static ref<GpuTimer> create(ref<Device> pDevice);

    /**
     * Destroy a new object
     */
    ~GpuTimer();

    /**
     * Begin the capture window.
     * If begin() is called in the middle of a begin()/end() pair, it will be ignored and a warning will be logged.
     */
    void begin();

    /**
     * End the capture window.
     * If end() is called before a begin() was called, it will be ignored and a warning will be logged.
     */
    void end();

    /**
     * Resolve time stamps.
     * This must be called after a pair of begin()/end() calls.
     * A new measurement can be started after calling resolve() even before getElapsedTime() is called.
     */
    void resolve();

    /**
     * Get the elapsed time in milliseconds for the last resolved pair of begin()/end() calls.
     * If this function called not after a begin()/end() pair, zero will be returned and a warning will be logged.
     * The resolve() function must be called prior to calling this function.
     * NOTE! The caller is responsible for inserting GPU synchronization between these two calls.
     */
    double getElapsedTime();

    void breakStrongReferenceToDevice();

private:
    GpuTimer(ref<Device> pDevice);

    enum class Status
    {
        Begin,
        End,
        Idle
    };

    BreakableReference<Device> mpDevice;
    Status mStatus = Status::Idle;
    uint32_t mStart = 0;
    uint32_t mEnd = 0;
    double mElapsedTime = 0.0;
    bool mDataPending = false; ///< Set to true when resolved timings are available for readback.

    ref<Buffer> mpResolveBuffer;        ///< GPU memory used as destination for resolving timestamp queries.
    ref<Buffer> mpResolveStagingBuffer; ///< CPU mappable memory for readback of resolved timings.
};
} // namespace Falcor
