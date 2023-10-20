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
#include "CopyContext.h"
#include "Handles.h"
#include "Buffer.h"
#include "LowLevelContextData.h"
#include "Core/Macros.h"
#include "Utils/Math/Vector.h"

namespace Falcor
{
class ComputeState;
class ProgramVars;
class ProgramKernels;
class UnorderedAccessView;

class FALCOR_API ComputeContext : public CopyContext
{
public:
    /**
     * Constructor.
     * Throws an exception if creation failed.
     * @param[in] pDevice Graphics device.
     * @param[in] pQueue Command queue.
     */
    ComputeContext(Device* pDevice, gfx::ICommandQueue* pQueue);
    ~ComputeContext();

    /**
     * Dispatch a compute task
     * @param[in] dispatchSize 3D dispatch group size
     */
    void dispatch(ComputeState* pState, ProgramVars* pVars, const uint3& dispatchSize);

    /**
     * Executes a dispatch call. Args to the dispatch call are contained in pArgBuffer
     */
    void dispatchIndirect(ComputeState* pState, ProgramVars* pVars, const Buffer* pArgBuffer, uint64_t argBufferOffset);

    /**
     * Clear an unordered-access view
     * @param[in] pUav The UAV to clear
     * @param[in] value The clear value
     */
    void clearUAV(const UnorderedAccessView* pUav, const float4& value);

    /**
     * Clear an unordered-access view
     * @param[in] pUav The UAV to clear
     * @param[in] value The clear value
     */
    void clearUAV(const UnorderedAccessView* pUav, const uint4& value);

    /**
     * Clear a structured buffer's UAV counter
     * @param[in] pBuffer Structured Buffer containing UAV counter
     * @param[in] value Value to clear counter to
     */
    void clearUAVCounter(const ref<Buffer>& pBuffer, uint32_t value);

    /**
     * Submit the command list
     */
    virtual void submit(bool wait = false) override;

protected:
    ComputeContext(gfx::ICommandQueue* pQueue);

    const ProgramVars* mpLastBoundComputeVars = nullptr;
};

} // namespace Falcor
