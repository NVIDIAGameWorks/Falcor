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
#include "Handles.h"
#include "Raytracing.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/Program/ProgramVersion.h"
#include <string>
#include <vector>

namespace Falcor
{

struct RtStateObjectDesc
{
    ref<const ProgramKernels> pProgramKernels;
    uint32_t maxTraceRecursionDepth = 0;
    RtPipelineFlags pipelineFlags = RtPipelineFlags::None;

    bool operator==(const RtStateObjectDesc& other) const
    {
        bool result = true;
        result = result && (pProgramKernels == other.pProgramKernels);
        result = result && (maxTraceRecursionDepth == other.maxTraceRecursionDepth);
        result = result && (pipelineFlags == other.pipelineFlags);
        return result;
    }
};

class FALCOR_API RtStateObject : public Object
{
    FALCOR_OBJECT(RtStateObject)
public:
    RtStateObject(ref<Device> pDevice, const RtStateObjectDesc& desc);
    ~RtStateObject();

    gfx::IPipelineState* getGfxPipelineState() const { return mGfxPipelineState; }

    const ref<const ProgramKernels>& getKernels() const { return mDesc.pProgramKernels; };
    uint32_t getMaxTraceRecursionDepth() const { return mDesc.maxTraceRecursionDepth; }
    void const* getShaderIdentifier(uint32_t index) const { return mEntryPointGroupExportNames[index].c_str(); }
    const RtStateObjectDesc& getDesc() const { return mDesc; }

private:
    ref<Device> mpDevice;
    RtStateObjectDesc mDesc;
    Slang::ComPtr<gfx::IPipelineState> mGfxPipelineState;
    std::vector<std::string> mEntryPointGroupExportNames;
};
} // namespace Falcor
