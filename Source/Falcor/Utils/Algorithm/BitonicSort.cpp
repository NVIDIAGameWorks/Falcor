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
#include "BitonicSort.h"
#include "Core/Macros.h"
#include "Core/API/RenderContext.h"
#include "Utils/Math/Common.h"
#include "Utils/Timing/Profiler.h"

namespace Falcor
{
    static const char kShaderFilename[] = "Utils/Algorithm/BitonicSort.cs.slang";

    BitonicSort::BitonicSort()
    {
#if !FALCOR_NVAPI_AVAILABLE
        throw RuntimeError("BitonicSort requires NVAPI. See installation instructions in README.");
#endif
        mSort.pState = ComputeState::create();

        // Create shaders
        Program::DefineList defines;
        defines.add("CHUNK_SIZE", "256");   // Dummy values just so we can get reflection data. We'll set the actual values in execute().
        defines.add("GROUP_SIZE", "256");
        mSort.pProgram = ComputeProgram::createFromFile(kShaderFilename, "main", defines);
        mSort.pState->setProgram(mSort.pProgram);
        mSort.pVars = ComputeVars::create(mSort.pProgram.get());
    }

    BitonicSort::SharedPtr BitonicSort::create()
    {
        return SharedPtr(new BitonicSort());
    }

    bool BitonicSort::execute(RenderContext* pRenderContext, Buffer::SharedPtr pData, uint32_t totalSize, uint32_t chunkSize, uint32_t groupSize)
    {
        FALCOR_PROFILE("BitonicSort::execute");

        // Validate inputs.
        FALCOR_ASSERT(pRenderContext);
        FALCOR_ASSERT(pData);
        FALCOR_ASSERT(chunkSize >= 1 && chunkSize <= groupSize && isPowerOf2(chunkSize));
        FALCOR_ASSERT(groupSize >= 1 && groupSize <= 1024 && isPowerOf2(groupSize));

        // Early out if there is nothing to be done.
        if (totalSize == 0 || chunkSize <= 1) return true;

        // Configure the shader for the specified chunk size.
        // This will trigger a re-compile if a new chunk size is encountered.
        mSort.pProgram->addDefine("CHUNK_SIZE", std::to_string(chunkSize));
        mSort.pProgram->addDefine("GROUP_SIZE", std::to_string(groupSize));

        // Determine dispatch dimensions.
        const uint32_t numGroups = div_round_up(totalSize, groupSize);
        const uint32_t groupsX = std::max((uint32_t)sqrt(numGroups), 1u);
        const uint32_t groupsY = div_round_up(numGroups, groupsX);
        FALCOR_ASSERT(groupsX * groupsY * groupSize >= totalSize);

        // Constants. The buffer size as a runtime constant as it may be variable and we don't want to recompile each time it changes.
        mSort.pVars["CB"]["gTotalSize"] = totalSize;
        mSort.pVars["CB"]["gDispatchX"] = groupsX;

        // Bind the data.
        bool success = mSort.pVars->setBuffer("gData", pData);
        FALCOR_ASSERT(success);

        // Execute.
        pRenderContext->dispatch(mSort.pState.get(), mSort.pVars.get(), {groupsX, groupsY, 1});

        return true;
    }
}
