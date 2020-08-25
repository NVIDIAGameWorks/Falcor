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
#include "PrefixSum.h"
#include <iostream>
#include <iomanip>

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Utils/Algorithm/PrefixSum.cs.slang";
        const uint32_t kGroupSize = 1024;
    }

    PrefixSum::PrefixSum()
    {
        // Create shaders and state.
        Program::DefineList defines = { {"GROUP_SIZE", std::to_string(kGroupSize)} };
        mpPrefixSumGroupProgram = ComputeProgram::createFromFile(kShaderFile, "groupScan", defines);
        mpPrefixSumGroupVars = ComputeVars::create(mpPrefixSumGroupProgram.get());
        mpPrefixSumFinalizeProgram = ComputeProgram::createFromFile(kShaderFile, "finalizeGroups", defines);
        mpPrefixSumFinalizeVars = ComputeVars::create(mpPrefixSumFinalizeProgram.get());

        mpComputeState = ComputeState::create();

        // Create and bind buffer for per-group sums.
        mpPrefixGroupSums = Buffer::create(kGroupSize * sizeof(uint32_t), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr);

        mpPrefixSumGroupVars["gPrefixGroupSums"] = mpPrefixGroupSums;
        mpPrefixSumFinalizeVars["gPrefixGroupSums"] = mpPrefixGroupSums;
    }

    PrefixSum::SharedPtr PrefixSum::create()
    {
        return SharedPtr(new PrefixSum());
    }

    bool PrefixSum::execute(RenderContext* pRenderContext, Buffer::SharedPtr pData, uint32_t elementCount, uint32_t* pTotalSum, Buffer::SharedPtr pTotalSumBuffer, uint64_t pTotalSumOffset)
    {
        PROFILE("PrefixSum::execute");

        assert(pRenderContext);
        assert(elementCount > 0);
        assert(pData && pData->getSize() >= elementCount * sizeof(uint32_t));

        // The current implementation is limited to N groups of 2N elements, where N = thread group size.
        // This is because we reuse the 1st pass to also compute the prefix sum across the thread groups.
        // It is easy to generalize this by adding an extra pass to compute the per-group prefix sum if needed
        // (with that large data sets, we probably want that for efficiency reasons anyway).
        const uint32_t maxElementCount = kGroupSize * kGroupSize * 2;
        if (elementCount > maxElementCount)
        {
            logError("PrefixSum::execute() - Maximum supported element count is " + std::to_string(maxElementCount) + ". Aborting.");
            return false;
        }

        // Compute number of thread groups in the first pass. Each thread operates on two elements.
        const uint32_t numPrefixGroups = std::max(1u, div_round_up(elementCount, kGroupSize * 2));
        assert(numPrefixGroups > 0 && numPrefixGroups < kGroupSize);

        // Pass 1: compute per-thread group prefix sums.
        {
            // Clear group sums to zero.
            pRenderContext->clearUAV(mpPrefixGroupSums->getUAV().get(), uint4(0));

            // Set constants and data.
            mpPrefixSumGroupVars["CB"]["gNumGroups"] = numPrefixGroups;
            mpPrefixSumGroupVars["CB"]["gNumElems"] = elementCount;
            mpPrefixSumGroupVars["gData"] = pData;

            mpComputeState->setProgram(mpPrefixSumGroupProgram);
            pRenderContext->dispatch(mpComputeState.get(), mpPrefixSumGroupVars.get(), { numPrefixGroups, 1, 1 });
        }

        // Add UAV barriers for our buffers to make sure writes from the previous pass finish before the next pass.
        // This is necessary since the buffers are bound as UAVs in both passes and there are no resource transitions.
        pRenderContext->uavBarrier(pData.get());
        pRenderContext->uavBarrier(mpPrefixGroupSums.get());

        // Pass 2: finalize prefix sum by adding the sums to the left to each group.
        // This is only necessary if we have more than one group.
        if (numPrefixGroups > 1)
        {
            // Compute number of thread groups. Each thread operates on one element.
            // Note that we're skipping the first group of 2N elements, as no add is needed (their group sum is zero).
            const uint dispatchSizeX = (numPrefixGroups - 1) * 2;
            assert(dispatchSizeX > 0);

            // Set constants and data.
            mpPrefixSumFinalizeVars["CB"]["gNumGroups"] = numPrefixGroups;
            mpPrefixSumFinalizeVars["CB"]["gNumElems"] = elementCount;
            mpPrefixSumFinalizeVars["gData"] = pData;

            mpComputeState->setProgram(mpPrefixSumFinalizeProgram);
            pRenderContext->dispatch(mpComputeState.get(), mpPrefixSumFinalizeVars.get(), { dispatchSizeX, 1, 1 });
        }

        // Copy total sum to separate destination buffer, if specified.
        if (pTotalSumBuffer)
        {
            if (pTotalSumOffset + 4 > pTotalSumBuffer->getSize())
            {
                logError("PrefixSum::execute() - Results buffer is too small. Aborting.");
                return false;
            }

            assert(numPrefixGroups > 0);
            uint64_t srcOffset = (numPrefixGroups - 1) * 4;
            pRenderContext->copyBufferRegion(pTotalSumBuffer.get(), pTotalSumOffset, mpPrefixGroupSums.get(), srcOffset, 4);
        }

        // Read back sum of all elements to the CPU, if requested.
        if (pTotalSum)
        {
            uint32_t* pGroupSums = (uint32_t*)mpPrefixGroupSums->map(Buffer::MapType::Read);
            assert(pGroupSums);
            assert(numPrefixGroups > 0);
            *pTotalSum = pGroupSums[numPrefixGroups - 1];
            mpPrefixGroupSums->unmap();
        }

        return true;
    }
}
