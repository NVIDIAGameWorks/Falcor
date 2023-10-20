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
#include "PrefixSum.h"
#include "Core/Error.h"
#include "Core/API/RenderContext.h"
#include "Utils/Math/Common.h"
#include "Utils/Timing/Profiler.h"

namespace Falcor
{
namespace
{
const char kShaderFile[] = "Utils/Algorithm/PrefixSum.cs.slang";
const uint32_t kGroupSize = 1024;
} // namespace

PrefixSum::PrefixSum(ref<Device> pDevice) : mpDevice(pDevice)
{
    // Create shaders and state.
    DefineList defines = {{"GROUP_SIZE", std::to_string(kGroupSize)}};
    mpPrefixSumGroupProgram = Program::createCompute(mpDevice, kShaderFile, "groupScan", defines);
    mpPrefixSumGroupVars = ProgramVars::create(mpDevice, mpPrefixSumGroupProgram.get());
    mpPrefixSumFinalizeProgram = Program::createCompute(mpDevice, kShaderFile, "finalizeGroups", defines);
    mpPrefixSumFinalizeVars = ProgramVars::create(mpDevice, mpPrefixSumFinalizeProgram.get());

    mpComputeState = ComputeState::create(mpDevice);

    // Create and bind buffer for per-group sums and total sum.
    mpPrefixGroupSums = mpDevice->createBuffer(
        kGroupSize * sizeof(uint32_t),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr
    );
    mpTotalSum = mpDevice->createBuffer(sizeof(uint32_t), ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr);
    mpPrevTotalSum = mpDevice->createBuffer(sizeof(uint32_t), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, nullptr);

    {
        auto var = mpPrefixSumGroupVars->getRootVar();
        var["gPrefixGroupSums"] = mpPrefixGroupSums;
        var["gTotalSum"] = mpTotalSum;
        var["gPrevTotalSum"] = mpPrevTotalSum;
    }
    {
        auto var = mpPrefixSumFinalizeVars->getRootVar();
        var["gPrefixGroupSums"] = mpPrefixGroupSums;
        var["gTotalSum"] = mpTotalSum;
        var["gPrevTotalSum"] = mpPrevTotalSum;
    }
}

void PrefixSum::execute(
    RenderContext* pRenderContext,
    ref<Buffer> pData,
    uint32_t elementCount,
    uint32_t* pTotalSum,
    ref<Buffer> pTotalSumBuffer,
    uint64_t pTotalSumOffset
)
{
    FALCOR_PROFILE(pRenderContext, "PrefixSum::execute");

    FALCOR_ASSERT(pRenderContext);
    FALCOR_ASSERT(elementCount > 0);
    FALCOR_ASSERT(pData && pData->getSize() >= elementCount * sizeof(uint32_t));

    // Clear total sum to zero.
    pRenderContext->clearUAV(mpTotalSum->getUAV().get(), uint4(0));

    uint32_t maxElementCountPerIteration = kGroupSize * kGroupSize * 2;
    uint32_t totalElementCount = elementCount;
    uint32_t iterationsCount = div_round_up(totalElementCount, maxElementCountPerIteration);

    for (uint32_t iter = 0; iter < iterationsCount; iter++)
    {
        // Compute number of thread groups in the first pass. Each thread operates on two elements.
        uint32_t numPrefixGroups = std::max(1u, div_round_up(std::min(elementCount, maxElementCountPerIteration), kGroupSize * 2));
        FALCOR_ASSERT(numPrefixGroups > 0 && numPrefixGroups <= kGroupSize);

        // Copy previus iterations total sum to read buffer.
        pRenderContext->copyResource(mpPrevTotalSum.get(), mpTotalSum.get());

        // Pass 1: compute per-thread group prefix sums.
        {
            // Clear group sums to zero.
            pRenderContext->clearUAV(mpPrefixGroupSums->getUAV().get(), uint4(0));

            // Set constants and data.
            auto var = mpPrefixSumGroupVars->getRootVar();
            var["CB"]["gNumGroups"] = numPrefixGroups;
            var["CB"]["gTotalNumElems"] = totalElementCount;
            var["CB"]["gIter"] = iter;
            var["gData"] = pData;

            mpComputeState->setProgram(mpPrefixSumGroupProgram);
            pRenderContext->dispatch(mpComputeState.get(), mpPrefixSumGroupVars.get(), {numPrefixGroups, 1, 1});
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
            const uint32_t dispatchSizeX = (numPrefixGroups - 1) * 2;
            FALCOR_ASSERT(dispatchSizeX > 0);

            // Set constants and data.
            auto var = mpPrefixSumFinalizeVars->getRootVar();
            var["CB"]["gNumGroups"] = numPrefixGroups;
            var["CB"]["gTotalNumElems"] = totalElementCount;
            var["CB"]["gIter"] = iter;
            var["gData"] = pData;

            mpComputeState->setProgram(mpPrefixSumFinalizeProgram);
            pRenderContext->dispatch(mpComputeState.get(), mpPrefixSumFinalizeVars.get(), {dispatchSizeX, 1, 1});
        }

        // Subtract the number of elements handled this iteration.
        elementCount -= maxElementCountPerIteration;
    }

    // Copy total sum to separate destination buffer, if specified.
    if (pTotalSumBuffer)
    {
        if (pTotalSumOffset + 4 > pTotalSumBuffer->getSize())
        {
            FALCOR_THROW("PrefixSum::execute() - Results buffer is too small.");
        }

        pRenderContext->copyBufferRegion(pTotalSumBuffer.get(), pTotalSumOffset, mpTotalSum.get(), 0, 4);
    }

    // Read back sum of all elements to the CPU, if requested.
    if (pTotalSum)
    {
        *pTotalSum = mpTotalSum->getElement<uint32_t>(0);
    }
}
} // namespace Falcor
