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
#include "ParallelReduction.h"
#include "ParallelReductionType.slangh"
#include "Core/Error.h"
#include "Core/API/RenderContext.h"
#include "Utils/Math/Common.h"
#include "Utils/Timing/Profiler.h"

namespace Falcor
{
static const char kShaderFile[] = "Utils/Algorithm/ParallelReduction.cs.slang";

ParallelReduction::ParallelReduction(ref<Device> pDevice) : mpDevice(pDevice)
{
    // Create the programs.
    // Set defines to avoid compiler warnings about undefined macros. Proper values will be assigned at runtime.
    DefineList defines = {{"REDUCTION_TYPE", "1"}, {"FORMAT_CHANNELS", "1"}, {"FORMAT_TYPE", "1"}};
    mpInitialProgram = Program::createCompute(mpDevice, kShaderFile, "initialPass", defines);
    mpFinalProgram = Program::createCompute(mpDevice, kShaderFile, "finalPass", defines);
    mpVars = ProgramVars::create(mpDevice, mpInitialProgram.get());

    // Check assumptions on thread group sizes. The initial pass is a 2D dispatch, the final pass a 1D.
    FALCOR_ASSERT(mpInitialProgram->getReflector()->getThreadGroupSize().z == 1);
    FALCOR_ASSERT(
        mpFinalProgram->getReflector()->getThreadGroupSize().y == 1 && mpFinalProgram->getReflector()->getThreadGroupSize().z == 1
    );

    mpState = ComputeState::create(mpDevice);
}

void ParallelReduction::allocate(uint32_t elementCount, uint32_t elementSize)
{
    if (mpBuffers[0] == nullptr || mpBuffers[0]->getElementCount() < elementCount * elementSize)
    {
        // Buffer 0 has one element per tile.
        mpBuffers[0] = mpDevice->createTypedBuffer<uint4>(elementCount * elementSize);
        mpBuffers[0]->setName("ParallelReduction::mpBuffers[0]");

        // Buffer 1 has one element per N elements in buffer 0.
        const uint32_t numElem1 = div_round_up(elementCount, mpFinalProgram->getReflector()->getThreadGroupSize().x);
        if (mpBuffers[1] == nullptr || mpBuffers[1]->getElementCount() < numElem1 * elementSize)
        {
            mpBuffers[1] = mpDevice->createTypedBuffer<uint4>(numElem1 * elementSize);
            mpBuffers[1]->setName("ParallelReduction::mpBuffers[1]");
        }
    }
}

template<typename T>
void ParallelReduction::execute(
    RenderContext* pRenderContext,
    const ref<Texture>& pInput,
    Type operation,
    T* pResult,
    ref<Buffer> pResultBuffer,
    uint64_t resultOffset
)
{
    FALCOR_PROFILE(pRenderContext, "ParallelReduction::execute");

    // Check texture array/mip/sample count.
    if (pInput->getArraySize() != 1 || pInput->getMipCount() != 1 || pInput->getSampleCount() != 1)
    {
        FALCOR_THROW("ParallelReduction::execute() - Input texture is unsupported.");
    }

    // Check texture format.
    uint32_t formatType = FORMAT_TYPE_UNKNOWN;
    switch (getFormatType(pInput->getFormat()))
    {
    case FormatType::Float:
    case FormatType::Unorm:
    case FormatType::Snorm:
        formatType = FORMAT_TYPE_FLOAT;
        break;
    case FormatType::Sint:
        formatType = FORMAT_TYPE_SINT;
        break;
    case FormatType::Uint:
        formatType = FORMAT_TYPE_UINT;
        break;
    default:
        FALCOR_THROW("ParallelReduction::execute() - Input texture format unsupported.");
    }

    // Check that reduction type T is compatible with the resource format.
    if (sizeof(typename T::value_type) != 4 || // The shader is written for 32-bit types
        (formatType == FORMAT_TYPE_FLOAT && !std::is_floating_point<typename T::value_type>::value) ||
        (formatType == FORMAT_TYPE_SINT &&
         (!std::is_integral<typename T::value_type>::value || !std::is_signed<typename T::value_type>::value)) ||
        (formatType == FORMAT_TYPE_UINT &&
         (!std::is_integral<typename T::value_type>::value || !std::is_unsigned<typename T::value_type>::value)))
    {
        FALCOR_THROW("ParallelReduction::execute() - Template type T is not compatible with resource format.");
    }

    uint32_t reductionType = REDUCTION_TYPE_UNKNOWN;
    uint32_t elementSize = 0;
    switch (operation)
    {
    case Type::Sum:
        reductionType = REDUCTION_TYPE_SUM;
        elementSize = 1;
        break;
    case Type::MinMax:
        reductionType = REDUCTION_TYPE_MINMAX;
        elementSize = 2;
        break;
    default:
        FALCOR_THROW("ParallelReduction::execute() - Unknown reduction type.");
    }

    // Allocate intermediate buffers if needed.
    const uint2 resolution = uint2(pInput->getWidth(), pInput->getHeight());
    FALCOR_ASSERT(resolution.x > 0 && resolution.y > 0);
    FALCOR_ASSERT(elementSize > 0);

    const uint2 numTiles = div_round_up(resolution, mpInitialProgram->getReflector()->getThreadGroupSize().xy());
    allocate(numTiles.x * numTiles.y, elementSize);
    FALCOR_ASSERT(mpBuffers[0]);
    FALCOR_ASSERT(mpBuffers[1]);

    // Configure program.
    const uint32_t channelCount = getFormatChannelCount(pInput->getFormat());
    FALCOR_ASSERT(channelCount >= 1 && channelCount <= 4);

    DefineList defines;
    defines.add("REDUCTION_TYPE", std::to_string(reductionType));
    defines.add("FORMAT_CHANNELS", std::to_string(channelCount));
    defines.add("FORMAT_TYPE", std::to_string(formatType));

    mpInitialProgram->addDefines(defines);
    mpFinalProgram->addDefines(defines);

    // Initial pass: Reduction over tiles of pixels in input texture.
    {
        auto var = mpVars->getRootVar();
        var["PerFrameCB"]["gResolution"] = resolution;
        var["PerFrameCB"]["gNumTiles"] = numTiles;
        var["gInput"] = pInput;
        var["gInputBuffer"].setBuffer(nullptr); // Unbind previously bound buffer from last call to execute()
        var["gResult"] = mpBuffers[0];

        mpState->setProgram(mpInitialProgram);
        uint3 numGroups = div_round_up(uint3(resolution.x, resolution.y, 1), mpInitialProgram->getReflector()->getThreadGroupSize());
        pRenderContext->dispatch(mpState.get(), mpVars.get(), numGroups);
    }

    uint32_t inputsBufferIndex = 0;

    // Final pass(es): Reduction by a factor N for each pass.
    {
        uint32_t elems = numTiles.x * numTiles.y;
        while (elems > 1)
        {
            auto var = mpVars->getRootVar();
            var["PerFrameCB"]["gElems"] = elems;
            var["gInputBuffer"] = mpBuffers[inputsBufferIndex];
            var["gResult"] = mpBuffers[1 - inputsBufferIndex];

            mpState->setProgram(mpFinalProgram);
            uint32_t numGroups = div_round_up(elems, mpFinalProgram->getReflector()->getThreadGroupSize().x);
            pRenderContext->dispatch(mpState.get(), mpVars.get(), {numGroups, 1, 1});

            inputsBufferIndex = 1 - inputsBufferIndex;
            elems = numGroups;
        }
    }

    size_t resultSize = elementSize * 16;

    // Copy the result to GPU buffer.
    if (pResultBuffer)
    {
        if (resultOffset + resultSize > pResultBuffer->getSize())
        {
            FALCOR_THROW("ParallelReduction::execute() - Results buffer is too small.");
        }

        pRenderContext->copyBufferRegion(pResultBuffer.get(), resultOffset, mpBuffers[inputsBufferIndex].get(), 0, resultSize);
    }

    // Read back the result to the CPU.
    if (pResult)
    {
        mpBuffers[inputsBufferIndex]->getBlob(pResult, 0, resultSize);
    }
}

// Explicit template instantiation of the supported types.
// clang-format off
template FALCOR_API void ParallelReduction::execute<float4>(RenderContext* pRenderContext, const ref<Texture>& pInput, Type operation, float4* pResult, ref<Buffer> pResultBuffer, uint64_t resultOffset);
template FALCOR_API void ParallelReduction::execute<int4>(RenderContext* pRenderContext, const ref<Texture>& pInput, Type operation, int4* pResult, ref<Buffer> pResultBuffer, uint64_t resultOffset);
template FALCOR_API void ParallelReduction::execute<uint4>(RenderContext* pRenderContext, const ref<Texture>& pInput, Type operation, uint4* pResult, ref<Buffer> pResultBuffer, uint64_t resultOffset);
// clang-format on
} // namespace Falcor
