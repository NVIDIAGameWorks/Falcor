/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "ComputeParallelReduction.h"
#include "ParallelReductionType.slangh"

namespace Falcor
{
    static const char kShaderFile[] = "Utils/Algorithm/ParallelReduction.cs.slang";

    ComputeParallelReduction::SharedPtr ComputeParallelReduction::create()
    {
        return SharedPtr(new ComputeParallelReduction());
    }

    ComputeParallelReduction::ComputeParallelReduction()
    {
        // Create the programs.
        // Set defines to avoid compiler warnings about undefined macros. Proper values will be assigned at runtime.
        Program::DefineList defines = { { "REDUCTION_TYPE", "1" }, { "FORMAT_CHANNELS", "1" }, { "FORMAT_TYPE", "1" } };
        mpInitialProgram = ComputeProgram::createFromFile(kShaderFile, "initialPass", defines, Shader::CompilerFlags::None);
        mpFinalProgram = ComputeProgram::createFromFile(kShaderFile, "finalPass", defines, Shader::CompilerFlags::None);
        mpVars = ComputeVars::create(mpInitialProgram.get());

        // Check assumptions on thread group sizes. The initial pass is a 2D dispatch, the final pass a 1D.
        assert(mpInitialProgram->getReflector()->getThreadGroupSize().z == 1);
        assert(mpFinalProgram->getReflector()->getThreadGroupSize().y == 1 && mpFinalProgram->getReflector()->getThreadGroupSize().z == 1);

        mpState = ComputeState::create();
    }

    void ComputeParallelReduction::allocate(uint32_t elementCount, uint32_t elementSize)
    {
        if (mpBuffers[0] == nullptr || mpBuffers[0]->getElementCount() < elementCount * elementSize)
        {
            // Buffer 0 has one element per tile.
            mpBuffers[0] = Buffer::createTyped<uint4>(elementCount * elementSize);

            // Buffer 1 has one element per N elements in buffer 0.
            const uint32_t numElem1 = div_round_up(elementCount, mpFinalProgram->getReflector()->getThreadGroupSize().x);
            if (mpBuffers[1] == nullptr || mpBuffers[1]->getElementCount() < numElem1 * elementSize)
            {
                mpBuffers[1] = Buffer::createTyped<uint4>(numElem1 * elementSize);
            }
        }
    }

    template<typename T>
    bool ComputeParallelReduction::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pInput, Type operation, T* pResult, Buffer::SharedPtr pResultBuffer, uint64_t resultOffset)
    {
        PROFILE("ComputeParallelReduction::execute");

        // Check texture array/mip/sample count.
        if (pInput->getArraySize() != 1 || pInput->getMipCount() != 1 || pInput->getSampleCount() != 1)
        {
            logError("ComputeParallelReduction::execute() - Input texture is unsupported. Aborting.");
            return false;
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
            logError("ComputeParallelReduction::execute() - Input texture format unsupported. Aborting.");
            return false;
        }

        // Check that reduction type T is compatible with the resource format.
        if (sizeof(typename T::value_type) != 4 ||     // The shader is written for 32-bit types
            (formatType == FORMAT_TYPE_FLOAT && !std::is_floating_point<T::value_type>::value) ||
            (formatType == FORMAT_TYPE_SINT && (!std::is_integral<T::value_type>::value || !std::is_signed<T::value_type>::value)) ||
            (formatType == FORMAT_TYPE_UINT && (!std::is_integral<T::value_type>::value || !std::is_unsigned<T::value_type>::value)))
        {
            logError("ComputeParallelReduction::execute() - Template type T is not compatible with resource format. Aborting.");
            return false;
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
            logError("ComputeParallelReduction::execute() - Unknown reduction type. Aborting.");
            return false;
        }

        // Allocate intermediate buffers if needed.
        const uint2 resolution = uint2(pInput->getWidth(), pInput->getHeight());
        assert(resolution.x > 0 && resolution.y > 0);
        assert(elementSize > 0);

        const uint2 numTiles = div_round_up(resolution, uint2(mpInitialProgram->getReflector()->getThreadGroupSize()));
        allocate(numTiles.x * numTiles.y, elementSize);
        assert(mpBuffers[0]);
        assert(mpBuffers[1]);

        // Configure program.
        const uint32_t channelCount = getFormatChannelCount(pInput->getFormat());
        assert(channelCount >= 1 && channelCount <= 4);

        Program::DefineList defines;
        defines.add("REDUCTION_TYPE", std::to_string(reductionType));
        defines.add("FORMAT_CHANNELS", std::to_string(channelCount));
        defines.add("FORMAT_TYPE", std::to_string(formatType));

        mpInitialProgram->addDefines(defines);
        mpFinalProgram->addDefines(defines);

        // Initial pass: Reduction over tiles of pixels in input texture.
        mpVars["PerFrameCB"]["gResolution"] = resolution;
        mpVars["PerFrameCB"]["gNumTiles"] = numTiles;
        mpVars["gInput"] = pInput;
        mpVars->setBuffer("gResult", mpBuffers[0]);

        mpState->setProgram(mpInitialProgram);
        uint3 numGroups = div_round_up(uint3(resolution.x, resolution.y, 1), mpInitialProgram->getReflector()->getThreadGroupSize());
        pRenderContext->dispatch(mpState.get(), mpVars.get(), numGroups);

        // Final pass(es): Reduction by a factor N for each pass.
        uint elems = numTiles.x * numTiles.y;
        uint inputsBufferIndex = 0;

        while (elems > 1)
        {
            mpVars["PerFrameCB"]["gElems"] = elems;
            mpVars->setBuffer("gInputBuffer", mpBuffers[inputsBufferIndex]);
            mpVars->setBuffer("gResult", mpBuffers[1 - inputsBufferIndex]);

            mpState->setProgram(mpFinalProgram);
            uint32_t numGroups = div_round_up(elems, mpFinalProgram->getReflector()->getThreadGroupSize().x);
            pRenderContext->dispatch(mpState.get(), mpVars.get(), { numGroups, 1, 1 });

            inputsBufferIndex = 1 - inputsBufferIndex;
            elems = numGroups;
        }

        size_t resultSize = elementSize * 16;

        // Copy the result to GPU buffer.
        if (pResultBuffer)
        {
            if (resultOffset + resultSize > pResultBuffer->getSize())
            {
                logError("ComputeParallelReduction::execute() - Results buffer is too small. Aborting.");
                return false;
            }

            pRenderContext->copyBufferRegion(pResultBuffer.get(), resultOffset, mpBuffers[inputsBufferIndex].get(), 0, resultSize);
        }

        // Read back the result to the CPU.
        if (pResult)
        {
            const T* pBuf = static_cast<const T*>(mpBuffers[inputsBufferIndex]->map(Buffer::MapType::Read));
            assert(pBuf);
            std::memcpy(pResult, pBuf, resultSize);
            mpBuffers[inputsBufferIndex]->unmap();
        }

        return true;
    }

    // Explicit template instantiation of the supported types.
    template dlldecl bool ComputeParallelReduction::execute<float4>(RenderContext* pRenderContext, const Texture::SharedPtr& pInput, Type operation, float4* pResult, Buffer::SharedPtr pResultBuffer, uint64_t resultOffset);
    template dlldecl bool ComputeParallelReduction::execute<int4>(RenderContext* pRenderContext, const Texture::SharedPtr& pInput, Type operation, int4* pResult, Buffer::SharedPtr pResultBuffer, uint64_t resultOffset);
    template dlldecl bool ComputeParallelReduction::execute<uint4>(RenderContext* pRenderContext, const Texture::SharedPtr& pInput, Type operation, uint4* pResult, Buffer::SharedPtr pResultBuffer, uint64_t resultOffset);
}
