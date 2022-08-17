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
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/State/ComputeState.h"
#include "Core/Program/ComputeProgram.h"
#include "Core/Program/ProgramVars.h"
#include <memory>

namespace Falcor
{
    /** Class that performs parallel reduction over all pixels in a texture.

        The reduction is done on recursively on blocks of n = 1024 elements.
        The total number of iterations is ceil(log2(N)/10), where N is the
        total number of elements (pixels).

        The numerical error for the summation operation lies between pairwise
        summation (blocks of size n = 2) and naive running summation.
    */
    class FALCOR_API ComputeParallelReduction
    {
    public:
        using SharedPtr = std::shared_ptr<ComputeParallelReduction>;
        using SharedConstPtr = std::shared_ptr<const ComputeParallelReduction>;
        virtual ~ComputeParallelReduction() = default;

        enum class Type
        {
            Sum,
            MinMax,
        };

        /** Create parallel reduction helper.
            \return Created object, or an exception is thrown on failure.
        */
        static SharedPtr create();

        /** Perform parallel reduction.
            The computations are performed in type T, which must be compatible with the texture format:
            - float4 for floating-point texture formats (float, snorm, unorm).
            - uint4 for unsigned integer texture formats.
            - int4 for signed integer texture formats.

            For the Sum operation, unused components are set to zero if texture format has < 4 components.

            For performance reasons, it is advisable to store the result in a buffer on the GPU,
            and then issue an asynchronous readback in user code to avoid a full GPU flush.

            The size of the result buffer depends on the executed operation:
            - Sum needs 16B
            - MinMax needs 32B

            \param[in] pRenderContext The render context.
            \param[in] pInput Input texture.
            \param[in] operation Reduction operation.
            \param[out] pResult (Optional) The result of the reduction operation is stored here if non-nullptr. Note that this requires a GPU flush!
            \param[out] pResultBuffer (Optional) Buffer on the GPU to which the result is copied (16B or 32B).
            \param[out] resultOffset (Optional) Byte offset into pResultBuffer to where the result should be stored.
        */
        template<typename T>
        void execute(RenderContext* pRenderContext, const Texture::SharedPtr& pInput, Type operation, T* pResult = nullptr, Buffer::SharedPtr pResultBuffer = nullptr, uint64_t resultOffset = 0);

    private:
        ComputeParallelReduction();
        void allocate(uint32_t elementCount, uint32_t elementSize);

        ComputeState::SharedPtr             mpState;
        ComputeProgram::SharedPtr           mpInitialProgram;
        ComputeProgram::SharedPtr           mpFinalProgram;
        ComputeVars::SharedPtr              mpVars;

        Buffer::SharedPtr                   mpBuffers[2];       ///< Intermediate buffers for reduction iterations.
    };
}
