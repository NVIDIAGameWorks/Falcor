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
#pragma once
#include "Core/API/Buffer.h"
#include "Core/State/ComputeState.h"
#include "Core/Program/ComputeProgram.h"
#include "Core/Program/ProgramVars.h"

namespace Falcor
{
    /** Computes the parallel prefix sum on the GPU.

        The prefix sum is computed in place using exclusive scan.
        Each new element is y[i] = x[0] + ... + x[i-1], for i=1..N and y[0] = 0.
    */
    class dlldecl PrefixSum : public std::enable_shared_from_this<PrefixSum>
    {
    public:
        using SharedPtr = std::shared_ptr<PrefixSum>;
        using SharedConstPtr = std::shared_ptr<const PrefixSum>;
        virtual ~PrefixSum() = default;

        /** Create a new prefix sum object.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create();

        /** Computes the parallel prefix sum over an array of uint32_t elements.
            \param[in] pRenderContext The render context.
            \param[in] pData The buffer to compute prefix sum over.
            \param[in] elementCount Number of elements to compute prefix sum over.
            \param[out] pTotalSum (Optional) The sum of all elements is stored to this variable if it is non-null. Note that this requires a GPU sync!
            \param[in] pTotalSumBuffer (Optional) Buffer on the GPU to which the total sum is copied (uint32_t).
            \param[in] pTotalSumOffset (Optional) Byte offset into pTotalSumBuffer to where the sum should be written.
        */
        bool execute(RenderContext* pRenderContext, Buffer::SharedPtr pData, uint32_t elementCount, uint32_t* pTotalSum = nullptr, Buffer::SharedPtr pTotalSumBuffer = nullptr, uint64_t pTotalSumOffset = 0);

    protected:
        PrefixSum();

        ComputeState::SharedPtr     mpComputeState;

        ComputeProgram::SharedPtr   mpPrefixSumGroupProgram;
        ComputeVars::SharedPtr      mpPrefixSumGroupVars;

        ComputeProgram::SharedPtr   mpPrefixSumFinalizeProgram;
        ComputeVars::SharedPtr      mpPrefixSumFinalizeVars;

        Buffer::SharedPtr           mpPrefixGroupSums;              ///< Temporary buffer for prefix sum computation.
    };
}
