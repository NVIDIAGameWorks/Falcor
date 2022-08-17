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
#include "Core/API/CopyContext.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"
#include <memory>
#include <vector>

namespace Falcor
{
    class RenderContext;

    class FALCOR_API ParallelReduction
    {
    public:
        using UniquePtr = std::unique_ptr<ParallelReduction>;

        enum class Type
        {
            MinMax
        };

        /** Create a new parallel reduction object.
            \param[in] reductionType The reduction operator.
            \param[in] readbackLatency The result is returned after this many calls to reduce().
            \param[in] width Width in pixels of the texture that will be used.
            \param[in] height Height in pixels of the texture that will be used.
            \param[in] sampleCount Multi-sample count for the texture that will be used.
            \return New object, or throws an exception if creation failed.
        */
        static UniquePtr create(Type reductionType, uint32_t readbackLatency, uint32_t width, uint32_t height, uint32_t sampleCount = 1);

        float4 reduce(RenderContext* pRenderCtx, Texture::SharedPtr pInput);

    private:
        ParallelReduction(Type reductionType, uint32_t readbackLatency, uint32_t width, uint32_t height, uint32_t sampleCount);
        FullScreenPass::SharedPtr mpFirstIterProg;
        FullScreenPass::SharedPtr mpRestIterProg;

        struct ResultData
        {
            CopyContext::ReadTextureTask::SharedPtr pReadTask;
            Fbo::SharedPtr pFbo;
        };
        std::vector<ResultData> mResultData;

        uint32_t mCurFbo = 0;
        Type mReductionType;
        Sampler::SharedPtr mpPointSampler;

        std::vector<Fbo::SharedPtr> mpTmpResultFbo;
        static const uint32_t kTileSize = 16;
    };
}
