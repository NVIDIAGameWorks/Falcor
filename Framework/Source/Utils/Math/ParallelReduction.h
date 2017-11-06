/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include "Framework.h"
#include "Graphics/FullScreenPass.h"
#include "Graphics/Program/ProgramVars.h"
#include "API/FBO.h"
#include "API/Sampler.h"

namespace Falcor
{
    class RenderContext;
    class Texture;

    class ParallelReduction
    {
    public:
        using UniquePtr = std::unique_ptr<ParallelReduction>;
        enum class Type
        {
            MinMax
        };
        static UniquePtr create(Type reductionType, uint32_t readbackLatency, uint32_t width, uint32_t height);
        glm::vec4 reduce(RenderContext* pRenderCtx, Texture::SharedPtr pInput);

    private:
        ParallelReduction(Type reductionType, uint32_t readbackLatency, uint32_t width, uint32_t height);
        FullScreenPass::UniquePtr mpFirstIterProg;
        FullScreenPass::UniquePtr mpRestIterProg;
        GraphicsVars::SharedPtr pVars;
        std::vector<Fbo::SharedPtr> mpResultFbo;
        uint32_t mCurFbo = 0;
        Type mReductionType;
        Sampler::SharedPtr mpPointSampler;

        std::vector<Fbo::SharedPtr> mpTmpResultFbo;
        static const uint32_t kTileSize = 16;
    };
}