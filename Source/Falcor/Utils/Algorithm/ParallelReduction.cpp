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
#include "ParallelReduction.h"
#include "Core/Assert.h"
#include "Core/API/RenderContext.h"

namespace Falcor
{
    static const char* psFilename = "Utils/Algorithm/ParallelReduction.ps.slang";

    ParallelReduction::ParallelReduction(ParallelReduction::Type reductionType, uint32_t readbackLatency, uint32_t width, uint32_t height, uint32_t sampleCount)
        : mReductionType(reductionType)
    {
        FALCOR_ASSERT(width > 0 && height > 0 && sampleCount > 0);
        ResourceFormat texFormat;
        Program::DefineList defines;
        defines.add("_SAMPLE_COUNT", std::to_string(sampleCount));
        defines.add("_TILE_SIZE", std::to_string(kTileSize));
        switch(reductionType)
        {
        case Type::MinMax:
           texFormat = ResourceFormat::RG32Float;
           defines.add("_MIN_MAX_REDUCTION");
           break;
        default:
            throw ArgumentError("Unknown parallel reduction operator");
        }

        Sampler::Desc samplerDesc;
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp).setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setLodParams(0, 0, 0);
        mpPointSampler = Sampler::create(samplerDesc);

        mResultData.resize(readbackLatency + 1);
        for(auto& res : mResultData)
        {
            Fbo::Desc fboDesc;
            fboDesc.setColorTarget(0, texFormat);
            res.pFbo = Fbo::create2D(1, 1, fboDesc);
        }
        mpFirstIterProg = FullScreenPass::create(psFilename, defines);
        mpFirstIterProg->addDefine("_FIRST_ITERATION");
        mpRestIterProg = FullScreenPass::create(psFilename, defines);

        // Calculate the number of reduction passes
        if(width > kTileSize || height > kTileSize)
        {
            while(width > 1 || height > 1)
            {
                width = (width + kTileSize - 1) / kTileSize;;
                height = (height + kTileSize - 1) / kTileSize;;

                width = std::max(width, 1u);
                height = std::max(height, 1u);

                Fbo::Desc fboDesc;
                fboDesc.setColorTarget(0, texFormat);
                mpTmpResultFbo.push_back(Fbo::create2D(width, height, fboDesc));
            }
        }
    }

    ParallelReduction::UniquePtr ParallelReduction::create(Type reductionType, uint32_t readbackLatency, uint32_t width, uint32_t height, uint32_t sampleCount)
    {
        return ParallelReduction::UniquePtr(new ParallelReduction(reductionType, readbackLatency, width, height, sampleCount));
    }

    void runProgram(RenderContext* pRenderCtx, Texture::SharedPtr pInput, const FullScreenPass::SharedPtr& pPass, Fbo::SharedPtr pDst, Sampler::SharedPtr pPointSampler)
    {
        pPass["gInputTex"] = pInput;
        pPass["gSampler"] = pPointSampler;
        pPass->execute(pRenderCtx, pDst);
     }

    float4 ParallelReduction::reduce(RenderContext* pRenderCtx, Texture::SharedPtr pInput)
    {
        FullScreenPass::SharedPtr pPass = mpFirstIterProg;

        for(size_t i = 0; i < mpTmpResultFbo.size(); i++)
        {
            runProgram(pRenderCtx, pInput, pPass, mpTmpResultFbo[i], mpPointSampler);
            pPass = mpRestIterProg;
            pInput = mpTmpResultFbo[i]->getColorTexture(0);
        }

        runProgram(pRenderCtx, pInput, pPass, mResultData[mCurFbo].pFbo, mpPointSampler);
        mResultData[mCurFbo].pReadTask = pRenderCtx->asyncReadTextureSubresource(mResultData[mCurFbo].pFbo->getColorTexture(0).get(), 0);
        // Read back the results
        mCurFbo = (mCurFbo + 1) % mResultData.size();
        float4 result(0);
        if(mResultData[mCurFbo].pReadTask)
        {
            auto texData = mResultData[mCurFbo].pReadTask->getData();
            mResultData[mCurFbo].pReadTask = nullptr;

            switch (mReductionType)
            {
            case Type::MinMax:
                result = float4(*reinterpret_cast<float2*>(texData.data()), 0, 0);
                break;
            default:
                FALCOR_UNREACHABLE();
            }
        }
        return result;
    }
}
