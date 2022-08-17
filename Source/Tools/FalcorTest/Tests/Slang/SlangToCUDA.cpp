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
#include "Testing/UnitTest.h"

#if FALCOR_HAS_CUDA

#include "Core/Program/CUDAProgram.h"

#include <random>

namespace Falcor
{
    namespace
    {
        const uint32_t kCount = 256;
        std::mt19937 r;
        std::uniform_real_distribution u;

        using Value = uint32_t;
    }

    GPU_TEST(SlangToCUDASimple, "Skipped due to crash with CUDA 11.3")
    {
        auto pProgram = CUDAProgram::createFromFile("Tests/Slang/SlangToCUDA.slang", "add", Program::DefineList(), Shader::CompilerFlags::None, "6_5");
        auto pVars = ComputeVars::create(pProgram.get());

        std::vector<Value> dataA(kCount);
        std::vector<Value> dataB(kCount);

        for (auto& v : dataA) v = r();
        for (auto& v : dataB) v = r();

        auto resourceBindFlags = Falcor::ResourceBindFlags::Shared
            | Falcor::ResourceBindFlags::ShaderResource
            | Falcor::ResourceBindFlags::UnorderedAccess;

        auto pBufferA = Buffer::createStructured(sizeof(Value), kCount, resourceBindFlags, Falcor::Buffer::CpuAccess::None, dataA.data());
        auto pBufferB = Buffer::createStructured(sizeof(Value), kCount, resourceBindFlags, Falcor::Buffer::CpuAccess::None, dataB.data());
        auto pBufferC = Buffer::createStructured(sizeof(Value), kCount, resourceBindFlags, Falcor::Buffer::CpuAccess::None);
        auto pBufferRead = Buffer::createStructured(sizeof(Value), kCount, Falcor::ResourceBindFlags::None, Falcor::Buffer::CpuAccess::Read);

        auto pEntryPointVars = pVars->getEntryPointGroupVars(0)->getRootVar();
        pVars["a"] = pBufferA;
        pEntryPointVars["b"] = pBufferB;
        pEntryPointVars["c"] = pBufferC;
        pEntryPointVars["count"] = kCount;

        auto pRenderContext = ctx.getRenderContext();
        pRenderContext->flush(true);

        pVars->dispatchCompute(pRenderContext, uint3(kCount, 1, 1));

        pRenderContext->flush(true);

        pRenderContext->copyBufferRegion(
            pBufferRead.get(), 0,
            pBufferC.get(), 0,
            kCount * sizeof(Value));

        pRenderContext->flush(true);

        // Verify results.
        const Value* result = (Value*)pBufferRead->map(Buffer::MapType::Read);
        for (uint32_t i = 0; i < kCount; i++)
        {
            Value a = dataA[i];
            Value b = dataB[i];
            Value expected = a + b;
            EXPECT_EQ(result[i], expected) << "i = " << i;
        }
        pBufferRead->unmap();
    }
}

#endif
