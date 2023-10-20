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
#include "Testing/UnitTest.h"
#include "DiffRendering/SceneGradients.h"

namespace Falcor
{
namespace
{
const char kShaderFile[] = "Tests/DiffRendering/SceneGradientsTest.cs.slang";

void testAggregateGradients(GPUUnitTestContext& ctx, const uint32_t hashSize)
{
    // We create a gradient vector with dimension = 3.
    // We add 10^i to the i-th element for 1024 times (using atomic add).
    // So the expected value of the i-th element is 1024 * (10^i).

    const uint32_t gradDim = 3;
    const uint32_t elemCount = 1024;

    ref<Device> pDevice = ctx.getDevice();
    RenderContext* pRenderContext = pDevice->getRenderContext();

    std::unique_ptr<SceneGradients> pSceneGradients = std::make_unique<SceneGradients>(pDevice, uint2(gradDim), uint2(hashSize));
    pSceneGradients->clearGrads(pRenderContext, GradientType::Material);

    ctx.createProgram(kShaderFile, "atomicAdd");
    ctx["CB"]["sz"] = uint2(gradDim, elemCount);
    ctx["CB"]["hashSize"] = hashSize;
    pSceneGradients->bindShaderData(ctx["gSceneGradients"]);
    ctx.runProgram(gradDim, elemCount, 1);

    pSceneGradients->aggregateGrads(pRenderContext, GradientType::Material);

    ctx.createProgram(kShaderFile, "testAggregateGradients");
    ctx["CB"]["sz"] = uint2(gradDim, elemCount);
    ctx["grads"] = pSceneGradients->getGradsBuffer(GradientType::Material);
    ctx.allocateStructuredBuffer("result", gradDim);
    ctx.runProgram(gradDim, 1, 1);

    std::vector<float> result = ctx.readBuffer<float>("result");
    for (uint32_t i = 0; i < gradDim; ++i)
    {
        float refValue = elemCount * std::pow(10.f, i);
        float relAbsDiff = std::abs(result[i] - refValue) / refValue;
        EXPECT_LE(relAbsDiff, 1e-6f);
    }
}
} // namespace

// Disabled on Vulkan for now as the compiler generates invalid code.
GPU_TEST(AggregateGradients, Device::Type::D3D12)
{
    testAggregateGradients(ctx, 64);
}
} // namespace Falcor
