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
#include "Core/Plugin.h"
#include "Testing/UnitTest.h"
#include "RenderGraph/RenderGraph.h"

namespace Falcor
{
GPU_TEST(InvalidPixelDetectionPass)
{
    PluginManager::instance().loadPluginByName("DebugPasses");

    ref<Device> pDevice = ctx.getDevice();

    float pInitData[8] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::signaling_NaN(),
        std::numeric_limits<float>::infinity(),
        -1 * std::numeric_limits<float>::infinity(),
        0.0f,
        255.0f,
        125.8f,
        1.0f,
    };

    RenderContext* pRenderContext = ctx.getRenderContext();
    ref<Fbo> pTargetFbo = Fbo::create2D(pDevice, 2, 4, ResourceFormat::BGRA8UnormSrgb);
    ref<Texture> pInput = pDevice->createTexture2D(2, 4, ResourceFormat::R32Float, 1, Resource::kMaxPossible, pInitData);
    ref<RenderGraph> pGraph = RenderGraph::create(ctx.getDevice(), "Invalid Pixel Detection");
    ref<RenderPass> pPass = RenderPass::create("InvalidPixelDetectionPass", ctx.getDevice());
    if (!pPass)
        FALCOR_THROW("Could not create render pass 'InvalidPixelDetectionPass'");
    pGraph->addPass(pPass, "InvalidPixelDetectionPass");
    pGraph->setInput("InvalidPixelDetectionPass.src", pInput);
    pGraph->markOutput("InvalidPixelDetectionPass.dst");
    pGraph->onResize(pTargetFbo.get());
    pGraph->execute(pRenderContext);
    ref<Resource> pOutput = pGraph->getOutput("InvalidPixelDetectionPass.dst");
    std::vector<uint8_t> color = pRenderContext->readTextureSubresource(pOutput->asTexture().get(), 0);
    uint32_t* output = (uint32_t*)color.data();

    for (uint32_t i = 0; i < 8; ++i)
    {
        uint32_t expected;
        switch (i)
        {
        case 0:
        case 1:
            expected = 0xFFFF0000;
            break;
        case 2:
        case 3:
            expected = 0xFF00FF00;
            break;
        default:
            expected = 0xFF000000;
            break;
        }
        EXPECT_EQ(output[i], expected);
    }
}
} // namespace Falcor
