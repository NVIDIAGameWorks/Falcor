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
#include "MultiSampling.h"

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
const uint32_t kTriangleCount = 16;
const uint32_t kSampleCount = 8;
} // namespace

MultiSampling::MultiSampling(const SampleAppConfig& config) : SampleApp(config) {}

MultiSampling::~MultiSampling() {}

void MultiSampling::onLoad(RenderContext* pRenderContext)
{
    // Load program
    mpRasterPass = RasterPass::create(getDevice(), "Samples/MultiSampling/MultiSampling.3d.slang", "vsMain", "psMain");

    // Create disk triangles
    float2 vertices[kTriangleCount * 3];
    for (uint32_t i = 0; i < kTriangleCount; ++i)
    {
        float theta0 = float(i) / kTriangleCount * M_2PI;
        float theta1 = float(i + 1) / kTriangleCount * M_2PI;
        vertices[i * 3 + 0] = float2(0, 0);
        vertices[i * 3 + 1] = float2(cos(theta0), sin(theta0)) * 0.75f;
        vertices[i * 3 + 2] = float2(cos(theta1), sin(theta1)) * 0.75f;
    }
    auto vertexBuffer = getDevice()->createTypedBuffer<float2>(
        kTriangleCount * 3, ResourceBindFlags::ShaderResource | ResourceBindFlags::Vertex, MemoryType::DeviceLocal, vertices
    );

    // Create vertex layout
    auto bufferLayout = VertexBufferLayout::create();
    bufferLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
    auto layout = VertexLayout::create();
    layout->addBufferLayout(0, bufferLayout);

    // Create VAO
    mpVao = Vao::create(Vao::Topology::TriangleList, layout, {vertexBuffer});

    // Create FBO
    mpFbo = Fbo::create(getDevice());
    ref<Texture> tex = getDevice()->createTexture2DMS(
        128, 128, ResourceFormat::RGBA32Float, kSampleCount, 1, ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget
    );
    mpFbo->attachColorTarget(tex, 0);
}

void MultiSampling::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    pRenderContext->clearFbo(mpFbo.get(), float4(0.f), 0.f, 0);

    mpRasterPass->getState()->setFbo(mpFbo);
    mpRasterPass->getState()->setVao(mpVao);
    mpRasterPass->draw(pRenderContext, kTriangleCount * 3, 0);

    pRenderContext->blit(mpFbo->getColorTexture(0)->getSRV(), pTargetFbo->getRenderTargetView(0));
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.width = 1024;
    config.windowDesc.height = 1024;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.enableVSync = true;
    config.windowDesc.title = "Falcor multi-sampling example";

    MultiSampling multiSample(config);
    return multiSample.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
