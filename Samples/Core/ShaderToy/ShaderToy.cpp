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
#include "ShaderToy.h"

ShaderToy::~ShaderToy()
{
}

void ShaderToy::onLoad()
{
    // create rasterizer state
    RasterizerState::Desc rsDesc;
    mpNoCullRastState = RasterizerState::create(rsDesc);

    // Depth test
    DepthStencilState::Desc dsDesc;
	dsDesc.setDepthTest(false);
	mpNoDepthDS = DepthStencilState::create(dsDesc);

    // Blend state
    BlendState::Desc blendDesc;
    mpOpaqueBS = BlendState::create(blendDesc);

    // Texture sampler
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear).setMaxAnisotropy(8);
    mpLinearSampler = Sampler::create(samplerDesc);

    // Load shaders
    mpMainPass = FullScreenPass::create(appendShaderExtension("toyContainer.ps"));

    // Create Constant buffer
    mpToyVars = GraphicsVars::create(mpMainPass->getProgram()->getActiveVersion()->getReflector());

    // Get buffer finding
    mToyCBBinding = mpMainPass->getProgram()->getActiveVersion()->getReflector()->getDefaultParameterBlock()->getResourceBinding("ToyCB");
}

void ShaderToy::onFrameRender()
{
    // iResolution
    float width = (float)mpDefaultFBO->getWidth();
    float height = (float)mpDefaultFBO->getHeight();
    ParameterBlock* pDefaultBlock = mpToyVars->getDefaultBlock().get();
    pDefaultBlock->getConstantBuffer(mToyCBBinding, 0)["iResolution"] = glm::vec2(width, height);;

    // iGlobalTime
    float iGlobalTime = (float)mCurrentTime;  
    pDefaultBlock->getConstantBuffer(mToyCBBinding, 0)["iGlobalTime"] = iGlobalTime;

    // run final pass
    mpRenderContext->setGraphicsVars(mpToyVars);
    mpMainPass->execute(mpRenderContext.get());
}

void ShaderToy::onShutdown()
{
}

bool ShaderToy::onKeyEvent(const KeyboardEvent& keyEvent)
{
    bool bHandled = false;
    {
        if(keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            //switch(keyEvent.key)
            //{
            //default:
            //    bHandled = false;
            //}
        }
    }
    return bHandled;
}

bool ShaderToy::onMouseEvent(const MouseEvent& mouseEvent)
{
    bool bHandled = false;
    return bHandled;
}

void ShaderToy::onResizeSwapChain()
{
    uint32_t width = mpDefaultFBO->getWidth();
    uint32_t height = mpDefaultFBO->getHeight();

    mAspectRatio = (float(width) / float(height));
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    ShaderToy sample;
    SampleConfig config;
    config.windowDesc.width = 1280;
    config.windowDesc.height = 720;
    config.deviceDesc.enableVsync = true;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.title = "Falcor Shader Toy";
#ifdef _WIN32
    sample.run(config);
#else
    sample.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}
