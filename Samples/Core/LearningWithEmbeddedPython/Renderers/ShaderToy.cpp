/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

void ShaderToyRenderer::onInitialize(RenderContext::SharedPtr)
{
    // Don't re-initialize if we already have.
    if (mIsInitialized) return;

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
    mpMainPass = FullScreenPass::create(appendShaderExtension("liveTrainingToyContainer.ps"));

    // Create Constant buffer
    mpToyVars = GraphicsVars::create(mpMainPass->getProgram()->getActiveVersion()->getReflector());

    // Get buffer finding
    mToyCBBinding = mpMainPass->getProgram()->getActiveVersion()->getReflector()->getDefaultParameterBlock()->getResourceBinding("ToyCB");

    // Create a text renderer
    mTextRender = TextRenderer::create();

    // Make sure to remember that we're initialized to avoid redoing this work
    mIsInitialized = true;
}

void ShaderToyRenderer::onFrameRender(RenderContext::SharedPtr pContext, Fbo::SharedPtr pTargetFbo)
{
    mpState->setFbo(pTargetFbo);

    // iResolution
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    ParameterBlock* pDefaultBlock = mpToyVars->getDefaultBlock().get();
    pDefaultBlock->getConstantBuffer(mToyCBBinding, 0)["iResolution"] = glm::vec2(width, height);

    // iGlobalTime
    float iGlobalTime = (float)mCurrentTime;
    pDefaultBlock->getConstantBuffer(mToyCBBinding, 0)["iGlobalTime"] = iGlobalTime;

    // run final pass
    pContext->setGraphicsVars(mpToyVars);
    mpMainPass->execute(pContext.get());

#if !FALCOR_USE_PYTHON
    mTextRender->begin(pContext, vec2(400, 400));
    mTextRender->renderLine("Please set FALCOR_USE_PYTHON to 1 in FalcorConfig.h to build this sample with an embedded Python for full functionality!\n");
    mTextRender->renderLine("Also, please read the README.txt in the LearningWithEmbeddedPython directory/project to ensure appropriate version of\n");
    mTextRender->renderLine("    Python, TensorFlow, and other utilities are installed and setup for use in Falcor!");
    mTextRender->end();
#endif
}
