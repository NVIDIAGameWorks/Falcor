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
#include "RenderContextTest.h"

//  
void RenderContextTest::addTests()
{
    addTestToList<TestDefaultRC>();
    addTestToList<TestDefaultRCGraphicsProgram>();
}

//  
testing_func(RenderContextTest, TestDefaultRC)
{
    return test_pass();
}

//  
testing_func(RenderContextTest, TestDefaultRCGraphicsProgram)
{
    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("MultipleOutVS.vs.hlsl", "MultipleOutPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    //  Set the Program, Graphics State and the Graphics Vars.
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.0);

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the Framebuffer Object, using only Depth.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setDepthStencilTarget(ResourceFormat::D32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    //
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
    expectedColor = vec4(1.0);

    //  Render.
    pCtx->draw(4, 0);

    //  
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active Graphics Program State!");


    //
    return test_pass();
}

//  
int main()
{
    RenderContextTest rcT;
    rcT.init(true);
    rcT.run();
    return 0;
}
