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
#include "GraphicsStateTest.h"
#include "TestHelper.h"

//  
void GraphicsStateTest::addTests()
{
    //  Test Blend Changes.
    addTestToList<TestBlendStateSimple>();
    addTestToList<TestBlendStateNullptr>();
    addTestToList<TestBlendStateChanges>();
    addTestToList<TestBlendStateMultipleRTChanges>();

    //  Test the Depth Stencil Changes.
    addTestToList<TestDepthSimple>();
    addTestToList<TestDepthNullptr>();
    addTestToList<TestDepthChanges>();

    //  Test Simple Changes.
    addTestToList<TestStencilSimple>();
    addTestToList<TestStencilNullptr>();
    addTestToList<TestStencilChanges>();

    //  Test the VAO Changes.
    addTestToList<TestVAOSimple>();
    addTestToList<TestVAONullptr>();
    addTestToList<TestVAOChanges>();

    //  Test Fbo Changes.
    addTestToList<TestFboSimple>();
    addTestToList<TestFboNullptr>();
    addTestToList<TestFboChanges>();

    //  Test Rasterizer Changes.
    addTestToList<TestRasterizerBasicChanges>();
    addTestToList<TestRasterizerDepthBiasChanges>();
    addTestToList<TestRasterizerScissorChanges>();


    //  Test Program Changes.
    addTestToList<TestGraphicsProgramBasic>();
    addTestToList<TestGraphicsProgramChanges>();


}

//  Test Blend State Simple.
testing_func(GraphicsStateTest, TestBlendStateSimple)
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

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the Framebuffer Object.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), glm::vec4(0.5f), 1.0f, 0, FboAttachmentType::Color);

    //  Create and test the blend state.
    BlendState::Desc blendDesc;
    blendDesc.setRenderTargetWriteMask(0, true, false, true, true);
    blendDesc.setBlendFactor(vec4(0.9f));
    blendDesc.setRtBlend(0, true);
    blendDesc.setRtParams(0, BlendState::BlendOp::Subtract, BlendState::BlendOp::Subtract, BlendState::BlendFunc::BlendFactor, BlendState::BlendFunc::BlendFactor, BlendState::BlendFunc::One, BlendState::BlendFunc::OneMinusBlendFactor);

    //  
    BlendState::SharedPtr pBS;
    pBS = BlendState::create(blendDesc);

    //  
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.0);
    expectedColor = vec4(0.45f, 0.5f, 0.45f, 0.95f);


    //  Get the Per Frame Constant Buffer, and set the Color Variable
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
    pGS->setBlendState(pBS);
    pCtx->draw(4, 0);

    //  Read Back the Color, and Verify.
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Simple Blend State!");

    //
    return test_pass();
}

//  Test Null Blend State.
testing_func(GraphicsStateTest, TestBlendStateNullptr)
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

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the Framebuffer Object.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.0);

    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), glm::vec4(0.0), 1.0, 0, FboAttachmentType::Color);

    //  Test Null Blend State Output.
    //  Get the Per Frame Constant Buffer, and set the Color Variable
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.0, 0.0, 0.0, 0.0));
    pGS->setBlendState(nullptr);
    pCtx->draw(4, 0);

    //  Read Back the Color, and Verify.
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Null Blend State!");

    //  Get the Per Frame Constant Buffer, and set the Color Variable
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
    expectedColor = vec4(1.0, 1.0, 1.0, 1.0);
    pCtx->draw(4, 0);

    //  Read Back the Color, and Verify.
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Null Blend State!");

    return test_pass();
}

//  Test Blend State Changes.
testing_func(GraphicsStateTest, TestBlendStateChanges)
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

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the Framebuffer Object.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.0);

    //  Test Null Blend State Output.
    //  Get the Per Frame Constant Buffer, and set the Color Variable
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.0, 0.0, 0.0, 0.0));
    pGS->setBlendState(nullptr);
    pCtx->draw(4, 0);

    //  Read Back the Color, and Verify.
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Null Blend State!");

    //  Get the Per Frame Constant Buffer, and set the Color Variable
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
    expectedColor = vec4(1.0, 1.0, 1.0, 1.0);
    pCtx->draw(4, 0);

    //  Read Back the Color, and Verify.
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Null Blend State!");


    //  
    uint32_t maskCombinations = (uint32_t)std::pow(2, 4);
    for (uint32_t i = 0; i < maskCombinations; i++)
    {
        //  RGBA
        bool wR = ((i & 1) == 1);
        bool wG = ((i & 2) == 2);
        bool wB = ((i & 4) == 4);
        bool wA = ((i & 8) == 8);

        //  Clear, Render and Blend Values.
        float clearColor = 0.5f;
        float renderValue = 1.0f;
        float blendFactor = 0.9f;

        //
        vec4 directResultColor = vec4();


        pCtx->clearRtv(pGS->getFbo()->getColorTexture(0)->getRTV().get(), vec4(clearColor));

        //  Create and test the blend state.
        BlendState::Desc blendDesc;
        blendDesc.setRenderTargetWriteMask(0, wR, wG, wB, wA);
        blendDesc.setBlendFactor(vec4(blendFactor));
        blendDesc.setRtBlend(0, true);
        blendDesc.setRtParams(0, BlendState::BlendOp::Subtract, BlendState::BlendOp::Subtract, BlendState::BlendFunc::BlendFactor, BlendState::BlendFunc::BlendFactor, BlendState::BlendFunc::One, BlendState::BlendFunc::OneMinusBlendFactor);


        float expecR = (((float)wR) * (blendFactor * (renderValue - clearColor))) + (1.0f - (float)(wR)) * clearColor;
        float expecG = (((float)wG) * (blendFactor * (renderValue - clearColor))) + (1.0f - (float)(wG)) * clearColor;
        float expecB = (((float)wB) * (blendFactor * (renderValue - clearColor))) + (1.0f - (float)(wB)) * clearColor;
        float expecA = (((float)wA) * (renderValue - ((1.0f - blendFactor) * clearColor))) + (1.0f - (float)(wA)) * clearColor;

        expectedColor = vec4(expecR, expecG, expecB, expecA);

        //  
        BlendState::SharedPtr pBS;
        pBS = BlendState::create(blendDesc);

        //  Set the Blend State.
        pGS->setBlendState(pBS);

        //  Get the Per Frame Constant Buffer, and set the Color Variable
        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(renderValue, renderValue, renderValue, renderValue));

        pCtx->draw(4, 0);

        //  
        outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
        if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Write Mask Blend State!");

        //  
        //  Null Blend State.

        //  Set the Blend State.
        pGS->setBlendState(nullptr);

        //  Get the Per Frame Constant Buffer, and set the Color Variable
        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
        expectedColor = vec4(1.0, 1.0, 1.0, 1.0);
        pCtx->draw(4, 0);

        //  
        outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
        if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Null Blend State!");

    }

    //  Return success.
    return test_pass();
}

//  Test Blend State Multiple RT Changes.
testing_func(GraphicsStateTest, TestBlendStateMultipleRTChanges)
{
    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("MultipleOutVS.vs.hlsl", "MultipleOutPS.ps.hlsl");

    //  Add the defines for the Render Targets.
    for (uint32_t rtIndex = 0; rtIndex < kRTCount; rtIndex++)
    {
        pGP->addDefine("_USE_SV_" + std::to_string(rtIndex));
    }

    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    //  Set the Program, Graphics State and the Graphics Vars.
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the Framebuffer Object.
    Fbo::Desc fboDesc;
    for (uint32_t rtIndex = 0; rtIndex < kRTCount; rtIndex++)
    {
        fboDesc.setColorTarget(rtIndex, ResourceFormat::RGBA32Float);
    }

    //  Create the FBO.
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  All possible combinations of blend states.
    uint32_t maskCombinations = (uint32_t)std::pow(2, 4);

    //  Clear, Render and Blend Values.
    float clearColor = 0.5f;
    float renderValue = 1.0f;
    float blendFactor = 0.9f;

    //  
    for (uint32_t currentMaskCombination = 0; currentMaskCombination < maskCombinations; currentMaskCombination++)
    {

        //  Create and test the blend state.
        BlendState::Desc blendDesc;
        BlendState::SharedPtr pBS;

        //  Output Color and Expected Color Set.
        vec4 outputColorSet[kRTCount];
        vec4 expectedColorSet[kRTCount];

        //  Set the Render Targets.
        for (uint32_t rtMaskIndex = 0; rtMaskIndex < kRTCount; rtMaskIndex++)
        {
            //  Offset the Mask, for independent blend state checking.
            uint32_t currentMaskOffset = (rtMaskIndex + currentMaskCombination) % maskCombinations;

            //  RGBA
            bool wR = ((currentMaskOffset & 1) == 1);
            bool wG = ((currentMaskOffset & 2) == 2);
            bool wB = ((currentMaskOffset & 4) == 4);
            bool wA = ((currentMaskOffset & 8) == 8);

            //  
            blendDesc.setRenderTargetWriteMask(rtMaskIndex, wR, wG, wB, wA);
            blendDesc.setIndependentBlend(true);
            blendDesc.setBlendFactor(vec4(blendFactor));
            blendDesc.setRtBlend(rtMaskIndex, true);
            blendDesc.setRtParams(rtMaskIndex, BlendState::BlendOp::Subtract, BlendState::BlendOp::Subtract, BlendState::BlendFunc::BlendFactor, BlendState::BlendFunc::BlendFactor, BlendState::BlendFunc::One, BlendState::BlendFunc::OneMinusBlendFactor);


            float expecR = (((float)wR) * (blendFactor * (renderValue - clearColor))) + (1.0f - (float)(wR)) * clearColor;
            float expecG = (((float)wG) * (blendFactor * (renderValue - clearColor))) + (1.0f - (float)(wG)) * clearColor;
            float expecB = (((float)wB) * (blendFactor * (renderValue - clearColor))) + (1.0f - (float)(wB)) * clearColor;
            float expecA = (((float)wA) * (renderValue - ((1.0f - blendFactor) * clearColor))) + (1.0f - (float)(wA)) * clearColor;


            //  
            expectedColorSet[rtMaskIndex] = vec4(expecR, expecG, expecB, expecA);
        }

        //  Clear the Render Target.
        for (uint32_t rtIndex = 0; rtIndex < kRTCount; rtIndex++)
        {
            pCtx->clearRtv(pGS->getFbo()->getColorTexture(rtIndex)->getRTV().get(), vec4(clearColor));
        }

        //  Create the Blend State.
        pBS = BlendState::create(blendDesc);

        //  Set the Blend State.
        pGS->setBlendState(pBS);

        //  Get the Per Frame Constant Buffer, and set the Color Variable
        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(renderValue));
        pCtx->draw(4, 0);

        //  
        for (uint32_t rtIndex = 0; rtIndex < kRTCount; rtIndex++)
        {
            //  
            outputColorSet[rtIndex] = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(rtIndex).get(), 0).data());
            if (outputColorSet[rtIndex] != expectedColorSet[rtIndex]) return test_fail("Output Color Not Equal to Expected Color - Write Mask Blend State!");
        }

        //  Set the Blend State.
        pGS->setBlendState(nullptr);

        //  Get the Per Frame Constant Buffer, and set the Color Variable
        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
        pCtx->draw(4, 0);

        //  
        for (uint32_t rtIndex = 0; rtIndex < kRTCount; rtIndex++)
        {
            //  
            outputColorSet[rtIndex] = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(rtIndex).get(), 0).data());
            if (outputColorSet[rtIndex] != vec4(1.0, 1.0, 1.0, 1.0)) return test_fail("Output Color Not Equal to Expected Color - Null Blend State!");
        }


    }

    //  Return success.
    return test_pass();
}

//  
//  Test Depth Stencil Simple State.
testing_func(GraphicsStateTest, TestDepthSimple)
{
    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("DSVS.vs.hlsl", "DSPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    //  Set the Program, Graphics State and the Graphics Vars.
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the Framebuffer Object.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setDepthStencilTarget(ResourceFormat::D32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    //  
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.0);

    //  Set the Depth Test.
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthTest(true);

    DepthStencilState::SharedPtr pDS = DepthStencilState::create(dsDesc);

    //  
    expectedColor = vec4(1.0);
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
    pGV->getConstantBuffer("PerFrameCB")->setVariable("depth_value", 0.0f);
    pCtx->draw(4, 0);

    //  
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active Depth State!");

    //  
    return test_pass();
}

//  Test Depth Stencil Nullptr State.
testing_func(GraphicsStateTest, TestDepthNullptr)
{
    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("DSVS.vs.hlsl", "DSPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    //  Set the Program, Graphics State and the Graphics Vars.
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the Framebuffer Object.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setDepthStencilTarget(ResourceFormat::D32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    //  
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.0);

    //  
    pGS->setDepthStencilState(nullptr);

    //  
    expectedColor = vec4(1.0);
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
    pGV->getConstantBuffer("PerFrameCB")->setVariable("depth_value", 0.0f);
    pCtx->draw(4, 0);

    //  
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active Depth State!");

    //  
    return test_pass();
}

//  Test Depth Stencil State Changes.
testing_func(GraphicsStateTest, TestDepthChanges)
{

    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("DSVS.vs.hlsl", "DSPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    //  Set the Program, Graphics State and the Graphics Vars.
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

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
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.0);

    float outputDepth = 1.0f;
    float expectedDepth = 1.0f;
    float newDepthVal = 0.0f;

    //  
    for (uint32_t i = 0; i < 9; i++)
    {

        pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

        DepthStencilState::Func depthFunc = ((DepthStencilState::Func)i);

        if (depthFunc == DepthStencilState::Func::Disabled)
            continue;

        //  
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthTest(true);
        dsDesc.setDepthFunc(DepthStencilState::Func(i));
        DepthStencilState::SharedPtr pDS = DepthStencilState::create(dsDesc);


        pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 0.5, 0, FboAttachmentType::All);
        pGS->setDepthStencilState(pDS);

        //  
        newDepthVal = 0.3f;
        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
        pGV->getConstantBuffer("PerFrameCB")->setVariable("depth_value", newDepthVal);
        if (depthFunc == DepthStencilState::Func::Less || depthFunc == DepthStencilState::Func::LessEqual || depthFunc == DepthStencilState::Func::NotEqual || depthFunc == DepthStencilState::Func::Always)
        {
            expectedColor = vec4(1.0, 1.0, 1.0, 1.0);
            expectedDepth = 0.3f;
        }
        else
        {
            expectedColor = vec4(0.0, 0.0, 0.0, 0.0);
            expectedDepth = 0.5f;
        }

        //  Render.
        pCtx->draw(4, 0);

        //  
        outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
        if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active Depth State!");

        //
        outputDepth = *(float*)(pCtx->readTextureSubresource(pGS->getFbo()->getDepthStencilTexture().get(), 0).data());
        if (outputDepth != expectedDepth) return test_fail("Output Depth Not Equal to Expected Depth - Active Depth State!");


        pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 0.5, 0, FboAttachmentType::All);
        pGS->setDepthStencilState(pDS);

        //  
        newDepthVal = 0.7f;
        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
        pGV->getConstantBuffer("PerFrameCB")->setVariable("depth_value", newDepthVal);
        if (depthFunc == DepthStencilState::Func::Greater || depthFunc == DepthStencilState::Func::GreaterEqual || depthFunc == DepthStencilState::Func::NotEqual || depthFunc == DepthStencilState::Func::Always)
        {
            expectedColor = vec4(1.0, 1.0, 1.0, 1.0);
            expectedDepth = 0.7f;
        }
        else
        {
            expectedColor = vec4(0.0, 0.0, 0.0, 0.0);
            expectedDepth = 0.5f;
        }

        //  Render.
        pCtx->draw(4, 0);

        //  
        outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
        if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active Depth State!");

        //
        outputDepth = *(float*)(pCtx->readTextureSubresource(pGS->getFbo()->getDepthStencilTexture().get(), 0).data());
        if (outputDepth != expectedDepth) return test_fail("Output Depth Not Equal to Expected Depth - Active Depth State!");


        //  Run the Depth Stencil State Null.
        newDepthVal = 0.25f;
        expectedDepth = 0.25f;
        expectedColor = vec4(0.5);

        //  
        pGS->setDepthStencilState(nullptr);

        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.5, 0.5, 0.5, 0.5));
        pGV->getConstantBuffer("PerFrameCB")->setVariable("depth_value", newDepthVal);

        //  Render.
        pCtx->draw(4, 0);

        //  
        outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
        if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Null Depth State!");

        //
        outputDepth = *(float*)(pCtx->readTextureSubresource(pGS->getFbo()->getDepthStencilTexture().get(), 0).data());
        if (outputDepth != expectedDepth) return test_fail("Output Depth Not Equal to Expected Depth - Null Depth State!");

    }

    return test_pass();
}

//  Test Stencil Simple.
testing_func(GraphicsStateTest, TestStencilSimple)
{
    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGPDS = GraphicsProgram::createFromFile("MultipleOutVS.vs.hlsl", "MultipleOutPS.ps.hlsl");
    GraphicsVars::SharedPtr pGVDS = GraphicsVars::create(pGPDS->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    //  Set the Program, Graphics State and the Graphics Vars.
    pCtx->setGraphicsState(pGS);
    pGS->setProgram(pGPDS);
    pCtx->setGraphicsVars(pGVDS);

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the Framebuffer Object.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setDepthStencilTarget(ResourceFormat::D24UnormS8);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.0);

    DepthStencilState::Desc dsDescReplace;
    dsDescReplace.setDepthTest(false).setStencilTest(true).setStencilRef(1);
    dsDescReplace.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Replace);
    DepthStencilState::SharedPtr dsStateReplace = DepthStencilState::create(dsDescReplace);


    DepthStencilState::Desc dsDescKeep;
    dsDescKeep.setDepthTest(false).setStencilTest(true).setStencilFunc(DepthStencilState::Face::FrontAndBack, DepthStencilState::Func::NotEqual).setStencilRef(0);
    dsDescKeep.setStencilOp(DepthStencilState::Face::FrontAndBack, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Keep, DepthStencilState::StencilOp::Replace);
    DepthStencilState::SharedPtr dsStateKeep = DepthStencilState::create(dsDescKeep);

    //  
    pGS->setDepthStencilState(dsStateReplace);

    //  Clear Fbo.
    pCtx->clearFbo(pFbo.get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    pGVDS->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.75f));
    expectedColor = vec4(0.75f);
    pCtx->draw(4, 0);

    //      
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active Stencil State!");


    //  
    pGS->setDepthStencilState(nullptr);

    pGVDS->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.25f));
    expectedColor = vec4(0.25f);
    pCtx->draw(4, 0);

    //      
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active Stencil State!");

    //  
    pGS->setDepthStencilState(dsStateKeep);

    pGVDS->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.5f));
    expectedColor = vec4(0.5f);
    pCtx->draw(4, 0);

    //      
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active Stencil State!");

    return test_pass();
}

//  Test Stencil Nullptr.
testing_func(GraphicsStateTest, TestStencilNullptr)
{
    return test_pass();
}

//  Test Stencil Changes.
testing_func(GraphicsStateTest, TestStencilChanges)
{
    return test_pass();
}

//  Test VAO Simple State. (Uses FullScreen VAO).
testing_func(GraphicsStateTest, TestVAOSimple)
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
    vec4 expectedColor = vec4(1.0);

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

    //  Render.
    pCtx->draw(4, 0);

    //  
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");

    //
    return test_pass();
}

//  Test VAO Nullptr State.
testing_func(GraphicsStateTest, TestVAONullptr)
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

    //  Create the Framebuffer Object, using only Depth.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setDepthStencilTarget(ResourceFormat::D32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);
    pGS->setVao(nullptr);

    //  
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.0);

    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    //
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));

    //  Render.
    pCtx->draw(4, 0);

    //  
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");

    //
    return test_pass();
}

//  Test VAO Changes State.
testing_func(GraphicsStateTest, TestVAOChanges)
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


    //  Create the pVAOs
    Vao::SharedPtr pVAOs[5];
    {
        pVAOs[0] = TestHelper::getBasicPointsVao();
        pVAOs[1] = TestHelper::getBasicLinesListVao();
        pVAOs[2] = TestHelper::getBasicLineStripVAO();
        pVAOs[3] = TestHelper::getBasicTriangleListVao();
        pVAOs[4] = TestHelper::getFullscreenQuadVao();
    }



    //  Create the Framebuffer Object.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setDepthStencilTarget(ResourceFormat::D32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(5, 5, fboDesc);
    pGS->setFbo(pFbo);

    expectedColor = vec4(1.0);
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));
    std::vector<vec4> colorResult;

    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    //  Run the Points Topology.    
    pGS->setVao(pVAOs[0]);
    pCtx->draw(1, 0);

    //  Read back the colors.
    colorResult = *(std::vector<vec4>*)& (pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0));

    //  Extract the Center Color pixel.
    outputColor = colorResult[12];
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");

    //
    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    //  Run the Lines List Topology. Two horizontal lines.
    expectedColor = vec4(1.0);
    pGS->setVao(pVAOs[1]);
    pCtx->draw(4, 0);

    //  Read back the colors.
    colorResult = *(std::vector<vec4>*)& (pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0));

    //  Extract the colored pixels.
    outputColor = colorResult[7];
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");
    outputColor = colorResult[18];
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");


    //
    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    //  Run the Lines Strip Topology. One horizontal line acorss the middle.
    expectedColor = vec4(1.0);
    pGS->setVao(pVAOs[2]);
    pCtx->draw(3, 0);

    //  Read back the colors.
    colorResult = *(std::vector<vec4>*)& (pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0));

    //  Extract the colored pixels.
    outputColor = colorResult[10];
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");
    outputColor = colorResult[14];
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");



    //
    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    //  Run the Triangle Strip Topology. One Triangle.
    expectedColor = vec4(1.0);
    pGS->setVao(pVAOs[3]);
    pCtx->drawIndexed(3, 0, 0);

    //  Read back the colors.
    colorResult = *(std::vector<vec4>*)& (pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0));

    //  Extract the colored pixels.
    outputColor = colorResult[11];
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");
    outputColor = colorResult[13];
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");
    outputColor = colorResult[17];
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");

    //
    return test_pass();
}

//  Test Rasterizer State Changes.
testing_func(GraphicsStateTest, TestRasterizerBasicChanges)
{

    //  Common State Setup, Sets Program, Graphics State and the Graphics Vars.

    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("MultipleOutVS.vs.hlsl", "MultipleOutPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    //  
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  Output Color and Expected Color to check for success.
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(1.0);

    //  Create the pVAO and the Reversed pRVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    Vao::SharedPtr pRVAO = TestHelper::getReversedFullscreenQuadVao();

    //  Create the Framebuffer Object, using only Depth.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setDepthStencilTarget(ResourceFormat::D32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  Create the Rasterizer States.   
    RasterizerState::SharedPtr pRS[2];

    RasterizerState::Desc rsDesc0;
    rsDesc0.setCullMode(RasterizerState::CullMode::Front);
    pRS[0] = RasterizerState::create(rsDesc0);

    RasterizerState::Desc rsDesc1;
    rsDesc1.setFrontCounterCW(false);
    pRS[1] = RasterizerState::create(rsDesc1);


    bool success = true;

    //  Test the Default Rasterizer State.
    success = renderDefaultRasterizerState(pCtx, pGS, pGV, pVAO);
    if (!success) return test_fail("Failed to Render Default Rasterizer State!");

    //  Test the Rasterizer State - with Front Culling.
    {
        //  
        //  Clear the Color Value.
        pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);
        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.25f, 0.25f, 0.25f, 0.25f));
        expectedColor = vec4(0.0, 0.0, 0.0, 0.0);

        //  
        pGS->setRasterizerState(pRS[0]);
        pGS->setVao(pVAO);
        pCtx->draw(4, 0);

        //  Get the Output Color.
        outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
        if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Rasterizer Front Culling State!");
    }

    //  Test the Default Rasterizer State.
    success = renderDefaultRasterizerState(pCtx, pGS, pGV, pVAO);
    if (!success) return test_fail("Failed to Render Default Rasterizer State!");

    //  Test the Rasterizer State - with Front Clockwise.
    {
        //  
        //  Clear the Color Value.
        pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);
        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.25f, 0.25f, 0.25f, 0.25f));
        expectedColor = vec4(0.25f, 0.25f, 0.25f, 0.25f);

        //  
        pGS->setRasterizerState(pRS[1]);
        pGS->setVao(pRVAO);
        pCtx->draw(4, 0);

        //  Get the Output Color.
        outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
        if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Rasterizer State!");

    }


    //  Test the Default Rasterizer State.
    success = renderDefaultRasterizerState(pCtx, pGS, pGV, pVAO);
    if (!success) return test_fail("Failed to Render Default Rasterizer State!");

    //  Return Test Pass.
    return test_pass();
}

//  Test the Rasterizer Depth Changes.
testing_func(GraphicsStateTest, TestRasterizerDepthBiasChanges)
{
    return test_pass();
}

//  Test the Rasterizer Fill Mode Changes
testing_func(GraphicsStateTest, TestRasterizerFillModeChanges)
{
    //
    return test_pass();
}

//  Test the Sample Count.
testing_func(GraphicsStateTest, TestRasterizerSampleCountChanges)
{
    //
    return test_pass();
}

//  Test the Conservative Raster.
testing_func(GraphicsStateTest, TestRasterizerConservativeRaster)
{
    //
    return test_pass();
}

//  Test the Line Anti-Aliasing.
testing_func(GraphicsStateTest, TestRasterizerLineAntiAliasing)
{
    return test_pass();
}

//  Test the Rasterizer Scissor.
testing_func(GraphicsStateTest, TestRasterizerScissorChanges)
{
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("MultipleOutVS.vs.hlsl", "MultipleOutPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    //  
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  Output Color and Expected Color to check for success.
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(1.0);

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();

    //  Create the Framebuffer Object, using only Depth.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setDepthStencilTarget(ResourceFormat::D32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(10, 10, fboDesc);

    pCtx->clearFbo(pFbo.get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    GraphicsState::Scissor scissor;
    scissor.bottom = 5;
    scissor.top = 0;
    scissor.left = 0;
    scissor.right = 5;

    RasterizerState::Desc rzDesc;
    rzDesc.setScissorTest(true);

    RasterizerState::SharedPtr rzState = RasterizerState::create(rzDesc);

    //  
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.5f, 0.5f, 0.5f, 0.5f));
    expectedColor = vec4(0.5, 0.5, 0.5, 0.5);
    pGS->setFbo(pFbo);
    pGS->setVao(pVAO);
    pGS->setRasterizerState(rzState);
    pGS->setScissors(0, scissor);

    //  
    pCtx->draw(4, 0);

    //
    std::vector<vec4> colorResult = *(std::vector<vec4>*)& (pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0));


    return test_pass();
}

//  Render the  Default Rasterizer State, without clearing.
bool GraphicsStateTest::renderDefaultRasterizerState(RenderContext::SharedPtr pCtx, GraphicsState::SharedPtr pGS, GraphicsVars::SharedPtr pGV, Vao::SharedPtr pVAO)
{
    //  Create the color variables.
    vec4 outputColor, expectedColor;

    //  
    //  Clear the Color Value.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.25f, 0.25f, 0.25f, 0.25f));
    expectedColor = vec4(0.25f, 0.25f, 0.25f, 0.25);

    //  Run the default Rasterizer State.
    pGS->setRasterizerState(nullptr);
    pGS->setVao(pVAO);
    pCtx->draw(4, 0);

    //  Get the Output Color.
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor)
    {
        return false;
    }
    else
    {
        return true;
    }
}

// Test Fbo Simple.
testing_func(GraphicsStateTest, TestFboSimple)
{

    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("MultipleOutVS.vs.hlsl", "MultipleOutPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  
    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(1.0);

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

    //  Set the Color Value.
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));

    //  Render.
    pCtx->draw(4, 0);

    //  Read back the Output Color.
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active VAO State!");

    //  
    return test_pass();
}

// Test Fbo Simple.
testing_func(GraphicsStateTest, TestFboNullptr)
{
    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("MultipleOutVS.vs.hlsl", "MultipleOutPS.ps.hlsl");

    //  
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Null Framebuffer.
    pGS->setFbo(nullptr);

    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    //  Set the Color Value.
    pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(1.0, 1.0, 1.0, 1.0));

    //  Render.
    pCtx->draw(4, 0);

    //  
    return test_pass();
}

// Test Fbo Changes in Render Target Numbers.
testing_func(GraphicsStateTest, TestFboChanges)
{
    //  Set the Program, Graphics State and the Graphics Vars.
        //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("MultipleOutVS.vs.hlsl", "MultipleOutPS.ps.hlsl");

    //  Add the defines for the Render Targets.
    for (uint32_t rtIndex = 0; rtIndex < kRTCount; rtIndex++)
    {
        pGP->addDefine("_USE_SV_" + std::to_string(rtIndex));
    }

    //  Add the defines for the Depth.
    pGP->addDefine("_USE_DEPTH");

    //  
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();

    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    Fbo::SharedPtr pFbos[kRTCount * 2];
    Texture::SharedPtr pTextures[kRTCount];
    Texture::SharedPtr pDepthTexture;

    //  Create the Textures.
    for (uint32_t rtIndex = 0; rtIndex < kRTCount; rtIndex++)
    {
        pTextures[rtIndex] = Texture::create2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Texture::BindFlags::RenderTarget);
    }

    //  
    pDepthTexture = Texture::create2D(1, 1, ResourceFormat::D32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Texture::BindFlags::DepthStencil);

    //  Create the Framebuffer Objects.
    for (uint32_t dpEnabled = 0; dpEnabled < 2; dpEnabled++)
    {
        for (uint32_t fboIndex = 0; fboIndex < kRTCount; fboIndex++)
        {
            //  Create the empty framebuffer.
            pFbos[kRTCount * dpEnabled + fboIndex] = Fbo::create();

            //  Add the appropriate color targets.
            for (uint32_t rtIndex = 0; rtIndex < fboIndex; rtIndex++)
            {
                pFbos[kRTCount * dpEnabled + fboIndex]->attachColorTarget(pTextures[rtIndex], rtIndex, 0, 0, Fbo::kAttachEntireMipLevel);
            }

            //  Add the appropriate depth targets.
            if (dpEnabled)
            {
                pFbos[kRTCount * dpEnabled + fboIndex]->attachDepthStencilTarget(pDepthTexture);
            }

        }
    }

    //  Render all of the Framebuffers.
    for (uint32_t dpEnabled = 0; dpEnabled < 2; dpEnabled++)
    {
        for (uint32_t fboIndex = 0; fboIndex < kRTCount; fboIndex++)
        {
            if((fboIndex + kRTCount * dpEnabled) == 0)
                continue;

            pGS->setFbo(pFbos[fboIndex + kRTCount * dpEnabled]);

            //  Clear the Fbo.
            pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

            //  
            vec4 outputColor = vec4(0.0);
            vec4 expectedColor = vec4(1.0);

            float outputDepth = 0.0;
            float expectedDepth = 0.0;

            //  Set the Color Value.
            pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value", expectedColor);

            pGV->getConstantBuffer("PerFrameCB")->setVariable("depth_value", 0.0f);

            //  Render.
            pCtx->draw(4, 0);

            //  
            for (uint32_t rtIndex = 0; rtIndex < fboIndex; rtIndex++)
            {
                //  Read back the Output Color.
                outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(rtIndex).get(), 0).data());
                if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Active Fbo State!");
            }

            if (dpEnabled)
            {
                //
                outputDepth = *(float*)(pCtx->readTextureSubresource(pGS->getFbo()->getDepthStencilTexture().get(), 0).data());
                if (outputDepth != expectedDepth) return test_fail("Output Depth Not Equal to Expected Depth - Active Fbo State!");
            }

            pGS->setFbo(nullptr);

            //  Clear the Fbo.
            pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

            for (uint32_t rtIndex = 0; rtIndex < fboIndex; rtIndex++)
            {
                //  Read back the Output Color.
                outputColor = *(vec4*)(pCtx->readTextureSubresource(pFbos[fboIndex + kRTCount * dpEnabled]->getColorTexture(rtIndex).get(), 0).data());
                if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Null Fbo State!");
            }

            if (dpEnabled)
            {
                //
                outputDepth = *(float*)(pCtx->readTextureSubresource(pFbos[fboIndex + kRTCount * dpEnabled]->getDepthStencilTexture().get(), 0).data());
                if (outputDepth != expectedDepth) return test_fail("Output Depth Not Equal to Expected Depth - Null Fbo State!");
            }
        }
    }


    //  Render all of the Framebuffers.
    for (uint32_t dpEnabled = 0; dpEnabled < 2; dpEnabled++)
    {
        for (uint32_t fboIndex = 0; fboIndex < kRTCount; fboIndex++)
        {
            pGS->setFbo(pFbos[fboIndex + kRTCount * dpEnabled]);

            //  Clear the Fbo.
            pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);
        }
    }


    //  
    return test_pass();
}

//  Test the Graphics Program Vars Basic.
testing_func(GraphicsStateTest, TestGraphicsProgramBasic)
{
    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();

    //  Create the first program.
    GraphicsProgram::SharedPtr pGP1 = GraphicsProgram::createFromFile("GPVS.vs.hlsl", "GPPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV1 = GraphicsVars::create(pGP1->getActiveVersion()->getReflector());

    //  Create the second program.
    GraphicsProgram::SharedPtr pGP2 = GraphicsProgram::createFromFile("MultipleOutVS.vs.hlsl", "MultipleOutPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV2 = GraphicsVars::create(pGP2->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();
    pCtx->setGraphicsState(pGS);

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the Framebuffer Object, using only Depth.
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setDepthStencilTarget(ResourceFormat::D32Float);
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  
    pCtx->clearFbo(pFbo.get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

    vec4 outputColor = vec4(0.0);
    vec4 expectedColor = vec4(0.5f);


    //  Set the Default Color Value.
    pGV1->getConstantBuffer("PerFrameCB")->setVariable("default_color_value", vec4(0.5));

    //  Set the Graphics Program and the Graphics Vars of the first program.
    pGS->setProgram(pGP1);
    pCtx->setGraphicsVars(pGV1);
    outputColor = vec4(0.25f);
    expectedColor = vec4(0.5f);

    // Draw.
    pCtx->draw(4, 0);

    //  Read back the Output Color.
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Graphics Vars!");

    pGV2->getConstantBuffer("PerFrameCB")->setVariable("color_value", vec4(0.25f));

    //  Set the Graphics Program and the Graphics Vars of the second program.
    pGS->setProgram(pGP2);
    pCtx->setGraphicsVars(pGV2);
    expectedColor = vec4(0.25f);

    // Draw.
    pCtx->draw(4, 0);

    //  Read back the Output Color.
    outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color - Graphics Vars!");

    //  
    return test_pass();
}

//  Test the Graphics Vars Change.
testing_func(GraphicsStateTest, TestGraphicsProgramChanges)
{
    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsState::SharedPtr pGS = GraphicsState::create();
    pCtx->setGraphicsState(pGS);

    //  Create the Framebuffer Object, using only Depth.
    Fbo::Desc fboDesc;
    for (uint32_t i = 0; i < kRTCount; i++)
    {
        fboDesc.setColorTarget(i, ResourceFormat::RGBA32Float);
    }
    Fbo::SharedPtr pFbo = FboHelper::create2D(1, 1, fboDesc);
    pGS->setFbo(pFbo);

    //  Create the pVAO.
    Vao::SharedPtr pVAO = TestHelper::getFullscreenQuadVao();
    pGS->setVao(pVAO);

    //  Create the first program.
    GraphicsProgram::SharedPtr pGPBase = GraphicsProgram::createFromFile("GPVS.vs.hlsl", "GPPS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGPBase->getActiveVersion()->getReflector());

    //  The Graphics Programs based on the same shaders.
    GraphicsProgram::SharedPtr pGPs[kRTCount];
    for (uint32_t i = 0; i < kRTCount; i++)
    {
        pGPs[i] = GraphicsProgram::createFromFile("GPVS.vs.hlsl", "GPPS.ps.hlsl");
    }

    //  Set the Graphics Vars - Constant for all the Shaders.
    pCtx->setGraphicsVars(pGV);
    for (uint32_t i = 0; i < kRTCount; i++)
    {
        pGV->getConstantBuffer("PerFrameCB")->setVariable("color_value" + std::to_string(i), vec4(0.1f));

        for (uint32_t j = 0; j < i; j++)
        {
            pGPs[i]->addDefine("_USE_COLOR_INPUT" + std::to_string(j));
        }
    }

    //  
    for (uint32_t currentProgramIndex = 0; currentProgramIndex < kRTCount; currentProgramIndex++)
    {
        //  
        pCtx->clearFbo(pFbo.get(), vec4(0.0), 1.0, 0, FboAttachmentType::All);

        //  Set the Graphics Program.
        pGS->setProgram(pGPs[currentProgramIndex]);

        vec4 outputColors[kRTCount];
        vec4 expectedColors[kRTCount];

        //  
        for (uint32_t expecColorIndex = 0; expecColorIndex < currentProgramIndex; expecColorIndex++)
        {
            expectedColors[expecColorIndex] = vec4(0.1f);
        }

        expectedColors[0] = vec4(0.1f) * (float)(currentProgramIndex + 1);

        //  
        pCtx->draw(4, 0);


        //  
        for (uint32_t verifyIndex = 0; verifyIndex < currentProgramIndex; verifyIndex++)
        {
            //  Read back the Output Color.
            outputColors[verifyIndex] = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(verifyIndex).get(), 0).data());
            if (outputColors[verifyIndex] != expectedColors[verifyIndex]) return test_fail("Output Color Not Equal to Expected Color - Graphics Vars!");
        }
    }

    return test_pass();
}

//
int main()
{
    GraphicsStateTest gpT;
    gpT.init(true);
    gpT.run();
    return 0;
}