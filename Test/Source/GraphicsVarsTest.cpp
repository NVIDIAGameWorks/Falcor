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
#include "GraphicsVarsTest.h"

//  Add the Tests.
void GraphicsVarsTest::addTests()
{
    //  Test Get/Set Constant Buffers.
    addTestToList<TestGetCBs>();
    addTestToList<TestSetCBsBasic>();


    //  Test Get/Set Samplers.
    addTestToList<TestSetSamplers>();
    addTestToList<TestGetSamplers>();


    addTestToList<TestSetTextures>();
    addTestToList<TestGetTextures>();

    addTestToList<TestSetStructuredBuffers>();
    addTestToList<TestGetStructuredBuffers>();

    addTestToList<TestGetShaderResourceViews>();
    addTestToList<TestSetShaderResourceViews>();
}

//  Write the Vertex Shader to File.
void GraphicsVarsTest::writeVertexShaderToFile(ShaderProgramMaker & sMaker)
{
    HLSLResourcesDesc inputDesc("val_position_input");
    inputDesc.semanticDesc.hasSemanticValue = true;
    inputDesc.semanticDesc.semanticValue = "POSITION";
    sMaker.getVSStage()->getHLSLShaderData().addInputVariable(inputDesc);

    sMaker.getVSStage()->getHLSLShaderData().defineVariable("float4", "position_value");
    sMaker.getVSStage()->getHLSLShaderData().useInputVariable("position_value", "val_position_input");
    sMaker.getVSStage()->getHLSLShaderData().setOutputVariable("vs_output", "position_value");

    HLSLResourcesDesc outputDesc("vs_output");
    outputDesc.semanticDesc.hasSemanticValue = true;
    outputDesc.semanticDesc.semanticValue = "SV_POSITION";
    sMaker.getVSStage()->getHLSLShaderData().addOutputVariable(outputDesc);

    //  
    std::string vsCode = sMaker.getVSStage()->writeShaderStageGLSLCode();
    vsCode = sMaker.getVSStage()->writeShaderStageHLSLCode();

    std::ofstream ofvs;
    ofvs.open("Data/VS.vs.hlsl", std::ofstream::trunc);
    ofvs << vsCode;
    ofvs.close();

}

//  Test Basic Get Constant Buffers.
testing_func(GraphicsVarsTest, TestGetCBs)
{
    //  Create the vertex and pixel shaders. 
    HLSLProgramMaker::HLSLProgramDesc sDesc;
    sDesc.hasVSStage = true;
    sDesc.hasPSStage = true;
    sDesc.cbsCount = 5;
    sDesc.bRegistersMaxPerSpace = 10;
    sDesc.bRegisterMaxSpace = 2;

    //  Create Shader Stages.
    HLSLProgramMaker sMaker(sDesc);

    //  Create the Constant Buffers.
    const std::vector<ConstantBufferData> & cbsData = sMaker.getConstantBuffers();

    //  Generate the Vertex Shader Code.
    writeVertexShaderToFile(sMaker);

    //  Generate the Pixel Shader Code.
    {

        HLSLResourcesDesc outputDesc("ps_output");
        outputDesc.semanticDesc.hasSemanticValue = true;
        outputDesc.semanticDesc.semanticValue = "SV_TARGET0";
        sMaker.getPSStage()->getHLSLShaderData().addOutputVariable(outputDesc);

        //  Create buffers.
        for (uint32_t i = 0; i < sDesc.cbsCount; i++)
        {
            sMaker.getPSStage()->getHLSLShaderData().addConstantBuffer(cbsData[i]);
        }

        sMaker.getPSStage()->getHLSLShaderData().defineVariable("float4", "base_output");
        sMaker.getPSStage()->getHLSLShaderData().executeOperate("base_output", "float4(0.0, 0.0, 0.0, 0.0)", "", "");
        sMaker.getPSStage()->getHLSLShaderData().setOutputVariable("ps_output", "base_output");

        //  
        std::string psCode = sMaker.getPSStage()->writeShaderStageHLSLCode();

        std::ofstream ofps;
        ofps.open("Data/PS.ps.hlsl", std::ofstream::trunc);
        ofps << psCode;
        ofps.close();

    }

    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("VS.vs.hlsl", "PS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  Check if we have the correct number of constants buffers.
    uint32_t assignedCBCount = 0;
    const ProgramVars::ResourceMap<ConstantBuffer>& assignedCBs = pGV->getAssignedCbs();
    for (const auto & currentCB : assignedCBs)
    {
        assignedCBCount++;
    }

    if (assignedCBCount == sDesc.cbsCount)
    {
        return test_pass()
    }
    else
    {
        return test_fail("Invalid Number of Constant Buffers Assigned!");
    }

}

//  Test Basic Set Constant Buffers.
testing_func(GraphicsVarsTest, TestSetCBsBasic)
{
    //  Create the vertex and pixel shaders. 
    HLSLProgramMaker::HLSLProgramDesc sDesc;
    sDesc.hasVSStage = true;
    sDesc.hasPSStage = true;
    sDesc.cbsCount = 10;
    sDesc.bRegistersMaxPerSpace = 10;
    sDesc.bRegisterMaxSpace = 2;

    //  Create Shader Stages.
    HLSLProgramMaker sMaker(sDesc);

    //  Create the Constant Buffers.
    const std::vector<ConstantBufferData> & cbsData = sMaker.getConstantBuffers();

    //  Generate the Vertex Shader Code.
    writeVertexShaderToFile(sMaker);

    //  Generate the Pixel Shader Code.
    {
        //  
        HLSLResourcesDesc outputDesc("ps_output");
        outputDesc.semanticDesc.hasSemanticValue = true;
        outputDesc.semanticDesc.semanticValue = "SV_TARGET0";
        sMaker.getPSStage()->getHLSLShaderData().addOutputVariable(outputDesc);

        //  
        sMaker.getPSStage()->getHLSLShaderData().defineVariable("float4", "base_output");
        sMaker.getPSStage()->getHLSLShaderData().executeOperate("base_output", "float4(0.0, 0.0, 0.0, 0.0)", "", "");
        sMaker.getPSStage()->getHLSLShaderData().setOutputVariable("ps_output", "base_output");

        //  Create buffers.
        for (uint32_t i = 0; i < sDesc.cbsCount; i++)
        {
            sMaker.getPSStage()->getHLSLShaderData().addConstantBuffer(cbsData[i]);
        }

        //  
        for (uint32_t i = 0; i < sDesc.cbsCount; i++)
        {
            if (cbsData[i].viewStructDesc().structVariables.size() && (i % 2) == 0)
            {
                //  Get the Struct Variable.
                uint32_t structVariableIndex = (uint32_t)cbsData[i].viewStructDesc().structVariables.size() - 1u;

                //  Access the Constant Buffer.
                sMaker.getPSStage()->getHLSLShaderData().accessConstantBuffer("cb_val" + std::to_string(i), cbsData[i].getCBVariable(), {}, "cb" + std::to_string(i) + "_val" + std::to_string(structVariableIndex));

                //  Define a variable.
                sMaker.getPSStage()->getHLSLShaderData().defineVariable("float4", "cb_val" + std::to_string(i));

                //  Add it to the single output.
                sMaker.getPSStage()->getHLSLShaderData().executeOperate("base_output", "base_output", "cb_val" + std::to_string(i), "+");

            }
        }

        //  
        std::string psCode = sMaker.getPSStage()->writeShaderStageGLSLCode();
        psCode = sMaker.getPSStage()->writeShaderStageHLSLCode();

        //  
        std::ofstream ofps;
        ofps.open("Data/PS.ps.hlsl", std::ofstream::trunc);
        ofps << psCode;
        ofps.close();

    }

    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("VS.vs.hlsl", "PS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();
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

    //  Check if we have the correct number of constants buffers.
    uint32_t assignedCBCount = 0;

    const ProgramVars::ResourceMap<ConstantBuffer>& assignedCBs = pGV->getAssignedCbs();

    for (const auto & currentCB : assignedCBs)
    {
        assignedCBCount++;
    }
    
    //  
    if (assignedCBCount != sDesc.cbsCount)
    {
        return test_fail("Invalid Number of Constant Buffers Assigned!");
    }

    //  
    vec4 expectedColor = vec4(0.0);

    //  Initialize all the variables.
    for (uint32_t i = 0; i < sDesc.cbsCount; i++)
    {
        //  
        if ((i % 2 == 0) && cbsData[i].viewStructDesc().structVariables.size())
        {
            //  Get the Struct Variable Index.
            uint32_t structVariableIndex = (uint32_t)cbsData[i].viewStructDesc().structVariables.size() - 1u;

            //  Get the 
            pGV->getConstantBuffer(cbsData[i].getCBVariable())->setVariable("cb" + std::to_string(i) + "_val" + std::to_string(structVariableIndex), vec4(0.1f));

            //  Add the Constant Buffer Index.
            expectedColor = expectedColor + vec4(0.1f);
        }
    }

    //  Clear the Fbo.
    pCtx->clearFbo(pGS->getFbo().get(), vec4(0.0f), 1.0f, 0, FboAttachmentType::All);

    //  Render.
    pCtx->draw(4, 0);

    //  Read back the Output Color.
    vec4 outputColor = *(vec4*)(pCtx->readTextureSubresource(pGS->getFbo()->getColorTexture(0).get(), 0).data());

    //  
    if (outputColor != expectedColor) return test_fail("Output Color Not Equal to Expected Color!");

    //
    return test_pass();

}

//  Test Get Samplers.
testing_func(GraphicsVarsTest, TestGetSamplers)
{
    srand(25);

    //  Create the vertex and pixel shaders. 
    HLSLProgramMaker::HLSLProgramDesc sDesc;
    sDesc.hasVSStage = true;
    sDesc.hasPSStage = true;

    //  Generate a set of samplers.
    sDesc.sRegistersMaxSpace = 1;
    sDesc.sRegistersMaxPerSpace = 20;
    sDesc.samplerCount = 10;
    sDesc.allowResourceArrays = true;

    //  Create Shader Stages.
    HLSLProgramMaker sMaker(sDesc);

    const std::vector<HLSLResourcesDesc> samplers = sMaker.getSamplers();

    //  Generate the Default Vertex Shader Code.
    writeVertexShaderToFile(sMaker);

    //  Generate the Pixel Shader Code.
    {
        //  Define and add the Pixel Shader Variables.
        HLSLResourcesDesc outputDesc("ps_output");
        outputDesc.semanticDesc.hasSemanticValue = true;
        outputDesc.semanticDesc.semanticValue = "SV_TARGET0";
        sMaker.getPSStage()->getHLSLShaderData().addOutputVariable(outputDesc);

        //  
        sMaker.getPSStage()->getHLSLShaderData().defineVariable("float4", "base_output");
        sMaker.getPSStage()->getHLSLShaderData().executeOperate("base_output", "float4(0.0, 0.0, 0.0, 0.0)", "", "");
        sMaker.getPSStage()->getHLSLShaderData().setOutputVariable("ps_output", "base_output");

        //  
        sMaker.getPSStage()->getHLSLShaderData().setSamplers(sMaker.getSamplers());


        //  Write the Shader Stage Code.
        std::string psCode = sMaker.getPSStage()->writeShaderStageGLSLCode();
        psCode = sMaker.getPSStage()->writeShaderStageHLSLCode();

        //  Write the Pixel Stage Code to file.
        std::ofstream ofps;
        ofps.open("Data/PS.ps.hlsl", std::ofstream::trunc);
        ofps << psCode;
        ofps.close();
    }

    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("VS.vs.hlsl", "PS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  
    //  Verify that the number of samplers is correct.
    uint32_t assignedSamplersCount = 0;

    //  Get the Assigned Samplers.
    const auto & assignedSamplers = pGV->getAssignedSamplers();
    //  
    for (const auto & currentSampler : assignedSamplers)
    {
        assignedSamplersCount = assignedSamplersCount + (uint32_t)currentSampler.second.size();
    }

    //  Verify that the Samplers exist.
    if (assignedSamplersCount != sDesc.samplerCount)
    {
        return test_fail("Invalid Samplers Count!");
    }


    //  
    return test_pass();
}

//  Test Set Samplers.
testing_func(GraphicsVarsTest, TestSetSamplers)
{
    srand(25);

    //  Create the vertex and pixel shaders. 
    HLSLProgramMaker::HLSLProgramDesc sDesc;
    sDesc.hasVSStage = true;
    sDesc.hasPSStage = true;

    //  Generate a set of samplers.
    sDesc.sRegistersMaxSpace = 1;
    sDesc.sRegistersMaxPerSpace = 20 ;
    sDesc.samplerCount = 10;
    sDesc.allowResourceArrays = true;

    //  Create Shader Stages.
    HLSLProgramMaker sMaker(sDesc);

    const std::vector<HLSLResourcesDesc> samplers = sMaker.getSamplers();

    //  Generate the Default Vertex Shader Code.
    writeVertexShaderToFile(sMaker);


    //  Generate the Pixel Shader Code.
    {
        //  Define and add the Pixel Shader Variables.
        HLSLResourcesDesc outputDesc("ps_output");
        outputDesc.semanticDesc.hasSemanticValue = true;
        outputDesc.semanticDesc.semanticValue = "SV_TARGET0";
        sMaker.getPSStage()->getHLSLShaderData().addOutputVariable(outputDesc);

        //  
        sMaker.getPSStage()->getHLSLShaderData().defineVariable("float4", "base_output");
        sMaker.getPSStage()->getHLSLShaderData().executeOperate("base_output", "float4(0.0, 0.0, 0.0, 0.0)", "", "");
        sMaker.getPSStage()->getHLSLShaderData().setOutputVariable("ps_output", "base_output");

        sMaker.getPSStage()->getHLSLShaderData().setSamplers(sMaker.getSamplers());

        //  Write the Shader Stage Code.
        std::string psCode = sMaker.getPSStage()->writeShaderStageGLSLCode();
        psCode = sMaker.getPSStage()->writeShaderStageHLSLCode();

        //  Write the Pixel Stage Code to file.
        std::ofstream ofps;
        ofps.open("Data/PS.ps.hlsl", std::ofstream::trunc);
        ofps << psCode;
        ofps.close();
    }

    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("VS.vs.hlsl", "PS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();
    pGS->setProgram(pGP);
    pCtx->setGraphicsState(pGS);
    pCtx->setGraphicsVars(pGV);

    //  
    //  Verify that the number of samplers is correct.
    uint32_t assignedSamplersCount = 0;

    //  Get the Assigned Samplers.
    const auto & assignedSamplers = pGV->getAssignedSamplers();
    //  
    for (const auto & currentSampler : assignedSamplers)
    {
        assignedSamplersCount = assignedSamplersCount + (uint32_t)currentSampler.second.size();
    }

    //  Verify that the Samplers exist.
    if (assignedSamplersCount != sDesc.samplerCount)
    {
        return test_fail("Invalid Samplers Count!");
    }


    //  
    return test_pass();
}

//  Test Get Textures.
testing_func(GraphicsVarsTest, TestGetTextures)
{
    //  Get the Render Context, and the Graphics Program. and the Graphics Vars.
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsProgram::SharedPtr pGP = GraphicsProgram::createFromFile("VS.vs.hlsl", "PS.ps.hlsl");
    GraphicsVars::SharedPtr pGV = GraphicsVars::create(pGP->getActiveVersion()->getReflector());

    //  Create the Graphics State.
    GraphicsState::SharedPtr pGS = GraphicsState::create();
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
    


    return test_pass();
}

//  Test Set Textures.
testing_func(GraphicsVarsTest, TestSetTextures)
{
    //  
    return test_pass();
}

//  Test Set Structured Buffers.
testing_func(GraphicsVarsTest, TestSetStructuredBuffers)
{
    return test_pass();
}

//  Test Get Structured Buffers.
testing_func(GraphicsVarsTest, TestGetStructuredBuffers)
{
    return test_pass();
}

//  Test Get Shader Resource Views.
testing_func(GraphicsVarsTest, TestGetShaderResourceViews)
{
    return test_pass();
}

//  Test Set Shader Resource Views.
testing_func(GraphicsVarsTest, TestSetShaderResourceViews)
{
    return test_pass();
}

//  Test Get Unordered Access Views.
testing_func(GraphicsVarsTest, TestSetUnorderedAccessViews)
{
    return test_pass();
}

//  Test Set Unordered Access Views.
testing_func(GraphicsVarsTest, TestGetUnorderedAccessViews)
{
    return test_pass();
}


int main()
{
    GraphicsVarsTest gvT;
    gvT.init(true);
    gvT.run();
    return 0;
}