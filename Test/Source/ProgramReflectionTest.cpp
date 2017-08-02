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
#include "ProgramReflectionTest.h"

using namespace Falcor;
//  
void ProgramReflectionTest::addTests()
{
    addTestToList<TestHLSLBasics>();
    addTestToList<TestGLSLBasics>();
}

//  Test the HLSL Basics.
testing_func(ProgramReflectionTest, TestHLSLBasics)
{
    srand(23);
    //  Create the vertex and pixel shaders. 
    HLSLProgramMaker::HLSLProgramDesc sDesc;
    sDesc.hasVSStage = true;
    sDesc.hasPSStage = true;
    sDesc.allowExplicitSpaces = true;
    sDesc.allowResourceArrays = true;
    sDesc.cbsCount = 5;
    sDesc.bRegistersMaxPerSpace = 10;
    sDesc.bRegisterMaxSpace = 2;


    sDesc.samplerCount = 15;
    sDesc.sRegistersMaxPerSpace = 5;
    sDesc.sRegistersMaxSpace = 4;

    sDesc.texturesCount = 10;
    sDesc.sbsCount = 10;
    sDesc.tRegistersMaxPerSpace = 40;
    sDesc.tRegistersMaxSpace = 1;


   //  Create Shader Stages.
    HLSLProgramMaker sMaker(sDesc);
    
    //  
    HLSLResourcesDesc inputVertDesc("val_position_input");
    inputVertDesc.semanticDesc.hasSemanticValue = true;
    inputVertDesc.semanticDesc.semanticValue = "POSITION";
    sMaker.getVSStage()->getHLSLShaderData().addInputVariable(inputVertDesc);
    
    sMaker.getVSStage()->getHLSLShaderData().defineVariable("float4", "position_value");
    sMaker.getVSStage()->getHLSLShaderData().useInputVariable("position_value", "val_position_input");
    sMaker.getVSStage()->getHLSLShaderData().setOutputVariable("vs_output", "position_value");

    HLSLResourcesDesc outputVertDesc("vs_output");
    outputVertDesc.semanticDesc.hasSemanticValue = true;
    outputVertDesc.semanticDesc.semanticValue = "SV_POSITION";
    sMaker.getVSStage()->getHLSLShaderData().addOutputVariable(outputVertDesc);

    //  
    std::string vsCode = sMaker.getVSStage()->writeShaderStageHLSLCode();

    std::ofstream ofvs;
    ofvs.open("Data/VS.vs.hlsl", std::ofstream::trunc);
    ofvs << vsCode;
    ofvs.close();


    HLSLDescs::HLSLResourcesDesc outputPixelDesc("ps_output");
    outputPixelDesc.semanticDesc.hasSemanticValue = true;
    outputPixelDesc.semanticDesc.semanticValue = "SV_TARGET0";
    sMaker.getPSStage()->getHLSLShaderData().addOutputVariable(outputPixelDesc);


    //  
    sMaker.getPSStage()->getHLSLShaderData().setConstantBuffers(sMaker.getConstantBuffers());
    sMaker.getPSStage()->getHLSLShaderData().setTextures(sMaker.getTextures());
    sMaker.getPSStage()->getHLSLShaderData().setSamplers(sMaker.getSamplers());
    sMaker.getPSStage()->getHLSLShaderData().setStructuredBuffers(sMaker.getStructuredBuffers());
    sMaker.getPSStage()->getHLSLShaderData().setRawBuffers(sMaker.getRawBuffers());
    
    //  
    sMaker.getPSStage()->getHLSLShaderData().defineVariable("float4", "base_output");
    sMaker.getPSStage()->getHLSLShaderData().executeOperate("base_output", "float4(0.0, 0.0, 0.0, 0.0)", "", "");
    sMaker.getPSStage()->getHLSLShaderData().setOutputVariable("ps_output", "base_output");

    //  
    std::string psCode = sMaker.getPSStage()->writeShaderStageHLSLCode();

    std::ofstream ofps;
    ofps.open("Data/PS.ps.hlsl", std::ofstream::trunc);
    ofps << psCode;
    ofps.close();


    return test_pass();
}


//  Test the GLSL Basics.
testing_func(ProgramReflectionTest, TestGLSLBasics)
{

    //  Create the vertex and pixel shaders. 
    GLSLProgramMaker::GLSLProgramDesc sDesc;
    sDesc.hasVSStage = true;
    sDesc.hasPSStage = true;
    sDesc.ubsCount = 15;
    sDesc.maxBindingsPerSet = 10;
    sDesc.maxSets = 2;

    //  Create Shader Stages.
    GLSLProgramMaker sMaker(sDesc);


    //  Create the input vert desc.
    GLSLDescs::GLSLResourcesDesc inputVertDesc("val_position_input");
    sMaker.getVSStage()->getGLSLShaderData().addInputVariable(inputVertDesc);

    //  
    sMaker.getVSStage()->getGLSLShaderData().defineVariable("float4", "position_value");
    sMaker.getVSStage()->getGLSLShaderData().useInputVariable("position_value", "val_position_input");
    sMaker.getVSStage()->getGLSLShaderData().setOutputVariable("vs_output", "position_value");

    //  
    GLSLDescs::GLSLResourcesDesc outputVertDesc("vs_output");
    sMaker.getVSStage()->getGLSLShaderData().addOutputVariable(outputVertDesc);

    //  
    std::string vsCode = sMaker.getVSStage()->writeShaderStageGLSLCode();

    //  Output the Vertes Shader Stage.
    std::ofstream ofvs;
    ofvs.open("Data/VS.vert", std::ofstream::trunc);
    ofvs << vsCode;
    ofvs.close();


    //  Create the Constant Buffers.
    const std::vector<UniformBufferData> & cbsData = sMaker.getUniformBuffers();

    GLSLDescs::GLSLResourcesDesc outputFragDesc("ps_output");
    sMaker.getPSStage()->getGLSLShaderData().addOutputVariable(outputFragDesc);

    //  Create buffers.
    for (uint32_t i = 0; i < cbsData.size(); i++)
    {
        sMaker.getPSStage()->getGLSLShaderData().addUniformBuffer(cbsData[i]);
    }

    sMaker.getPSStage()->getGLSLShaderData().defineVariable("float4", "base_output");
    sMaker.getPSStage()->getGLSLShaderData().executeOperate("base_output", "float4(0.0, 0.0, 0.0, 0.0)", "", "");

    //  
    std::string psCode = sMaker.getPSStage()->writeShaderStageGLSLCode();

    //  Output the Fragment Shader.
    std::ofstream ofps;
    ofps.open("Data/PS.frag", std::ofstream::trunc);
    ofps << psCode;
    ofps.close();

    return test_pass();
}


int main()
{
    ProgramReflectionTest prT;
    prT.init(true);
    prT.run();
    return 0;
}