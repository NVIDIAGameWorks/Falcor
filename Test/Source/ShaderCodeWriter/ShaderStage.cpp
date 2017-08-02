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

#include "ShaderStage.h"


//  Shader Stage Maker Constructor.
ShaderStage::ShaderStage(const std::string & inputStructVariableType, const std::string & outputStructVariableType)
{
    hlslShaderData = HLSLShaderData(inputStructVariableType, outputStructVariableType);
    glslShaderData = GLSLShaderData(inputStructVariableType, outputStructVariableType);
}


HLSLShaderData & ShaderStage::getHLSLShaderData()
{
    return hlslShaderData;

}

const HLSLShaderData & ShaderStage::viewHLSLShaderData() const
{
    return hlslShaderData;
}

GLSLShaderData & ShaderStage::getGLSLShaderData()
{
    return glslShaderData;
}

const GLSLShaderData & ShaderStage::viewGLSLShaderData() const
{
    return glslShaderData;
}

//  Write Shader Code.
std::string ShaderStage::writeShaderCode() const
{
    std::string scode = "";

    //  Get the code for the local variables defined.
    scode = scode + CommonShaderDescs::getCodeBlock(viewHLSLShaderData().getCreateLocalVariablesCode()) + " \n";

    //  Get the input variables.
    scode = scode + CommonShaderDescs::getCodeBlock(viewHLSLShaderData().getUseInputsCode()) + " \n";

    //  Get the access variables.
    scode = scode + CommonShaderDescs::getCodeBlock(viewHLSLShaderData().getAccessResourcesCode()) + " \n";

    //  Get the operate variables.
    scode = scode + CommonShaderDescs::getCodeBlock(viewHLSLShaderData().getOperateVariablesCode()) + " \n";

    //  Get the output variables.
    scode = scode + CommonShaderDescs::getCodeBlock(viewHLSLShaderData().getWriteOutputsCode()) + " \n";

    //  Return the shader code.
    return scode;

}

//  Vertex Shader Stage Maker Constructor.
VertexShaderStage::VertexShaderStage(const std::string & inputStructVariableType, const std::string & outputStructVariableType) : ShaderStage(inputStructVariableType, outputStructVariableType)
{

}

//  Write the Vertex Shader Stage Code Data.
std::string VertexShaderStage::writeShaderStageHLSLCode() const
{
    std::string vscode = "\n";

    vscode = vscode + mHLSLCodeWriter.writeInputStructCode(viewHLSLShaderData());
    vscode = vscode + mHLSLCodeWriter.writeOutputStructCode(viewHLSLShaderData());
    vscode = vscode + " \n";

    vscode = vscode + mHLSLCodeWriter.writeConstantBuffersDeclareCode(viewHLSLShaderData());
    vscode = vscode + mHLSLCodeWriter.writeBaseResourcesDeclareCode(viewHLSLShaderData());
    vscode = vscode + " \n";

    //  
    std::string inputVariable = viewHLSLShaderData().getShaderInputStructType();
    std::string outputVariable = viewHLSLShaderData().getShaderOutputStructType();

    //      
    std::transform(inputVariable.begin(), inputVariable.end(), inputVariable.begin(), ::tolower);
    std::transform(outputVariable.begin(), outputVariable.end(), outputVariable.begin(), ::tolower);

    //  
    vscode = vscode + "//   Vertex Shader Code. \n";
    vscode = vscode + viewHLSLShaderData().getShaderOutputStructType() + " main ( " + viewHLSLShaderData().getShaderInputStructType() + " " + inputVariable + " ) \n";
    vscode = vscode + "{ \n";
    vscode = vscode + "    " + viewHLSLShaderData().getShaderOutputStructType() + " " + outputVariable + "; \n";

    vscode = vscode + writeShaderCode();

    vscode = vscode + "    " + "return " + outputVariable + "; \n";
    vscode = vscode + "} \n";
    vscode = vscode + "\n";

    return vscode;
}

//  Write the Shader Stage HLSL Code.
std::string VertexShaderStage::writeShaderStageGLSLCode() const
{
    //  
    std::string vscode = "\n";

    vscode = vscode + "#version 450 \n \n";
    vscode = vscode + "#define VULKAN 100 \n \n";

    vscode = vscode + "// Vertex Shader Code.";
    vscode = vscode + "void main () \n";
    vscode = vscode + "{ \n";

    //  Write the Contents of the Code.
    vscode = vscode + writeShaderCode();

    vscode = vscode + "} \n";

    //  
    return vscode;
}


//  
HullShaderStage::HullShaderStage(const std::string & inputStructVariableType, const std::string & outputStructVariableType)
{

}

std::string HullShaderStage::writeShaderStageHLSLCode() const
{
    return "";
}

std::string HullShaderStage::writeShaderStageGLSLCode() const
{
    return "";
}


//  
DomainShaderStage::DomainShaderStage(const std::string & inputStructVariableType, const std::string & outputStructVariableType)
{

}

std::string DomainShaderStage::writeShaderStageHLSLCode() const
{
    return "";
}

std::string DomainShaderStage::writeShaderStageGLSLCode() const
{
    return "";
}


//  
GeometryShaderStage::GeometryShaderStage(const std::string & inputStructVariableType, const std::string & outputStructVariableType)
{

}

std::string GeometryShaderStage::writeShaderStageHLSLCode() const
{
    return "";
}

std::string GeometryShaderStage::writeShaderStageGLSLCode() const
{
    return "";
}


//  Pixel Shader Stage.
PixelShaderStage::PixelShaderStage(const std::string & inputStructVariableType, const std::string & outputStructVariableType) : ShaderStage(inputStructVariableType, outputStructVariableType)
{

}

//  Write Pixel Shader Stage Code.
std::string PixelShaderStage::writeShaderStageHLSLCode() const
{
    //  
    std::string pscode = "\n";

    //  Add the Shader Input and Output Struct Code.
    pscode = pscode + mHLSLCodeWriter.writeInputStructCode(viewHLSLShaderData());
    pscode = pscode + mHLSLCodeWriter.writeOutputStructCode(viewHLSLShaderData());
    pscode = pscode + "\n";

    //  Add the Constant Buffer and Shader Resource.
    pscode = pscode + mHLSLCodeWriter.writeConstantBuffersDeclareCode(viewHLSLShaderData());
    pscode = pscode + mHLSLCodeWriter.writeBaseResourcesDeclareCode(viewHLSLShaderData());
    pscode = pscode + "\n";

    //  Create the input variable - just the input type in lowercase format.
    std::string inputVariable = viewHLSLShaderData().getShaderInputStructType();
    std::transform(inputVariable.begin(), inputVariable.end(), inputVariable.begin(), ::tolower);

    //  Create the output variable - just the output variable in lowercase format.
    std::string outputVariable = viewHLSLShaderData().getShaderOutputStructType();
    std::transform(outputVariable.begin(), outputVariable.end(), outputVariable.begin(), ::tolower);

    //  Create the main pixel shader function,
    pscode = pscode + "// Pixel Shader Code. \n";
    pscode = pscode + viewHLSLShaderData().getShaderOutputStructType() + " main ( " + viewHLSLShaderData().getShaderInputStructType() + " " + inputVariable + " ) \n";
    pscode = pscode + "{ \n";

    //  Define the output variable.
    pscode = pscode + "    " + viewHLSLShaderData().getShaderOutputStructType() + " " + outputVariable + "; \n \n";

    //  Write the Contents of the Code.
    pscode = pscode + writeShaderCode();

    //  Return the input variable.
    pscode = pscode + "    " + "return " + outputVariable + "; \n \n";

    //  End the main pixel shader function.
    pscode = pscode + "} \n";
    pscode = pscode + "\n";

    //  
    return pscode;
}


//  Write Pixel Shader Stage Code.
std::string PixelShaderStage::writeShaderStageGLSLCode() const
{
    //  
    std::string pscode = "";

    pscode = pscode + "#version 450 \n \n";
    pscode = pscode + "#define VULKAN 100 \n \n";

    //  Add the Uniform Buffer and Shader Resources.
    pscode = pscode + mGLSLCodeWriter.writeUniformResourcesCode(viewGLSLShaderData());
    pscode = pscode + mGLSLCodeWriter.writeBaseResourcesDeclareCode(viewGLSLShaderData());

    //  Create the main pixel shader function,
    pscode = pscode + "// Pixel Shader Code. \n";
    pscode = pscode + "void main() \n";
    pscode = pscode + "{ \n";

    //  Write the Contents of the Code.
    pscode = pscode + writeShaderCode();

    //  End the main pixel shader function.
    pscode = pscode + "} \n";
    pscode = pscode + "\n";

    //  
    return pscode;
}


//  Compute Shader Stage.
ComputeShaderStage::ComputeShaderStage(const std::string & inputStructVariableType, const std::string & ouputStructVariableType)
{

}

//  Write Shader Stage.
std::string ComputeShaderStage::writeShaderStageHLSLCode() const
{
    return "";
}

//  Write Shader Stage.
std::string ComputeShaderStage::writeShaderStageGLSLCode() const
{
    return "";
}
