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
#include "GLSLShaderData.h"

//  
GLSLShaderData::GLSLShaderData()
{
    inSemantics.definesStructType = false;

    outSemantics.definesStructType = false;
}

//  
GLSLShaderData::GLSLShaderData(std::string inputStructVariableType, std::string outputStructVariableType)
{
    //
    inSemantics.definesStructType = true;
    inSemantics.structVariableType = inputStructVariableType;

    //  
    outSemantics.definesStructType = true;
    outSemantics.structVariableType = outputStructVariableType;
}

//  Add a Constant Buffer.
void GLSLShaderData::addUniformBuffer(const UniformBufferData & ubDesc)
{
    mUBs.push_back(ubDesc);
}


//  
void GLSLShaderData::setTextures(const std::vector<GLSLResourcesDesc> & textures)
{
    mTextures = textures;
}

void GLSLShaderData::setSSBOs(const std::vector<GLSLResourcesDesc> & ssbos)
{

    mSSBOs = ssbos;
}

//
void GLSLShaderData::setImages(const std::vector<GLSLResourcesDesc> & images)
{
    mImages = images;
}

//  
void GLSLShaderData::setSamplers(const std::vector<GLSLResourcesDesc> & samplers)
{
    mSamplers = samplers;
}


//  Add an Input Variable.
void GLSLShaderData::addInputVariable(const GLSLDescs::GLSLResourcesDesc & hlslResourceDesc)
{
    inSemantics.structVariables.push_back(hlslResourceDesc);
}

//  Add an Output Variable.
void GLSLShaderData::addOutputVariable(const GLSLDescs::GLSLResourcesDesc & hlslResourceDesc)
{
    outSemantics.structVariables.push_back(hlslResourceDesc);
}

//  
void GLSLShaderData::useInputVariable(const std::string & useVar, const std::string & inputVar)
{
    std::string inputVariable = getShaderInputStructType();
    std::transform(inputVariable.begin(), inputVariable.end(), inputVariable.begin(), ::tolower);

    inputVariables.push_back("    " + useVar + " = " + inputVariable + "." + inputVar + "; \n");
}

//  Access a Constant Buffer.
void GLSLShaderData::accessUniformBuffer(const std::string & outVar, const std::string & cbVar, std::vector<uint32_t> dimensions, const std::string & accessVar)
{
    //  
    uint32_t cbIndex = 0;
    for (cbIndex = 0; mUBs.size(); cbIndex++)
    {
        if (mUBs[cbIndex].getUBVariable() == cbVar)
        {
            break;
        }
    }

    if (cbIndex == (uint32_t)mUBs.size())
    {
        return;
    }

    //  Change the Access Style Depending on the Constant Buffer Maker.
    if (mUBs[cbIndex].getIsGlobalUBType())
    {
        std::string accessVariable = "    " + outVar + " = " + accessVar + "; \n";
        accessVariables.push_back(accessVariable);
    }
    else
    {
        std::string arrayAccess = "";

        //  Add the Array Access.
        for (uint32_t i = 0; i < dimensions.size(); i++)
        {
            arrayAccess = arrayAccess + "[" + std::to_string(dimensions[i]) + "]";
        }

        arrayAccess = arrayAccess + ".";

        std::string accessVariable = "    " + outVar + " = " + cbVar + arrayAccess + accessVar + "; \n";

        accessVariables.push_back(accessVariable);
    }

}

//  Access the Texture by load.
void GLSLShaderData::loadTexture(const std::string & outVar, const std::string & textureVar, const std::string & loadTextureAccess)
{
    accessVariables.push_back("    " + outVar + " = " + textureVar + ".Load(" + loadTextureAccess + ");");
}

//  Access the Texture by sample.
void GLSLShaderData::sampleTexture(const std::string & outVar, const std::string & textureVar, const std::string & samplerVar, const std::string & sampleTextureAccess)
{
    accessVariables.push_back("    " + outVar + " = " + textureVar + "." + "Sample(" + samplerVar + "," + sampleTextureAccess + ");");
}

//  Load a Texture from an Array.
void GLSLShaderData::loadTextureFromArray(const std::string & outVar, const std::string & arrayOfTexturesVar, const std::string & arrayAccess, const std::string & loadTextureAccess)
{
    accessVariables.push_back("    " + outVar + " = " + arrayOfTexturesVar + arrayAccess + ".Load(" + loadTextureAccess + ");");
}

//  Sample a Texture from an Array.
void GLSLShaderData::sampleTextureFromArray(const std::string & outVar, const std::string & arrayOfTexturesVar, const std::string & arrayAccess, const std::string & samplerAccessVariable, const std::string & samplerAccessLocation)
{
    accessVariables.push_back("    " + outVar + " = " + arrayOfTexturesVar + arrayAccess + "." + "Sample(" + samplerAccessVariable + "," + samplerAccessLocation + ");");
}

//  Write the code to define variable.  
void GLSLShaderData::defineVariable(const std::string & newVariableType, const std::string & outVariable)
{
    defineVariables.push_back("    " + newVariableType + " " + outVariable + "; ");
}

//  Execute Operate Variables.
void GLSLShaderData::executeOperate(const std::string & outVariable, const std::string & varleft, const std::string & varright, const std::string & op)
{
    std::string operation = "    " + outVariable + " = " + varleft + " " + op + " " + varright + "; \n";
    operateVariables.push_back(operation);
}

//  Set the Output Variable.
void GLSLShaderData::setOutputVariable(const std::string & outVariable, const std::string & inVariable)
{

    std::string outputVariable = getShaderOutputStructType();
    std::transform(outputVariable.begin(), outputVariable.end(), outputVariable.begin(), ::tolower);

    std::string setOutputCode = "    " + outputVariable + "." + outVariable + " = " + inVariable + "; \n";

    outputVariables.push_back(setOutputCode);
}


//  Return the Input Struct Type.
std::string GLSLShaderData::getShaderInputStructType() const
{
    return inSemantics.structVariableType;
}

//  Return the Output Struct Type.
std::string GLSLShaderData::getShaderOutputStructType() const
{
    return outSemantics.structVariableType;
}


//  
const std::vector<UniformBufferData> & GLSLShaderData::getUniformBuffers() const
{
    return mUBs;
}


//  
const std::vector<std::string> & GLSLShaderData::getUseInputsCode() const
{
    //  Return Input Variables.
    return inputVariables;
}

const std::vector<std::string> & GLSLShaderData::getAccessResourcesCode() const
{
    //  Return Access Variables.
    return accessVariables;
}

const std::vector<std::string> & GLSLShaderData::getCreateLocalVariablesCode() const
{
    //  Define Variables.
    return defineVariables;
}

const std::vector<std::string> & GLSLShaderData::getOperateVariablesCode() const
{
    //  Operate Variables.
    return operateVariables;
}

const std::vector<std::string> & GLSLShaderData::getWriteOutputsCode() const
{
    //  Output Variables.
    return outputVariables;
}

//  Return the input semantics.
const GLSLStructDesc & GLSLShaderData::getInputStructDesc() const
{
    return inSemantics;
}

//  Return the output semantics.
const GLSLStructDesc & GLSLShaderData::getOutputStructDesc() const
{
    return outSemantics;
}

