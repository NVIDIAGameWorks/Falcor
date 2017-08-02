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
#include "HLSLShaderData.h"

//  
HLSLShaderData::HLSLShaderData()
{
    inSemantics.definesStructType = false;

    outSemantics.definesStructType = false;

}

//  
HLSLShaderData::HLSLShaderData(std::string inputStructVariableType, std::string outputStructVariableType)
{
    //
    inSemantics.definesStructType = true;
    inSemantics.structVariableType = inputStructVariableType;

    //  
    outSemantics.definesStructType = true;
    outSemantics.structVariableType = outputStructVariableType;
}

//  Add a Constant Buffer.
void HLSLShaderData::addConstantBuffer(const ConstantBufferData & cbDesc)
{
    mUCBs.push_back(cbDesc);
}

//  
void HLSLShaderData::setConstantBuffers(const std::vector<ConstantBufferData> & cbs)
{
    mUCBs = cbs;
}

//  Set the Samplers.
void HLSLShaderData::setSamplers(const std::vector<HLSLResourcesDesc> & samplers)
{
    mSamplers = samplers;
}

//  Set the Textures.
void HLSLShaderData::setTextures(const std::vector<HLSLResourcesDesc> & textures)
{
    mTextures = textures;
}

//  Set the Read Write Textures.
void HLSLShaderData::setRWTextures(const std::vector<HLSLResourcesDesc> & rwTextures)
{
    mRWTextures = rwTextures;
}

//  Set the Structured Buffers.
void HLSLShaderData::setStructuredBuffers(const std::vector<HLSLResourcesDesc> & structuredBuffers)
{
    mSBs = structuredBuffers;
}

//  Set the Read Write Structured Buffers.
void HLSLShaderData::setRWStructuredBuffers(const std::vector<HLSLResourcesDesc> & rwStructuredBuffers)
{
    mRWSBs = rwStructuredBuffers;
}

//  Set the Raw Buffers.
void HLSLShaderData::setRawBuffers(const std::vector<HLSLResourcesDesc> & rawBuffers)
{
    mRawBuffers = rawBuffers;
}

//  Set the Read Write Raw Buffers.
void HLSLShaderData::setRWRawBuffers(const std::vector<HLSLResourcesDesc> & rwRawBuffers)
{
    mRWRawBuffers = rwRawBuffers;
}

//  Add an Input Variable.
void HLSLShaderData::addInputVariable(const HLSLDescs::HLSLResourcesDesc & hlslResourceDesc)
{
    inSemantics.structVariables.push_back(hlslResourceDesc);
}

//  Add an Output Variable.
void HLSLShaderData::addOutputVariable(const HLSLDescs::HLSLResourcesDesc & hlslResourceDesc)
{
    outSemantics.structVariables.push_back(hlslResourceDesc);
}

//  
void HLSLShaderData::useInputVariable(const std::string & useVar, const std::string & inputVar)
{
    std::string inputVariable = getShaderInputStructType();
    std::transform(inputVariable.begin(), inputVariable.end(), inputVariable.begin(), ::tolower);

    inputVariables.push_back("    " + useVar + " = " + inputVariable + "." + inputVar + "; \n");
}

//  Access a Constant Buffer.
void HLSLShaderData::accessConstantBuffer(const std::string & outVar, const std::string & cbVar, std::vector<uint32_t> dimensions, const std::string & accessVar)
{
    //  
    uint32_t cbIndex = 0;
    for (cbIndex = 0; mUCBs.size(); cbIndex++)
    {
        if (mUCBs[cbIndex].getCBVariable() == cbVar)
        {
            break;
        }
    }

    if (cbIndex == (uint32_t)mUCBs.size())
    {
        return;
    }

    //  Change the Access Style Depending on the Constant Buffer Maker.
    if (mUCBs[cbIndex].getIsGlobalCBType())
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
void HLSLShaderData::loadTexture(const std::string & outVar, const std::string & textureVar, const std::string & loadTextureAccess)
{
    accessVariables.push_back("    " + outVar + " = " + textureVar + ".Load(" + loadTextureAccess + ");");
}

//  Access the Texture by sample.
void HLSLShaderData::sampleTexture(const std::string & outVar, const std::string & textureVar, const std::string & samplerVar, const std::string & sampleTextureAccess)
{
    accessVariables.push_back("    " + outVar + " = " + textureVar + "." + "Sample(" + samplerVar + "," + sampleTextureAccess + ");");
}

//  Load a Texture from an Array.
void HLSLShaderData::loadTextureFromArray(const std::string & outVar, const std::string & arrayOfTexturesVar, const std::string & arrayAccess, const std::string & loadTextureAccess)
{
    accessVariables.push_back("    " + outVar + " = " + arrayOfTexturesVar + arrayAccess + ".Load(" + loadTextureAccess + ");");
}

//  Sample a Texture from an Array.
void HLSLShaderData::sampleTextureFromArray(const std::string & outVar, const std::string & arrayOfTexturesVar, const std::string & arrayAccess, const std::string & samplerAccessVariable, const std::string & samplerAccessLocation)
{
    accessVariables.push_back("    " + outVar + " = " + arrayOfTexturesVar + arrayAccess + "." + "Sample(" + samplerAccessVariable + "," + samplerAccessLocation + ");");
}

//  Write the code to define variable.  
void HLSLShaderData::defineVariable(const std::string & newVariableType, const std::string & outVariable)
{
    defineVariables.push_back("    " + newVariableType + " " + outVariable + "; ");
}

//  Execute Operate Variables.
void HLSLShaderData::executeOperate(const std::string & outVariable, const std::string & varleft, const std::string & varright, const std::string & op)
{
    std::string operation = "    " + outVariable + " = " + varleft + " " + op + " " + varright + "; \n";
    operateVariables.push_back(operation);
}

//  Set the Output Variable.
void HLSLShaderData::setOutputVariable(const std::string & outVariable, const std::string & inVariable)
{
    std::string outputVariable = getShaderOutputStructType();

    std::transform(outputVariable.begin(), outputVariable.end(), outputVariable.begin(), ::tolower);

    std::string setOutputCode = "    " + outputVariable + "." + outVariable + " = " + inVariable + "; \n";

    outputVariables.push_back(setOutputCode);
}


//  Return the Input Struct Type.
std::string HLSLShaderData::getShaderInputStructType() const
{
    return inSemantics.structVariableType;
}

//  Return the Output Struct Type.
std::string HLSLShaderData::getShaderOutputStructType() const
{
    return outSemantics.structVariableType;
}


//  Get Constant Buffer Resources.
const std::vector<ConstantBufferData> & HLSLShaderData::getConstantBuffers() const
{
    return mUCBs;
}

//  
const std::vector<std::string> & HLSLShaderData::getUseInputsCode() const
{
    //  Return Input Variables.
    return inputVariables;
}

const std::vector<std::string> & HLSLShaderData::getAccessResourcesCode() const
{
    //  Return Access Variables.
    return accessVariables;
}

const std::vector<std::string> & HLSLShaderData::getCreateLocalVariablesCode() const
{
    //  Define Variables.
    return defineVariables;
}

const std::vector<std::string> & HLSLShaderData::getOperateVariablesCode() const
{
    //  Operate Variables.
    return operateVariables;
}

const std::vector<std::string> & HLSLShaderData::getWriteOutputsCode() const
{
    //  Output Variables.
    return outputVariables;
}

//  Return the Samplers.
const std::vector<HLSLDescs::HLSLResourcesDesc> & HLSLShaderData::getSamplers() const
{
    return mSamplers;
}

//  Return the input semantics.
const HLSLStructDesc & HLSLShaderData::getInputStructDesc() const
{
    return inSemantics;
}

//  Return the output semantics.
const HLSLStructDesc & HLSLShaderData::getOutputStructDesc() const
{
    return outSemantics;
}

//  Return the Structured Buffer Resources.
const std::vector<Falcor::HLSLDescs::HLSLResourcesDesc> & HLSLShaderData::getStructuredBufferResources() const
{
    return mSBs;
}

//  Return the Read Write Structured Buffers.
const std::vector<Falcor::HLSLDescs::HLSLResourcesDesc> & HLSLShaderData::getRWStructuredBufferResources() const
{
    return mRWSBs;
}

//  Return the Read Write Textures.
const std::vector<Falcor::HLSLDescs::HLSLResourcesDesc> & HLSLShaderData::getTextures() const
{
    return mTextures;
}

//  Return the Textures.
const std::vector<Falcor::HLSLDescs::HLSLResourcesDesc> & HLSLShaderData::getRWTextures() const
{
    return mRWTextures;
}

