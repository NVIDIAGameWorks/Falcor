// /***************************************************************************
// # Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
// #
// # Redistribution and use in source and binary forms, with or without
// # modification, are permitted provided that the following conditions
// # are met:
// #  * Redistributions of source code must retain the above copyright
// #    notice, this list of conditions and the following disclaimer.
// #  * Redistributions in binary form must reproduce the above copyright
// #    notice, this list of conditions and the following disclaimer in the
// #    documentation and/or other materials provided with the distribution.
// #  * Neither the name of NVIDIA CORPORATION nor the names of its
// #    contributors may be used to endorse or promote products derived
// #    from this software without specific prior written permission.
// #
// # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ***************************************************************************/
// #include "ShaderCommonData.h"

// //  
// ShaderCommonData::ShaderCommonData()
// {
//     inSemantics.definesStructType = false;

//     outSemantics.definesStructType = false;

// }

// //  
// ShaderCommonData::ShaderCommonData(std::string inputStructVariableType, std::string outputStructVariableType)
// {
//     //
//     inSemantics.definesStructType = true;
//     inSemantics.structVariableType = inputStructVariableType;

//     //  
//     outSemantics.definesStructType = true;
//     outSemantics.structVariableType = outputStructVariableType;
// }

// //  Add a Constant Buffer.
// void ShaderCommonData::addUCBuffer(const UCBufferData & cbDesc)
// {
//     mUCBs.push_back(cbDesc);
// }

// //  Add a Constant Buffer 
// void ShaderCommonData::addResourceVariable(const ShaderCodeWriterDescs::ShaderResourceDesc & shaderResourceDesc)
// {
//     mSRs.push_back(shaderResourceDesc);
// }


// //  Add an Input Variable.
// void ShaderCommonData::addInputVariable(const ShaderCodeWriterDescs::ShaderResourceDesc & shaderResourceDesc)
// {
//     inSemantics.structVariables.push_back(shaderResourceDesc);
// }

// //  Add an Output Variable.
// void ShaderCommonData::addOutputVariable(const ShaderCodeWriterDescs::ShaderResourceDesc & shaderResourceDesc) 
// {
//     outSemantics.structVariables.push_back(shaderResourceDesc);
// }

// //  
// void ShaderCommonData::useInputVariable(const std::string & useVar, const std::string & inputVar)
// {
//     std::string inputVariable = getShaderInputStructType();
//     std::transform(inputVariable.begin(), inputVariable.end(), inputVariable.begin(), ::tolower);

//     inputVariables.push_back("    " + useVar + " = " + inputVariable + "." + inputVar + "; \n");
// }

// //  Access a Constant Buffer.
// void ShaderCommonData::accessConstantBuffer(const std::string & outVar, const std::string & cbVar, std::vector<uint32_t> dimensions, const std::string & accessVar)
// {
//     //  
//     uint32_t cbIndex = 0;
//     for (cbIndex = 0; mUCBs.size(); cbIndex++)
//     {
//         if (mUCBs[cbIndex].getUCBVariable() == cbVar)
//         {
//             break;
//         }
//     }

//     if (cbIndex == (uint32_t)mUCBs.size())
//     {
//         return;
//     }

//     //  Change the Access Style Depending on the Constant Buffer Maker.
//     if (mUCBs[cbIndex].getIsGlobalUCBType())
//     {
//         std::string accessVariable = "    " + outVar + " = " + accessVar + "; \n";
//         accessVariables.push_back(accessVariable);
//     }
//     else
//     {
//         std::string arrayAccess = "";

//         //  Add the Array Access.
//         for (uint32_t i = 0; i < dimensions.size(); i++)
//         {
//             arrayAccess = arrayAccess + "[" + std::to_string(dimensions[i]) + "]";
//         }

//         arrayAccess = arrayAccess + ".";
        
//         std::string accessVariable = "    " + outVar + " = " + cbVar + arrayAccess + accessVar + "; \n";

//         accessVariables.push_back(accessVariable);
//     }
        
// }

// //  Access the Texture by load.
// void ShaderCommonData::loadTexture(const std::string & outVar, const std::string & textureVar, const std::string & loadTextureAccess)
// {
//     accessVariables.push_back("    " + outVar + " = " + textureVar + ".Load(" + loadTextureAccess + ");");
// }

// //  Access the Texture by sample.
// void ShaderCommonData::sampleTexture(const std::string & outVar, const std::string & textureVar, const std::string & samplerVar, const std::string & sampleTextureAccess)
// {
//     accessVariables.push_back("    " + outVar + " = " + textureVar + "." + "Sample(" + samplerVar + "," + sampleTextureAccess + ");");
// }

// //  Load a Texture from an Array.
// void ShaderCommonData::loadTextureFromArray(const std::string & outVar, const std::string & arrayOfTexturesVar, const std::string & arrayAccess, const std::string & loadTextureAccess)
// {
//     accessVariables.push_back("    " + outVar + " = " + arrayOfTexturesVar + arrayAccess + ".Load(" + loadTextureAccess + ");");
// }

// //  Sample a Texture from an Array.
// void ShaderCommonData::sampleTextureFromArray(const std::string & outVar, const std::string & arrayOfTexturesVar, const std::string & arrayAccess, const std::string & samplerAccessVariable, const std::string & samplerAccessLocation)
// {
//     accessVariables.push_back("    " + outVar + " = " + arrayOfTexturesVar + arrayAccess + "." + "Sample(" + samplerAccessVariable +  "," + samplerAccessLocation + ");");
// }

// //  Write the code to define variable.  
// void ShaderCommonData::defineVariable(const std::string & newVariableType, const std::string & outVariable)
// {
//     defineVariables.push_back("    " + newVariableType + " " + outVariable + "; ");
// }

// //  Execute Operate Variables.
// void ShaderCommonData::executeOperate(const std::string & outVariable, const std::string & varleft, const std::string & varright, const std::string & op)
// {
//     std::string operation = "    " + outVariable + " = " + varleft + " " + op + " " + varright + "; \n";
//     operateVariables.push_back(operation);
// }

// //  Set the Output Variable.
// void ShaderCommonData::setOutputVariable(const std::string & outVariable, const std::string & inVariable)
// {

//     std::string outputVariable = getShaderOutputStructType();
//     std::transform(outputVariable.begin(), outputVariable.end(), outputVariable.begin(), ::tolower);

//     std::string setOutputCode = "    " + outputVariable + "." + outVariable + " = " + inVariable + "; \n";

//     outputVariables.push_back(setOutputCode);
// }


// //  Return the Input Struct Type.
// std::string ShaderCommonData::getShaderInputStructType() const
// {
//     return inSemantics.structVariableType;
// }

// //  Return the Output Struct Type.
// std::string ShaderCommonData::getShaderOutputStructType() const
// {
//     return outSemantics.structVariableType;
// }


// //  
// const std::vector<UCBufferData> & ShaderCommonData::getConstantBuffers() const
// {
//     return mUCBs;
// }

// //  
// const std::vector<ShaderCodeWriterDescs::ShaderResourceDesc> & ShaderCommonData::getShaderResources() const
// {
//     return mSRs;
// }

// //  
// const std::vector<std::string> & ShaderCommonData::getUseInputsCode() const
// {
//     //  Return Input Variables.
//     return inputVariables;
// }

// const std::vector<std::string> & ShaderCommonData::getAccessResourcesCode() const
// {
//     //  Return Access Variables.
//     return accessVariables;
// }

// const std::vector<std::string> & ShaderCommonData::getCreateLocalVariablesCode() const
// {
//     //  Define Variables.
//     return defineVariables;
// }

// const std::vector<std::string> & ShaderCommonData::getOperateVariablesCode() const
// {
//     //  Operate Variables.
//     return operateVariables;
// }

// const std::vector<std::string> & ShaderCommonData::getWriteOutputsCode() const
// {
//     //  Output Variables.
//     return outputVariables;
// }

// //  Return the input semantics.
// const ShaderCodeWriterDescs::ShaderStructDesc & ShaderCommonData::getInputStructDesc() const
// {
//     return inSemantics;
// }

// //  Return the output semantics.
// const ShaderCodeWriterDescs::ShaderStructDesc & ShaderCommonData::getOutputStructDesc() const
// {
//     return outSemantics;
// }

