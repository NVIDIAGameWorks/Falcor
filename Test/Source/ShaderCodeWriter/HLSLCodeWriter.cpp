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

#pragma once
#include "Falcor.h"
#include "HLSLCodeWriter.h"
#include <string>
#include <map>
#include <vector>
#include <set>
#include <random>
#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include "ShaderCommonData.h"             //  

using namespace HLSLDescs;

//  Write the Constant Buffers Declare Code.
std::string HLSLCodeWriter::writeConstantBufferDeclareCode(const HLSLDescs::ConstantBufferData & cbData) const
{
    //  
    std::string cbRegisterDeclareCode = "";

    //  Get the Constant Buffer Declaration Code.
    cbRegisterDeclareCode = writeRegisterSpecificationCode(cbData.viewAttachmentDesc());

    //  Determine the style of Constant Buffer to use.
    if (!cbData.getIsGlobalCBType())
    {
        std::string cbDeclareCode = "ConstantBuffer";

        //  
        std::string cbStructCode = "";

        //  Check if we need to define a struct type.
        if (cbData.viewStructDesc().definesStructType)
        {
            //  
            std::string cbStructCode = "\n// Constant Buffer Struct Code. \n";

            //  
            cbStructCode = writeGenericStructCode(cbData.viewStructDesc());
        }


        //  Use the Struct Type
        if (cbData.viewStructDesc().usesStructType)
        {
            cbDeclareCode = cbDeclareCode + "<" + cbData.viewStructDesc().structVariableType + "> " + cbData.getCBVariable();
        }
        else
        {
            cbDeclareCode = cbDeclareCode + "<float4> " + cbData.getCBVariable();
        }

        //  Add the Register Declare Code.
        cbDeclareCode = cbDeclareCode + cbRegisterDeclareCode + " ; \n";

        //  Return the Constant Buffer Declare Code.
        return cbDeclareCode;
    }
    else
    {
        //  Create the Constant Buffer Declare Code.
        std::string cbDeclareCode = "cbuffer " + cbData.getCBVariable() + " ";
        cbDeclareCode = cbDeclareCode + cbRegisterDeclareCode + " \n";
        cbDeclareCode = cbDeclareCode + "{ \n";

        //  Add the Variables to the Constant Buffer.
        for (uint32_t svIndex = 0; svIndex < cbData.viewStructDesc().structVariables.size(); svIndex++)
        {
            cbDeclareCode = cbDeclareCode + "    " + writeBaseResourceDeclareCode(cbData.viewStructDesc().structVariables[svIndex]) + " \n";
        }

        cbDeclareCode = cbDeclareCode + "}; \n";

        //  Return the Constant Buffer Declare Code.
        return cbDeclareCode;

    }
}



//  Write the Constant Buffers Declare Code.
std::string HLSLCodeWriter::writeConstantBuffersDeclareCode(const HLSLShaderData & shaderCodeData) const
{
    //  
    std::string cbDeclareCode = "// Constant Buffers Declare Code. Count = " + std::to_string(shaderCodeData.getConstantBuffers().size()) + " \n";

    for (uint32_t cbIndex = 0; cbIndex < shaderCodeData.getConstantBuffers().size(); cbIndex++)
    {
        cbDeclareCode = cbDeclareCode + writeConstantBufferDeclareCode(shaderCodeData.getConstantBuffers()[cbIndex]);
        cbDeclareCode = cbDeclareCode + " \n";
    }

    //  
    return cbDeclareCode;
}


//  Write the Base Resource Declare Code.
std::string HLSLCodeWriter::writeBaseResourceDeclareCode(const HLSLDescs::HLSLResourcesDesc & rDesc) const
{
    //  
    std::string resourceDeclareCode = "";

    if (rDesc.baseResourceType == HLSLBaseResourceType::BasicType)
    {
        //  Add the declaration for a basic resource variable.
        resourceDeclareCode = "float4 " + rDesc.resourceVariable + " ";
    }
    else if (rDesc.baseResourceType == HLSLBaseResourceType::TextureType)
    {
        resourceDeclareCode = writeTexturesDeclareCode(rDesc);
    }
    else if (rDesc.baseResourceType == HLSLBaseResourceType::BufferType)
    {
        //  Add the declaration for a Buffer.
        resourceDeclareCode = "Buffer";

        //  Change to Buffer Type.
        if (true)
        {
            resourceDeclareCode = resourceDeclareCode + "<float4> ";
        }

        resourceDeclareCode = resourceDeclareCode + rDesc.resourceVariable + "";

    }
    else if (rDesc.baseResourceType == HLSLBaseResourceType::BufferType && rDesc.bufferType == HLSLBufferType::StructuredBuffer)
    {
        resourceDeclareCode = writeStructuredBufferDeclareCode(rDesc);
    }
    else if (rDesc.baseResourceType == HLSLBaseResourceType::SamplerType)
    {
        //  Add the declaration for a Sampler.
        resourceDeclareCode = writeSamplersDeclareCode(rDesc);
    }


    //  Return the Declaration Code.
    return resourceDeclareCode;
}


//  Return the Base Resources Declare Code.
std::string HLSLCodeWriter::writeBaseResourcesDeclareCode(const HLSLShaderData & shaderCodeData) const
{

    std::string structuredBuffersDeclareCode = "// Structured Buffers Resources Declare Code. Count = " + std::to_string(shaderCodeData.getSamplers().size()) + " \n";

    //  Write the Structured Buffers
    for (uint32_t srIndex = 0; srIndex < shaderCodeData.getStructuredBufferResources().size(); srIndex++)
    {
        structuredBuffersDeclareCode = structuredBuffersDeclareCode + writeStructuredBufferDeclareCode(shaderCodeData.getStructuredBufferResources()[srIndex]);
    }

    //  
    std::string texturesDeclareCode = "// Texture Resources Declare Code. Count = " + std::to_string(shaderCodeData.getTextures().size()) + " \n";

    //  Write the Textures Resources Declare Code.
    for (uint32_t srIndex = 0; srIndex < shaderCodeData.getTextures().size(); srIndex++)
    {
        texturesDeclareCode = texturesDeclareCode + writeTexturesDeclareCode(shaderCodeData.getTextures()[srIndex]);
    }


    std::string samplersDeclareCode = "// Sampler Resources Declare Code. Count = " + std::to_string(shaderCodeData.getSamplers().size()) + " \n";

    //  Write the Samplers Resource Declare Code.
    for (uint32_t srIndex = 0; srIndex < shaderCodeData.getSamplers().size(); srIndex++)
    {
        samplersDeclareCode = samplersDeclareCode + writeSamplersDeclareCode(shaderCodeData.getSamplers()[srIndex]);
    }


    return structuredBuffersDeclareCode + " \n " + texturesDeclareCode + " \n " + samplersDeclareCode + " \n ";
}


//  Write Structured Buffer Declare Code.
std::string HLSLCodeWriter::writeStructuredBufferDeclareCode(const HLSLDescs::HLSLResourcesDesc & rDesc) const
{
    std::string resourceDeclareCode = "";

    //  Check whether we define a new struct type.
    if (rDesc.structDesc.definesStructType)
    {
        //  Create the Struct for the Structured Buffer.
        resourceDeclareCode = resourceDeclareCode + writeGenericStructCode(rDesc.structDesc) + " \n";
    }

    //  Set the appropriate Resource Type.
    std::string resourceTypeCode = rDesc.accessType == CommonShaderDescs::AccessType::ReadWrite ? "RWStructuredBuffer" : "StructuredBuffer";
    
    //  Check if we use the Struct.
    if (rDesc.structDesc.usesStructType)
    {
        //  Create the code.
        resourceDeclareCode = resourceDeclareCode + resourceTypeCode + "<" + rDesc.structDesc.structVariableType + "> ";
        resourceDeclareCode = resourceDeclareCode + rDesc.resourceVariable;
    }
    else
    {
        //  
        resourceDeclareCode = resourceDeclareCode + resourceTypeCode + "<" + "float4" + "> ";
        resourceDeclareCode = resourceDeclareCode + rDesc.resourceVariable;
    }

    //  Add the declaration for the Register Index and Space.
    resourceDeclareCode = resourceDeclareCode + writeArrayDeclareCode(rDesc.arrayDesc) + "";

    //  Add the declaration for the Register Index and Space.
    resourceDeclareCode = resourceDeclareCode + writeRegisterSpecificationCode(rDesc.attachmentDesc) + "";

    //  Return the Resource Declare Code.
    return resourceDeclareCode + " ; \n \n ";
}

//  
std::string HLSLCodeWriter::writeSamplersDeclareCode(const HLSLDescs::HLSLResourcesDesc & rDesc) const
{
    std::string samplersDeclareCode = "SamplerState " + rDesc.resourceVariable;

    samplersDeclareCode = samplersDeclareCode + writeArrayDeclareCode(rDesc.arrayDesc);
    samplersDeclareCode = samplersDeclareCode + writeRegisterSpecificationCode(rDesc.attachmentDesc);

    samplersDeclareCode = samplersDeclareCode + "; \n";

    return samplersDeclareCode;
}

//  Write Textures Declare Code.
std::string HLSLCodeWriter::writeTexturesDeclareCode(const HLSLDescs::HLSLResourcesDesc & rDesc) const
{
    std::string textureDeclareCode = "Texture2D<float4> " + rDesc.resourceVariable + " ";

    textureDeclareCode = textureDeclareCode + writeArrayDeclareCode(rDesc.arrayDesc);
    textureDeclareCode = textureDeclareCode + writeRegisterSpecificationCode(rDesc.attachmentDesc);

    textureDeclareCode = textureDeclareCode + "; \n";

    return textureDeclareCode;
}

//  Write the Shader Struct Code.
std::string HLSLCodeWriter::writeGenericStructCode(const HLSLDescs::HLSLStructDesc & shaderStructDesc) const
{
    //  Check whether we have to define a new struct.
    if (shaderStructDesc.definesStructType)
    {
        //  
        std::string structCode = "// Struct Declare Code. \n";
        structCode = structCode + "struct " + shaderStructDesc.structVariableType + " \n";
        structCode = structCode + "{  \n";

        //  
        for (uint32_t svIndex = 0; svIndex < shaderStructDesc.structVariables.size(); svIndex++)
        {
            structCode = structCode + "    " + writeBaseResourceDeclareCode(shaderStructDesc.structVariables[svIndex]) + "; \n";
        }

        //  
        structCode = structCode + "};  \n";

        //  
        return structCode;
    }
    else
    {
        return "";
    }
}



//  Write the Shader Semantic Struct Code.
std::string HLSLCodeWriter::writeGenericSemanticStructCode(const HLSLDescs::HLSLStructDesc & shaderStructDesc) const
{
    //      
    if (shaderStructDesc.definesStructType)
    {
        std::string structCode = "// Struct Declare Code. \n";
        structCode = structCode + "struct " + shaderStructDesc.structVariableType + " \n";
        structCode = structCode + "{  \n";

        //  
        for (uint32_t svIndex = 0; svIndex < shaderStructDesc.structVariables.size(); svIndex++)
        {
            std::string semanticPostFix = "";
            if (shaderStructDesc.structVariables[svIndex].semanticDesc.hasSemanticValue)
            {
                semanticPostFix = semanticPostFix + " : " + shaderStructDesc.structVariables[svIndex].semanticDesc.semanticValue;
            }

            structCode = structCode + "float4 " + shaderStructDesc.structVariables[svIndex].resourceVariable + semanticPostFix + "; \n";
        }

        //  
        structCode = structCode + "};  \n";

        //  
        return structCode;
    }
    else
    {
        return "";
    }
}


//  Write the Attachment Code.
std::string HLSLCodeWriter::writeArrayDeclareCode(const CommonShaderDescs::ArrayDesc & rDesc) const
{
    std::string arrayCode = "";
    
    //  Add the Array Declarations.
    if (rDesc.isArray)
    {
        if (rDesc.dimensions.size())
        {
            for (uint32_t dimensionIndex = 0; dimensionIndex < rDesc.dimensions.size(); dimensionIndex++)
            {
                arrayCode = arrayCode + "[" + std::to_string(rDesc.dimensions[dimensionIndex]) + "]";
            }
        }
        else
        {
            arrayCode = arrayCode + "[]";
        }
    }

    return arrayCode;
}

//  Write the Register Specification Code.
std::string HLSLCodeWriter::writeRegisterSpecificationCode(const CommonShaderDescs::ResourceAttachmentDesc & shaderRegisterDesc) const
{
    //  Write the Register Index and Space Code.
    if (shaderRegisterDesc.isAttachmentSubpointExplicit || shaderRegisterDesc.isAttachmentPointExplicit)
    {
        std::string registerIndexAndSpaceCode = " : register ( ";

        if (shaderRegisterDesc.isAttachmentSubpointExplicit)
        {
            registerIndexAndSpaceCode = registerIndexAndSpaceCode + shaderRegisterDesc.registerType + std::to_string(shaderRegisterDesc.attachmentSubpoint);
        }

        //  
        if (shaderRegisterDesc.isAttachmentSubpointExplicit && shaderRegisterDesc.isAttachmentPointExplicit)
        {
            registerIndexAndSpaceCode = registerIndexAndSpaceCode + " , ";
        }

        if (shaderRegisterDesc.isAttachmentPointExplicit)
        {
            registerIndexAndSpaceCode = registerIndexAndSpaceCode + "space" + std::to_string(shaderRegisterDesc.attachmentPoint);
        }

        registerIndexAndSpaceCode = registerIndexAndSpaceCode + " ) ";

        //  
        return registerIndexAndSpaceCode;
    }
    else
    {
        return "";
    }
}


//  Return the Generic Semantics Input Code.
std::string HLSLCodeWriter::writeInputStructCode(const HLSLShaderData & shaderCodeData) const
{
    return writeGenericSemanticStructCode(shaderCodeData.getInputStructDesc());
}

//  Return the Generic Semantics Output Code.
std::string HLSLCodeWriter::writeOutputStructCode(const HLSLShaderData & shaderCodeData) const
{
    return writeGenericSemanticStructCode(shaderCodeData.getOutputStructDesc());
}
