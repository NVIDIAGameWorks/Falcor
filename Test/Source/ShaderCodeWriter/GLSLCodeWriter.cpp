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
#include "GLSLCodeWriter.h"
#include <string>
#include <map>
#include <vector>
#include <set>
#include <random>
#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include "CommonShaderDescs.h"        
#include "GLSLDescs.h"

//  
std::string GLSLCodeWriter::writeUniformBufferDeclareCode(const GLSLDescs::UniformBufferData & ubData) const
{
    std::string ubAttachmentCode = "";

    //  Get the Layout(set = , binding = ) code.
    ubAttachmentCode = writeLayoutSpecificationCode(ubData.viewAttachmentDesc());

    //  Check if this is not a global type uniform buffer.
    if (!ubData.getIsGlobalUBType())
    {
        //  Create the Uniform Buffer Variable.
        std::string ubDeclareCode = "\n";

        //  
        ubDeclareCode = ubAttachmentCode + "uniform " + ubData.viewStructDesc().structVariableType + " \n ";
        ubDeclareCode = ubDeclareCode + "{ \n";

        //  Add the Variables to the Uniform.
        for (uint32_t svIndex = 0; svIndex < ubData.viewStructDesc().structVariables.size(); svIndex++)
        {

            ubDeclareCode = ubDeclareCode + "    " + writeBaseResourceDeclareCode(ubData.viewStructDesc().structVariables[svIndex]) + " \n";
        }
        
        //  
        ubDeclareCode = ubDeclareCode + "} ";

        //  Add the syntax for the array.
        if (ubData.viewArrayDesc().isArray)
        {
            ubDeclareCode = ubDeclareCode + " " + ubData.getUBVariable() + " ; ";
        }
        else
        {
            //  
            std::string dimensions = "";

            //  Add the dimensions to the declaration.
            for (uint32_t dimensionIndex = 0; dimensionIndex < ubData.viewArrayDesc().dimensions.size(); dimensionIndex++)
            {
                dimensions = dimensions + "[" + std::to_string(ubData.viewArrayDesc().dimensions[dimensionIndex]) + "]";
            }
            
            //  
            ubDeclareCode = ubDeclareCode + " " + ubData.getUBVariable() +  dimensions + ";";
        }

        return ubDeclareCode;
    }    
    else
    {
        //  Create the Uniform Buffer Variable.
        std::string ubDeclareCode = "\n";

        //  
        ubDeclareCode = ubAttachmentCode + "uniform " + ubData.getUBVariable() + " ";
        ubDeclareCode = ubDeclareCode + "{ \n";

        //  Add the Variables to the Uniform.
        for (uint32_t svIndex = 0; svIndex < ubData.viewStructDesc().structVariables.size(); svIndex++)
        {
            ubDeclareCode = ubDeclareCode + "    " + writeBaseResourceDeclareCode(ubData.viewStructDesc().structVariables[svIndex]) + " \n";
        }
        
        //  
        ubDeclareCode = ubDeclareCode + "}; \n";

        //  Return the Uniform Declare Code.
        return ubDeclareCode;
    }
}



//  Write the Constant Buffers Declare Code.
std::string GLSLCodeWriter::writeUniformResourcesCode(const GLSLShaderData & shaderCodeData) const
{
    //  
    std::string ubDeclareCode = "// Constant Buffers Declare Code. Count = " + std::to_string(shaderCodeData.getUniformBuffers().size()) + " \n";

    //  Add the Code for the Uniform Buffers.
    for (uint32_t ubIndex = 0; ubIndex < shaderCodeData.getUniformBuffers().size(); ubIndex++)
    {
        ubDeclareCode = ubDeclareCode + writeUniformBufferDeclareCode(shaderCodeData.getUniformBuffers()[ubIndex]);
        ubDeclareCode = ubDeclareCode + " \n";
    }

    //  Return the 
    return ubDeclareCode;
}


//  Write the Base Resource Declare Code.
std::string GLSLCodeWriter::writeBaseResourceDeclareCode(const GLSLDescs::GLSLResourcesDesc & shaderResourceDesc) const
{
    //  
    std::string resourceDeclareCode = "";

    //  Check for a Basic Type.
    if (shaderResourceDesc.baseResourceType == GLSLDescs::GLSLBaseResourceType::BasicType)
    {

    }

    //  Check for a basic Sampler Type.
    if (shaderResourceDesc.baseResourceType == GLSLDescs::GLSLBaseResourceType::SamplerType)
    {
        resourceDeclareCode = " uniform sampler2D " + shaderResourceDesc.resourceVariable + " ";
    }


    //  Check for a SSBO Type resource.
    if (shaderResourceDesc.baseResourceType == GLSLDescs::GLSLBaseResourceType::BufferBackedType)
    {
        if (shaderResourceDesc.bufferType == GLSLBufferType::SSBO)
        {
            
        }
    }

    //  Check for a Texture Type Resource
    if (shaderResourceDesc.baseResourceType == GLSLBaseResourceType::TextureType)
    {
        
    }


    //  Check for a Image Type resource.
    if (shaderResourceDesc.baseResourceType == GLSLDescs::GLSLBaseResourceType::ImageType)
    {

    }


    //  Add the Array Declarations.
    if (shaderResourceDesc.arrayDesc.isArray)
    {
        if (shaderResourceDesc.arrayDesc.dimensions.size())
        {
            for (uint32_t dimensionIndex = 0; dimensionIndex < shaderResourceDesc.arrayDesc.dimensions.size(); dimensionIndex++)
            {
                resourceDeclareCode = resourceDeclareCode + "[" + std::to_string(shaderResourceDesc.arrayDesc.dimensions[dimensionIndex]) + "]";
            }
        }
        else
        {
            resourceDeclareCode = resourceDeclareCode + "[]";
        }
    }


    //  Add the declaration for the Register Index and Space.
    resourceDeclareCode =  writeLayoutSpecificationCode(shaderResourceDesc.attachmentDesc) + resourceDeclareCode + "; \n";

    //  Return the Declaration Code.
    return resourceDeclareCode;
}


//  Return the Base Resources Declare Code.
std::string GLSLCodeWriter::writeBaseResourcesDeclareCode(const GLSLShaderData & shaderCodeData) const
{

    return "";
}





//  Write the Shader Struct Code.
std::string GLSLCodeWriter::writeGenericStructCode(const GLSLDescs::GLSLStructDesc & shaderStructDesc) const
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



//  Write the Layout Specification Declare Code.
std::string GLSLCodeWriter::writeLayoutSpecificationCode(const CommonShaderDescs::ResourceAttachmentDesc & shaderResourceDesc) const
{
    
    //  Write the Set and Binding Code.
    if (shaderResourceDesc.isAttachmentSubpointExplicit || shaderResourceDesc.isAttachmentPointExplicit)
    {
        std::string layoutSetAndBindingCode = "layout(";

        if (shaderResourceDesc.isAttachmentPointExplicit)
        {
            layoutSetAndBindingCode = layoutSetAndBindingCode + "set = " + std::to_string(shaderResourceDesc.attachmentPoint);
        }

        //  
        if (shaderResourceDesc.isAttachmentSubpointExplicit && shaderResourceDesc.isAttachmentPointExplicit)
        {
            layoutSetAndBindingCode = layoutSetAndBindingCode + " , ";
        }

        if (shaderResourceDesc.isAttachmentSubpointExplicit)
        {
            layoutSetAndBindingCode = layoutSetAndBindingCode + "binding = " + std::to_string(shaderResourceDesc.attachmentSubpoint);
        }

        layoutSetAndBindingCode = layoutSetAndBindingCode + ") ";

        //  
        return layoutSetAndBindingCode;
    }
    else
    {
        return "";
    }
}


//  Write the Shader Storage Block Object.
std::string GLSLCodeWriter::writeSSBODeclareCode(const GLSLDescs::GLSLResourcesDesc & ssboDesc)
{


    return "";
}

