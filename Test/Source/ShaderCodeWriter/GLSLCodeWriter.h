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
#include "GLSLShaderData.h"

using namespace Falcor;


//  GLSL Code Writer.
class GLSLCodeWriter
{

public:

    //  GLSL Code Writer Constructor.
    GLSLCodeWriter() = default;

    //  GLSL Code Writer Destructor.
    ~GLSLCodeWriter() = default;


    //  Write a Uniform Buffer Declare Code.
    std::string writeUniformBufferDeclareCode(const GLSLDescs::UniformBufferData & ubData) const;

    //  Write the Uniform Buffers Declare Code.
    std::string writeUniformResourcesCode(const GLSLShaderData & shaderData) const;

    //  Write the Base Resources Declare Code.
    std::string writeBaseResourceDeclareCode(const GLSLDescs::GLSLResourcesDesc & rDesc) const;

    //  Write the Base Resources Declare Code.
    std::string writeBaseResourcesDeclareCode(const GLSLShaderData & shaderData) const;

    //  Write the Shader Struct Code.
    std::string writeGenericStructCode(const GLSLDescs::GLSLStructDesc & structDesc) const;

    //  Write the Generic Attachments.  
    std::string writeLayoutSpecificationCode(const CommonShaderDescs::ResourceAttachmentDesc & layoutDesc) const;

    //  Write the Shader Storage Block Attachment.
    std::string writeSSBODeclareCode(const GLSLDescs::GLSLResourcesDesc & ssboDesc);

};