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

#include "ShaderCommonData.h"
#include "HLSLDescs.h"
#include "HLSLShaderData.h"

using namespace Falcor;


//  HLSL Code Writer.
class HLSLCodeWriter
{

public:

    //  HLSL Code Writer Constructor.
    HLSLCodeWriter() = default;

    //  HLSL Code Writer Destructor.
    ~HLSLCodeWriter() = default;

    //  Write a Constant Buffer Declare Code.
    std::string writeConstantBufferDeclareCode(const HLSLDescs::ConstantBufferData & cbData) const;

    //  Write the Constant Buffers Declare Code.
    std::string writeConstantBuffersDeclareCode(const HLSLShaderData & hlslCodeData) const;

    //  Write the Base Resources Declare Code.
    std::string writeBaseResourceDeclareCode(const HLSLDescs::HLSLResourcesDesc & shaderResourceDesc) const;

    //  Write the Base Resources Declare Code.
    std::string writeBaseResourcesDeclareCode(const HLSLShaderData & hlslCodeData) const;

    //  Write the Structured Buffer Declare Code.
    std::string writeStructuredBufferDeclareCode(const HLSLDescs::HLSLResourcesDesc & rDesc) const;

    //  Return the Samplers Declare Code.
    std::string writeSamplersDeclareCode(const HLSLDescs::HLSLResourcesDesc & rDesc) const;

    //  Return the Textures Declare Code.
    std::string writeTexturesDeclareCode(const HLSLDescs::HLSLResourcesDesc & rDesc) const;

    //  Write the Generic Struct Code.
    std::string writeGenericStructCode(const HLSLDescs::HLSLStructDesc & shaderStructDesc) const;

    //  Write the Generic Semantic Struct Code.
    std::string writeGenericSemanticStructCode(const HLSLDescs::HLSLStructDesc & shaderStructDesc) const;

    //  Write the Attachment Code.
    std::string writeArrayDeclareCode(const CommonShaderDescs::ArrayDesc & rDesc) const;

    //  Write the Register Code.
    std::string writeRegisterSpecificationCode(const CommonShaderDescs::ResourceAttachmentDesc & shaderResourceDesc) const;

    //  Write the Input Struct Code.
    std::string writeInputStructCode(const HLSLShaderData & hlslCodeData) const;

    //  Write the Output Struct Code.
    std::string writeOutputStructCode(const HLSLShaderData & hlslCodeData) const;


};
