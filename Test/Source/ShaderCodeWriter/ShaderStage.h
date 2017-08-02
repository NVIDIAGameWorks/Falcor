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
#include "CommonShaderDescs.h"
#include "HLSLCodeWriter.h"
#include "GLSLCodeWriter.h"
#include <string>
#include <map>
#include <vector>
#include <set>
#include <random>
#include <algorithm> // std::random_shuffle
#include <ctime>     // std::time
#include <cstdlib>   // std::rand, std::srand

using namespace Falcor;

class ShaderStage
{
public:
    //  Default ShaderStageMaker Constructor.
    ShaderStage() = default;

    //
    ShaderStage(const std::string &inputStructVariableType, const std::string &outputStructVariableType);

    //  Default ShaderStageMaker Destructor.
    ~ShaderStage() = default;

    //  Get HLSL Shader Data.
    HLSLShaderData & getHLSLShaderData();

    //  View HLSL Shader Data.
    const HLSLShaderData & viewHLSLShaderData() const;
    
    //  Get GLSL Shader Data.
    GLSLShaderData & getGLSLShaderData();

    //  View GLSL Shader Data.
    const GLSLShaderData & viewGLSLShaderData() const;


    //  Return the Shader Stage GLSL Code.
    virtual std::string writeShaderStageGLSLCode() const = 0;

    //  Return the Shader Stage GLSL Code.
    virtual std::string writeShaderStageHLSLCode() const = 0;

    //  Return the Shader Code.
    virtual std::string writeShaderCode() const;

protected:
    HLSLCodeWriter mHLSLCodeWriter;
    
    //  Write the GL
    GLSLCodeWriter mGLSLCodeWriter;

private:

    //  HLSL Shader Data.
    HLSLShaderData hlslShaderData;

    //  GLSL Shader Data.
    GLSLShaderData glslShaderData;

};

//  Vertex Shader Stage.
class VertexShaderStage : public ShaderStage
{

public:
    //  Construct the Vertex Shader Stage.
    VertexShaderStage(const std::string &inputStructVariableType, const std::string &outputStructVariableType);

    //  Return the Vertex Shader Stage Code.
    virtual std::string writeShaderStageHLSLCode() const;

    //  Return the Vertex Shader Stage Code.
    virtual std::string writeShaderStageGLSLCode() const;
};

//  Hull Shader Stage.
class HullShaderStage : public ShaderStage
{
public:

    //  Construct the Geometry Shader Stage.
    HullShaderStage(const std::string & inputStructVariableType, const std::string & outputStructVariableType);

    //  Return the Geometry Shader Stage Code.
    virtual std::string writeShaderStageHLSLCode() const;

    //  Return the Geometry Shader Stage Code.
    virtual std::string writeShaderStageGLSLCode() const;

};

//  Domain Shader Stage.
class DomainShaderStage : public ShaderStage
{
public:

    //  Construct the Domain Shader Stage.
    DomainShaderStage(const std::string & inputStructVariableType, const std::string & outputStructVariableType);

    //  Return the Domain Shader Stage Code.
    virtual std::string writeShaderStageHLSLCode() const; 

    //  Return the Domain Shader Stage Code.
    virtual std::string writeShaderStageGLSLCode() const;

};

class GeometryShaderStage : public ShaderStage
{

public:
    
    //  Construct the Geometry Shader Stage.
    GeometryShaderStage(const std::string & inputStructVariableType, const std::string & outputStructVariableType);

    //  
    virtual std::string writeShaderStageHLSLCode() const;

    //  
    virtual std::string writeShaderStageGLSLCode() const;
};


//  Pixel Shader Stage.
class PixelShaderStage : public ShaderStage
{

public:
    //  Construct the Pixel Shader Stage.
    PixelShaderStage(const std::string &inputStructVariableType, const std::string &outputStructVariableType);

    //  Return the Pixel Shader Stage HLSL Code.
    virtual std::string writeShaderStageHLSLCode() const;

    //  Return the Pixel Shader Stage GLSL Code.
    virtual std::string writeShaderStageGLSLCode() const;
};


//  Compute Shader Stage.
class ComputeShaderStage : public ShaderStage
{
public:
    // Construct the Compute Shader Stage.
    ComputeShaderStage(const std::string & inputStructVariableType, const std::string & ouputStructVariableType);

    //  Return the Compute Shader Stage HLSL Code.
    virtual std::string writeShaderStageHLSLCode() const;

    //  Return the Compute Shader Stage GLSL Code.
    virtual std::string writeShaderStageGLSLCode() const;
};

