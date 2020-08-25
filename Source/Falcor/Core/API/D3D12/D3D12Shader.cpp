/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#include "stdafx.h"
#include "Core/API/Shader.h"
#include "Slang/slang.h"

namespace Falcor
{
    struct ShaderData
    {
        ID3DBlobPtr pBlob;
    };

    std::string getTargetString(ShaderType type, const std::string& shaderModel)
    {
        switch (type)
        {
        case ShaderType::Vertex:
            return "vs_" + shaderModel;
        case ShaderType::Pixel:
            return "ps_" + shaderModel;
        case ShaderType::Hull:
            return "hs_" + shaderModel;
        case ShaderType::Domain:
            return "ds_" + shaderModel;
        case ShaderType::Geometry:
            return "gs_" + shaderModel;
        case ShaderType::Compute:
            return "cs_" + shaderModel;
        default:
            should_not_get_here();
            return "";
        }
    }

    Shader::Shader(ShaderType type) : mType(type)
    {
        mpPrivateData = new ShaderData;
    }

    Shader::~Shader()
    {
        ShaderData* pData = (ShaderData*)mpPrivateData;
        safe_delete(pData);
    }

    bool Shader::init(const Blob& shaderBlob, const std::string& entryPointName, CompilerFlags flags, std::string& log)
    {
        // Compile the shader
        ShaderData* pData = (ShaderData*)mpPrivateData;
        pData->pBlob = shaderBlob.get();

        if (pData->pBlob == nullptr)
        {
            return false;
        }

        mApiHandle = { pData->pBlob->GetBufferPointer(), pData->pBlob->GetBufferSize() };
        return true;
    }

    ID3DBlobPtr Shader::getD3DBlob() const
    {
        const ShaderData* pData = (ShaderData*)mpPrivateData;
        return pData->pBlob;
    }
}
