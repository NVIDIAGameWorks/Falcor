/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/API/Shader.h"

#include <slang.h>

namespace Falcor
{
    struct ShaderData
    {
        ID3DBlobPtr pBlob;
    };

    Shader::Shader(ShaderType type) : mType(type)
    {
        mpPrivateData = std::make_unique<ShaderData>();
    }

    Shader::~Shader()
    {
    }

    bool Shader::init(ComPtr<slang::IComponentType> slangEntryPoint, const std::string& entryPointName, CompilerFlags flags, std::string& log)
    {
        // Compile the shader kernel.
        ComPtr<slang::IBlob> pSlangDiagnostics;
        ComPtr<slang::IBlob> pShaderBlob;

        bool succeeded = SLANG_SUCCEEDED(slangEntryPoint->getEntryPointCode(
            /* entryPointIndex: */ 0,
            /* targetIndex: */ 0,
            pShaderBlob.writeRef(),
            pSlangDiagnostics.writeRef()));
        if (pSlangDiagnostics && pSlangDiagnostics->getBufferSize() > 0)
        {
            log += static_cast<char const*>(pSlangDiagnostics->getBufferPointer());
        }
        if (succeeded)
        {
            mpPrivateData->pBlob = pShaderBlob.get();
        }
        return succeeded;
    }

    ID3DBlobPtr Shader::getD3DBlob() const
    {
        return mpPrivateData->pBlob;
    }

    Shader::BlobData Shader::getBlobData() const
    {
        BlobData result;
        auto blob = getD3DBlob();
        result.data = blob->GetBufferPointer();
        result.size = blob->GetBufferSize();
        return result;
    }
}
