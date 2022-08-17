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
#include "Core/API/GFX/GFXAPI.h"

namespace Falcor
{
    struct ShaderData
    {
        Shader::Blob pBlob;
        Slang::ComPtr<slang::IComponentType> pLinkedSlangEntryPoint;
        ISlangBlob* getBlob()
        {
            if (!pBlob)
            {
                Slang::ComPtr<ISlangBlob> pSlangBlob;
                Slang::ComPtr<ISlangBlob> pDiagnostics;

                if (SLANG_FAILED(pLinkedSlangEntryPoint->getEntryPointCode(0, 0, pSlangBlob.writeRef(), pDiagnostics.writeRef())))
                {
                    throw RuntimeError(std::string("Shader compilation failed. \n") + (const char*)pDiagnostics->getBufferPointer());
                }
                pBlob = Shader::Blob(pSlangBlob.get());
            }
            return pBlob.get();
        }
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
        // In GFX, we do not generate actual shader code at program creation.
        // The actual shader code will only be generated and cached when all specialization arguments
        // are known, which is right before a draw/dispatch command is issued, and this is done
        // internally within GFX.
        // The `Shader` implementation here serves as a helper utility for application code that
        // uses raw graphics API to get shader kernel code from an ordinary slang source.
        // Since most users/render-passes do not need to get shader kernel code, we defer
        // the call to slang's `getEntryPointCode` function until it is actually needed.
        // to avoid redundant shader compiler invocation.
        mpPrivateData->pBlob = nullptr;
        mpPrivateData->pLinkedSlangEntryPoint = slangEntryPoint;
        return slangEntryPoint != nullptr;
    }

#if FALCOR_HAS_D3D12
    ID3DBlobPtr Shader::getD3DBlob() const
    {
        ID3DBlobPtr result = mpPrivateData->getBlob();
        return result;
    }
#endif

    Shader::BlobData Shader::getBlobData() const
    {
        auto blob = mpPrivateData->getBlob();

        BlobData result;
        result.data = blob->getBufferPointer();
        result.size = blob->getBufferSize();
        return result;
    }
}
