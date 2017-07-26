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
#include "Framework.h"
#include <vector>
#include "API/Shader.h"

namespace Falcor
{
    class IApiHandle
    {
    public:
        virtual ~IApiHandle() {};
    };

    template<typename T>
    class ShaderHandle :public IApiHandle
    {
    public:
        ShaderHandle(T handle) { mHandle = handle; }
        T getHandle() const { return mHandle; }

    private:
        T mHandle;
    };

    struct DxShaderData
    {
        ID3DBlobPtr pBlob;
        ID3D11ShaderReflectionPtr pReflector;
        std::unique_ptr<IApiHandle> pHandle = nullptr;
    };

    static const char* kEntryPoint = "main";

    static ID3DBlob* compileShader(const std::string& source, const std::string& target, std::string& errorLog)
    {
        ID3DBlob* pCode;
        ID3DBlobPtr pErrors;

        UINT flags = D3DCOMPILE_WARNINGS_ARE_ERRORS;
#ifdef _DEBUG
        flags |= D3DCOMPILE_DEBUG;
#endif

        HRESULT hr = D3DCompile(source.c_str(), source.size(), nullptr, nullptr, nullptr, kEntryPoint, target.c_str(), flags, 0, &pCode, &pErrors);
        if(FAILED(hr))
        {
            std::vector<char> infoLog(pErrors->GetBufferSize() + 1);
            memcpy(infoLog.data(), pErrors->GetBufferPointer(), pErrors->GetBufferSize());
            infoLog[pErrors->GetBufferSize()] = 0;
            errorLog = std::string(infoLog.data());
            return nullptr;
        }

        return pCode;
    }

    Shader::Shader(ShaderType type) : mType(type)
    {
        mpPrivateData = new DxShaderData;
    }

    Shader::~Shader()
    {
        DxShaderData* pData = (DxShaderData*)mpPrivateData;
        safe_delete(pData);
    }

    const std::string getTargetString(ShaderType type)
    {
        switch(type)
        {
        case ShaderType::Vertex:
            return "vs_5_0";
        case ShaderType::Pixel:
            return "ps_5_0";
        case ShaderType::Hull:
            return "hs_5_0";
        case ShaderType::Domain:
            return "ds_5_0";
        case ShaderType::Geometry:
            return "gs_5_0";
        case ShaderType::Compute:
            return "cs_5_0";
        default:
            should_not_get_here();
            return "";
        }
    }

#define create_shader(handle_type_, falcor_func, dx_func_) \
    std::unique_ptr<ShaderHandle<handle_type_>> falcor_func(ID3DBlob* pBlob) \
    {                               \
        handle_type_ handle;        \
        d3d_call(getD3D11Device()->dx_func_(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &handle)); \
        if(handle == nullptr) return nullptr;                         \
        return std::make_unique<ShaderHandle<handle_type_>>(handle);  \
    }

    create_shader(VertexShaderHandle, createVertexShader, CreateVertexShader);
    create_shader(FragmentShaderHandle, createPixelShader, CreatePixelShader);
    create_shader(HullShaderHandle, createHullShader, CreateHullShader);
    create_shader(DomainShaderHandle, createDomainShader, CreateDomainShader);
    create_shader(GeometryShaderHandle, createGeometryShader, CreateGeometryShader);
    create_shader(ComputeShaderHandle, createComputeShader, CreateComputeShader);
        
    Shader::SharedPtr Shader::create(const std::string& shaderString, ShaderType type, std::string& log)
    {
        SharedPtr pShader = SharedPtr(new Shader(type));

        // Compile the shader
        DxShaderData* pData = (DxShaderData*)pShader->mpPrivateData;
        pData->pBlob = compileShader(shaderString, getTargetString(type), log);

        if(pData->pBlob == nullptr)
        {
            return nullptr;
        }

        // create the shader object
        switch(type)
        {
        case ShaderType::Vertex:
            pData->pHandle = createVertexShader(pData->pBlob);
            break;
        case ShaderType::Pixel:
            pData->pHandle = createPixelShader(pData->pBlob);
            break;
        case ShaderType::Hull:
            pData->pHandle = createHullShader(pData->pBlob);
            break;
        case ShaderType::Domain:
            pData->pHandle = createDomainShader(pData->pBlob);
            break;
        case ShaderType::Geometry:
            pData->pHandle = createGeometryShader(pData->pBlob);
            break;
        case ShaderType::Compute:
            pData->pHandle = createComputeShader(pData->pBlob);
            break;
        default:
            should_not_get_here();
            return pShader;
        }

        if(pData->pHandle == nullptr)
        {
            return nullptr;
        }

        // Get the reflection object
        d3d_call(D3DReflect(pData->pBlob->GetBufferPointer(), pData->pBlob->GetBufferSize(), __uuidof(ID3D11ShaderReflection), (void**)&pData->pReflector));

        return pShader;
    }

    template<typename HandleType>
    const HandleType getHandle(void* pPrivateData, ShaderType expectedType, ShaderType actualType)
    {
#ifdef _LOG_ENABLED
        if(expectedType != actualType)
        {
            logError("Can't retrieve shader API handle. Requested handle Type doesn't match actual shader Type");
            return nullptr;
        }
#endif
        DxShaderData* pData = (DxShaderData*)pPrivateData;
        const IApiHandle* pApiHandle = pData->pHandle.get();
        const ShaderHandle<HandleType>* pHandle = (ShaderHandle<HandleType>*)pApiHandle;
        return pHandle->getHandle();
    }

#define get_api_handle(handle_type, expected_type)   \
    template<>                                                  \
    handle_type Shader::getApiHandle() const                    \
    {                                                           \
        return getHandle<handle_type>(mpPrivateData, expected_type, mType);   \
    }

    get_api_handle(VertexShaderHandle, ShaderType::Vertex);
    get_api_handle(FragmentShaderHandle, ShaderType::Pixel);
    get_api_handle(GeometryShaderHandle, ShaderType::Geometry);
    get_api_handle(DomainShaderHandle, ShaderType::Domain);
    get_api_handle(HullShaderHandle, ShaderType::Hull);
    get_api_handle(ComputeShaderHandle, ShaderType::Compute);
#undef get_api_handle
}
