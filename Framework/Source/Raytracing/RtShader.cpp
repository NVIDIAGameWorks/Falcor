/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
#ifdef FALCOR_DXR
#include "Framework.h"
#include "RtShader.h"
#include "Utils/StringUtils.h"
#include "../../../Externals/DXR/dxcapi.use.h"

static dxc::DxcDllSupport gDxrDllHelper;

namespace Falcor
{
    RtShader::~RtShader() = default;
    MAKE_SMART_COM_PTR(IDxcCompiler);
    MAKE_SMART_COM_PTR(IDxcLibrary);
    MAKE_SMART_COM_PTR(IDxcBlobEncoding);
    MAKE_SMART_COM_PTR(IDxcOperationResult);
    MAKE_SMART_COM_PTR(IDxcContainerReflection);

    RtShader::RtShader(ShaderType type, const std::string& entryPointName) : Shader(type), mEntryPoint(entryPointName) {}

    RtShader::SharedPtr RtShader::create(const Blob& shaderBlob, const std::string& entryPointName, ShaderType type, Shader::CompilerFlags flags, std::string& log)
    {
        SharedPtr pShader = SharedPtr(new RtShader(type, entryPointName));
        return pShader->init(shaderBlob, entryPointName, flags, log) ? pShader : nullptr;
    }

    ID3DBlobPtr RtShader::compile(const Blob& blob, const std::string&  entryPointName, Shader::CompilerFlags flags, std::string& log)
    {
        d3d_call(gDxrDllHelper.Initialize());
        IDxcCompilerPtr pCompiler;
        IDxcLibraryPtr pLibrary;
        d3d_call(gDxrDllHelper.CreateInstance(CLSID_DxcCompiler, &pCompiler));
        d3d_call(gDxrDllHelper.CreateInstance(CLSID_DxcLibrary, &pLibrary));

        // Create blob from the string
        IDxcBlobEncodingPtr pTextBlob;
        d3d_call(pLibrary->CreateBlobWithEncodingFromPinned((LPBYTE)blob.data.data(), (uint32_t)blob.data.size(), 0, &pTextBlob));

        // Compile
        std::vector<const WCHAR*> argv;
        argv.push_back(L"-Zpr");
        if (is_set(flags, Shader::CompilerFlags::TreatWarningsAsErrors))
        {
            argv.push_back(L"-WX");
        }
        IDxcOperationResultPtr pResult;
        std::wstring entryPoint = string_2_wstring(entryPointName);
        d3d_call(pCompiler->Compile(pTextBlob, L"RT Shader", L"", L"lib_6_1", argv.size() ? argv.data() : nullptr, (uint32_t)argv.size(), nullptr, 0, nullptr, &pResult));

        // Verify the result
        HRESULT resultCode;
        d3d_call(pResult->GetStatus(&resultCode));
        if (FAILED(resultCode))
        {
            IDxcBlobEncodingPtr pError;
            d3d_call(pResult->GetErrorBuffer(&pError));
            log += convertBlobToString(pError.GetInterfacePtr());
            return nullptr;
        }

        IDxcBlobPtr pBlob;
        d3d_call(pResult->GetResult(&pBlob));
        return pBlob;
    }

    RtShader::SharedPtr createRtShaderFromBlob(const std::string& filename, const std::string& entryPoint, const Shader::Blob& blob, Shader::CompilerFlags flags, ShaderType shaderType, std::string& log)
    {
        assert(blob.type == Shader::Blob::Type::String);
        std::string msg;
        RtShader::SharedPtr pShader = RtShader::create(blob, entryPoint, shaderType, flags, msg);

        if(pShader == nullptr)
        {
            log = "Error when creating " + to_string(shaderType) + " shader from file \"" + filename + "\"\nError log:\n";
            log += msg;
        }
        return pShader;
    }
}
#endif