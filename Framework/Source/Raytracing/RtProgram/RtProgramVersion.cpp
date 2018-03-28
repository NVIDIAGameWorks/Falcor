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
#include "Framework.h"
#include "RtProgramVersion.h"
#include "API/Device.h"
#include "API/D3D12/D3D12State.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
    uint64_t RtProgramVersion::sProgId = 0;
    ProgramReflection::SharedPtr createProgramReflection(const Shader::SharedConstPtr pShaders[], std::string& log);

    RtProgramVersion::RtProgramVersion(Type progType, RtShader::SharedPtr const* ppShaders, size_t shaderCount, const std::string& name, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
        : ProgramVersion(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, name)
        , mType(progType)
        , mMaxAttributeSize(maxAttributeSize)
        , mMaxPayloadSize(maxPayloadSize)
    {
        for( size_t ii = 0; ii < shaderCount; ++ii )
        {
            auto pShader = ppShaders[ii];
            mpShaders[uint32_t(pShader->getType())] = pShader;
        }

        switch (mType)
        {
        case Falcor::RtProgramVersion::Type::RayGeneration:
            mExportName = string_2_wstring(ppShaders[0]->getEntryPoint());
            break;
        case Falcor::RtProgramVersion::Type::Hit:
            mExportName = L"RtProgramVersion" + std::to_wstring(sProgId++);
            break;
        case Falcor::RtProgramVersion::Type::Miss:
            mExportName = string_2_wstring(ppShaders[0]->getEntryPoint());
            break;
        default:
            should_not_get_here();
        }
    }

    bool RtProgramVersion::initCommon(std::string& log, ProgramReflection::SharedPtr pReflector)
    {
        if (init(log) == false)
        {
            return false;
        }

        mpReflector = pReflector;

        // Create the root signature
        mpRootSignature = RootSignature::create(mpReflector.get(), true);

        return true;
    }

    RtProgramVersion::Type getProgTypeFromShader(ShaderType shaderType)
    {
        switch (shaderType)
        {
        case ShaderType::Miss:
            return RtProgramVersion::Type::Miss;
        case ShaderType::RayGeneration:
            return RtProgramVersion::Type::RayGeneration;
        default:
            should_not_get_here();
            return RtProgramVersion::Type(-1);
        }
    }
    
    template<ShaderType shaderType>
    RtProgramVersion::SharedPtr RtProgramVersion::createSingleShaderProgram(RtShader::SharedPtr pShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        // We are using the RayGeneration structure in the union to avoid code duplication, these asserts make sure that our assumptions are correct

        if (pShader == nullptr)
        {
            log = to_string(shaderType) + " shader is null. Can't create a " + to_string(shaderType) + " RtProgramVersion";
            return nullptr;
        }

        SharedPtr pProgram = SharedPtr(new RtProgramVersion(getProgTypeFromShader(shaderType), &pShader, 1, name, maxPayloadSize, maxAttributeSize));
        if (pProgram->initCommon(log, pReflector) == false)
        {
            return nullptr;
        }

        return pProgram;
    }

    RtShader::SharedConstPtr RtProgramVersion::getShader(ShaderType type) const
    {
        Shader::SharedConstPtr pShader = mpShaders[(uint32_t)type];
        RtShader::SharedConstPtr pRtShader = std::dynamic_pointer_cast<const RtShader>(pShader);
        assert(!pShader || pRtShader);
        return pRtShader;
    }

    RtProgramVersion::SharedPtr RtProgramVersion::createRayGen(RtShader::SharedPtr pRayGenShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        return createSingleShaderProgram<ShaderType::RayGeneration>(pRayGenShader, log, name, pReflector, maxPayloadSize, maxAttributeSize);
    }

    RtProgramVersion::SharedPtr RtProgramVersion::createMiss(RtShader::SharedPtr pMissShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        return createSingleShaderProgram<ShaderType::Miss>(pMissShader, log, name, pReflector, maxPayloadSize, maxAttributeSize);
    }
    
    RtProgramVersion::SharedPtr RtProgramVersion::createHit(RtShader::SharedPtr pAnyHit, RtShader::SharedPtr pClosestHit, RtShader::SharedPtr pIntersection, std::string& log, const std::string& name, ProgramReflection::SharedPtr pReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        size_t shaderCount = 0;
        RtShader::SharedPtr pShaders[3];

        if(pAnyHit)         pShaders[shaderCount++] = pAnyHit;
        if(pClosestHit)     pShaders[shaderCount++] = pClosestHit;
        if(pIntersection)   pShaders[shaderCount++] = pIntersection;

        if (shaderCount == 0)
        {
            log = "Error when creating " + to_string(Type::Hit) + " RtProgramVersion for program" + name + ". At least one of the shaders must be valid.";
            return nullptr;
        }

        SharedPtr pProgram = SharedPtr(new RtProgramVersion(Type::Hit, pShaders, shaderCount, name, maxPayloadSize, maxAttributeSize));
        if (pProgram->initCommon(log, pReflector) == false)
        {
            return nullptr;
        }

        return pProgram;
    }
}
