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
#pragma once
#include "Graphics/Program/ProgramVersion.h"
#include "../RtShader.h"
#include "API/VAO.h"

namespace Falcor
{
    class RootSignature;

    // The inheritence here is just so that we could work with Program. We're not actually using anything from ProgramVersion
    class RtProgramVersion : public ProgramVersion, inherit_shared_from_this<ProgramVersion, RtProgramVersion>
    {
    public:
        using SharedPtr = std::shared_ptr<RtProgramVersion>;
        using SharedConstPtr = std::shared_ptr<const RtProgramVersion>;

        ~RtProgramVersion() = default;

        enum class Type
        {
            RayGeneration,
            Hit,
            Miss,
        };

        static SharedPtr createRayGen(RtShader::SharedPtr pRayGenShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize);
        static SharedPtr createMiss(RtShader::SharedPtr pMissShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize);
        static SharedPtr createHit(RtShader::SharedPtr pAnyHit, RtShader::SharedPtr pClosestHit, RtShader::SharedPtr pIntersection, std::string& log, const std::string& name, ProgramReflection::SharedPtr pReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize);

        RtShader::SharedConstPtr getShader(ShaderType type) const;

        std::shared_ptr<RootSignature> getRootSignature() const { return mpRootSignature; }

        const std::wstring& getExportName() const { return mExportName; }

        Type getType() const { return mType; }

        uint32_t getMaxPayloadSize() const { return mMaxPayloadSize; }
        uint32_t getMaxAttributesSize() const { return mMaxAttributeSize; }
    private:
        template<ShaderType shaderType>
        static SharedPtr createSingleShaderProgram(RtShader::SharedPtr pShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize);
        bool initCommon(std::string& log, ProgramReflection::SharedPtr pReflector);

        RtProgramVersion(Type progType, RtShader::SharedPtr const* ppShaders, size_t shaderCount, const std::string& name, uint32_t maxPayloadSize, uint32_t maxAttributeSize);
        std::shared_ptr<RootSignature> mpRootSignature;
        Type mType;
        std::wstring mExportName;

        static uint64_t sProgId;
        uint32_t mMaxPayloadSize;
        uint32_t mMaxAttributeSize;
    };

    inline std::string to_string(RtProgramVersion::Type t)
    {
        switch (t)
        {
        case RtProgramVersion::Type::RayGeneration:
            return "RayGen";
        case RtProgramVersion::Type::Hit:
            return "Hit";
        case RtProgramVersion::Type::Miss:
            return "Miss";
        default:
            should_not_get_here();
            return "";
        }
    }
}
