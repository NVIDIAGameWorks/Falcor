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
#include "HitProgram.h"
#include "..\RtShader.h"
#include "RtProgramVersion.h"

namespace Falcor
{
    HitProgram::SharedPtr HitProgram::createFromFile(const std::string& filename, const std::string& closestHitEntry, const std::string& anyHitEntry, const std::string& intersectionEntry, const DefineList& programDefines, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        return createCommon(filename, closestHitEntry, anyHitEntry, intersectionEntry, programDefines, true, maxPayloadSize, maxAttributeSize);
    }
    
    HitProgram::SharedPtr HitProgram::createCommon(const std::string& filename, const std::string& closestHitEntry, const std::string& anyHitEntry, const std::string& intersectionEntry, const DefineList& programDefines, bool fromFile, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        SharedPtr pProg = SharedPtr(new HitProgram(maxPayloadSize, maxAttributeSize));

        if ((closestHitEntry.size() + anyHitEntry.size()) == 0)
        {
            logError("HitProgram must have either a closest-hit or any-hit shader");
            return nullptr;
        }
        
        Program::Desc desc;
        desc.sourceFile(filename);
        if (closestHitEntry.size())     desc.entryPoint(ShaderType::ClosestHit,   closestHitEntry);
        if (anyHitEntry.size())         desc.entryPoint(ShaderType::AnyHit,       anyHitEntry);
        if (intersectionEntry.size())   desc.entryPoint(ShaderType::Intersection, intersectionEntry);

        pProg->init(desc, programDefines);
        return pProg;
    }

    // #DXR_FIX add the filename
#define create_shader(_type, _pshader)                          \
    if (shaderBlob[uint32_t(_type)].data.size())                \
    {                                                           \
        _pshader = createRtShaderFromBlob(                      \
        mDesc.getShaderSource(_type),                           \
        mDesc.getShaderEntryPoint(_type),                       \
            shaderBlob[uint32_t(_type)],                        \
            flags,                                              \
            _type,                                              \
            log);                                               \
        OK = OK && (_pshader != nullptr);                       \
    }

    ProgramVersion::SharedPtr HitProgram::createProgramVersion(std::string& log, const Shader::Blob shaderBlob[kShaderCount]) const
    {
        RtShader::SharedPtr pAnyHit, pIntersect, pClosestHit;
        bool OK = true;
        Shader::CompilerFlags flags = mDesc.getCompilerFlags();
        create_shader(ShaderType::Intersection, pIntersect);
        create_shader(ShaderType::AnyHit, pAnyHit);
        create_shader(ShaderType::ClosestHit, pClosestHit);

        return OK ? RtProgramVersion::createHit(pAnyHit, pClosestHit, pIntersect, log, getProgramDescString(), mPreprocessedReflector, mMaxPayloadSize, mMaxAttributeSize) : nullptr;
    }
}
