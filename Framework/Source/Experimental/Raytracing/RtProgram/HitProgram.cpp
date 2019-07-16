/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "HitProgram.h"
#include "..\RtShader.h"
#include "RtProgramVersion.h"
#include "Graphics/Program/ShaderLibrary.h"

namespace Falcor
{
    HitProgram::SharedPtr HitProgram::createFromFile(const std::string& filename, const std::string& closestHitEntry, const std::string& anyHitEntry, const std::string& intersectionEntry, const DefineList& programDefines, uint32_t maxPayloadSize, uint32_t maxAttributeSize, Shader::CompilerFlags flags)
    {
        return createCommon(filename, closestHitEntry, anyHitEntry, intersectionEntry, programDefines, true, maxPayloadSize, maxAttributeSize, flags);
    }
    
    HitProgram::SharedPtr HitProgram::createCommon(const std::string& filename, const std::string& closestHitEntry, const std::string& anyHitEntry, const std::string& intersectionEntry, const DefineList& programDefines, bool fromFile, uint32_t maxPayloadSize, uint32_t maxAttributeSize, Shader::CompilerFlags flags)
    {
        SharedPtr pProg = SharedPtr(new HitProgram(maxPayloadSize, maxAttributeSize));

        if ((closestHitEntry.size() + anyHitEntry.size()) == 0)
        {
            logError("HitProgram must have either a closest-hit or any-hit shader");
            return nullptr;
        }
        
        Program::Desc desc;
        desc.setCompilerFlags(flags);
        desc.addShaderLibrary(filename);
        if (closestHitEntry.size())     desc.entryPoint(ShaderType::ClosestHit,   closestHitEntry);
        if (anyHitEntry.size())         desc.entryPoint(ShaderType::AnyHit,       anyHitEntry);
        if (intersectionEntry.size())   desc.entryPoint(ShaderType::Intersection, intersectionEntry);
#ifdef FALCOR_VK
        desc.setShaderModel("460");
#else
        desc.setShaderModel("6_3");
#endif
        pProg->init(desc, programDefines);
        return pProg;
    }

    // #DXR_FIX add the filename
#define create_shader(_type, _pshader)                          \
    if (shaderBlob[uint32_t(_type)])                            \
    {                                                           \
        _pshader = createRtShaderFromBlob(                      \
        mDesc.getShaderLibrary(_type)->getFilename(),           \
        mDesc.getShaderEntryPoint(_type),                       \
            shaderBlob[uint32_t(_type)],                        \
            flags,                                              \
            _type,                                              \
            log);                                               \
        OK = OK && (_pshader != nullptr);                       \
    }

    ProgramVersion::SharedPtr HitProgram::createProgramVersion(std::string& log, const Shader::Blob shaderBlob[kShaderCount], const ProgramReflectors& reflectors) const
    {
        RtShader::SharedPtr pAnyHit, pIntersect, pClosestHit;
        bool OK = true;
        Shader::CompilerFlags flags = mDesc.getCompilerFlags();
        create_shader(ShaderType::Intersection, pIntersect);
        create_shader(ShaderType::AnyHit, pAnyHit);
        create_shader(ShaderType::ClosestHit, pClosestHit);

        return OK ? RtProgramVersion::createHit(pAnyHit, pClosestHit, pIntersect, log, getProgramDescString(), reflectors.pLocalReflector, mMaxPayloadSize, mMaxAttributeSize) : nullptr;
    }
}
