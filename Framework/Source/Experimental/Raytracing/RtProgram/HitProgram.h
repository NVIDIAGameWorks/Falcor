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
#pragma once
#include "Graphics/Program/Program.h"
#include "RtProgramVersion.h"

namespace Falcor
{
    class HitProgram : public Program, public inherit_shared_from_this<Program, HitProgram>
    {
    public:
        using SharedPtr = std::shared_ptr<HitProgram>;
        using SharedConstPtr = std::shared_ptr<const HitProgram>;
        ~HitProgram() = default;

        static HitProgram::SharedPtr createFromFile(const std::string& filename,
            const std::string& closestHitEntry, 
            const std::string& anyHitEntry, 
            const std::string& intersectionEntry, 
            const DefineList& programDefines = DefineList(),
            uint32_t maxPayloadSize = FALCOR_RT_MAX_PAYLOAD_SIZE_IN_BYTES,
            uint32_t maxAttributeSize = FALCOR_RT_MAX_ATTRIBUTE_SIZE_IN_BYTES,
            Shader::CompilerFlags flags = Shader::CompilerFlags::None
            );

        RtProgramVersion::SharedConstPtr getActiveVersion() const { return std::dynamic_pointer_cast<const RtProgramVersion>(Program::getActiveVersion()); }

    private:
        HitProgram(uint32_t maxPayloadSize, uint32_t maxAttributeSize) : mMaxPayloadSize(maxPayloadSize), mMaxAttributeSize(maxAttributeSize) {}
        uint32_t mMaxPayloadSize;
        uint32_t mMaxAttributeSize;

        static HitProgram::SharedPtr createCommon(const std::string& filename, const std::string& closestHitEntry, const std::string& anyHitEntry, const std::string& intersectionEntry, const DefineList& programDefines, bool fromFile, uint32_t maxPayloadSize, uint32_t maxAttributeSize, Shader::CompilerFlags flags);
        virtual ProgramVersion::SharedPtr createProgramVersion(std::string& log, const Shader::Blob shaderBlob[kShaderCount], const ProgramReflectors& reflectors) const override;
    };
}