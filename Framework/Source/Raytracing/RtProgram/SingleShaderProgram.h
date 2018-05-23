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
#include "..\RtShader.h"
#include "RtProgramKernels.h"
#include "Graphics/Program/ShaderLibrary.h"

namespace Falcor
{
    template<ShaderType shaderType>
    class RtSingleShaderProgram: public Program, public inherit_shared_from_this<Program, RtSingleShaderProgram<shaderType>>
    {
    public:
        using SharedPtr = std::shared_ptr<RtSingleShaderProgram>;
        using SharedConstPtr = std::shared_ptr<const RtSingleShaderProgram>;
        ~RtSingleShaderProgram() = default;

        static SharedPtr createFromFile(const char* filename, const char* entryPoint, const DefineList& programDefines = DefineList(), uint32_t maxPayloadSize = FALCOR_RT_MAX_PAYLOAD_SIZE_IN_BYTES, uint32_t maxAttributesSize = D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES)
        {
            return createCommon(filename, entryPoint, programDefines, maxPayloadSize, maxAttributesSize);
        }

    private:
        RtSingleShaderProgram(uint32_t maxPayloadSize, uint32_t maxAttributesSize) : mMaxPayloadSize(maxPayloadSize), mMaxAttributesSize(maxAttributesSize) {}

        uint32_t mMaxPayloadSize;
        uint32_t mMaxAttributesSize;

        static SharedPtr createCommon(const char* str, const char* entryPoint, const DefineList& programDefines, uint32_t maxPayloadSize, uint32_t maxAttributesSize)
        {
            SharedPtr pProg = SharedPtr(new RtSingleShaderProgram(maxPayloadSize, maxAttributesSize));
            Desc d(str);
            d.entryPoint((ShaderType)shaderType, entryPoint);
            pProg->init(d, programDefines);
            return pProg;
        }

        virtual ProgramKernels::SharedPtr createProgramKernels(std::string& log, const Shader::Blob shaderBlob[kShaderCount], ProgramReflection::SharedPtr pReflector) const override
        {
            RtShader::SharedPtr pShader;
            pShader = createRtShaderFromBlob(mDesc.getShaderLibrary(ShaderType(shaderType))->getFilename(), mDesc.getShaderEntryPoint(ShaderType(shaderType)), shaderBlob[uint32_t(shaderType)], mDesc.getCompilerFlags(), shaderType, log);
            auto rootSignature = RootSignature::create(pReflector.get());

            if (pShader)
            {
                switch (shaderType)
                {
                case ShaderType::RayGeneration:
                    return RtProgramKernels::createRayGen(pShader, log, getProgramDescString(), pReflector, rootSignature, mMaxPayloadSize, mMaxAttributesSize);
                case ShaderType::Miss:
                    return RtProgramKernels::createMiss(pShader, log, getProgramDescString(), pReflector, rootSignature, mMaxPayloadSize, mMaxAttributesSize);
                default:
                    should_not_get_here();
                }
            }
            return nullptr;
        }
    };

    using RayGenProgram = RtSingleShaderProgram<ShaderType::RayGeneration>;
    using MissProgram = RtSingleShaderProgram<ShaderType::Miss>;
}
