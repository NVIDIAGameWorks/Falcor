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
#include "API/Shader.h"

namespace Falcor
{
    class RtShader : public Shader, public inherit_shared_from_this<Shader, RtShader>
    {
    public:
        using SharedPtr = std::shared_ptr<RtShader>;
        using SharedConstPtr = std::shared_ptr<const RtShader>;

        static SharedPtr create(const Blob& shaderBlob, const std::string& entryPointName, ShaderType Type, Shader::CompilerFlags flags, std::string& log);
        ~RtShader();

        const std::string& getEntryPoint() const { return mEntryPoint; }

#ifdef FALCOR_VK
        VkPipelineShaderStageCreateInfo getShaderStage(VkShaderStageFlagBits stage) const;
#endif

    private:
        RtShader(ShaderType type, const std::string& entryPointName);
        std::string mEntryPoint;
    };

    RtShader::SharedPtr createRtShaderFromBlob(
        const std::string& filename,
        const std::string& entryPoint,
        const Shader::Blob& blob,
        Shader::CompilerFlags flags,
        ShaderType shaderType,
        std::string& log);
}
