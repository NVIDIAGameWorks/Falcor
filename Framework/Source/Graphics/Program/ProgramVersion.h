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
#pragma once
#include "Framework.h"
#include <string>
#include <map>
#include <vector>
#include "API/Shader.h"
#include "Graphics/Program//ProgramReflection.h"

namespace Falcor
{
    class ConstantBuffer;

    /** Low-level program object
        This class abstracts the API's program creation and management
    */
    class ProgramVersion : public std::enable_shared_from_this<ProgramVersion>
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramVersion>;
        using SharedConstPtr = std::shared_ptr<const ProgramVersion>;

        /** Create a new program object for graphics.
            \param[in] pVS Vertex shader object
            \param[in] pPS Fragment shader object
            \param[in] pGS Geometry shader object
            \param[in] pHS Hull shader object
            \param[in] pDS Domain shader object
            \param[out] Log In case of error, this will contain the error log string
            \param[in] DebugName Optional. A meaningful name to use with log messages
            \return New object in case of success, otherwise nullptr
        */
        static SharedPtr create(
            ProgramReflection::SharedPtr const& pReflector,
            const Shader::SharedPtr& pVS,
            const Shader::SharedPtr& pPS,
            const Shader::SharedPtr& pGS,
            const Shader::SharedPtr& pHS,
            const Shader::SharedPtr& pDS,
            std::string& log, 
            const std::string& name = "");

        /** Create a new program object for compute.
            \param[in] pCs Compute shader object
            \param[out] Log In case of error, this will contain the error log string
            \param[in] DebugName Optional. A meaningful name to use with log messages
            \return New object in case of success, otherwise nullptr
        */
        static SharedPtr create(
            ProgramReflection::SharedPtr const& pReflector,
            const Shader::SharedPtr& pCS,
            std::string& log,
            const std::string& name = "");

        virtual ~ProgramVersion();

        /** Get an attached shader object, or nullptr if no shader is attached to the slot.
        */
        const Shader* getShader(ShaderType Type) const { return mpShaders[(uint32_t)Type].get(); }

        /** Get the program name
        */
        const std::string& getName() const {return mName;}

        /** Get the reflection object
        */
        ProgramReflection::SharedConstPtr getReflector() const { return mpReflector; }
    protected:
        ProgramVersion(const Shader::SharedPtr& pVS,
            const Shader::SharedPtr& pPS,
            const Shader::SharedPtr& pGS,
            const Shader::SharedPtr& pHS,
            const Shader::SharedPtr& pDS,
            const Shader::SharedPtr& pCS,
            const std::string& name = "");

        virtual bool init(std::string& log);
        void deleteApiHandle();
        ProgramHandle mApiHandle = ProgramHandle();
        const std::string mName;

        static const uint32_t kShaderCount = (uint32_t)ShaderType::Count;
        Shader::SharedConstPtr mpShaders[kShaderCount];

        ProgramReflection::SharedPtr mpReflector;
        void* mpPrivateData;
    };
}