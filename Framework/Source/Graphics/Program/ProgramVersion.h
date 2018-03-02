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
#include "API/LowLevel/RootSignature.h"

namespace Falcor
{
    class ConstantBuffer;
    class Program;
    class ProgramVars;

    /** Low-level program object
        This class abstracts the API's program creation and management
    */
    class ProgramKernels : public std::enable_shared_from_this<ProgramKernels>
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramKernels>;
        using SharedConstPtr = std::shared_ptr<const ProgramKernels>;

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
            const RootSignature::SharedPtr& pRootSignature,
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
            const RootSignature::SharedPtr& pRootSignature,
            std::string& log,
            const std::string& name = "");

        virtual ~ProgramKernels();

        /** Get an attached shader object, or nullptr if no shader is attached to the slot.
        */
        const Shader* getShader(ShaderType Type) const { return mpShaders[(uint32_t)Type].get(); }

        /** Get the program name
        */
        const std::string& getName() const {return mName;}

        /** Get the reflection object
        */

        ProgramReflection::SharedConstPtr getReflector() const { return mpReflector; }
        
        // SLANG-INTEGRATION
        // move root signature to be a member of ProgramKernels
        RootSignature::SharedPtr mpRootSignature;
        RootSignature::SharedPtr getRootSignature() const
        {
            return mpRootSignature;
        }

    protected:
        ProgramKernels(const Shader::SharedPtr& pVS,
            const Shader::SharedPtr& pPS,
            const Shader::SharedPtr& pGS,
            const Shader::SharedPtr& pHS,
            const Shader::SharedPtr& pDS,
            const Shader::SharedPtr& pCS,
            const RootSignature::SharedPtr& pRootSignature,
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

    class ProgramVersion : public std::enable_shared_from_this<ProgramVersion>
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramVersion>;
        using SharedConstPtr = std::shared_ptr<const ProgramVersion>;
        // SLANG-INTEGRATION:
        // ProgramVersion now holds a SlangCompileRequest
        // to support querying/creating new types and type layouts
        // during ParameterBlock creation for shader components.
        SlangCompileRequest* slangRequest = nullptr;
        using DefineList = Shader::DefineList;

        static SharedPtr create(
            std::shared_ptr<Program>     const& pProgram,
            DefineList                   const& defines,
            ProgramReflection::SharedPtr const& pReflector,
            std::string                  const& name,
            SlangCompileRequest * compileReq);

        /** Get the program that this version was created from
        */
        std::shared_ptr<Program> getProgram() const { return mpProgram; }

        /** Get the defines that were used to create this version
        */
        DefineList const& getDefines() const { return mDefines; }

        /** Get the program name
        */
        const std::string& getName() const {return mName;}

        /** Get the reflection object
        */
        ProgramReflection::SharedConstPtr getReflector() const { return mpReflector; }

        /** Get executable kernels based on state in a `ProgramVars`
        */
        ProgramKernels::SharedConstPtr getKernels(ProgramVars const* pVars) const;

    protected:
        ProgramVersion(
            std::shared_ptr<Program>     const& pProgram,
            DefineList                   const& defines,
            ProgramReflection::SharedPtr const& pReflector,
            std::string                  const& name,
            SlangCompileRequest*         compileReq);

        std::shared_ptr<Program>        mpProgram;
        DefineList                      mDefines;
        ProgramReflection::SharedPtr    mpReflector;
        std::string                     mName;

        // Cached version of compiled kernels for this program version
        mutable std::unordered_map<int, ProgramKernels::SharedPtr> mpKernels;
    };

}