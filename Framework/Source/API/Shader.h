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
#include <string>
#include <unordered_set>

namespace Falcor
{
    /** Low-level shader object
    This class abstracts the API's shader creation and management
    */

    class Shader : public std::enable_shared_from_this<Shader>
    {
    public:
        using SharedPtr = std::shared_ptr<Shader>;
        using SharedConstPtr = std::shared_ptr<const Shader>;
        using ApiHandle = ShaderHandle;

        struct Blob
        {
            enum class Type
            {
                Undefined,
                String,
                Bytecode
            };
            std::vector<uint8_t> data;
            Type type = Type::Undefined;
        };

        enum class CompilerFlags
        {
            None                  = 0x0,
            TreatWarningsAsErrors = 0x1,
            DumpIntermediates     = 0x2,
        };

        class DefineList : public std::map<std::string, std::string>
        {
        public:
            void add(const std::string& name, const std::string& val = "") { (*this)[name] = val; }
            void remove(const std::string& name) {(*this).erase(name); }
        };

        /** create a shader object
            \param[in] shaderBlog A blob containing the shader code
            \param[in] Type The Type of the shader
            \param[out] log This string will contain the error log message in case shader compilation failed
            \return If success, a new shader object, otherwise nullptr
        */
        static SharedPtr create(const Blob& shaderBlob, ShaderType type, std::string const&  entryPointName, CompilerFlags flags, std::string& log)
        {
            SharedPtr pShader = SharedPtr(new Shader(type));
            return pShader->init(shaderBlob, entryPointName, flags, log) ? pShader : nullptr;
        }

        virtual ~Shader();

        /** Get the API handle.
        */
        ApiHandle getApiHandle() const { return mApiHandle; }

        /** Get the shader Type
        */
        ShaderType getType() const { return mType; }



#ifdef FALCOR_D3D
        ID3DBlobPtr getD3DBlob() const;
        virtual ID3DBlobPtr compile(const Blob& blob, const std::string&  entryPointName, CompilerFlags flags, std::string& errorLog);
#endif

    protected:
        // API handle depends on the shader Type, so it stored be stored as part of the private data
        bool init(const Blob& shaderBlob, const std::string&  entryPointName, CompilerFlags flags, std::string& log);
        Shader(ShaderType Type);
        ShaderType mType;
        ApiHandle mApiHandle;
        void* mpPrivateData = nullptr;
    };
    enum_class_operators(Shader::CompilerFlags);
}