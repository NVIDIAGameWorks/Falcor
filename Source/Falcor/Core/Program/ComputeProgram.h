/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#pragma once
#include "Program.h"

namespace Falcor
{
    /** Compute program. See GraphicsProgram to manage graphics programs.
    */
    class dlldecl ComputeProgram : public Program
    {
    public:
        using SharedPtr = std::shared_ptr<ComputeProgram>;
        using SharedConstPtr = std::shared_ptr<const ComputeProgram>;

        ~ComputeProgram() = default;

        /** Create a new compute program from file.
            Note that this call merely creates a program object. The actual compilation and link happens at a later time.
            \param[in] filename Compute program filename.
            \param[in] csEntry Name of the entry point in the program.
            \param[in] programDefines Optional list of macro definitions to set into the program.
            \param[in] flags Optional program compilation flags.
            \param[in] shaderModel Optional string describing which shader model to use.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr createFromFile(const std::string& filename, const std::string& csEntry, const DefineList& programDefines = DefineList(), Shader::CompilerFlags flags = Shader::CompilerFlags::None, const std::string& shaderModel = "");

        /** Create a new compute program.
            Note that this call merely creates a program object. The actual compilation and link happens at a later time.
            \param[in] desc The program description.
            \param[in] programDefines Optional list of macro definitions to set into the program.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(const Program::Desc& desc, const DefineList& programDefines = DefineList());

    private:
        ComputeProgram() = default;
    };
}
