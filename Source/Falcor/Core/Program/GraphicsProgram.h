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
    /** Graphics program. See ComputeProgram to manage compute programs.
    */
    class dlldecl GraphicsProgram : public Program
    {
    public:
        using SharedPtr = std::shared_ptr<GraphicsProgram>;
        using SharedConstPtr = std::shared_ptr<const GraphicsProgram>;

        ~GraphicsProgram() = default;

        /** Create a new graphics program.
            Note that this call merely creates a program object. The actual compilation and link happens at a later time.
            \param[in] desc Description of the source files and entry points to use.
            \param[in] programDefines Optional list of macro definitions to set into the program. The macro definitions will be set on all shader stages.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(const Desc& desc, const Program::DefineList& programDefines = DefineList());

        /** Create a new graphics program from file.
            \param[in] filename Graphics program filename.
            \param[in] vsEntry Vertex shader entry point. If this string is empty (""), it will use a default vertex shader, which transforms and outputs all default vertex attributes.
            \param[in] psEntry Pixel shader entry point
            \param[in] programDefines Optional list of macro definitions to set into the program. The macro definitions will be set on all shader stages.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr createFromFile(const std::string& filename, const std::string& vsEntry, const std::string& psEntry, const DefineList& programDefines = DefineList());

    private:
        GraphicsProgram() = default;
    };
}
